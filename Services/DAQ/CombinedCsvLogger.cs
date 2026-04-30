using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.DAQ
{
    /// <summary>
    /// 토크(1ms 폴링)와 가속도(DAQ 블록)를 원래 해상도로 단일 CSV에 기록합니다.
    ///
    /// ▶ 두 신호 해상도 동시 보존 방법:
    ///   1. 토크 폴링 루프(1ms)가 링 버퍼에 (Stopwatch 틱, torque[]) 를 연속 저장
    ///   2. 가속도 블록 도착 시 각 샘플의 Stopwatch 추정 타임스탬프를 계산
    ///   3. 투 포인터로 링 버퍼를 순회 → 각 Accel 샘플에 시간적으로 가장 가까운 토크 매핑
    ///   → Accel 원해상도 + Torque ≈1ms 해상도 모두 보존
    ///
    /// CSV 컬럼:
    ///   time_s, Ax0_Trq(%), ..., Ax0_x, Ax0_y, Ax0_z, ..., Op_Ax0, ...
    /// </summary>
    public sealed class CombinedCsvLogger : IDisposable
    {
        [DllImport("winmm.dll")] private static extern uint timeBeginPeriod(uint uPeriod);
        [DllImport("winmm.dll")] private static extern uint timeEndPeriod(uint uPeriod);

        // ── 외부 주입 ──────────────────────────────────────────────
        private readonly Func<int, double> _getTorque;
        private readonly Func<int, string> _getAxisOp;
        private readonly int[]             _axes;
        private readonly Action<string>    _log;

        // ── 토크 링 버퍼 ───────────────────────────────────────────
        // 공유 Stopwatch 기준으로 각 폴링 시점을 기록
        // 크기 = 8192: 1kHz 기준 ≈8초, 빠른 샘플레이트에도 여유
        private const int RING = 8192;
        private readonly double[] _tBuf;     // [ringIdx * nAxes + axIdx]
        private readonly long[]   _tTick;    // 각 폴링의 Stopwatch 틱
        private long _tTotal = 0;            // 누적 폴링 횟수 (volatile-like, lock 보호)
        private readonly object _tLock = new object();

        // ── Accel 샘플 카운터 (타임스탬프 계산용) ─────────────────
        private readonly long[] _accelSamples;  // [axisArrayIdx]

        // ── 다축: 이 모듈 외 다른 축의 최신 가속도 ────────────────
        private readonly double[,] _latestAccel; // [axisArrayIdx, 0/1/2]

        // ── 파일 ──────────────────────────────────────────────────
        private StreamWriter _writer;
        private readonly object _writerLock = new object();
        private string _filePath;

        // ── 공유 Stopwatch ────────────────────────────────────────
        private Stopwatch _sw;

        // ── 상태 ──────────────────────────────────────────────────
        private CancellationTokenSource _cts;
        private Task _pollTask;
        private bool _running;
        private bool _disposed;

        public bool   IsLogging  => _running;
        public string OutputPath => _filePath;

        public CombinedCsvLogger(
            Func<int, double> getTorque,
            Func<int, string> getAxisOp,
            int[]             axes,
            Action<string>    log = null)
        {
            _getTorque = getTorque ?? throw new ArgumentNullException(nameof(getTorque));
            _getAxisOp = getAxisOp;
            _axes      = axes ?? new int[0];
            _log       = log ?? (_ => { });

            int n = _axes.Length;
            _tBuf        = new double[RING * n];
            _tTick       = new long[RING];
            _accelSamples = new long[n];
            _latestAccel  = new double[n, 3];
        }

        // ── 시작 ──────────────────────────────────────────────────
        public bool Start(string dir, string baseName)
        {
            if (_running) return false;
            try
            {
                Directory.CreateDirectory(dir);
                _filePath = Path.Combine(dir, baseName + "_Combined.csv");

                var hdr = new StringBuilder("time_s");
                foreach (int ax in _axes) hdr.Append($",Ax{ax}_Trq(%)");
                foreach (int ax in _axes)
                {
                    hdr.Append($",Ax{ax}_x");
                    hdr.Append($",Ax{ax}_y");
                    hdr.Append($",Ax{ax}_z");
                }
                foreach (int ax in _axes) hdr.Append($",Op_Ax{ax}");

                _writer = new StreamWriter(
                    _filePath, false, Encoding.UTF8, bufferSize: 65536);
                _writer.WriteLine(hdr.ToString());

                _sw = Stopwatch.StartNew();

                _cts      = new CancellationTokenSource();
                _pollTask = Task.Run(() => TorquePollLoop(_cts.Token), _cts.Token);

                _running = true;
                return true;
            }
            catch (Exception ex)
            {
                _log($"[CombinedLog] 시작 실패: {ex.Message}");
                _writer?.Dispose();
                _writer = null;
                return false;
            }
        }

        // ── 정지 ──────────────────────────────────────────────────
        public void Stop()
        {
            if (!_running) return;
            _running = false;

            try { _cts?.Cancel(); _pollTask?.Wait(2000); }
            catch { }
            finally { _cts?.Dispose(); _cts = null; _pollTask = null; }

            lock (_writerLock)
            {
                try { _writer?.Flush(); } catch { }
                try { _writer?.Dispose(); } catch { }
                _writer = null;
            }
        }

        // ── 가속도 블록 처리 (DAQ BlockReceived에서 호출) ──────────
        /// <summary>
        /// 블록 내 N개 샘플 각각에 대해 시간적으로 가장 가까운 토크를
        /// 링 버퍼에서 찾아 한 행씩 기록합니다.
        /// </summary>
        public void ProcessAccelBlock(int modIdx, double[,] block, int n, double sampleRate)
        {
            if (!_running || _writer == null || n <= 0) return;
            if (modIdx < 0 || modIdx >= _axes.Length) return;
            if (sampleRate <= 0) sampleRate = 1000.0;

            // 이 블록 최신 샘플 → 다축 보정용 버퍼 갱신
            _latestAccel[modIdx, 0] = block[0, n - 1];
            _latestAccel[modIdx, 1] = block[1, n - 1];
            _latestAccel[modIdx, 2] = block[2, n - 1];

            // 블록 도착 시각 (공유 Stopwatch 기준)
            long blockArrivalTick = _sw.ElapsedTicks;
            double ticksPerSample = (double)Stopwatch.Frequency / sampleRate;

            // 토크 링 버퍼 스냅샷 (이 블록 처리에 필요한 범위만)
            long tTotal;
            lock (_tLock) { tTotal = _tTotal; }
            int tAvail = (int)Math.Min(tTotal, RING);
            if (tAvail == 0)
            {
                // 토크 데이터 아직 없음 → 0으로 채움
                WriteBlockNoTorque(modIdx, block, n, sampleRate, blockArrivalTick, ticksPerSample);
                return;
            }

            // Op 스냅샷
            string[] opSnap = new string[_axes.Length];
            for (int i = 0; i < _axes.Length; i++)
                opSnap[i] = _getAxisOp?.Invoke(_axes[i]) ?? "Pos";

            int nAxes    = _axes.Length;
            long baseCount = _accelSamples[modIdx];

            // ── 투 포인터 매핑: Accel 샘플(오름차순) ↔ 토크 링(오름차순) ──
            // 링 버퍼에서 유효한 가장 오래된 인덱스부터 시작
            long riStart = Math.Max(0, tTotal - tAvail);
            long ri = riStart;

            lock (_writerLock)
            {
                if (_writer == null) return;
                var sb = new StringBuilder(128);

                for (int i = 0; i < n; i++)
                {
                    // 이 샘플의 추정 타임스탬프
                    long sampleTick = blockArrivalTick - (long)((n - 1 - i) * ticksPerSample);

                    // ri 전진: 다음 항목이 현재보다 sampleTick에 더 가까우면 전진
                    while (ri + 1 < tTotal)
                    {
                        int ci = (int)(ri       % RING);
                        int ni = (int)((ri + 1) % RING);
                        long dCurr = Abs64(_tTick[ci] - sampleTick);
                        long dNext = Abs64(_tTick[ni] - sampleTick);
                        if (dNext <= dCurr) ri++;
                        else break;
                    }

                    int rIdx = (int)(ri % RING);

                    // 한 행 구성
                    double t_s = (baseCount + i) / sampleRate;
                    sb.Clear();
                    sb.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                    // 토크 (링 버퍼에서)
                    int tBase = rIdx * nAxes;
                    for (int ai = 0; ai < nAxes; ai++)
                        sb.Append(",").Append(_tBuf[tBase + ai].ToString("F4", CultureInfo.InvariantCulture));

                    // 가속도 (이 모듈: 현재 샘플 / 다른 모듈: 최신 버퍼)
                    for (int ai = 0; ai < nAxes; ai++)
                    {
                        if (ai == modIdx)
                        {
                            sb.Append(",").Append(block[0, i].ToString("G6", CultureInfo.InvariantCulture));
                            sb.Append(",").Append(block[1, i].ToString("G6", CultureInfo.InvariantCulture));
                            sb.Append(",").Append(block[2, i].ToString("G6", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            sb.Append(",").Append(_latestAccel[ai, 0].ToString("G6", CultureInfo.InvariantCulture));
                            sb.Append(",").Append(_latestAccel[ai, 1].ToString("G6", CultureInfo.InvariantCulture));
                            sb.Append(",").Append(_latestAccel[ai, 2].ToString("G6", CultureInfo.InvariantCulture));
                        }
                    }

                    // Op
                    for (int ai = 0; ai < nAxes; ai++)
                        sb.Append(",").Append(opSnap[ai]);

                    _writer.WriteLine(sb.ToString());
                }
            }

            _accelSamples[modIdx] += n;
        }

        // 토크 링 버퍼 미준비 시 0으로 채운 행 기록
        private void WriteBlockNoTorque(
            int modIdx, double[,] block, int n,
            double sampleRate, long blockArrivalTick, double ticksPerSample)
        {
            int nAxes = _axes.Length;
            long baseCount = _accelSamples[modIdx];

            lock (_writerLock)
            {
                if (_writer == null) return;
                var sb = new StringBuilder(128);
                string op = _getAxisOp?.Invoke(_axes[modIdx]) ?? "Pos";

                for (int i = 0; i < n; i++)
                {
                    double t_s = (baseCount + i) / sampleRate;
                    sb.Clear();
                    sb.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));
                    for (int ai = 0; ai < nAxes; ai++) sb.Append(",0");
                    for (int ai = 0; ai < nAxes; ai++)
                    {
                        if (ai == modIdx)
                        {
                            sb.Append(",").Append(block[0, i].ToString("G6", CultureInfo.InvariantCulture));
                            sb.Append(",").Append(block[1, i].ToString("G6", CultureInfo.InvariantCulture));
                            sb.Append(",").Append(block[2, i].ToString("G6", CultureInfo.InvariantCulture));
                        }
                        else { sb.Append(",0,0,0"); }
                    }
                    for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(op);
                    _writer.WriteLine(sb.ToString());
                }
            }
            _accelSamples[modIdx] += n;
        }

        // ── 토크 전용 1ms 폴링 루프 ───────────────────────────────
        private void TorquePollLoop(CancellationToken token)
        {
            int nAxes = _axes.Length;
            long ticksPerMs  = Stopwatch.Frequency / 1000;
            long nextTick    = _sw.ElapsedTicks + ticksPerMs;

            Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
            timeBeginPeriod(1);
            try
            {
                while (!token.IsCancellationRequested)
                {
                    long tNow;
                    int  wi;
                    lock (_tLock)
                    {
                        wi   = (int)(_tTotal % RING);
                        tNow = _sw.ElapsedTicks;
                        _tTick[wi] = tNow;
                        for (int ai = 0; ai < nAxes; ai++)
                            _tBuf[wi * nAxes + ai] = SafeGet(_getTorque, _axes[ai]);
                        _tTotal++;
                    }

                    long rem;
                    while ((rem = nextTick - _sw.ElapsedTicks) > 0)
                    {
                        if (rem > ticksPerMs / 2) Thread.Sleep(1);
                    }
                    nextTick += ticksPerMs;
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { _log($"[CombinedLog] 토크 폴링 오류: {ex.Message}"); }
            finally { timeEndPeriod(1); }
        }

        private static long Abs64(long v) { return v < 0 ? -v : v; }

        private static double SafeGet(Func<int, double> fn, int ax)
        {
            try { return fn(ax); }
            catch { return double.NaN; }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Stop();
            _disposed = true;
        }
    }
}
