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
    /// 토크(1ms 폴링)와 가속도(DAQ 블록, 단일 센서)를 원래 해상도로 단일 CSV에 기록합니다.
    ///
    /// ▶ 두 신호 해상도 동시 보존 방법:
    ///   1. 폴링 루프(1ms)가 링 버퍼에 (Stopwatch 틱, torque[], op[]) 를 연속 저장
    ///   2. 가속도 블록 도착 시 각 샘플의 Stopwatch 추정 타임스탬프를 계산
    ///   3. 투 포인터로 링 버퍼를 순회 → 각 Accel 샘플에 가장 가까운 토크/Op 매핑
    ///   → Accel 원해상도 + Torque/Op ≈1ms 해상도 정합
    ///
    /// CSV 컬럼:
    ///   time_s, Ax0_Trq(%), Ax1_Trq(%), ..., x, y, z, Op_Ax0, Op_Ax1, ...
    ///
    /// ▶ 가속도는 물리적으로 단일 센서 → x, y, z 3열만 기록 (축별 구분 없음)
    /// ▶ 토크와 Op는 모터별로 별도 열 유지
    /// ▶ Op 상태도 링 버퍼에 저장하여 블록 지연(≈ N/sampleRate 초) 보정
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
        /// <summary>제어기 연결 시 true → Op_Ax 컬럼 포함. 미연결 시 false → Op_Ax 컬럼 생략.</summary>
        private readonly bool              _hasOpColumns;

        // ── 링 버퍼 (토크 + Op 동시 저장) ────────────────────────
        // 크기 = 8192: 1kHz 기준 ≈8초, 빠른 샘플레이트에도 여유
        private const int RING = 8192;
        private readonly double[] _tBuf;   // [ringIdx * nAxes + axIdx] 토크
        private readonly string[] _opBuf;  // [ringIdx * nAxes + axIdx] Op 상태 (intern)
        private readonly long[]   _tTick;  // 각 폴링의 Stopwatch 틱
        private long _tTotal = 0;
        private readonly object _tLock = new object();

        // ── Accel 샘플 카운터 (타임스탬프 계산용, 단일 센서) ──────
        private long _accelTotal = 0;

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

        /// <summary>
        /// 토크 폴링 루프가 각 샘플을 기록할 때 추가로 호출할 콜백.
        /// 파라미터: (axisIndex, torqueValue, sampleTimeUtc)
        /// InfluxDB 피드 등 외부 처리에 사용.
        /// </summary>
        public Action<int, double, DateTime> TorqueSampled { get; set; }

        public CombinedCsvLogger(
            Func<int, double> getTorque,
            Func<int, string> getAxisOp,
            int[]             axes,
            Action<string>    log = null)
        {
            _getTorque    = getTorque ?? throw new ArgumentNullException(nameof(getTorque));
            _getAxisOp    = getAxisOp;
            _axes         = axes ?? new int[0];
            _log          = log ?? (_ => { });
            _hasOpColumns = (getAxisOp != null);   // 제어기 연결 여부

            int n  = _axes.Length;
            _tBuf  = new double[RING * n];
            _opBuf = new string[RING * n];
            _tTick = new long[RING];

            // 초기값: "Pos" (intern으로 GC 부담 최소화)
            for (int i = 0; i < _opBuf.Length; i++)
                _opBuf[i] = string.Intern("Pos");
        }

        // ── 시작 ──────────────────────────────────────────────────
        public bool Start(string dir, string baseName)
        {
            if (_running) return false;
            try
            {
                Directory.CreateDirectory(dir);
                _filePath = Path.Combine(dir, baseName + "_Combined.csv");

                // 헤더: 토크(축별) | 가속도(단일 센서 x/y/z) | Op(축별, 제어기 연결 시만)
                var hdr = new StringBuilder("time_s");
                foreach (int ax in _axes) hdr.Append($",Ax{ax}_Trq(%)");
                hdr.Append(",x,y,z");
                if (_hasOpColumns)
                    foreach (int ax in _axes) hdr.Append($",Op_Ax{ax}");

                _writer = new StreamWriter(
                    _filePath, false, Encoding.UTF8, bufferSize: 65536);
                _writer.WriteLine(hdr.ToString());

                _sw = Stopwatch.StartNew();

                _cts      = new CancellationTokenSource();
                _pollTask = Task.Run(() => TorqueOpPollLoop(_cts.Token), _cts.Token);

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
        /// 단일 가속도 센서의 블록 N개 샘플 각각에 대해
        /// 시간적으로 가장 가까운 토크/Op를 링 버퍼에서 찾아 한 행씩 기록합니다.
        /// </summary>
        public void ProcessAccelBlock(int modIdx, double[,] block, int n, double sampleRate)
        {
            if (!_running || _writer == null || n <= 0) return;
            if (sampleRate <= 0) sampleRate = 1000.0;

            // 블록 도착 시각 (공유 Stopwatch 기준)
            long blockArrivalTick = _sw.ElapsedTicks;
            double ticksPerSample = (double)Stopwatch.Frequency / sampleRate;

            // 링 버퍼 현재 크기 스냅샷
            long tTotal;
            lock (_tLock) { tTotal = _tTotal; }
            int tAvail = (int)Math.Min(tTotal, RING);
            if (tAvail == 0)
            {
                // 링 버퍼 미준비: 현재 상태로 채움
                WriteBlockNoRingData(block, n, sampleRate);
                return;
            }

            int nAxes = _axes.Length;
            long baseCount = _accelTotal;

            // ── 투 포인터 매핑: Accel 샘플(오름차순) ↔ 링 버퍼(오름차순) ──
            // 블록 첫 샘플 타임스탬프 = 도착 시각 - (n-1) * ticksPerSample
            // 링에서 해당 시각에 가장 가까운 폴링 위치부터 시작
            long riStart = Math.Max(0, tTotal - tAvail);
            long ri = riStart;

            lock (_writerLock)
            {
                if (_writer == null) return;
                var sb = new StringBuilder(128);

                for (int i = 0; i < n; i++)
                {
                    // 이 샘플의 추정 타임스탬프 (Stopwatch 틱)
                    long sampleTick = blockArrivalTick - (long)((n - 1 - i) * ticksPerSample);

                    // ri 전진: 다음 항목이 sampleTick에 더 가까우면 전진
                    while (ri + 1 < tTotal)
                    {
                        int ci = (int)(ri       % RING);
                        int ni = (int)((ri + 1) % RING);
                        long dCurr = Abs64(_tTick[ci] - sampleTick);
                        long dNext = Abs64(_tTick[ni] - sampleTick);
                        if (dNext <= dCurr) ri++;
                        else break;
                    }

                    int rIdx  = (int)(ri % RING);
                    int rBase = rIdx * nAxes;

                    // time_s
                    double t_s = (baseCount + i) / sampleRate;
                    sb.Clear();
                    sb.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                    // 토크: 축별 (링 버퍼 → 해당 시점의 값)
                    for (int ai = 0; ai < nAxes; ai++)
                        sb.Append(",").Append(_tBuf[rBase + ai].ToString("F4", CultureInfo.InvariantCulture));

                    // 가속도: 단일 센서 x, y, z
                    sb.Append(",").Append(block[0, i].ToString("G6", CultureInfo.InvariantCulture));
                    sb.Append(",").Append(block[1, i].ToString("G6", CultureInfo.InvariantCulture));
                    sb.Append(",").Append(block[2, i].ToString("G6", CultureInfo.InvariantCulture));

                    // Op: 축별 (링 버퍼 → 해당 시점의 상태) — 제어기 연결 시만 기록
                    if (_hasOpColumns)
                        for (int ai = 0; ai < nAxes; ai++)
                            sb.Append(",").Append(_opBuf[rBase + ai] ?? "Pos");

                    _writer.WriteLine(sb.ToString());
                }
            }

            _accelTotal += n;
        }

        // 링 버퍼 미준비(수집 시작 직후 첫 블록) 시 현재 상태로 채운 행 기록
        private void WriteBlockNoRingData(double[,] block, int n, double sampleRate)
        {
            int nAxes = _axes.Length;
            long baseCount = _accelTotal;

            // 현재 Op 상태 스냅샷 (링 버퍼 없으므로 할 수 없음, 현재값 사용)
            string[] opNow = new string[nAxes];
            for (int ai = 0; ai < nAxes; ai++)
                opNow[ai] = SafeGetOp(_getAxisOp, _axes[ai]);

            lock (_writerLock)
            {
                if (_writer == null) return;
                var sb = new StringBuilder(128);

                for (int i = 0; i < n; i++)
                {
                    double t_s = (baseCount + i) / sampleRate;
                    sb.Clear();
                    sb.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                    // 토크: 0 채움 (링 버퍼 미준비)
                    for (int ai = 0; ai < nAxes; ai++) sb.Append(",0");

                    // 가속도: 단일 센서 x, y, z
                    sb.Append(",").Append(block[0, i].ToString("G6", CultureInfo.InvariantCulture));
                    sb.Append(",").Append(block[1, i].ToString("G6", CultureInfo.InvariantCulture));
                    sb.Append(",").Append(block[2, i].ToString("G6", CultureInfo.InvariantCulture));

                    // Op: 현재 상태 — 제어기 연결 시만 기록
                    if (_hasOpColumns)
                        for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(opNow[ai]);

                    _writer.WriteLine(sb.ToString());
                }
            }
            _accelTotal += n;
        }

        // ── 토크 + Op 1ms 폴링 루프 ──────────────────────────────
        private void TorqueOpPollLoop(CancellationToken token)
        {
            int nAxes = _axes.Length;
            long ticksPerMs = Stopwatch.Frequency / 1000;
            long nextTick   = _sw.ElapsedTicks + ticksPerMs;

            Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
            timeBeginPeriod(1);
            try
            {
                while (!token.IsCancellationRequested)
                {
                    long tNow;
                    int  wi;
                    double[] torqueSnap = null;
                    lock (_tLock)
                    {
                        wi   = (int)(_tTotal % RING);
                        tNow = _sw.ElapsedTicks;
                        _tTick[wi] = tNow;
                        if (TorqueSampled != null) torqueSnap = new double[nAxes];
                        int wBase = wi * nAxes;
                        for (int ai = 0; ai < nAxes; ai++)
                        {
                            double v = SafeGet(_getTorque, _axes[ai]);
                            _tBuf[wBase + ai] = v;
                            if (torqueSnap != null) torqueSnap[ai] = v;

                            // Op 상태도 동일 링 슬롯에 저장 (intern으로 중복 할당 방지)
                            _opBuf[wBase + ai] = string.Intern(SafeGetOp(_getAxisOp, _axes[ai]));
                        }
                        _tTotal++;
                    }

                    // InfluxDB 등 외부 콜백 (lock 밖에서 호출)
                    if (torqueSnap != null)
                    {
                        var cb     = TorqueSampled;
                        var utcNow = DateTime.UtcNow;
                        if (cb != null)
                            for (int ai = 0; ai < nAxes; ai++)
                                try { cb(_axes[ai], torqueSnap[ai], utcNow); } catch { }
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
            catch (Exception ex) { _log($"[CombinedLog] 폴링 오류: {ex.Message}"); }
            finally { timeEndPeriod(1); }
        }

        private static long Abs64(long v) => v < 0 ? -v : v;

        private static double SafeGet(Func<int, double> fn, int ax)
        {
            try { return fn(ax); }
            catch { return double.NaN; }
        }

        private static string SafeGetOp(Func<int, string> fn, int ax)
        {
            try { return fn?.Invoke(ax) ?? "Pos"; }
            catch { return "Pos"; }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Stop();
            _disposed = true;
        }
    }
}
