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
    /// 토크(1ms 폴링 버퍼)와 가속도(DAQ 블록 기반)를 단일 CSV에 기록합니다.
    ///
    /// ▶ 마스터 클럭 = DAQ 가속도 블록 (accel-master 방식)
    ///   - 1ms 토크 폴링 루프는 최신 토크값만 메모리에 저장 (파일 I/O 없음)
    ///   - ProcessAccelBlock() 이 호출될 때마다 블록 내 N개 샘플 전부 즉시 기록
    ///   → 가속도 원래 해상도 완전 보존
    ///   → 토크는 최신 폴링값으로 각 샘플 행에 채움
    ///
    /// CSV 컬럼:
    ///   time_s, Ax0_Trq(%), Ax1_Trq(%), ..., Ax0_x, Ax0_y, Ax0_z, Ax1_x, ..., Op_Ax0, ...
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

        // ── 토크 버퍼 (1ms 폴링 루프가 채움, 파일 I/O 없음) ───────
        private readonly double[] _latestTorque;   // [axisArrayIdx]
        private readonly object   _torqueLock = new object();

        // ── 가속도 최신값 (다축일 때 다른 축 행 채우기용) ─────────
        private readonly double[,] _latestAccel;   // [axisArrayIdx, 0=x/1=y/2=z]

        // ── 샘플 카운터 (축별 타임스탬프 계산용) ──────────────────
        private readonly long[] _totalSamples;     // [axisArrayIdx]

        // ── 파일 ──────────────────────────────────────────────────
        private StreamWriter _writer;
        private readonly object _writerLock = new object();
        private string _filePath;

        // ── 토크 폴링 태스크 ──────────────────────────────────────
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
            _latestTorque  = new double[n];
            _latestAccel   = new double[n, 3];
            _totalSamples  = new long[n];
        }

        // ── 시작 ──────────────────────────────────────────────────
        public bool Start(string dir, string baseName)
        {
            if (_running) return false;
            try
            {
                Directory.CreateDirectory(dir);
                _filePath = Path.Combine(dir, baseName + "_Combined.csv");

                // 헤더 구성: time_s, Ax{n}_Trq(%), ..., Ax{n}_x/y/z, ..., Op_Ax{n}, ...
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

                // 토크 전용 1ms 폴링 태스크 시작
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

        // ── 가속도 블록 처리 (DAQ BlockReceived 에서 호출) ────────
        /// <summary>
        /// DAQ 콜백에서 전달된 블록 전체를 CSV에 씁니다.
        /// 블록 안의 N개 샘플이 모두 원본 해상도로 기록됩니다.
        /// </summary>
        /// <param name="modIdx">_axes 배열 내 모듈 인덱스 (축 순서)</param>
        /// <param name="block">shape [3, N] — [0]=x, [1]=y, [2]=z</param>
        /// <param name="n">유효 샘플 수</param>
        /// <param name="sampleRate">DAQ 샘플레이트 (Hz)</param>
        public void ProcessAccelBlock(int modIdx, double[,] block, int n, double sampleRate)
        {
            if (!_running || _writer == null || n <= 0) return;
            if (modIdx < 0 || modIdx >= _axes.Length) return;
            if (sampleRate <= 0) sampleRate = 1000.0;

            // 이 블록의 최신 샘플을 다축 보정용 버퍼에 저장
            _latestAccel[modIdx, 0] = block[0, n - 1];
            _latestAccel[modIdx, 1] = block[1, n - 1];
            _latestAccel[modIdx, 2] = block[2, n - 1];

            // 토크 스냅샷 (블록 단위로 한 번만)
            double[] torqueSnap = new double[_axes.Length];
            lock (_torqueLock)
            {
                for (int i = 0; i < _axes.Length; i++)
                    torqueSnap[i] = _latestTorque[i];
            }

            // Op 스냅샷
            string[] opSnap = new string[_axes.Length];
            for (int i = 0; i < _axes.Length; i++)
                opSnap[i] = _getAxisOp?.Invoke(_axes[i]) ?? "Pos";

            long baseCount = _totalSamples[modIdx];

            lock (_writerLock)
            {
                if (_writer == null) return;
                var sb = new StringBuilder(128);

                for (int i = 0; i < n; i++)
                {
                    double t = (baseCount + i) / sampleRate;
                    sb.Clear();
                    sb.Append(t.ToString("F6", CultureInfo.InvariantCulture));

                    // 토크 (모든 축)
                    for (int ai = 0; ai < _axes.Length; ai++)
                        sb.Append(",").Append(torqueSnap[ai].ToString("F4", CultureInfo.InvariantCulture));

                    // 가속도: 이 모듈은 현재 샘플, 다른 모듈은 최신 버퍼값
                    for (int ai = 0; ai < _axes.Length; ai++)
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
                    for (int ai = 0; ai < _axes.Length; ai++)
                        sb.Append(",").Append(opSnap[ai]);

                    _writer.WriteLine(sb.ToString());
                }
            }

            _totalSamples[modIdx] += n;
        }

        // ── 토크 전용 1ms 폴링 루프 (파일 I/O 없음) ───────────────
        private void TorquePollLoop(CancellationToken token)
        {
            var sw = Stopwatch.StartNew();
            long ticksPerMs  = Stopwatch.Frequency / 1000;
            long nextTick    = sw.ElapsedTicks + ticksPerMs;

            Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
            timeBeginPeriod(1);
            try
            {
                while (!token.IsCancellationRequested)
                {
                    // 토크만 폴링, 파일에는 안 씀
                    lock (_torqueLock)
                    {
                        for (int i = 0; i < _axes.Length; i++)
                            _latestTorque[i] = SafeGet(_getTorque, _axes[i]);
                    }

                    long rem;
                    while ((rem = nextTick - sw.ElapsedTicks) > 0)
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
