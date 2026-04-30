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
    /// 토크(1ms 폴링)와 가속도(DAQ 비동기 블록)를 하나의 CSV 파일에 동기 기록합니다.
    /// 마스터 클럭 = Stopwatch 기반 1ms 폴링 루프 (AjinCsvLogger와 동일한 방식).
    /// 가속도는 DAQ BlockReceived 콜백에서 UpdateAccel() 을 통해 최신값을 채웁니다.
    ///
    /// CSV 컬럼:
    ///   time_s, Ax0_Trq(%), Ax1_Trq(%), ..., Ax0_x, Ax0_y, Ax0_z, Ax1_x, ..., Op_Ax0, Op_Ax1, ...
    /// </summary>
    public sealed class CombinedCsvLogger : IDisposable
    {
        // ── Windows 멀티미디어 타이머 해상도 ───────────────────────
        [DllImport("winmm.dll")] private static extern uint timeBeginPeriod(uint uPeriod);
        [DllImport("winmm.dll")] private static extern uint timeEndPeriod(uint uPeriod);

        // ── 폴링 주기 ──────────────────────────────────────────────
        public int IntervalMs { get; set; } = 1;

        // ── 외부 주입 ──────────────────────────────────────────────
        private readonly Func<int, double> _getTorque;   // axis → torque(%)
        private readonly Func<int, string> _getAxisOp;   // axis → "Pos" | "Idle"
        private readonly int[]             _axes;
        private readonly Action<string>    _log;

        // ── 최신 가속도 버퍼 (DAQ 콜백에서 채움) ──────────────────
        private readonly object   _accelLock;
        private readonly double[] _latX;
        private readonly double[] _latY;
        private readonly double[] _latZ;
        private readonly bool[]   _accelReady;  // 최초 데이터 수신 여부

        // ── 상태 ──────────────────────────────────────────────────
        private CancellationTokenSource _cts;
        private Task   _task;
        private string _filePath;
        private bool   _disposed;

        public bool   IsLogging  => _task != null && !_task.IsCompleted;
        public string OutputPath => _filePath;

        // ── TorqueSampled 이벤트 (InfluxDB 연동용) ────────────────
        public Action<string, int, double, DateTime> TorqueSampled;

        public CombinedCsvLogger(
            Func<int, double> getTorque,
            Func<int, string> getAxisOp,
            int[]             axes,
            Action<string>    log = null)
        {
            _getTorque  = getTorque ?? throw new ArgumentNullException(nameof(getTorque));
            _getAxisOp  = getAxisOp;
            _axes       = axes ?? new int[0];
            _log        = log ?? (_ => { });

            int n = _axes.Length;
            _accelLock  = new object();
            _latX       = new double[n];
            _latY       = new double[n];
            _latZ       = new double[n];
            _accelReady = new bool[n];
        }

        // ── DAQ BlockReceived 에서 호출 ────────────────────────────
        /// <param name="axisIdx">_axes 배열 내 인덱스 (모듈 순서 = 축 순서)</param>
        public void UpdateAccel(int axisIdx, double x, double y, double z)
        {
            if (axisIdx < 0 || axisIdx >= _latX.Length) return;
            lock (_accelLock)
            {
                _latX[axisIdx]       = x;
                _latY[axisIdx]       = y;
                _latZ[axisIdx]       = z;
                _accelReady[axisIdx] = true;
            }
        }

        // ── 시작 ──────────────────────────────────────────────────
        public bool Start(string dir, string baseName)
        {
            if (IsLogging) return false;
            try
            {
                Directory.CreateDirectory(dir);
                _filePath = Path.Combine(dir, baseName + "_Combined.csv");
                _cts      = new CancellationTokenSource();
                _task     = Task.Run(() => PollLoop(_cts.Token), _cts.Token);
                return true;
            }
            catch (Exception ex)
            {
                _log($"[CombinedLog] 시작 실패: {ex.Message}");
                return false;
            }
        }

        // ── 정지 ──────────────────────────────────────────────────
        public void Stop()
        {
            if (!IsLogging) return;
            try { _cts?.Cancel(); _task?.Wait(2000); }
            catch { }
            finally { _cts?.Dispose(); _cts = null; _task = null; }
        }

        // ── 1ms 폴링 루프 ──────────────────────────────────────────
        private void PollLoop(CancellationToken token)
        {
            // 헤더 구성:
            //   time_s, Ax0_Trq(%), ..., Ax0_x, Ax0_y, Ax0_z, ..., Op_Ax0, ...
            var hdr = new StringBuilder("time_s");
            foreach (int ax in _axes)
                hdr.Append($",Ax{ax}_Trq(%)");
            foreach (int ax in _axes)
            {
                hdr.Append($",Ax{ax}_x");
                hdr.Append($",Ax{ax}_y");
                hdr.Append($",Ax{ax}_z");
            }
            foreach (int ax in _axes)
                hdr.Append($",Op_Ax{ax}");

            var sw = Stopwatch.StartNew();
            long ticksPerInterval = (long)Math.Round(IntervalMs * 0.001 * Stopwatch.Frequency);
            long ticksPer1ms      = Stopwatch.Frequency / 1000;
            long nextTick         = sw.ElapsedTicks + ticksPerInterval;

            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            timeBeginPeriod(1);
            try
            {
                using (var writer = new StreamWriter(
                    _filePath, false, Encoding.UTF8, bufferSize: 65536))
                {
                    writer.WriteLine(hdr.ToString());

                    while (!token.IsCancellationRequested)
                    {
                        double t_s = (double)sw.ElapsedTicks / Stopwatch.Frequency;
                        var line   = new StringBuilder(128);
                        line.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                        // 토크
                        DateTime now = DateTime.UtcNow;
                        for (int i = 0; i < _axes.Length; i++)
                        {
                            int    ax  = _axes[i];
                            double trq = SafeGet(_getTorque, ax);
                            line.Append(",").Append(trq.ToString("F4", CultureInfo.InvariantCulture));
                            TorqueSampled?.Invoke("Combined", ax, trq, now);
                        }

                        // 가속도 (최신 스냅샷)
                        lock (_accelLock)
                        {
                            for (int i = 0; i < _axes.Length; i++)
                            {
                                line.Append(",").Append(_latX[i].ToString("G6", CultureInfo.InvariantCulture));
                                line.Append(",").Append(_latY[i].ToString("G6", CultureInfo.InvariantCulture));
                                line.Append(",").Append(_latZ[i].ToString("G6", CultureInfo.InvariantCulture));
                            }
                        }

                        // 동작 상태
                        for (int i = 0; i < _axes.Length; i++)
                        {
                            string op = _getAxisOp?.Invoke(_axes[i]) ?? "Pos";
                            line.Append(",").Append(op);
                        }

                        writer.WriteLine(line.ToString());

                        // 고정 인터벌 대기
                        long rem;
                        while ((rem = nextTick - sw.ElapsedTicks) > 0)
                        {
                            if (rem > ticksPer1ms) Thread.Sleep(1);
                        }
                        nextTick += ticksPerInterval;
                    }
                    writer.Flush();
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { _log($"[CombinedLog] 폴링 오류: {ex.Message}"); }
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
