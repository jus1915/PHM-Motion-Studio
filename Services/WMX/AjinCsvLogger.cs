using PHM_Project_DockPanel.Services;
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.WMX
{
    /// <summary>
    /// Ajin AMP 전용 CSV 로거.
    /// SDK 로그 API가 없으므로 폴링 방식으로 기록합니다.
    /// 속도는 AxmStatusReadActVel 대신 위치 차분으로 계산합니다 (단위 보장).
    /// 기록 항목: Timestamp(ms), Pos(mm), Vel(mm/s), Torque(%)
    /// </summary>
    public class AjinCsvLogger : IDisposable
    {
        // ── Windows 멀티미디어 타이머 해상도 (winmm.dll) ───────────
        [DllImport("winmm.dll")] private static extern uint timeBeginPeriod(uint uPeriod);
        [DllImport("winmm.dll")] private static extern uint timeEndPeriod(uint uPeriod);

        // ── 폴링 주기 ──────────────────────────────────────────────
        public int IntervalMs { get; set; } = 1;   // 기본 1ms (1000 Hz)

        // ── 외부 주입 ──────────────────────────────────────────────
        private readonly Func<int, double> _getPos;     // axis → actual pos(mm)
        private readonly Func<int, double> _getVel;     // axis → actual vel(mm/s), null이면 위치 차분으로 계산
        private readonly Func<int, double> _getTorque;  // axis → torque(%)
        private readonly Func<int, double> _getCmdPos;  // axis → command pos(mm), null이면 기록 안 함
        private readonly Action<string> _log;
        private readonly string _fileSuffix;            // CSV 파일명 접미사 (예: "AjinMotion", "Simulator")

        // ── 상태 ──────────────────────────────────────────────────
        private CancellationTokenSource _cts;
        private Task _task;
        private bool _disposed;
        private int[] _axes;
        private string _filePath;

        public bool IsLogging => _task != null && !_task.IsCompleted;
        public string OutputPath => _filePath;

        /// <summary>외부(CombinedCsvLogger 등)에서 토크값을 폴링할 수 있도록 공개합니다.</summary>
        public double ReadTorque(int ax) => SafeGet(_getTorque, ax);

        /// <summary>장치 식별자 (InfluxDB device 태그). 미설정 시 _fileSuffix 사용.</summary>
        public string Device { get; set; }

        /// <summary>
        /// 각 폴링 샘플마다 호출됩니다: (device, axis, fbtrq%, timestampUtc).
        /// InfluxDB 실시간 토크 게시에 사용.
        /// </summary>
        public Action<string, int, double, DateTime> TorqueSampled;
        /// <summary>異??몃뜳?ㅻ? 諛쏆븘 "Idle" ?먮뒗 "Pos"瑜?諛섑솚. null?대㈃ "Pos"濡?媛꾩＜.</summary>
        public Func<int, string> GetAxisOperation { get; set; }

        public AjinCsvLogger(
            Func<int, double> getPos,
            Func<int, double> getTorque,
            Action<string> log = null,
            Func<int, double> getVel = null,
            Func<int, double> getCmdPos = null,
            string fileSuffix = "AjinMotion")
        {
            _getPos    = getPos    ?? throw new ArgumentNullException(nameof(getPos));
            _getTorque = getTorque ?? throw new ArgumentNullException(nameof(getTorque));
            _getVel    = getVel;
            _getCmdPos = getCmdPos;
            _log       = log ?? (_ => { });
            _fileSuffix = string.IsNullOrWhiteSpace(fileSuffix) ? "AjinMotion" : fileSuffix;
        }

        public bool Start(int[] axes, string dir, string baseName)
        {
            if (IsLogging) return false;

            try
            {
                Directory.CreateDirectory(dir);
                _axes = axes;
                _filePath = Path.Combine(dir, baseName + "_" + _fileSuffix + ".csv");
                _cts = new CancellationTokenSource();
                _task = Task.Run(() => PollLoop(_cts.Token), _cts.Token);
                return true;
            }
            catch (Exception ex)
            {
                _log($"[AjinLog] 시작 실패: {ex.Message}");
                return false;
            }
        }

        public void Stop()
        {
            if (!IsLogging) return;
            try { _cts?.Cancel(); _task?.Wait(2000); }
            catch { }
            finally { _cts?.Dispose(); _cts = null; _task = null; }
        }

        // ── 폴링 루프 ─────────────────────────────────────────────
        private void PollLoop(CancellationToken token)
        {
            // 헤더: time_s(초) + 축별 토크 + Op(제어기 연결 시만)
            bool hasOp = GetAxisOperation != null;
            var header = new StringBuilder("time_s");
            foreach (int ax in _axes)
                header.Append($",Ax{ax}_Trq(%)");
            if (hasOp)
                foreach (int ax in _axes)
                    header.Append($",Op_Ax{ax}");

            var sw = Stopwatch.StartNew();

            // Stopwatch 틱 기반 인터벌 (드리프트 없는 누적 타이밍)
            long ticksPerInterval = (long)Math.Round(IntervalMs * 0.001 * Stopwatch.Frequency);
            long ticksPer1ms      = Stopwatch.Frequency / 1000;
            long nextTick         = sw.ElapsedTicks + ticksPerInterval;

            // 폴링 스레드 우선순위 상향 → OS 스케줄러에 의한 선점 최소화
            Thread.CurrentThread.Priority = ThreadPriority.Highest;

            // Windows 멀티미디어 타이머 해상도를 1ms로 설정 → Thread.Sleep(1) 정밀도 확보
            timeBeginPeriod(1);
            try
            {
                using (var writer = new StreamWriter(_filePath, false, Encoding.UTF8,
                                                     bufferSize: 65536))  // 64KB 버퍼로 I/O 부담 감소
                {
                    writer.WriteLine(header.ToString());

                    while (!token.IsCancellationRequested)
                    {
                        long nowTick = sw.ElapsedTicks;
                        double t_s   = (double)nowTick / Stopwatch.Frequency;   // 초 단위 (가속도와 동일)

                        var line = new StringBuilder(64);
                        line.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                        for (int i = 0; i < _axes.Length; i++)
                        {
                            int    ax  = _axes[i];
                            double trq = SafeGet(_getTorque, ax);
                            line.Append($",{trq:F4}");
                            TorqueSampled?.Invoke(Device ?? _fileSuffix, ax, trq, DateTime.UtcNow);
                        }

                        if (hasOp)
                            for (int _oi = 0; _oi < _axes.Length; _oi++)
                                line.Append("," + (GetAxisOperation?.Invoke(_axes[_oi]) ?? "Pos"));
                        writer.WriteLine(line.ToString());

                        // ── 고정 인터벌 대기 ──────────────────────────────
                        // ① remaining > 1ms → Sleep(1) 으로 CPU 양보
                        // ② remaining ≤ 1ms → 순수 tight spin → nextTick 정확히 통과
                        long remaining;
                        while ((remaining = nextTick - sw.ElapsedTicks) > 0)
                        {
                            if (remaining > ticksPer1ms)
                                Thread.Sleep(1);
                            // else: tight spin
                        }
                        nextTick += ticksPerInterval;
                    }

                    writer.Flush();
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { _log($"[AjinLog] 폴링 오류: {ex.Message}"); }
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