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

        /// <summary>장치 식별자 (InfluxDB device 태그). 미설정 시 _fileSuffix 사용.</summary>
        public string Device { get; set; }

        /// <summary>
        /// 각 폴링 샘플마다 호출됩니다: (device, axis, fbtrq%, timestampUtc).
        /// InfluxDB 실시간 토크 게시에 사용.
        /// </summary>
        public Action<string, int, double, DateTime> TorqueSampled;

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
            if (IsLogging) { _log("[AjinLog] 이미 로깅 중입니다."); return false; }

            try
            {
                Directory.CreateDirectory(dir);
                _axes = axes;
                _filePath = Path.Combine(dir, baseName + "_" + _fileSuffix + ".csv");
                _cts = new CancellationTokenSource();
                _task = Task.Run(() => PollLoop(_cts.Token), _cts.Token);
                _log($"[AjinLog] 시작 → {_filePath}");
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
            try { _cts?.Cancel(); _task?.Wait(2000); _log("[AjinLog] 중지 완료"); }
            catch { }
            finally { _cts?.Dispose(); _cts = null; _task = null; }
        }

        // ── 폴링 루프 ─────────────────────────────────────────────
        private void PollLoop(CancellationToken token)
        {
            // 헤더
            bool hasCmdPos = _getCmdPos != null;
            var header = new StringBuilder("Timestamp_ms");
            foreach (int ax in _axes)
            {
                header.Append($",Ax{ax}_Pos(mm),Ax{ax}_Vel(mm/s),Ax{ax}_Trq(%)");
                if (hasCmdPos) header.Append($",Ax{ax}_CmdPos(mm),Ax{ax}_CmdVel(mm/s)");
            }

            // 이전 위치 (차분 속도 계산용)
            double[] prevPos    = new double[_axes.Length];
            double[] prevCmdPos = new double[_axes.Length];
            long prevTick = 0;
            bool first = true;

            var sw = Stopwatch.StartNew();

            // Stopwatch 틱 기반 인터벌 (드리프트 없는 누적 타이밍)
            long ticksPerInterval = (long)Math.Round(IntervalMs * 0.001 * Stopwatch.Frequency);
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
                        long t       = nowTick * 1000L / Stopwatch.Frequency; // ms
                        double dtSec = first ? 0.0 : (double)(nowTick - prevTick) / Stopwatch.Frequency;

                        var line = new StringBuilder(128);
                        line.Append(t.ToString(CultureInfo.InvariantCulture));

                        for (int i = 0; i < _axes.Length; i++)
                        {
                            int ax = _axes[i];
                            double pos = SafeGet(_getPos, ax);
                            double trq = SafeGet(_getTorque, ax);

                            double vel = (_getVel != null)
                                ? SafeGet(_getVel, ax)
                                : ((first || dtSec <= 0) ? 0.0 : (pos - prevPos[i]) / dtSec);

                            line.Append($",{pos:F4},{vel:F4},{trq:F4}");
                            TorqueSampled?.Invoke(Device ?? _fileSuffix, ax, trq, DateTime.UtcNow);
                            if (hasCmdPos)
                            {
                                double cmdPos = SafeGet(_getCmdPos, ax);
                                double cmdVel = (first || dtSec <= 0) ? 0.0 : (cmdPos - prevCmdPos[i]) / dtSec;
                                line.Append($",{cmdPos:F4},{cmdVel:F4}");
                                prevCmdPos[i] = cmdPos;
                            }
                            prevPos[i] = pos;
                        }

                        writer.WriteLine(line.ToString());
                        prevTick = nowTick;
                        first    = false;

                        // ── 고정 인터벌 대기 ──────────────────────────────
                        // ① 남은 시간 > 1ms  → Sleep(1) 으로 CPU 양보
                        //    timeBeginPeriod(1) 보장으로 ~1ms 후 복귀.
                        //    오버슈트 시 remaining < 0 → 즉시 탈출 (tick 누적이 다음 주기 보정).
                        // ② 남은 시간 ≤ 1ms  → 순수 tight spin (SpinWait 없음)
                        //    Stopwatch 틱 단위 정밀 대기로 nextTick 정확히 통과.
                        long ticksPer1ms = Stopwatch.Frequency / 1000;
                        long remaining;
                        while ((remaining = nextTick - sw.ElapsedTicks) > 0)
                        {
                            if (remaining > ticksPer1ms)
                                Thread.Sleep(1);
                            // else: tight spin — 아무것도 하지 않고 조건 재확인
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