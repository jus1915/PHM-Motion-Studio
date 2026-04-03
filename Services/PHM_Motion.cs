using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.DebugTools;
using PHM_Project_DockPanel.Services.WMX;
using PHM_Project_DockPanel.Services.DAQ;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WMX3ApiCLR;
using PHM_Project_DockPanel.Windows;
using System.Drawing;
using System.Windows.Forms;
using System.Globalization;

namespace PHM_Project_DockPanel.Services
{
    public class PHM_Motion
    {
        private readonly ControllerManager _controller;
        private AxisConfig[] _axisConfigs;

        private WmxTorqueLogger _torqueLogger;
        private DaqAccelCsvLogger _accelLogger;
        private DaqAccelHttpSender _accelHttpSender;
        private AjinCsvLogger _ajinLogger;   // Ajin 전용 폴링 로거

        // ▶ 분리된 로깅 토글 (주입식)
        private readonly Func<bool> _isAccelEnabled;   // 가속도 수집 여부
        private readonly Func<bool> _isTorqueEnabled;  // 토크 수집 여부

        public ControllerManager Controller => _controller;
        public AxisConfig[] AxisConfigs => _axisConfigs;

        public event Action<int[]> MotionStarted;
        public event Action MotionEnded;

        // === 신규 CTOR: 가속도/토크 각각의 토글을 주입 ===
        public PHM_Motion(ControllerManager controller,
                          AxisConfig[] axisConfigs,
                          WmxTorqueLogger torqueLogger,
                          Func<bool> isAccelEnabled,
                          Func<bool> isTorqueEnabled)
        {
            _controller = controller ?? new ControllerManager();
            _axisConfigs = axisConfigs;
            _torqueLogger = torqueLogger;
            _isAccelEnabled = isAccelEnabled ?? (() => false);
            _isTorqueEnabled = isTorqueEnabled ?? (() => false);
        }

        // === 구 CTOR 호환(기존 단일 토글): 둘 다 동일 토글을 사용 ===
        [Obsolete("Use ctor with separate isAccelEnabled / isTorqueEnabled.")]
        public PHM_Motion(ControllerManager controller,
                          AxisConfig[] axisConfigs,
                          WmxTorqueLogger torqueLogger,
                          Func<bool> isLogEnabled)
            : this(controller, axisConfigs, torqueLogger,
                   () => (isLogEnabled?.Invoke() ?? false),
                   () => (isLogEnabled?.Invoke() ?? false))
        { }

        public void SetAccelLogger(DaqAccelCsvLogger accelLogger)
        {
            _accelLogger = accelLogger;
        }

        public void SetAccelHttpSender(DaqAccelHttpSender accelHttpSender)
        {
            _accelHttpSender = accelHttpSender;
        }

        /// <summary>Ajin 전용 폴링 로거를 주입합니다. MainForm에서 Ajin 선택 시 호출.</summary>
        public void SetAjinLogger(AjinCsvLogger ajinLogger)
        {
            _ajinLogger = ajinLogger;
        }
        public void SetAxisConfigs(AxisConfig[] configs)
        {
            _axisConfigs = configs;
            _controller?.SetAxisConfigs(configs);
        }

        public double GetAxisCurrentPos(int axisIndex)
        {
            if (axisIndex < 0)
                throw new IndexOutOfRangeException("축 인덱스가 유효하지 않습니다.");

            var status = _controller.GetStatus();
            double rawPos = status.AxesStatus[axisIndex].ActualPos;

            // ✅ Ajin/Simulation이면 그대로 사용 (이미 mm 단위)
            if (_controller.PosIsAlreadyMm)
            {
                return rawPos;
            }

            // ✅ WMX 등은 encoder → mm 변환
            var cfg = (_axisConfigs != null && axisIndex < _axisConfigs.Length) ? _axisConfigs[axisIndex] : new AxisConfig();
            return UnitConverter.EncoderToMm(rawPos, cfg.PitchMmPerRev);
        }

        public bool IsWithinRange(double pos, double max)
        {
            if (pos < 0 || pos > max)
            {
                AppEvents.RaiseLog($"[경고] 위치가 범위를 벗어났습니다. (0 ~ {max} mm)");
                return false;
            }
            return true;
        }

        private bool ShouldLogAccel() => _isAccelEnabled?.Invoke() == true;
        private bool ShouldLogTorque() => _isTorqueEnabled?.Invoke() == true;

        // 특정 예외 메시지는 잡음이라 무시
        private static bool IsNotCollectingError(Exception ex)
            => (ex?.Message?.IndexOf("Currently not collecting log data", StringComparison.OrdinalIgnoreCase) ?? -1) >= 0;

        public Task<bool> RunMotionWithLogging(
            int[] axes,
            bool isAbs,
            double value,
            Func<Task> extraWaitAfterMotion = null,
            bool isLoopMode = false)
        {
            if (axes == null || axes.Length == 0) return Task.FromResult(false);
            // 브로드캐스트
            var values = Enumerable.Repeat(value, axes.Length).ToArray();
            return RunMotionWithLogging(axes, isAbs, values, extraWaitAfterMotion, isLoopMode);
        }

        public async Task<bool> RunMotionWithLogging(
            int[] axes,
            bool isAbs,
            double[] values,
            Func<Task> extraWaitAfterMotion = null,
            bool isLoopMode = false)
        {
            if (axes == null || axes.Length == 0) return false;
            if (values == null || values.Length == 0) return false;

            if (values.Length == 1 && axes.Length > 1)
                values = Enumerable.Repeat(values[0], axes.Length).ToArray();
            if (values.Length != axes.Length)
                throw new ArgumentException("values length must be 1 or equal to axes length.");

            bool logAccel = ShouldLogAccel();
            bool logTorque = ShouldLogTorque();
            bool anyLog = (logAccel || logTorque) && _axisConfigs != null;

            var status = _controller.GetStatus();
            var active = new List<int>();
            var startPos = new List<double>();
            var targetPos = new List<double>();
            var vmax = new List<double>();
            var acc = new List<double>();
            var dec = new List<double>();

            foreach (var (ax, idx) in axes.Distinct().OrderBy(a => a).Select((a, i) => (a, i)))
            {
                if (ax < 0) continue;
                if (!(status?.AxesStatus[ax].ServoOn ?? false)) continue;

                var cfg = (_axisConfigs != null && ax < _axisConfigs.Length) ? _axisConfigs[ax] : new AxisConfig();
                double sp = GetAxisCurrentPos(ax);
                double tp = isAbs ? values[idx] : sp + values[idx];
                if (!IsWithinRange(tp, cfg.PositionMax)) continue;

                active.Add(ax);
                startPos.Add(sp);
                targetPos.Add(tp);
                vmax.Add(cfg.MaxVel);
                acc.Add(cfg.Acc);
                dec.Add(cfg.Dec);
            }

            if (active.Count == 0)
            {
                AppEvents.RaiseLog("[안내] 유효한 축이 없습니다.");
                return false;
            }

            string robotId = ResolveRobotIdForActiveAxes(active);

            // === 세션 시작 (HTTP 실시간 모드와 연동) ===
            try
            {
                string sessionId = $"{robotId}_{DateTime.Now:yyyyMMdd_HHmmssfff}";

                if (logAccel) // logAccel이 true일 때만 실행
                    _accelHttpSender?.BeginSession(sessionId, robotId, active.ToArray());

                MotionStarted?.Invoke(active.ToArray());
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[세션 시작 오류] {ex.Message}");
            }

            double maxMoveSec = 0;
            for (int i = 0; i < active.Count; i++)
                maxMoveSec = Math.Max(
                    maxMoveSec,
                    EstimateMotionTime(Math.Abs(targetPos[i] - startPos[i]), vmax[i], acc[i], dec[i]));
            int moveTimeMs = (int)Math.Round(maxMoveSec * 1000.0);

            bool startedAccelCsvRun = false;
            bool startedTorqueRun = false;
            bool startedAjinRun = false;

            try
            {
                if (anyLog)
                {
                    // === 기존 CSV/토크 로깅 준비 (로깅 활성화 시에만) ===
                    string folderName = $"{DateTime.Now:yyyyMMdd}_Axis{active[0]}";
                    string baseRoot = @"C:\Data\PHM_Logs\Signals";
                    string rootDir = Path.Combine(baseRoot, folderName);
                    string torqueDir = Path.Combine(rootDir, "Torque");
                    string accelDir = Path.Combine(rootDir, "Accel");
                    Directory.CreateDirectory(torqueDir);
                    Directory.CreateDirectory(accelDir);

                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
                    string baseName = $"{timestamp}_Axis{active[0]}_T{moveTimeMs}ms";
                    string torqueCsvPath = Path.Combine(torqueDir, baseName + "_Torque.csv");
                    bool isAjin = _controller.IsAjin;

                    // ── Ajin: 폴링 로거 ────────────────────────────────────
                    if (logTorque && isAjin && _ajinLogger != null)
                    {
                        try
                        {
                            startedAjinRun = _ajinLogger.Start(active.ToArray(), torqueDir, baseName);
                        }
                        catch (Exception ex) { AppEvents.RaiseLog($"[로깅 오류] (Ajin) {ex.Message}"); }
                    }

                    // ── WMX3: SDK 토크 로거 ───────────────────────────────
                    if (logTorque && !isAjin)
                    {
                        if (_torqueLogger == null)
                            _torqueLogger = new WmxTorqueLogger(new Log(), 0, msg => AppEvents.RaiseLog(msg));
                        try
                        {
                            _torqueLogger.Start(active.ToArray(), torqueDir, Path.GetFileName(torqueCsvPath), 5000);
                            startedTorqueRun = true;
                        }
                        catch (Exception ex) { AppEvents.RaiseLog($"[로깅 오류] (WMX) {ex.Message}"); }
                    }

                    // ── DAQ 가속도 CSV 로거 (공통) ─────────────────────────
                    if (logAccel && _accelLogger != null)
                    {
                        try { startedAccelCsvRun = _accelLogger.Start(active.ToArray(), accelDir, baseName, 0); }
                        catch (Exception ex) { AppEvents.RaiseLog($"[로깅 오류] (DAQ CSV) {ex.Message}"); }
                    }
                }

                // === 모션 실행 ===
                _controller.MoveAbs(active.ToArray(), targetPos.ToArray(), vmax.ToArray(), acc.ToArray(), dec.ToArray());
                await WaitForMotionsEnd(active);
                if (extraWaitAfterMotion != null) await extraWaitAfterMotion();
            }
            finally
            {
                // === 모션 종료 후 정리 ===
                try
                {
                    if (logAccel) // logAccel이 true일 때만 실행
                        _accelHttpSender?.EndSession();

                    MotionEnded?.Invoke();
                }
                catch (Exception ex)
                {
                    AppEvents.RaiseLog($"[세션 종료 오류] {ex.Message}");
                }

                await Task.Delay(200);
                var stopTasks = new List<Task>();
                if (startedTorqueRun)
                    stopTasks.Add(Task.Run(() => { try { _torqueLogger.Stop(); } catch { } }));

                if (startedAjinRun)
                    stopTasks.Add(Task.Run(() => { try { _ajinLogger.Stop(); } catch { } }));

                if (startedAccelCsvRun)
                    stopTasks.Add(Task.Run(() => { try { _accelLogger.Stop(); } catch { } }));

                if (stopTasks.Count > 0) await Task.WhenAll(stopTasks);
            }

            return true;
        }


        private async Task WaitForMotionEnd(int axisIndex)
        {
            while (true)
            {
                var status = _controller.GetStatus();
                if (status.AxesStatus[axisIndex].OpState == OperationState.Idle)
                    break;
                await Task.Delay(10);
            }
        }

        private async Task WaitForMotionsEnd(IEnumerable<int> axes)
        {
            var arr = axes?.ToArray() ?? Array.Empty<int>();
            while (true)
            {
                var st = _controller.GetStatus();
                if (arr.All(a => st.AxesStatus[a].OpState == OperationState.Idle))
                    break;
                await Task.Delay(10);
            }
        }

        private static double EstimateMotionTime(double distanceMm, double vmax, double acc, double dec)
        {
            // 방어코드
            if (distanceMm <= 0) return 0.0;
            if (vmax <= 0 || acc <= 0 || dec <= 0) return 0.0;

            // 가속/감속 구간에서 소요 거리
            double da = 0.5 * (vmax * vmax) / acc; // mm
            double dd = 0.5 * (vmax * vmax) / dec; // mm

            // 크루즈 존재(트라페zoid)
            if (distanceMm >= da + dd)
            {
                double ta = vmax / acc;                          // s
                double td = vmax / dec;                          // s
                double dc = distanceMm - da - dd;                // mm
                double tc = dc / vmax;                           // s
                return ta + tc + td;
            }
            // 크루즈 없음(삼각)
            else
            {
                // 최고속도 = sqrt( 2*D / (1/a + 1/d) )
                double vpeak = Math.Sqrt(2.0 * distanceMm / (1.0 / acc + 1.0 / dec));
                double ta = vpeak / acc;
                double td = vpeak / dec;
                return ta + td;
            }
        }

        string ResolveRobotIdForActiveAxes(IList<int> activeAxes)
        {
            if (activeAxes == null || activeAxes.Count == 0) return "RB01";

            // 서로 다른 로봇(축)이 섞여 있으면 경고만 남기고 가장 작은 축 인덱스 기준으로 보냄
            // (축=로봇 1:1 가정이라 사실상 첫 축이 대표)
            int repAxis = activeAxes.Min(); // 대표 축
            if (activeAxes.Any(a => a != repAxis))
                AppEvents.RaiseLog("[안내] 여러 로봇 축이 함께 요청되었습니다. HTTP 송신은 대표 축 기준 한 로봇으로 전송합니다.");

            return string.Format("RB{0:00}", repAxis);
        }
    }
}