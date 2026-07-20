using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.DebugTools;
using PHM_Project_DockPanel.Services.Core;
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
        private AccelInfluxPublisher _accelInfluxPublisher;
        private AjinCsvLogger _ajinLogger;         // Ajin 전용 폴링 로거
        private CombinedCsvLogger _combinedLogger; // 통합 CSV 로거 (accel+torque 동시)
        // 통합 모드에서 교체하기 전 BlockReceived 백업 (Stop 시 복원용)
        private Action<string, double[,], DateTime> _savedBlockReceived;

        // ▶ 분리된 로깅 토글 (주입식)
        private readonly Func<bool> _isAccelEnabled;      // 가속도 수집 여부
        private readonly Func<bool> _isVelocityEnabled;   // 속도 수집 여부
        private readonly Func<bool> _isTorqueEnabled;     // 토크 수집 여부

        // ▶ 연속 수집 상태
        private bool _continuousLoggingActive;
        private ContinuousInferenceService _inferenceService;

        // ▶ 축별 MotionID (CSV Ax{n}_MotionID용) — 프로그램 실행 후 축마다 독립적으로 1부터 증가.
        //   실제 모션 명령을 내리는 지점(RunMotionWithLogging의 MoveAbs 호출부)에서 채번하므로
        //   토크 임계값 등 간접 추정 없이 "명령 시작" 이벤트를 직접 반영합니다.
        private const int MaxTrackedAxes = 32; // 실사용 축 수보다 넉넉히 — 배열 재할당(스레드 경합) 방지용 고정 크기
        private readonly int[] _motionIdCounters = new int[MaxTrackedAxes];
        private readonly int[] _currentMotionId  = new int[MaxTrackedAxes];

        /// <summary>해당 축의 현재 MotionID. Idle이면 0, Pos 진행 중이면 해당 모션의 ID.</summary>
        public int GetAxisMotionId(int axisIndex)
            => (axisIndex >= 0 && axisIndex < MaxTrackedAxes) ? _currentMotionId[axisIndex] : 0;

        // ▶ SequenceID (CSV SequenceID용) — 여러 축이 조합된 장비 시퀀스 1회 반복 식별자.
        //   TeachingForm이 스텝 목록을 1회 반복 시작할 때마다 AppEvents.SequenceStarted를 발생시키고,
        //   여기서 구독해 1씩 증가시킵니다. 그 외 경로(AxisInfoForm 단발 이동 등)로 실행된 모션은
        //   "시퀀스"에 속하지 않으므로 0을 유지합니다 (임의 추정 대신 명확한 시작 이벤트가 있을 때만 채번).
        private int _sequenceId = 0;
        public int GetSequenceId() => _sequenceId;

        public ControllerManager Controller => _controller;
        public AxisConfig[] AxisConfigs => _axisConfigs;

        public event Action<int[]> MotionStarted;
        public event Action MotionEnded;

        // === 신규 CTOR: 가속도/속도/토크 각각의 토글을 주입 ===
        public PHM_Motion(ControllerManager controller,
                          AxisConfig[] axisConfigs,
                          WmxTorqueLogger torqueLogger,
                          Func<bool> isAccelEnabled,
                          Func<bool> isTorqueEnabled,
                          Func<bool> isVelocityEnabled = null)
        {
            _controller = controller ?? new ControllerManager();
            _axisConfigs = axisConfigs;
            _torqueLogger = torqueLogger;
            _isAccelEnabled    = isAccelEnabled    ?? (() => false);
            _isVelocityEnabled = isVelocityEnabled ?? (() => false);
            _isTorqueEnabled   = isTorqueEnabled   ?? (() => false);

            // PHM_Motion은 앱 수명 동안 1회만 생성되는 싱글턴이므로 구독 해제 없이 유지합니다.
            AppEvents.SequenceStarted += () => _sequenceId++;
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

        public void SetAccelInfluxPublisher(AccelInfluxPublisher publisher)
        {
            _accelInfluxPublisher = publisher;
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

        private bool ShouldLogAccel()    => _isAccelEnabled?.Invoke()    == true;
        private bool ShouldLogVelocity() => _isVelocityEnabled?.Invoke() == true;
        private bool ShouldLogTorque()   => _isTorqueEnabled?.Invoke()   == true;

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

            bool logAccel    = ShouldLogAccel();
            bool logVelocity = ShouldLogVelocity();
            bool logTorque   = ShouldLogTorque();
            bool anyLog      = (logAccel || logVelocity || logTorque) && _axisConfigs != null;

            // 연속 수집 중이면 모션별 파일 생성 억제 (NI-DAQ 채널 충돌 방지)
            if (_continuousLoggingActive)
            {
                logAccel = false; logVelocity = false; logTorque = false; anyLog = false;
            }

            var status = _controller.GetStatus();
            var active = new List<int>();
            var startPos = new List<double>();
            var targetPos = new List<double>();   // 실제(물리) 목표 위치 — 범위체크/시간추정용
            var commandPos = new List<double>();  // 컨트롤러에 실제로 보낼 목표 위치 (축별 보정 배율 적용)
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

                // 드라이브 쪽 원인(모드/기어비 불일치 등)으로 명령 거리와 실제 이동 거리가
                // 일정 비율로 어긋나는 축을 위한 보정. 피드백(sp/tp)은 그대로 실제 mm를 쓰고,
                // 컨트롤러로 보내는 목표값(cp)에만 배율을 곱해 실제 이동량이 tp에 맞도록 한다.
                double scale = cfg.MoveCommandScale > 0 ? cfg.MoveCommandScale : 1.0;
                double cp = sp + (tp - sp) * scale;

                active.Add(ax);
                startPos.Add(sp);
                targetPos.Add(tp);
                commandPos.Add(cp);
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
                    _accelInfluxPublisher?.BeginSession(sessionId, robotId, active.ToArray());

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
                    string axisTag = active.Count == 1
                        ? $"Axis{active[0]}"
                        : $"Axes{string.Join("-", active)}";
                    string folderName = $"{DateTime.Now:yyyyMMdd}_{axisTag}";
                    string baseRoot = @"C:\Data\PHM_Logs\Signals";
                    // 레이블이 있으면 서브폴더로 분류 — DL 학습 시 폴더명 = 레이블로 자동 인식
                    string labelTag = string.IsNullOrEmpty(AppState.CurrentLabel) ? "unlabeled" : AppState.CurrentLabel;
                    string rootDir = Path.Combine(baseRoot, folderName, labelTag);
                    string torqueDir   = Path.Combine(rootDir, "Torque");
                    string velocityDir = Path.Combine(rootDir, "Velocity");
                    string accelDir    = Path.Combine(rootDir, "Accel");
                    if (logTorque)   Directory.CreateDirectory(torqueDir);
                    if (logVelocity) Directory.CreateDirectory(velocityDir);
                    if (logAccel)    Directory.CreateDirectory(accelDir);

                    string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
                    string baseName = $"{timestamp}_{axisTag}_T{moveTimeMs}ms";
                    string torqueCsvPath = Path.Combine(torqueDir, baseName + "_Torque.csv");
                    bool isAjin = _controller.IsAjin;
                    bool usePollingLogger = isAjin || _controller.IsSimulationMode;

                    // ── Ajin / Simulation: 폴링 로거 ─────────────────────────
                    // 속도 단독: Velocity 폴더 / 토크(±속도): Torque 폴더
                    if ((logTorque || logVelocity) && usePollingLogger && _ajinLogger != null)
                    {
                        string ajinDir = logTorque ? torqueDir : velocityDir;
                        _ajinLogger.IncludeTorque   = logTorque;
                        _ajinLogger.IncludeVelocity = logVelocity;
                        try
                        {
                            startedAjinRun = _ajinLogger.Start(active.ToArray(), ajinDir, baseName);
                        }
                        catch (Exception ex) { AppEvents.RaiseLog($"[로깅 오류] (Ajin) {ex.Message}"); }
                    }

                    // ── WMX3: SDK 토크 로거 ───────────────────────────────
                    if (logTorque && !usePollingLogger)
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
                        // Passive Monitor가 같은 NI 모듈을 사용 중일 수 있으므로 먼저 해제 요청
                        AppEvents.RaisePassiveMonitorSuspend();
                        await Task.Delay(300); // DAQmx 리소스 해제 대기

                        try { startedAccelCsvRun = _accelLogger.Start(active.ToArray(), accelDir, baseName, 0); }
                        catch (Exception ex) { AppEvents.RaiseLog($"[로깅 오류] (DAQ CSV) {ex.Message}"); }
                    }
                }

                // === MotionID 채번 === 실제 명령을 내리는 이 지점에서 축별로 독립적으로 +1
                // (다른 축의 동작은 이 축의 MotionID 증가에 영향을 주지 않음 — active에 포함된 축만 갱신)
                foreach (int ax in active)
                {
                    if (ax < 0 || ax >= MaxTrackedAxes) continue;
                    _motionIdCounters[ax]++;
                    _currentMotionId[ax] = _motionIdCounters[ax];
                }

                // === 모션 실행 === (commandPos: 축별 MoveCommandScale 보정이 적용된 실제 명령값)
                _controller.MoveAbs(active.ToArray(), commandPos.ToArray(), vmax.ToArray(), acc.ToArray(), dec.ToArray());
                await WaitForMotionsEnd(active);
                if (extraWaitAfterMotion != null) await extraWaitAfterMotion();
            }
            finally
            {
                // 모션 종료(정상/예외 무관) → 해당 축들의 MotionID를 0(Idle)으로 되돌림
                foreach (int ax in active)
                    if (ax >= 0 && ax < MaxTrackedAxes) _currentMotionId[ax] = 0;

                // === 모션 종료 후 정리 ===
                try
                {
                    if (logAccel) // logAccel이 true일 때만 실행
                        _accelInfluxPublisher?.EndSession();

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

                string ajinOutputPath = startedAjinRun ? _ajinLogger?.OutputPath : null;

                if (startedAjinRun)
                    stopTasks.Add(Task.Run(() => { try { _ajinLogger.Stop(); } catch { } }));

                // Capture accel paths BEFORE Stop() which nulls CsvPathByModule
                string[] accelCsvPaths = startedAccelCsvRun ? _accelLogger?.CsvPathByModule : null;

                if (startedAccelCsvRun)
                    stopTasks.Add(Task.Run(() => { try { _accelLogger.Stop(); } catch { } }));

                if (stopTasks.Count > 0) await Task.WhenAll(stopTasks);

                // Passive Monitor 재시작 (모션 전에 suspend 했던 경우)
                if (startedAccelCsvRun)
                    AppEvents.RaisePassiveMonitorResume();

                // 로깅 완료 → Log Graph Viewer에 파일 전달
                if (!string.IsNullOrEmpty(ajinOutputPath) && File.Exists(ajinOutputPath))
                    AppEvents.RaiseShowLogGraph(AppEvents.LogDataKind.Torque, ajinOutputPath);

                // Accel CSV → Log Graph (accel paths survived via LastCsvPaths)
                var validAccelPaths = (accelCsvPaths ?? _accelLogger?.LastCsvPaths)
                    ?.Where(p => !string.IsNullOrEmpty(p) && File.Exists(p))
                    .ToList();
                if (validAccelPaths != null && validAccelPaths.Count > 0)
                {
                    AppState.LastAccelCsvs = validAccelPaths;
                    AppEvents.RaiseShowLogGraph(AppEvents.LogDataKind.Accel, validAccelPaths[0]);
                }
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

        // ======================= 연속 수집 =======================

        /// <summary>
        /// 모션 트리거 없이 DAQ/토크 로거를 연속으로 시작합니다.
        /// _isAccelEnabled / _isTorqueEnabled 플래그를 그대로 존중합니다.
        /// </summary>
        /// <param name="forceAccel">null이면 _isAccelEnabled 델리게이트 사용. UI 스레드 외부에서 호출 시 미리 캡처한 값을 전달하세요.</param>
        /// <param name="forceTorque">null이면 _isTorqueEnabled 델리게이트 사용.</param>
        /// <summary>
        /// NI-DAQmx 가속도 하드웨어를 재시도와 함께 시작합니다.
        /// 직전에 StopContinuousLogging()으로 방금 정지한 직후 곧바로 다시 시작하면, 드라이버가
        /// 이전 태스크의 리소스(채널 예약)를 아직 완전히 해제하지 못해 Start()가 즉시 실패하는
        /// 경우가 있다 — 연속 수집 체크박스를 해제했다가 바로 다시 체크하면 데이터가 전혀
        /// 수집되지 않던 문제의 원인. 짧은 지연 후 재시도해 이 경합을 흡수한다.
        /// </summary>
        private bool StartAccelHardwareWithRetry(string dir, string baseName, int maxAttempts = 3, int retryDelayMs = 400)
        {
            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                bool ok = _accelLogger.Start(new int[0], dir, baseName, 0);
                if (ok) return true;

                if (attempt < maxAttempts)
                {
                    AppEvents.RaiseLog(
                        $"[연속 수집] DAQ 하드웨어 시작 실패 (시도 {attempt}/{maxAttempts}) — " +
                        $"직전 정지 이후 하드웨어 리소스 해제 대기 중, {retryDelayMs}ms 후 재시도합니다.");
                    System.Threading.Thread.Sleep(retryDelayMs);
                }
            }
            return false;
        }

        public bool StartContinuousLogging(string label = "", bool? forceAccel = null, bool? forceTorque = null)
        {
            if (_continuousLoggingActive) return true;

            bool logAccel  = forceAccel  ?? ShouldLogAccel();
            bool logTorque = forceTorque ?? ShouldLogTorque();

            if (!logAccel && !logTorque)
            {
                AppEvents.RaiseLog("[연속 수집] 가속도 또는 토크 수집 체크박스를 먼저 활성화하세요.");
                return false;
            }

            // ── 연속 수집 모드 즉시 활성화 ───────────────────────────────
            // CSV 저장 성공 여부와 무관하게, 체크박스를 켠 순간부터
            // 모션별(RunMotionWithLogging) 수집을 억제합니다.
            _continuousLoggingActive = true;

            string labelTag  = string.IsNullOrEmpty(label) ? "unlabeled" : label;
            string today     = DateTime.Now.ToString("yyyyMMdd");
            string baseRoot  = ServerSettings.Current.ContinuousDataPath;
            if (string.IsNullOrWhiteSpace(baseRoot))
                baseRoot = @"C:\Data\PHM_Logs\Signals";
            string rootDir   = Path.Combine(baseRoot, today + "_Continuous", labelTag);
            string accelDir  = Path.Combine(rootDir, "Accel");
            string torqueDir = Path.Combine(rootDir, "Torque");
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            string baseName  = timestamp + "_AllAxes_Continuous";

            // ── InfluxDB label 태그 설정 (연속 수집 기간 동안 유지) ──────────
            if (_accelInfluxPublisher != null)
                _accelInfluxPublisher.Label = labelTag;

            // ── 저장 경로 접근성 사전 확인 ────────────────────────────────
            bool pathOk = false;
            try
            {
                Directory.CreateDirectory(rootDir);
                pathOk = true;
            }
            catch (Exception ex)
            {
                string hint = baseRoot.StartsWith(@"\\")
                    ? $"\n[해결 방법] 서버 PC에서 해당 폴더를 Windows 공유(우클릭→공유)하거나\n"
                      + "연결 설정의 '연속 수집 데이터 저장 경로'를 로컬 경로로 변경하세요."
                    : "";
                AppEvents.RaiseLog($"[연속 수집] ⚠ 저장 경로 접근 실패 — CSV는 저장되지 않습니다.\n"
                                 + $"  경로: {rootDir}\n  오류: {ex.Message}{hint}");
            }

            bool anyStarted = false;

            // WMX / Ajin 두 제어기 모두 미연결이면 Op_Ax 컬럼 생략
            // (pathOk 블록 안팎 모두 사용하므로 먼저 선언)
            bool controllerConnected = _controller.IsConnected;

            if (pathOk)
            {
                int axisCount = AxisConfig.AxisCount > 0
                    ? AxisConfig.AxisCount
                    : Math.Max(1, _axisConfigs?.Length ?? 1);
                int[] allAxes = new int[axisCount];
                for (int i = 0; i < axisCount; i++) allAxes[i] = i;

                bool canPollingTorque = _ajinLogger != null &&
                                        (_controller.IsAjin || _controller.IsSimulationMode);
                bool useCombined = logAccel && logTorque && _accelLogger != null && canPollingTorque;

                if (useCombined)
                {
                    // ── 통합 모드: Accel + Torque → 단일 CSV ─────────────
                    CombinedCsvLogger combined = null;
                    bool blockReceivedOverridden = false;
                    try
                    {
                        combined = new CombinedCsvLogger(
                            getTorque:     ax => _ajinLogger.ReadTorque(ax),
                            getAxisOp:     controllerConnected ? (Func<int, string>)GetAxisOperation : null,
                            axes:          allAxes,
                            log:           msg => AppEvents.RaiseLog(msg),
                            getActPos:     controllerConnected ? (Func<int, double>)(ax => _ajinLogger.ReadActPos(ax)) : null,
                            getCmdPos:     controllerConnected ? (Func<int, double>)(ax => _ajinLogger.ReadCmdPos(ax)) : null,
                            getActVel:     controllerConnected ? (Func<int, double>)(ax => _ajinLogger.ReadActVel(ax)) : null,
                            getMotionId:   controllerConnected ? (Func<int, int>)GetAxisMotionId : null,
                            getSequenceId: GetSequenceId);

                        // DAQ 하드웨어 시작 (CSV 쓰기는 억제, 블록만 combined로 전달)
                        _accelLogger.SuppressCsvWrite = true;
                        string[] modules = _accelLogger.Modules;
                        double accelRate = _accelLogger.SampleRate > 0
                            ? _accelLogger.SampleRate : 1000.0;
                        // ★ 기존 MainForm의 InfluxDB 구독(BlockReceived)을 보존하면서 combined 처리 추가
                        var localCombined = combined;
                        _savedBlockReceived = _accelLogger.BlockReceived;
                        var prevBlockReceived = _savedBlockReceived;
                        _accelLogger.BlockReceived = (module, block, ts) =>
                        {
                            // 1) InfluxDB 피드 (MainForm 구독 체이닝)
                            prevBlockReceived?.Invoke(module, block, ts);

                            // 2) Combined CSV 기록 (원해상도)
                            int modIdx = System.Array.IndexOf(modules, module);
                            if (modIdx < 0) return;
                            int n = block.GetLength(1);
                            if (n <= 0) return;
                            localCombined.ProcessAccelBlock(modIdx, block, n, accelRate);
                        };
                        blockReceivedOverridden = true;
                        // ★ InfluxDB 토크 피드 연결: combined 내부 1ms 폴링 → InfluxDB
                        {
                            string influxModule = _accelLogger.Modules.Length > 0
                                ? _accelLogger.Modules[0] : "combined";
                            combined.TorqueSampled = (axisIdx, torqueVal, utcTs) =>
                            {
                                // 실시간 신호 모니터 브로드캐스트
                                AppEvents.RaiseTorqueSample(axisIdx, torqueVal, utcTs);
                                if (_accelInfluxPublisher != null)
                                    _accelInfluxPublisher.FeedTorqueSample(
                                        influxModule, axisIdx, torqueVal, utcTs);
                            };
                        }

                        // DAQ 하드웨어가 실제로 돌아야 combined CSV에도 실제 행이 쓰인다.
                        // (예전엔 combined.Start()만 성공하면 "시작됨"으로 취급해, DAQ가 실패해도
                        //  헤더만 있는 빈 CSV가 만들어지는 채로 "연속 수집 중"이라고 표시되었음.)
                        bool accelOk = StartAccelHardwareWithRetry(rootDir, baseName);

                        if (!accelOk)
                        {
                            AppEvents.RaiseLog("[연속 수집] DAQ 하드웨어 시작 실패 — 통합 CSV를 시작하지 않습니다.");
                            _accelLogger.BlockReceived = prevBlockReceived;
                            _savedBlockReceived = null;
                            _accelLogger.SuppressCsvWrite = false;
                        }
                        else
                        {
                            bool combOk = combined.Start(rootDir, baseName);
                            if (combOk)
                            {
                                _combinedLogger = combined;
                                anyStarted = true;
                                AppEvents.RaiseLog("[연속 수집] 통합 CSV 시작 → " + rootDir);
                            }
                            else
                            {
                                AppEvents.RaiseLog("[연속 수집] 통합 CSV 시작 실패");
                                _accelLogger.BlockReceived = prevBlockReceived;
                                _savedBlockReceived = null;
                                _accelLogger.SuppressCsvWrite = false;
                                try { _accelLogger.Stop(); } catch { }   // 방금 켠 DAQ 하드웨어 정리
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        AppEvents.RaiseLog("[연속 수집] 통합 오류: " + ex.Message);
                        // 부분 초기화 상태 정리 — BlockReceived가 실패한 combined를 계속 가리키지 않도록 복원
                        if (blockReceivedOverridden)
                        {
                            _accelLogger.BlockReceived = _savedBlockReceived;
                            _savedBlockReceived = null;
                        }
                        _accelLogger.SuppressCsvWrite = false;
                        try { if (_accelLogger.IsRunning) _accelLogger.Stop(); } catch { }
                        try { combined?.Stop(); } catch { }
                    }
                }
                else
                {
                    // ── 단독 모드: 각각 별도 CSV ─────────────────────────

                    // DAQ 가속도
                    if (logAccel && _accelLogger != null)
                    {
                        try
                        {
                            _accelLogger.SuppressCsvWrite = false;
                            // ★ Start() 전에 Op 컬럼 설정 — 헤더에 반영되도록
                            // (Start() 내부에서 헤더를 즉시 작성하므로, 이후에 설정하면 헤더/데이터 컬럼 수 불일치)
                            _accelLogger.GetAxisOperation = controllerConnected ? (Func<int, string>)GetAxisOperation : null;
                            int _axCntAccel = AxisConfig.AxisCount > 0 ? AxisConfig.AxisCount : (_axisConfigs?.Length ?? 0);
                            _accelLogger.LoggedAxes = controllerConnected
                                ? System.Linq.Enumerable.Range(0, _axCntAccel).ToArray()
                                : null;
                            Directory.CreateDirectory(accelDir);
                            bool ok = StartAccelHardwareWithRetry(accelDir, baseName);
                            if (ok)
                            {
                                anyStarted = true;
                                AppEvents.RaiseLog("[연속 수집] DAQ 가속도 시작 → " + accelDir);
                            }
                            else AppEvents.RaiseLog("[연속 수집] DAQ 가속도 시작 실패");
                        }
                        catch (Exception ex)
                        {
                            AppEvents.RaiseLog("[연속 수집] DAQ 오류: " + ex.Message);
                        }
                    }

                    // 토크
                    if (logTorque && canPollingTorque)
                    {
                        try
                        {
                            Directory.CreateDirectory(torqueDir);
                            bool ok = _ajinLogger.Start(allAxes, torqueDir, baseName);
                            if (ok)
                            {
                                anyStarted = true;
                                AppEvents.RaiseLog("[연속 수집] 토크 로거 시작 → " + torqueDir);
                            }
                            else AppEvents.RaiseLog("[연속 수집] 토크 로거 시작 실패");
                        }
                        catch (Exception ex)
                        {
                            AppEvents.RaiseLog("[연속 수집] 토크 오류: " + ex.Message);
                        }
                    }
                    else if (logTorque && !_controller.IsAjin && !_controller.IsSimulationMode)
                    {
                        AppEvents.RaiseLog("[연속 수집] WMX3 토크 로거는 연속 수집을 지원하지 않습니다.");
                    }
                }
            }

            if (!anyStarted)
            {
                // 요청한 로거가 하나도 실제로 시작되지 못했으면 "연속 수집 중"으로 위장하지 않는다.
                // (이전에는 여기서 플래그만 켜둔 채 항상 true를 반환해, 체크박스는 켜져 있는데
                //  실제로는 아무 데이터도 기록되지 않는 상태가 될 수 있었다.)
                AppEvents.RaiseLog("[연속 수집] 시작 실패 — 로거가 하나도 시작되지 않았습니다.");
                _continuousLoggingActive = false;
                return false;
            }

            // ── 추론 서비스 시작 ─────────────────────────────────────────────
            string inferUrl = ServerSettings.Current.InferenceServerUrl;
            if (!string.IsNullOrWhiteSpace(inferUrl))
            {
                _inferenceService?.Dispose();
                if (_accelLogger != null)
                {
                    // 단독 모드에서는 이미 Start() 전에 설정됨.
                    // 통합(combined) 모드에서는 SuppressCsvWrite=true이므로 CSV 헤더 불일치 없음.
                    // 여기서는 추론 서비스가 참조하는 콜백/축 목록만 최신 상태로 동기화.
                    if (_accelLogger.LoggedAxes == null && controllerConnected)
                    {
                        _accelLogger.GetAxisOperation = (Func<int, string>)GetAxisOperation;
                        int _axCnt = AxisConfig.AxisCount > 0 ? AxisConfig.AxisCount : (_axisConfigs?.Length ?? 0);
                        _accelLogger.LoggedAxes = System.Linq.Enumerable.Range(0, _axCnt).ToArray();
                    }
                }
                if (_ajinLogger != null)
                    _ajinLogger.GetAxisOperation = controllerConnected ? (Func<int, string>)GetAxisOperation : null;

                int _axCntInfer = AxisConfig.AxisCount > 0 ? AxisConfig.AxisCount : (_axisConfigs?.Length ?? 0);
                // 제어기 연결 시: per-axis 추론 / 미연결 시: axis 없이 단일 추론
                int[] _inferAxes = controllerConnected
                    ? System.Linq.Enumerable.Range(0, _axCntInfer).ToArray()
                    : null;
                _inferenceService = new ContinuousInferenceService(
                    inferUrl,
                    _combinedLogger ?? (object)null,
                    _accelLogger,
                    _ajinLogger,
                    getAxisOperation: controllerConnected ? (Func<int, string>)GetAxisOperation : null,
                    axes:             _inferAxes);
                _inferenceService.Start();
            }

            return true;  // anyStarted == true 인 경우에만 여기 도달
        }

        /// <summary>?뱀젙 異뺤쓽 ?숈옉 ?곹깭瑜?諛섑솚?⑸땲?? "Pos" ?먮뒗 "Idle".</summary>
        public string GetAxisOperation(int axisIndex)
        {
            try
            {
                var status = _controller.GetStatus();
                if (status?.AxesStatus == null || axisIndex >= status.AxesStatus.Length) return "Idle";
                return status.AxesStatus[axisIndex].OpState != OperationState.Idle ? "Pos" : "Idle";
            }
            catch { return "Idle"; }
        }

        /// <summary>?섎굹?쇰룄 ?吏곸씠硫?"Pos", ?꾨? Idle?대㈃ "Idle".</summary>
        public string GetCurrentOperation()
        {
            try
            {
                var status = _controller.GetStatus();
                if (status?.AxesStatus == null) return "Idle";
                foreach (var ax in status.AxesStatus)
                    if (ax.OpState != OperationState.Idle) return "Pos";
                return "Idle";
            }
            catch { return "Idle"; }
        }
        /// <summary>연속 수집을 중지하고 CSV를 닫습니다.</summary>
        public void StopContinuousLogging()
        {
            if (!_continuousLoggingActive) return;

            try { if (_combinedLogger?.IsLogging  == true) _combinedLogger.Stop(); } catch { }
            try { if (_accelLogger?.IsRunning   == true) _accelLogger.Stop();   } catch { }
            try { if (_ajinLogger?.IsLogging    == true) _ajinLogger.Stop();    } catch { }

            // 통합 모드 상태 정리
            if (_combinedLogger != null)
            {
                _accelLogger.SuppressCsvWrite = false;
                // 통합 모드에서 교체한 BlockReceived를 원래 구독(InfluxDB Feed 등)으로 복원
                _accelLogger.BlockReceived    = _savedBlockReceived;
                _savedBlockReceived           = null;
                _combinedLogger               = null;
            }

            // ── InfluxDB label 태그 초기화 ───────────────────────────────
            if (_accelInfluxPublisher != null)
                _accelInfluxPublisher.Label = "";

            // ── 추론 서비스 중지 ─────────────────────────────────────────────
            _inferenceService?.Stop();
            _inferenceService?.Dispose();
            _inferenceService = null;

            _continuousLoggingActive = false;
            AppEvents.RaiseLog("[연속 수집] 종료");
        }

        /// <summary>현재 연속 수집 중 여부.</summary>
        public bool IsContinuousLogging => _continuousLoggingActive;

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