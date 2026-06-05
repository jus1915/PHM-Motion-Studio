using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using PHM_Project_DockPanel.Services.DAQ;   // DaqAccelCsvLogger, CombinedCsvLogger
using PHM_Project_DockPanel.Services.WMX;

namespace PHM_Project_DockPanel.Services.Core
{
    public sealed class ContinuousInferenceService : IDisposable
    {
        private readonly InferenceServerClient _client;
        private readonly CombinedCsvLogger _combinedLogger; // 통합 CSV (accel+torque)
        private readonly DaqAccelCsvLogger _accelLogger;
        private readonly AjinCsvLogger _torqueLogger;

        private readonly Func<int, string> _getAxisOperation;
        private readonly int[] _axes;

        private CancellationTokenSource _cts;
        private Task _loopTask;

        // ── 윈도우 크기: Start() 후 서버 /model_info 로 자동 설정 ────────────
        private const int DefaultWindowSize = 512;  // 서버 쿼리 실패 시 폴백
        private const int DefaultIntervalMs = 256;  // DefaultWindowSize / 2
        private readonly System.Collections.Generic.Dictionary<string, int> _windowSizes
            = new System.Collections.Generic.Dictionary<string, int>();
        private int _intervalMs = DefaultIntervalMs;

        // ── 활동성 임계값: sensor_type → activity_threshold (_meta.json 기반) ──
        // 0.0 = 게이팅 없음 (정지 구간 포함 학습 모델).
        // 0 초과 = ReadLastWindow에서 채널별 std 최댓값 < 이 값인 행을 제외.
        private readonly System.Collections.Generic.Dictionary<string, double> _activityThresholds
            = new System.Collections.Generic.Dictionary<string, double>();

        // ── 윈도우 상태 분류 임계값: sensor_type → WindowStateThresholds ─────
        // 서버 /model_info 에서 학습 파이프라인 통계 기반 값을 동기화.
        // 없으면 WindowStateThresholds 기본값 사용.
        private readonly System.Collections.Generic.Dictionary<string, WindowStateThresholds> _stateThresholds
            = new System.Collections.Generic.Dictionary<string, WindowStateThresholds>();

        /// <summary>sensor_type별 window_size 반환. 서버 쿼리 전이거나 없으면 기본값.</summary>
        private int GetWindowSize(string sensorType)
        {
            int ws;
            return _windowSizes.TryGetValue(sensorType, out ws) ? ws : DefaultWindowSize;
        }

        /// <summary>sensor_type별 activity_threshold 반환. 없으면 0.0(필터 없음).</summary>
        private double GetActivityThreshold(string sensorType)
        {
            double thr;
            return _activityThresholds.TryGetValue(sensorType, out thr) ? thr : 0.0;
        }

        /// <summary>sensor_type별 WindowStateThresholds 반환. 없으면 기본값.</summary>
        private WindowStateThresholds GetStateThresholds(string sensorType)
        {
            WindowStateThresholds thr;
            return _stateThresholds.TryGetValue(sensorType, out thr)
                ? thr
                : new WindowStateThresholds();
        }

        private int? _lastMovingAxis = null;

        // ── 개별 AE 스코어 캐시 (combined 폴백용) ─────────────────────────────
        private float _latestAccelScore = 0f;
        private readonly System.Collections.Generic.Dictionary<int, float> _latestTorqueScores
            = new System.Collections.Generic.Dictionary<int, float>();

        /// <summary>
        /// CLS(결함진단) 추론 활성화 여부. 기본 false — AE 이상탐지만 실행.
        /// true로 설정 시 RunClsInferenceAll / RunCombinedClsAll 호출.
        /// </summary>
        public bool EnableCls { get; set; } = false;

        public bool IsRunning => _loopTask != null && !_loopTask.IsCompleted;

        public ContinuousInferenceService(
            string inferenceServerUrl,
            object combinedLoggerOrNull,       // CombinedCsvLogger (object 로 받아 C#6 호환)
            DaqAccelCsvLogger accelLogger,
            AjinCsvLogger torqueLogger,
            Func<int, string> getAxisOperation = null,
            int[] axes = null)
        {
            _client           = new InferenceServerClient(inferenceServerUrl);
            _combinedLogger   = combinedLoggerOrNull as CombinedCsvLogger;
            _accelLogger      = accelLogger;
            _torqueLogger     = torqueLogger;
            _getAxisOperation = getAxisOperation;
            _axes             = axes;
        }

        // 기존 코드와의 하위 호환 오버로드
        public ContinuousInferenceService(
            string inferenceServerUrl,
            DaqAccelCsvLogger accelLogger,
            AjinCsvLogger torqueLogger,
            Func<int, string> getAxisOperation = null,
            int[] axes = null)
            : this(inferenceServerUrl, null, accelLogger, torqueLogger, getAxisOperation, axes)
        { }

        /// <summary>현재 Op 상태: 하나라도 Pos면 "Pos", 아니면 "Idle"</summary>
        private string GetCurrentOp()
        {
            if (_getAxisOperation == null || _axes == null)
                return "Pos";   // 제어기 없음 → 항상 Pos 취급 (CLS도 항상 실행)

            foreach (int ax in _axes)
                if (_getAxisOperation(ax) == "Pos")
                    return "Pos";

            return "Idle";
        }

        /// <summary>현재 움직이는 축 번호 (없으면 null)</summary>
        private int? GetMovingAxis()
        {
            if (_getAxisOperation == null || _axes == null)
                return null;

            foreach (int ax in _axes)
                if (_getAxisOperation(ax) == "Pos")
                    return ax;

            return null;
        }

        public void Start()
        {
            if (IsRunning) return;
            _cts = new CancellationTokenSource();
            _loopTask = Task.Run(() => LoopAsync(_cts.Token));
        }

        public void Stop()
        {
            _cts?.Cancel();
        }

        /// <summary>서버 /model_info 를 조회해 _windowSizes, _activityThresholds, _intervalMs 를 갱신합니다.</summary>
        private async Task RefreshWindowSizesAsync(CancellationToken ct)
        {
            try
            {
                var info = await _client.GetModelInfoFullAsync().ConfigureAwait(false);
                if (info == null || info.Count == 0) return;

                foreach (var kv in info)
                {
                    _windowSizes[kv.Key]          = kv.Value.WindowSize;
                    _activityThresholds[kv.Key]   = kv.Value.ActivityThreshold;

                    // 학습 파이프라인에서 저장한 윈도우 상태 분류 파라미터 동기화
                    _stateThresholds[kv.Key] = new WindowStateThresholds
                    {
                        MotionScoreThreshold = kv.Value.MotionScoreThreshold,
                        IdleScoreThreshold   = kv.Value.IdleScoreThreshold,
                        RefAccMagRms         = kv.Value.RefAccMagRms,
                        RefTrqDetrendedRms   = kv.Value.RefTrqDetrendedRms,
                        RefTrqPeakToPeakMax  = kv.Value.RefTrqPeakToPeakMax,
                    };
                }

                // IntervalMs = 가장 작은 window_size / 2 (stride 50%)
                int minWs = int.MaxValue;
                foreach (var ws in _windowSizes.Values)
                    if (ws < minWs) minWs = ws;
                _intervalMs = minWs / 2;

                AppEvents.RaiseLog(
                    "[추론] 윈도우 크기 동기화: " +
                    string.Join(", ", System.Linq.Enumerable.Select(
                        _windowSizes, kv =>
                        {
                            double act;
                            _activityThresholds.TryGetValue(kv.Key, out act);
                            return $"{kv.Key}=ws{kv.Value}/act{act:F4}";
                        })) +
                    $"  intervalMs={_intervalMs}");
            }
            catch { /* 실패 시 기본값 유지 */ }
        }

        private async Task LoopAsync(CancellationToken ct)
        {
            // 루프 시작 전 서버에서 window_size 동기화 (재학습 후 자동 반영)
            await RefreshWindowSizesAsync(ct);

            while (!ct.IsCancellationRequested)
            {
                try { await Task.Delay(_intervalMs, ct); }
                catch { break; }

                // ── (1) AE 추론: Idle/Pos 무관하게 항상 실행 ──────────────────
                //   • 가속도: 단일 센서 → axis = null, Op 필터 없음
                //   • 토크:   축별     → axis = n,    Op 필터 없음
                await RunAeInferenceAll(ct);

                // ── (2) 결합 이상 스코어: 항상 실행 (/predict/combined → AE 스코어) ──
                await RunCombinedClsAll(ct);

                // ── (3) 가속도/토크 CLS: EnableCls=true + Pos 상태일 때만 실행 ────
                if (EnableCls)
                {
                    string op = GetCurrentOp();
                    if (op == "Pos")
                    {
                        int? curAxis = GetMovingAxis();
                        if (curAxis.HasValue)
                            _lastMovingAxis = curAxis;
                        int? clsAxis = _lastMovingAxis;

                        await RunClsInferenceAll(clsAxis, ct);
                    }
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        //  AE 추론 (항상 실행, Op 필터 없음)
        // ──────────────────────────────────────────────────────────────────────

        private async Task RunAeInferenceAll(CancellationToken ct)
        {
            if (_combinedLogger != null && _combinedLogger.IsLogging)
            {
                string cp = _combinedLogger.OutputPath;
                if (string.IsNullOrEmpty(cp) || !File.Exists(cp)) return;

                // 가속도 AE: 단일 센서 → axis = null
                await RunAeInference(cp, "accel", null, ct);

                // 토크 AE: 전역 단일 모델 + 축별 모델 모두 실행
                if (_axes != null)
                {
                    await RunAeInference(cp, "torque", null, ct);   // 전역 단일 (ae_torque_global)
                    foreach (int ax in _axes)
                        await RunAeInference(cp, "torque", ax, ct); // per-axis (ae_torque_axN)
                }
                else
                    await RunAeInference(cp, "torque", null, ct);
            }
            else
            {
                // 단독 가속도 AE
                if (_accelLogger != null && _accelLogger.IsRunning)
                {
                    string[] paths = _accelLogger.CsvPathByModule;
                    if (paths != null)
                        foreach (string p in paths)
                        {
                            if (!string.IsNullOrEmpty(p) && File.Exists(p))
                            { await RunAeInference(p, "accel", null, ct); break; }
                        }
                }

                // 단독 토크 AE: 전역 단일 + 축별 모두 실행
                if (_torqueLogger != null && _torqueLogger.IsLogging)
                {
                    string p = _torqueLogger.OutputPath;
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                    {
                        if (_axes != null)
                        {
                            await RunAeInference(p, "torque", null, ct);   // 전역 단일
                            foreach (int ax in _axes)
                                await RunAeInference(p, "torque", ax, ct); // per-axis
                        }
                        else
                            await RunAeInference(p, "torque", null, ct);
                    }
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        //  CLS 추론 (Pos 상태 전용, Op 필터 있음)
        // ──────────────────────────────────────────────────────────────────────

        private async Task RunClsInferenceAll(int? axis, CancellationToken ct)
        {
            if (_combinedLogger != null && _combinedLogger.IsLogging)
            {
                string cp = _combinedLogger.OutputPath;
                if (string.IsNullOrEmpty(cp) || !File.Exists(cp)) return;

                // 가속도 CLS: 단일 센서 → axis = null
                await RunClsInference(cp, "accel",  null, ct);
                // 토크 CLS: 축별
                await RunClsInference(cp, "torque", axis, ct);
                // 결합 CLS: RunCombinedClsAll에서 처리 (항상 실행, 여기서는 제외)
            }
            else
            {
                if (_accelLogger != null && _accelLogger.IsRunning)
                {
                    string[] paths = _accelLogger.CsvPathByModule;
                    if (paths != null)
                        foreach (string p in paths)
                        {
                            if (!string.IsNullOrEmpty(p) && File.Exists(p))
                            { await RunClsInference(p, "accel", null, ct); break; }
                        }
                }

                if (_torqueLogger != null && _torqueLogger.IsLogging)
                {
                    string p = _torqueLogger.OutputPath;
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                        await RunClsInference(p, "torque", axis, ct);
                }
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        //  결합 CLS 추론 (항상 실행 — AE 스코어 갱신, Op 필터 없음)
        // ──────────────────────────────────────────────────────────────────────

        private async Task RunCombinedClsAll(CancellationToken ct)
        {
            // 결합 CSV가 없으면 스킵 (단독 가속도/토크 로거만 있을 때)
            if (_combinedLogger == null || !_combinedLogger.IsLogging) return;
            string cp = _combinedLogger.OutputPath;
            if (string.IsNullOrEmpty(cp) || !File.Exists(cp)) return;

            // 처리할 축 목록 구성
            System.Collections.Generic.List<int?> axList;
            if (_axes != null)
            {
                axList = new System.Collections.Generic.List<int?>();
                foreach (int ax in _axes) axList.Add(ax);
            }
            else
            {
                axList = new System.Collections.Generic.List<int?> { _lastMovingAxis };
            }

            foreach (int? ax in axList)
            {
                // ── 1차 시도: 서버의 ae_combined / cls_combined 모델 사용 ──────────
                int nCh;
                int    ws     = GetWindowSize("combined");
                double actThr = GetActivityThreshold("combined");
                float[] window = ReadLastWindow(cp, "combined", ws, ax, out nCh,
                    filterOp: false, activityThreshold: actThr);

                // ── 윈도우 상태 분류 (서버 동기화 후에만 적용) ───────────────
                bool serverSuccess = false;
                if (window != null)
                {
                    WindowStateThresholds stateThrC;
                    if (_stateThresholds.TryGetValue("combined", out stateThrC))
                    {
                        int[] accelIdx, torqueIdx;
                        WindowStateClassifier.GetChannelRoles("combined", nCh, out accelIdx, out torqueIdx);
                        WindowFeatures features = WindowStateClassifier.ComputeFeatures(
                            window, ws, nCh, accelIdx, torqueIdx, stateThrC);
                        WindowState winState = WindowStateClassifier.Classify(features, stateThrC);

                        AppEvents.RaiseWindowState("combined", ax, winState, features);

                        if (winState != WindowState.Motion)
                            continue;  // Idle/Ambiguous — 추론 스킵
                    }
                }

                if (window != null)
                {
                    CombinedInferenceResult combined = await _client.PredictCombinedAsync(
                        window, ws, nCh, "combined", ax, ct);

                    if (!combined.IsError)
                    {
                        AppEvents.RaiseClsInferenceResult("combined", combined);
                        serverSuccess = true;
                    }
                    // 404 등 서버 오류는 조용히 폴백 (반복 로그 스팸 방지)
                }

                if (serverSuccess) continue;

                // ── 2차 폴백: 개별 accel/torque AE 스코어의 최댓값으로 합성 ─────────
                // 서버에 ae_combined 모델이 없어도 combined 차트는 항상 갱신됨.
                float torqueScore = 0f;
                if (ax.HasValue)
                    _latestTorqueScores.TryGetValue(ax.Value, out torqueScore);
                float combinedScore = Math.Max(_latestAccelScore, torqueScore);

                var synth = new CombinedInferenceResult
                {
                    SensorType   = "combined",
                    Axis         = ax,
                    IsAnomaly    = combinedScore >= 1.0f,
                    AnomalyScore = combinedScore,
                    Threshold    = 1.0f,
                    ClsAvailable = false,
                };
                AppEvents.RaiseClsInferenceResult("combined", synth);
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        //  개별 추론 메서드
        // ──────────────────────────────────────────────────────────────────────

        /// <summary>
        /// AE 이상탐지 추론 — /predict 엔드포인트 사용.
        /// 1) 마지막 windowSize 행 읽기
        /// 2) 윈도우 상태 분류 (idle/motion/ambiguous)
        /// 3) Motion 상태일 때만 서버 AE 추론 실행
        /// </summary>
        private async Task RunAeInference(
            string csvPath, string sensorType, int? axis, CancellationToken ct)
        {
            int nCh;
            int    ws     = GetWindowSize(sensorType);
            double actThr = GetActivityThreshold(sensorType);
            float[] window = ReadLastWindow(csvPath, sensorType, ws, axis, out nCh,
                filterOp: false, activityThreshold: actThr);
            if (window == null) return;

            // ── 윈도우 상태 분류 (서버 동기화 후에만 적용) ───────────────────
            // _stateThresholds 는 RefreshWindowSizesAsync 에서 /model_info 를 통해
            // 학습 데이터 기반 ref값을 받아야 채워진다.
            // 재학습 전(서버 메타 없음)에는 기본값으로 분류하면 오탐이 발생하므로
            // 해당 sensor_type 의 임계값이 명시적으로 로드된 경우에만 게이팅 적용.
            WindowStateThresholds stateThr;
            if (_stateThresholds.TryGetValue(sensorType, out stateThr))
            {
                int[] accelIdx, torqueIdx;
                WindowStateClassifier.GetChannelRoles(sensorType, nCh, out accelIdx, out torqueIdx);

                WindowFeatures features = WindowStateClassifier.ComputeFeatures(
                    window, ws, nCh, accelIdx, torqueIdx, stateThr);
                WindowState winState = WindowStateClassifier.Classify(features, stateThr);

                // 상태 이벤트 발행 (Dashboard / 로그 표시용)
                AppEvents.RaiseWindowState(sensorType, axis, winState, features);

                // Motion 상태가 아니면 AE 추론 스킵
                // - Idle      : 정지 구간, rule 기반 감시만 수행
                // - Ambiguous : 경계 구간, 학습/알람에서 제외
                if (winState != WindowState.Motion) return;
            }
            // 임계값 미로드 시: 기존 activityThreshold 행 필터만 적용 (ReadLastWindow 내부)

            InferenceResult result = await _client.PredictAsync(
                window, ws, nCh, sensorType, axis, ct);

            // 모델 없음 (404): 학습 안 된 (sensor_type, axis) — 조용히 무시.
            // 클라이언트가 캐시에 등록하므로 이후엔 네트워크 호출도 없음.
            if (result.IsModelMissing)
                return;

            if (result.IsError)
                AppEvents.RaiseLog(
                    $"[AE 추론 오류] sensorType={sensorType} axis={axis?.ToString() ?? "null"}" +
                    $"  nCh={nCh}  windowLen={window.Length}  →  {result.Error}");
            else
            {
                // combined 폴백용: 최신 개별 AE 스코어 캐시
                if (sensorType == "accel")
                    _latestAccelScore = result.AnomalyScore;
                else if (sensorType == "torque" && axis.HasValue)
                    _latestTorqueScores[axis.Value] = result.AnomalyScore;
            }

            AppEvents.RaiseInferenceResult(sensorType, result);
        }

        /// <summary>
        /// CLS 결함진단 추론 — filterOp=true (Pos 행만), /predict/combined 엔드포인트 사용.
        /// </summary>
        private async Task RunClsInference(
            string csvPath, string sensorType, int? axis, CancellationToken ct,
            bool? filterOp = null)
        {
            int nCh;
            // filterOp 명시 없으면: accel=false(전체), torque=true(Pos행만), combined=true(Pos행만)
            bool   useFilterOp = filterOp ?? (sensorType != "accel");
            int    ws          = GetWindowSize(sensorType);
            double actThr      = GetActivityThreshold(sensorType);
            float[] window = ReadLastWindow(csvPath, sensorType, ws, axis, out nCh,
                filterOp: useFilterOp, activityThreshold: actThr);
            if (window == null) return;

            CombinedInferenceResult combined = await _client.PredictCombinedAsync(
                window, ws, nCh, sensorType, axis, ct);

            // 모델 없음 (404): 조용히 무시 (이후엔 캐시로 차단)
            if (combined.IsModelMissing)
                return;

            if (combined.IsError)
            {
                AppEvents.RaiseLog(
                    $"[CLS 추론 오류] sensorType={sensorType} axis={axis?.ToString() ?? "null"}" +
                    $"  nCh={nCh}  windowLen={window.Length}  →  {combined.Error}");
                return;
            }

            // CLS 결과만 발행 (AE는 RunAeInference에서 별도 발행)
            AppEvents.RaiseClsInferenceResult(sensorType, combined);
        }

        /// <summary>
        /// CSV 끝 windowSize 행에서 신호 윈도우를 읽습니다.
        /// filterOp=true          : Op==Pos 행만 사용 (CLS용)
        /// activityThreshold &gt; 0 : 채널별 std 최댓값 &lt; 임계값인 정지 행 제외 (학습과 동일 분포 유지)
        ///   → 후보 행이 부족하면 null 반환 (해당 사이클 추론 스킵)
        /// 성공 시 float 배열 반환, 실패 시 null.
        /// </summary>
        private static float[] ReadLastWindow(
            string csvPath,
            string sensorType,
            int windowSize,
            int? axis,
            out int nChannels,
            bool filterOp = true,
            double activityThreshold = 0.0)
        {
            nChannels = 0;
            try
            {
                string[] lines;
                using (FileStream fs = new FileStream(
                    csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (StreamReader sr = new StreamReader(fs))
                {
                    lines = sr.ReadToEnd()
                        .Split(new char[] { '\r', '\n' },
                               StringSplitOptions.RemoveEmptyEntries);
                }

                if (lines.Length < 2) return null;

                string[] headers = lines[0].Split(',');

                int[] signalCols = GetSignalColumnIndices(headers, sensorType, axis);
                if (signalCols.Length == 0) return null;

                // ── Op 필터 (filterOp=true일 때만 적용) ──────────────────
                List<string> dataLines;
                if (!filterOp)
                {
                    dataLines = new List<string>(lines.Length - 1);
                    for (int li = 1; li < lines.Length; li++)
                        dataLines.Add(lines[li]);
                }
                else
                {
                    dataLines = FilterPosByOp(lines, headers, axis);
                }

                // ── 활동성 필터 (activityThreshold > 0일 때만 적용) ──────
                // 학습: max(std per channel) < percentile_threshold 인 윈도우 제거
                // 추론: 동일 기준을 행 단위로 적용 — 충분히 활성인 행만 사용
                // 구현: CSV 끝에서 최대 windowSize*4 행을 읽고,
                //        각 행의 신호값 절댓값이 threshold를 넘는 행만 유지.
                // (행별 max(|x|) ≈ std 대리변수 — 계산 간단, 정지 판별에 충분)
                if (activityThreshold > 0.0 && dataLines.Count > 0)
                {
                    // 끝에서 최대 windowSize*4 후보만 검사 (전체 CSV를 다 순회하지 않음)
                    int scanStart = Math.Max(0, dataLines.Count - windowSize * 4);
                    var activeLines = new List<string>(windowSize * 2);
                    for (int li = scanStart; li < dataLines.Count; li++)
                    {
                        string[] cols = dataLines[li].Split(',');
                        // 신호 채널 중 하나라도 |값| > threshold 이면 활성 행
                        bool active = false;
                        foreach (int ci in signalCols)
                        {
                            float v;
                            if (ci < cols.Length &&
                                float.TryParse(cols[ci], NumberStyles.Float,
                                    CultureInfo.InvariantCulture, out v) &&
                                Math.Abs(v) >= activityThreshold)
                            {
                                active = true;
                                break;
                            }
                        }
                        if (active) activeLines.Add(dataLines[li]);
                    }
                    dataLines = activeLines;
                }

                if (dataLines.Count < windowSize) return null;

                int startIdx = dataLines.Count - windowSize;
                int nCh = signalCols.Length;
                float[] w = new float[windowSize * nCh];

                int wIdx = 0;
                for (int ri = startIdx; ri < dataLines.Count; ri++)
                {
                    string[] cols = dataLines[ri].Split(',');
                    for (int si = 0; si < signalCols.Length; si++)
                    {
                        int ci = signalCols[si];
                        float v;
                        if (ci < cols.Length &&
                            float.TryParse(cols[ci], NumberStyles.Float,
                                CultureInfo.InvariantCulture, out v))
                        {
                            w[wIdx] = v;
                        }
                        wIdx++;
                    }
                }

                nChannels = nCh;
                return w;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Op 필터: axis 지정 시 Op_Ax{n}==Pos 행, null 시 임의 Op_Ax* 컬럼 중 하나라도 Pos 행.
        /// Op 컬럼이 없으면 모든 행 반환 (제어기 미연결 케이스).
        /// </summary>
        private static List<string> FilterPosByOp(string[] lines, string[] headers, int? axis)
        {
            var result = new List<string>();

            // Op 컬럼 인덱스 탐색
            List<int> opCols = new List<int>();
            if (axis.HasValue)
            {
                // Op_Ax{n} 단일 컬럼
                string opName = "Op_Ax" + axis.Value;
                for (int i = 0; i < headers.Length; i++)
                    if (string.Equals(headers[i].Trim(), opName, StringComparison.OrdinalIgnoreCase))
                    { opCols.Add(i); break; }
            }
            else
            {
                // 임의 Op_Ax* 컬럼 전체 (accel 단일 센서: 어느 축이든 움직이면 Pos)
                for (int i = 0; i < headers.Length; i++)
                    if (headers[i].Trim().StartsWith("Op_Ax", StringComparison.OrdinalIgnoreCase)
                        || string.Equals(headers[i].Trim(), "Op", StringComparison.OrdinalIgnoreCase))
                        opCols.Add(i);
            }

            for (int li = 1; li < lines.Length; li++)
            {
                string line = lines[li];
                if (opCols.Count == 0)
                {
                    // Op 컬럼 없음 → 제어기 미연결 → 모든 행 포함
                    result.Add(line);
                    continue;
                }
                string[] cols = line.Split(',');
                bool isPos = false;
                foreach (int oc in opCols)
                {
                    if (oc < cols.Length &&
                        string.Equals(cols[oc].Trim(), "Pos", StringComparison.OrdinalIgnoreCase))
                    { isPos = true; break; }
                }
                if (isPos) result.Add(line);
            }

            return result;
        }

        private static int[] GetSignalColumnIndices(
            string[] headers, string sensorType, int? axis)
        {
            List<int> result = new List<int>();

            if (sensorType == "accel")
            {
                // x, y, z (combined CSV 신규 포맷 / 단독 accel CSV)
                for (int i = 0; i < headers.Length; i++)
                {
                    string h = headers[i].Trim().ToLower();
                    if (h == "x" || h == "y" || h == "z")
                        result.Add(i);
                }
                // 폴백: 구 포맷 Ax{n}_x, Ax{n}_y, Ax{n}_z
                if (result.Count == 0 && axis.HasValue)
                {
                    string px = "ax" + axis.Value + "_x";
                    string py = "ax" + axis.Value + "_y";
                    string pz = "ax" + axis.Value + "_z";
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].Trim().ToLower();
                        if (h == px || h == py || h == pz)
                            result.Add(i);
                    }
                }
            }
            else if (sensorType == "combined")
            {
                // 가속도: x, y, z
                for (int i = 0; i < headers.Length; i++)
                {
                    string h = headers[i].Trim().ToLower();
                    if (h == "x" || h == "y" || h == "z")
                        result.Add(i);
                }
                // 토크: Ax{n}_Trq(%) — 채널 순서 학습과 동일 (accel 다음에 torque)
                if (axis.HasValue)
                {
                    string target = "ax" + axis.Value.ToString() + "_trq(%)";
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (headers[i].Trim().ToLower() == target)
                            result.Add(i);
                    }
                }
            }
            else  // "torque"
            {
                if (axis.HasValue)
                {
                    // per-axis: 해당 축 토크 채널 1개
                    string target = "ax" + axis.Value.ToString() + "_trq(%)";
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (headers[i].Trim().ToLower() == target)
                            result.Add(i);
                    }
                }
                else
                {
                    // axis=null: 전역 단일 모델(ae_torque_global) — 현재 1ch 모델이므로 첫 토크 채널만 전송.
                    // ae_torque_global이 다채널(전 축 통합)로 재학습된 후에는 모든 Ax*_Trq(%) 컬럼을 수집하도록 변경 필요.
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].Trim().ToLower();
                        if (h.Contains("trq") || h.Contains("torque"))
                        { result.Add(i); break; }   // 첫 번째 토크 채널만 (1ch 모델 매칭)
                    }
                }
            }

            return result.ToArray();
        }

        public void Dispose()
        {
            Stop();
            if (_client != null) _client.Dispose();
        }
    }
}
