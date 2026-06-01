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

        /// <summary>sensor_type별 activity_rms_thr 캐시. 0.0 이면 필터 비활성.</summary>
        private readonly System.Collections.Generic.Dictionary<string, double> _activityRmsThresholds
            = new System.Collections.Generic.Dictionary<string, double>();

        /// <summary>sensor_type별 window_size 반환. 서버 쿼리 전이거나 없으면 기본값.</summary>
        private int GetWindowSize(string sensorType)
        {
            int ws;
            return _windowSizes.TryGetValue(sensorType, out ws) ? ws : DefaultWindowSize;
        }

        /// <summary>sensor_type별 activity_rms_thr 반환. 캐시 없으면 0.0 (필터 비활성).</summary>
        private double GetActivityRmsThr(string sensorType)
        {
            double thr;
            return _activityRmsThresholds.TryGetValue(sensorType, out thr) ? thr : 0.0;
        }

        private int? _lastMovingAxis = null;

        // ── 개별 AE 스코어 캐시 (combined 폴백용) ─────────────────────────────
        private float _latestAccelScore = 0f;
        private readonly System.Collections.Generic.Dictionary<int, float> _latestTorqueScores
            = new System.Collections.Generic.Dictionary<int, float>();

        // ── 세그먼트 상태 (RMS 기반 Active/Idle 전환 감지) ──────────────────
        // activity_rms_thr > 0: RMS 가 threshold 이상이면 Active, 미만이면 Idle
        // activity_rms_thr = 0: 학습 시 필터 미사용 → per-window 동작 유지

        // key = "{sensorType}|{axis}" 별 이전 Active 상태
        private readonly System.Collections.Generic.Dictionary<string, bool>
            _segmentWasActive = new System.Collections.Generic.Dictionary<string, bool>();

        // key별 누적 window score 목록
        private readonly System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<float>>
            _segmentScores = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<float>>();

        // key별 최신 InferenceResult (세그먼트 결과 구성용 메타데이터 보존)
        private readonly System.Collections.Generic.Dictionary<string, InferenceResult>
            _segmentMeta = new System.Collections.Generic.Dictionary<string, InferenceResult>();

        private static string SegKey(string sensorType, int? axis) =>
            axis.HasValue ? $"{sensorType}|{axis.Value}" : sensorType;

        /// <summary>세그먼트 시작: 해당 key 의 누적 버퍼를 초기화합니다.</summary>
        private void OnSegmentStart(string key, string sensorType, int? axis)
        {
            _segmentScores[key]  = new System.Collections.Generic.List<float>();
            _segmentMeta.Remove(key);
            AppEvents.RaiseLog(
                $"[세그먼트] 시작 — {sensorType}" +
                (axis.HasValue ? $" Ax{axis.Value}" : "") +
                "  윈도우 누적 시작");
        }

        /// <summary>
        /// 세그먼트 종료: 누적 window score 를 90th pct 로 집계하고
        /// IsSegmentResult=true 인 InferenceResult 를 발행합니다.
        /// </summary>
        private void OnSegmentEnd(string key)
        {
            System.Collections.Generic.List<float> scores;
            if (!_segmentScores.TryGetValue(key, out scores) || scores.Count == 0)
            {
                _segmentScores.Remove(key);
                _segmentMeta.Remove(key);
                return;
            }

            InferenceResult meta;
            if (!_segmentMeta.TryGetValue(key, out meta))
            {
                _segmentScores.Remove(key);
                return;
            }

            // 90th percentile — 소수의 노이즈 window 무시, 전반적 패턴 반영
            var sorted = new System.Collections.Generic.List<float>(scores);
            sorted.Sort();
            int pctIdx = (int)Math.Ceiling(sorted.Count * 0.9) - 1;
            if (pctIdx < 0) pctIdx = 0;
            float segScore = sorted[pctIdx];

            var segResult = new InferenceResult
            {
                ModelType       = meta.ModelType,
                SensorType      = meta.SensorType,
                Axis            = meta.Axis,
                ModelFile       = meta.ModelFile,
                AnomalyScore    = segScore,
                Threshold       = meta.Threshold,
                IsAnomaly       = meta.Threshold > 0 && segScore >= meta.Threshold,
                ClassName       = (meta.Threshold > 0 && segScore >= meta.Threshold) ? "anomaly" : "normal",
                RawMae          = segScore,
                RawThreshold    = meta.Threshold,
                IsSegmentResult = true,
                WindowCount     = scores.Count,
            };

            AppEvents.RaiseLog(
                $"[세그먼트] 종료 — {meta.SensorType}" +
                (meta.Axis.HasValue ? $" Ax{meta.Axis.Value}" : "") +
                $"  windows={scores.Count}" +
                $"  90pct={segScore:F3}  thr={meta.Threshold:F3}" +
                $"  → {(segResult.IsAnomaly ? "⚠ 이상" : "✓ 정상")}");

            AppEvents.RaiseInferenceResult(meta.SensorType, segResult);

            _segmentScores.Remove(key);
            _segmentMeta.Remove(key);
        }

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

        /// <summary>서버 /model_info 를 조회해 _windowSizes, _activityRmsThresholds, _intervalMs 를 갱신합니다.</summary>
        private async Task RefreshWindowSizesAsync(CancellationToken ct)
        {
            try
            {
                var info = await _client.GetModelInfoAsync().ConfigureAwait(false);
                if (info == null || info.Count == 0) return;

                foreach (var kv in info)
                {
                    _windowSizes[kv.Key]           = kv.Value.WindowSize;
                    _activityRmsThresholds[kv.Key] = kv.Value.ActivityRmsThr;
                }

                // IntervalMs = 가장 작은 window_size / 2 (stride 50%)
                int minWs = int.MaxValue;
                foreach (var ws in _windowSizes.Values)
                    if (ws < minWs) minWs = ws;
                _intervalMs = minWs / 2;

                AppEvents.RaiseLog(
                    "[추론] 윈도우 크기 동기화: " +
                    string.Join(", ", System.Linq.Enumerable.Select(
                        _windowSizes, kv => $"{kv.Key}={kv.Value}")) +
                    $"  intervalMs={_intervalMs}");

                // activity_rms_thr 가 0 초과인 항목만 로그 출력
                foreach (var kv in _activityRmsThresholds)
                    if (kv.Value > 0.0)
                        AppEvents.RaiseLog(
                            $"[추론] activity_rms_thr 동기화: {kv.Key}={kv.Value:F6} " +
                            "(RMS 미만 윈도우는 Idle로 간주해 추론 건너뜀)");
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

                // ── (1) AE 추론: 항상 실행 (세그먼트 경계는 RunAeInference 내부에서 RMS 기반 감지) ──
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
                int ws = GetWindowSize("combined");
                float[] window = ReadLastWindow(cp, "combined", ws, ax, out nCh,
                    filterOp: false);   // 학습과 동일: Idle+Pos 전체 행 사용

                bool serverSuccess = false;
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
        /// activity_rms_thr 가 설정된 경우 윈도우 RMS 를 먼저 계산하고,
        /// 임계값 미만이면 Idle로 간주해 추론을 건너뜁니다 (훈련 조건과 일치).
        /// </summary>
        private async Task RunAeInference(
            string csvPath, string sensorType, int? axis, CancellationToken ct)
        {
            int nCh;
            int ws = GetWindowSize(sensorType);
            // 학습: Idle+Pos 전체 → 추론도 전체 행 사용 (분포 일치)
            float[] window = ReadLastWindow(csvPath, sensorType, ws, axis, out nCh,
                filterOp: false);
            if (window == null) return;

            // ── RMS 기반 세그먼트 경계 감지 + Idle 필터 ─────────────────────
            double actThr = GetActivityRmsThr(sensorType);
            string segKey = SegKey(sensorType, axis);

            bool useSegmentMode = actThr > 0.0;
            bool nowActive      = !useSegmentMode || ComputeWindowRms(window) >= actThr;

            if (useSegmentMode)
            {
                bool wasActive;
                _segmentWasActive.TryGetValue(segKey, out wasActive);

                if (!wasActive && nowActive)
                    OnSegmentStart(segKey, sensorType, axis);   // Idle → Active
                else if (wasActive && !nowActive)
                    OnSegmentEnd(segKey);                        // Active → Idle: 집계 발행

                _segmentWasActive[segKey] = nowActive;

                if (!nowActive)
                {
                    // Idle 윈도우: score=0 UI 갱신 후 반환
                    AppEvents.RaiseInferenceResult(sensorType,
                        InferenceResult.Idle(sensorType, axis));
                    return;
                }
            }

            InferenceResult result = await _client.PredictAsync(
                window, ws, nCh, sensorType, axis, ct);

            // 모델 없음 (404): 학습 안 된 (sensor_type, axis) — 조용히 무시.
            // 클라이언트가 캐시에 등록하므로 이후엔 네트워크 호출도 없음.
            if (result.IsModelMissing)
                return;

            if (result.IsError)
            {
                AppEvents.RaiseLog(
                    $"[AE 추론 오류] sensorType={sensorType} axis={axis?.ToString() ?? "null"}" +
                    $"  nCh={nCh}  windowLen={window.Length}  →  {result.Error}");
                AppEvents.RaiseInferenceResult(sensorType, result);
                return;
            }

            // combined 폴백용: 최신 개별 AE 스코어 캐시
            if (sensorType == "accel")
                _latestAccelScore = result.AnomalyScore;
            else if (sensorType == "torque" && axis.HasValue)
                _latestTorqueScores[axis.Value] = result.AnomalyScore;

            if (useSegmentMode)
            {
                // ── 세그먼트 모드: score 누적 + 차트용(IsChartOnly) 발행 ──────
                // 경고 판정은 Active→Idle 전환(OnSegmentEnd) 시에만 수행.
                if (!_segmentScores.ContainsKey(segKey))
                    _segmentScores[segKey] = new System.Collections.Generic.List<float>();
                _segmentScores[segKey].Add(result.AnomalyScore);
                _segmentMeta[segKey] = result;

                AppEvents.RaiseInferenceResult(sensorType, new InferenceResult
                {
                    ModelType    = result.ModelType,
                    SensorType   = result.SensorType,
                    Axis         = result.Axis,
                    ModelFile    = result.ModelFile,
                    AnomalyScore = result.AnomalyScore,
                    Threshold    = result.Threshold,
                    IsAnomaly    = false,
                    ClassName    = result.ClassName,
                    RawMae       = result.RawMae,
                    RawThreshold = result.RawThreshold,
                    IsChartOnly  = true,
                });
            }
            else
            {
                // ── per-window 모드 (activity_rms_thr 미설정): 즉시 발행 ──────
                AppEvents.RaiseInferenceResult(sensorType, result);
            }
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
            bool useFilterOp = filterOp ?? (sensorType != "accel");
            int ws = GetWindowSize(sensorType);
            float[] window = ReadLastWindow(csvPath, sensorType, ws, axis, out nCh,
                filterOp: useFilterOp);
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
        /// filterOp=true: Op==Pos 행만 사용 (CLS용)
        /// filterOp=false: 전체 행 사용 (AE용)
        /// 성공 시 float 배열 반환, 실패 시 null.
        /// </summary>
        private static float[] ReadLastWindow(
            string csvPath,
            string sensorType,
            int windowSize,
            int? axis,
            out int nChannels,
            bool filterOp = true)
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
                    // AE: 모든 행 사용
                    dataLines = new List<string>(lines.Length - 1);
                    for (int li = 1; li < lines.Length; li++)
                        dataLines.Add(lines[li]);
                }
                else
                {
                    // CLS: Op==Pos 행만 사용
                    // axis 지정 시: Op_Ax{n} 컬럼 / null 시: 임의의 Op_Ax* 컬럼 중 하나라도 Pos
                    dataLines = FilterPosByOp(lines, headers, axis);
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
        /// 평탄화된 윈도우 배열의 전채널 RMS를 계산합니다.
        /// window: [t0c0, t0c1, … t1c0, …]  길이 = windowSize × nChannels
        /// </summary>
        private static double ComputeWindowRms(float[] window)
        {
            if (window == null || window.Length == 0) return 0.0;
            double sum = 0.0;
            for (int i = 0; i < window.Length; i++)
                sum += (double)window[i] * window[i];
            return Math.Sqrt(sum / window.Length);
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
