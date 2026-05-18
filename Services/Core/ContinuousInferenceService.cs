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

        /// <summary>sensor_type별 window_size 반환. 서버 쿼리 전이거나 없으면 기본값.</summary>
        private int GetWindowSize(string sensorType)
        {
            int ws;
            return _windowSizes.TryGetValue(sensorType, out ws) ? ws : DefaultWindowSize;
        }

        private int? _lastMovingAxis = null;

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

        /// <summary>서버 /model_info 를 조회해 _windowSizes, _intervalMs 를 갱신합니다.</summary>
        private async Task RefreshWindowSizesAsync(CancellationToken ct)
        {
            try
            {
                var info = await _client.GetModelInfoAsync().ConfigureAwait(false);
                if (info == null || info.Count == 0) return;

                foreach (var kv in info)
                    _windowSizes[kv.Key] = kv.Value;

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

                // 토크 AE: 축별 (제어기 미연결 시 axis=null로 단일 추론)
                if (_axes != null)
                    foreach (int ax in _axes)
                        await RunAeInference(cp, "torque", ax, ct);
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

                // 단독 토크 AE: 축별
                if (_torqueLogger != null && _torqueLogger.IsLogging)
                {
                    string p = _torqueLogger.OutputPath;
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                    {
                        if (_axes != null)
                            foreach (int ax in _axes)
                                await RunAeInference(p, "torque", ax, ct);
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

            // combined CLS 모델은 Pos(모션) 구간 데이터로만 학습됨
            // → filterOp 기본값(sensorType != "accel" = true) 그대로 사용: Pos 행만 입력
            // → 정지 상태에서는 Pos 행 없음 → window=null → 추론 스킵 (false alarm 방지)
            int? axis = _lastMovingAxis;

            // filterOp=false: Idle/Pos 전체 행 사용 — 결합 이상 스코어는 항상 갱신
            if (_axes != null)
            {
                foreach (int ax in _axes)
                    await RunClsInference(cp, "combined", ax, ct, filterOp: false);
            }
            else
            {
                await RunClsInference(cp, "combined", axis, ct, filterOp: false);
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        //  개별 추론 메서드
        // ──────────────────────────────────────────────────────────────────────

        /// <summary>
        /// AE 이상탐지 추론 — filterOp=false (전체 데이터), /predict 엔드포인트 사용.
        /// </summary>
        private async Task RunAeInference(
            string csvPath, string sensorType, int? axis, CancellationToken ct)
        {
            int nCh;
            // AE 추론: 모든 센서 타입 filterOp=false (전체 행 사용)
            // torque/combined 재학습 시 filter_op_column=None 적용 예정 → 학습과 일치
            // 현재 모델(Pos 학습)이라도 일단 데이터를 보여주는 것이 평가에 필요
            bool filterOp = false;
            int ws = GetWindowSize(sensorType);
            float[] window = ReadLastWindow(csvPath, sensorType, ws, axis, out nCh,
                filterOp: filterOp);
            if (window == null) return;

            InferenceResult result = await _client.PredictAsync(
                window, ws, nCh, sensorType, axis, ct);

            if (result.IsError)
                AppEvents.RaiseLog(
                    $"[AE 추론 오류] sensorType={sensorType} axis={axis?.ToString() ?? "null"}" +
                    $"  nCh={nCh}  windowLen={window.Length}  →  {result.Error}");

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
            bool useFilterOp = filterOp ?? (sensorType != "accel");
            int ws = GetWindowSize(sensorType);
            float[] window = ReadLastWindow(csvPath, sensorType, ws, axis, out nCh,
                filterOp: useFilterOp);
            if (window == null) return;

            CombinedInferenceResult combined = await _client.PredictCombinedAsync(
                window, ws, nCh, sensorType, axis, ct);

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
                    string target = "ax" + axis.Value.ToString() + "_trq(%)";
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (headers[i].Trim().ToLower() == target)
                            result.Add(i);
                    }
                }

                if (result.Count == 0)
                {
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].Trim().ToLower();
                        if (h.Contains("trq") || h.Contains("torque"))
                            result.Add(i);
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
