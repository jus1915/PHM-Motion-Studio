using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using PHM_Project_DockPanel.Services.DAQ;
using PHM_Project_DockPanel.Services.WMX;

namespace PHM_Project_DockPanel.Services.Core
{
    // =========================================================================
    //  ContinuousInferenceService
    //  연속 수집 중 주기적으로 최신 CSV 데이터를 읽어 추론 서버에 전송합니다.
    //
    //  동작:
    //    Start() → 백그라운드 루프 시작
    //    Stop()  → 루프 취소
    //
    //  루프마다:
    //    1. 가속도/토크 CSV 마지막 windowSize 행 읽기
    //    2. InferenceServerClient.PredictAsync() 호출
    //    3. AppEvents.RaiseInferenceResult() 발행
    // =========================================================================
    public sealed class ContinuousInferenceService : IDisposable
    {
        private readonly InferenceServerClient  _client;
        private readonly DaqAccelCsvLogger      _accelLogger;
        private readonly AjinCsvLogger          _torqueLogger;

        private CancellationTokenSource _cts;
        private Task                    _loopTask;
        private readonly Func<string>         _getOperation;      // 전역 (레거시)
        private readonly Func<int, string>    _getAxisOperation;  // per-axis
        private readonly int[]               _axes;              // 모니터링 대상 축

        // 추론 주기 (ms)
        private const int IntervalMs = 500;

        // 동일 오류 반복 로그 억제 (키 = "sensor_type" 또는 "sensor_type_ax{n}")
        private readonly System.Collections.Generic.Dictionary<string, string> _lastErrorByType
            = new System.Collections.Generic.Dictionary<string, string>();

        public bool IsRunning => _loopTask != null && !_loopTask.IsCompleted;

        /// <param name="getOperation">전역 동작 상태 콜백 (레거시 호환용, null 가능)</param>
        /// <param name="getAxisOperation">
        ///   per-axis 동작 상태 콜백 (int axisIndex → "Pos"/"Idle").
        ///   지정 시 axes 배열도 함께 전달해야 합니다.
        /// </param>
        /// <param name="axes">per-axis 모니터링 대상 축 인덱스 배열</param>
        public ContinuousInferenceService(
            string              inferenceServerUrl,
            DaqAccelCsvLogger   accelLogger,
            AjinCsvLogger       torqueLogger,
            Func<string>        getOperation     = null,
            Func<int, string>   getAxisOperation = null,
            int[]               axes             = null)
        {
            _client           = new InferenceServerClient(inferenceServerUrl);
            _accelLogger      = accelLogger;
            _torqueLogger     = torqueLogger;
            _getOperation     = getOperation;
            _getAxisOperation = getAxisOperation;
            _axes             = axes;
        }

        // ── 현재 전역 동작 상태 ─────────────────────────────────────────────────
        /// <summary>어느 축이라도 Pos 이면 "Pos", 모두 Idle 이면 "Idle".</summary>
        private string GetCurrentOp()
        {
            if (_getAxisOperation != null && _axes != null)
            {
                foreach (int ax in _axes)
                    if (_getAxisOperation(ax) == "Pos") return "Pos";
                return "Idle";
            }
            return _getOperation?.Invoke() ?? "Pos";
        }

        /// <summary>현재 Pos 상태인 첫 번째 축 인덱스. 없거나 per-axis 콜백 없으면 null.</summary>
        private int? GetMovingAxis()
        {
            if (_getAxisOperation == null || _axes == null) return null;
            foreach (int ax in _axes)
                if (_getAxisOperation(ax) == "Pos") return ax;
            return null;
        }

        // ── 시작 / 중지 ────────────────────────────────────────────────────────
        public void Start()
        {
            if (IsRunning) return;
            _cts      = new CancellationTokenSource();
            _loopTask = Task.Run(() => LoopAsync(_cts.Token));
        }

        public void Stop()
        {
            _cts?.Cancel();
        }

        // ── 추론 루프 ──────────────────────────────────────────────────────────
        private async Task LoopAsync(CancellationToken ct)
        {
            AppEvents.RaiseLog("[추론 서비스] 시작");

            string _prevOp = "Idle";
            while (!ct.IsCancellationRequested)
            {
                string _curOp      = GetCurrentOp();
                bool   _justStarted = (_prevOp == "Idle" && _curOp == "Pos");
                _prevOp = _curOp;

                if (_curOp == "Idle")
                {
                    // Idle: 100ms 간격으로 상태 재확인
                    try { await Task.Delay(100, ct).ConfigureAwait(false); }
                    catch (TaskCanceledException) { break; }
                    continue;
                }

                // Pos: 직전 호출 직후라면 지연 없이 바로 추론, 이후엔 IntervalMs 대기
                if (!_justStarted)
                {
                    try { await Task.Delay(IntervalMs, ct).ConfigureAwait(false); }
                    catch (TaskCanceledException) { break; }
                }

                // 현재 움직이는 축 (per-axis 모델 선택용)
                int? _movingAxis = GetMovingAxis();

                // ── 가속도 ────────────────────────────────────────────────────
                if (_accelLogger?.IsRunning == true)
                {
                    string[] paths = _accelLogger.CsvPathByModule;
                    if (paths != null)
                    {
                        foreach (string p in paths)
                        {
                            if (string.IsNullOrEmpty(p) || !File.Exists(p)) continue;
                            // per-axis 모델: 움직이는 축 전달 → ae_fd_ax{n}.onnx 우선
                            await RunInferenceForCsvAsync(p, "accel", _movingAxis, ct)
                                  .ConfigureAwait(false);
                            break; // 첫 번째 모듈만 사용
                        }
                    }
                }

                // ── 토크 ──────────────────────────────────────────────────────
                if (_torqueLogger?.IsLogging == true)
                {
                    string p = _torqueLogger.OutputPath;
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                        // 토크도 per-axis 모델 사용: ae_torque_ax{n}.onnx 우선
                        await RunInferenceForCsvAsync(p, "torque", _movingAxis, ct)
                              .ConfigureAwait(false);
                }
            }

            AppEvents.RaiseLog("[추론 서비스] 종료");
        }

        // ── CSV → 윈도우 추출 → 추론 ─────────────────────────────────────────
        private async Task RunInferenceForCsvAsync(
            string            csvPath,
            string            sensorType,
            int?              axis,
            CancellationToken ct)
        {
            const int WindowSize = 1024;
            string _errKey = axis.HasValue ? $"{sensorType}_ax{axis}" : sensorType;

            try
            {
                var (window, nCh) = ReadLastWindow(csvPath, sensorType, WindowSize, axis);
                if (window == null)
                    return;

                var result = await _client.PredictAsync(
                    window, WindowSize, nCh, sensorType, axis, ct).ConfigureAwait(false);

                if (result.IsError)
                {
                    // 센서 타입별 동일 오류 반복 억제
                    _lastErrorByType.TryGetValue(_errKey, out string prev);
                    if (result.Error != prev)
                    {
                        _lastErrorByType[_errKey] = result.Error;
                        AppEvents.RaiseLog($"[추론 서비스] {sensorType} 오류: {result.Error}");
                    }
                }
                else
                {
                    _lastErrorByType.Remove(_errKey); // 정상 응답이면 리셋
                }

                AppEvents.RaiseInferenceResult(sensorType, result);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                // 예외도 센서 타입별 반복 억제
                string msg = ex.Message;
                _lastErrorByType.TryGetValue(_errKey, out string prev);
                if (msg != prev)
                {
                    _lastErrorByType[_errKey] = msg;
                    AppEvents.RaiseLog($"[추론 서비스] {sensorType} 오류: {msg}");
                }
            }
        }

        // ── CSV 마지막 N 행 읽기 ───────────────────────────────────────────────
        /// <summary>
        /// CSV 파일 끝에서 windowSize 행을 읽어 (flat_array, n_channels) 반환합니다.
        /// 행이 부족하면 null 반환.
        /// </summary>
        private static (float[] window, int nChannels) ReadLastWindow(
            string csvPath,
            string sensorType,
            int    windowSize,
            int?   axis = null)
        {
            try
            {
                string[] lines;
                // 파일이 열려있을 수 있으므로 FileShare.ReadWrite
                using (var fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs))
                    lines = sr.ReadToEnd()
                              .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

                if (lines.Length < 2) return (null, 0); // 헤더만 있음

                // 헤더 파싱 → 신호 컬럼 인덱스 결정
                string[] headers = lines[0].Split(',');
                int[] signalCols = GetSignalColumnIndices(headers, sensorType, axis);
                if (signalCols.Length == 0) return (null, 0);

                // 데이터 행 (헤더 제외)
                // Op==Idle ???쒓굅 ???곗씠???쇱씤 援ъ꽦
                // Op_Ax{n} ?먮뒗 ?덇굅??Op 而щ읆 ??Idle ???쒓굅
                var _opCols = new System.Collections.Generic.List<int>();
                for (int _i = 0; _i < headers.Length; _i++)
                {
                    string _h = headers[_i].Trim();
                    if (_h.StartsWith("Op_Ax", System.StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(_h, "Op", System.StringComparison.OrdinalIgnoreCase))
                        _opCols.Add(_i);
                }
                int[] _opColArr = _opCols.ToArray();
                string[] dataLines = lines.Skip(1)
                    .Where(_ln =>
                    {
                        if (_opColArr.Length == 0) return true;
                        var _cols = _ln.Split(',');
                        foreach (int _oci in _opColArr)
                        {
                            if (_oci >= _cols.Length) continue;
                            if (!string.Equals(_cols[_oci].Trim(), "Idle",
                                    System.StringComparison.OrdinalIgnoreCase))
                                return true;
                        }
                        return false;
                    })
                    .ToArray();
                if (dataLines.Length < windowSize)
                    return (null, 0); // 데이터 부족

                // 마지막 windowSize 행
                string[] slice = dataLines.Skip(dataLines.Length - windowSize).ToArray();

                int nCh    = signalCols.Length;
                float[] w  = new float[windowSize * nCh];
                int writeIdx = 0;

                foreach (string row in slice)
                {
                    string[] cols = row.Split(',');
                    foreach (int ci in signalCols)
                    {
                        if (ci < cols.Length &&
                            float.TryParse(cols[ci], NumberStyles.Float,
                                           CultureInfo.InvariantCulture, out float v))
                            w[writeIdx] = v;
                        writeIdx++;
                    }
                }

                return (w, nCh);
            }
            catch
            {
                return (null, 0);
            }
        }

        // ── 컬럼 인덱스 결정 ──────────────────────────────────────────────────
        /// <param name="axis">
        ///   per-axis 모드 시 축 인덱스.
        ///   torque: axis 지정 → Ax{n}_Trq(%) 단일 컬럼, null → 전체 Trq 컬럼.
        ///   accel: axis 무관 — 항상 x/y/z 전체 반환 (모델이 3채널 고정).
        /// </param>
        private static int[] GetSignalColumnIndices(string[] headers, string sensorType, int? axis = null)
        {
            var result = new List<int>();

            if (sensorType == "accel")
            {
                // x, y, z 컬럼 (axis와 무관 — accel은 단일 센서 3채널 고정)
                for (int i = 0; i < headers.Length; i++)
                {
                    string h = headers[i].Trim().ToLowerInvariant();
                    if (h == "x" || h == "y" || h == "z")
                        result.Add(i);
                }
            }
            else // torque
            {
                if (axis.HasValue)
                {
                    // per-axis: Ax{n}_Trq(%) 단일 컬럼 우선
                    string target = $"ax{axis.Value}_trq(%)";
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (headers[i].Trim().ToLowerInvariant() == target)
                        {
                            result.Add(i);
                            break;
                        }
                    }
                }

                // per-axis 미발견(result 비어있음) 또는 axis=null → 전체 Trq 컬럼 폴백
                if (result.Count == 0)
                {
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].Trim().ToLowerInvariant();
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
            _client?.Dispose();
        }
    }
}
