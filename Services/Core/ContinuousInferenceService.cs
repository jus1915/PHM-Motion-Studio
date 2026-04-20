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
        private readonly InferenceServerClient _client;
        private readonly DaqAccelCsvLogger     _accelLogger;
        private readonly AjinCsvLogger         _torqueLogger;

        private CancellationTokenSource _cts;
        private Task                    _loopTask;

        // 추론 주기 (ms)
        private const int IntervalMs = 2000;

        // 동일 오류 반복 로그 억제 (센서 타입별)
        private readonly System.Collections.Generic.Dictionary<string, string> _lastErrorByType
            = new System.Collections.Generic.Dictionary<string, string>();

        public bool IsRunning => _loopTask != null && !_loopTask.IsCompleted;

        public ContinuousInferenceService(
            string             inferenceServerUrl,
            DaqAccelCsvLogger  accelLogger,
            AjinCsvLogger      torqueLogger)
        {
            _client       = new InferenceServerClient(inferenceServerUrl);
            _accelLogger  = accelLogger;
            _torqueLogger = torqueLogger;
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

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(IntervalMs, ct).ConfigureAwait(false);
                }
                catch (TaskCanceledException) { break; }

                // ── 가속도 ────────────────────────────────────────────────────
                if (_accelLogger?.IsRunning == true)
                {
                    string[] paths = _accelLogger.CsvPathByModule;
                    if (paths != null)
                    {
                        foreach (string p in paths)
                        {
                            if (string.IsNullOrEmpty(p) || !File.Exists(p)) continue;
                            await RunInferenceForCsvAsync(p, "accel", ct).ConfigureAwait(false);
                            break; // 첫 번째 모듈만 사용
                        }
                    }
                }

                // ── 토크 ──────────────────────────────────────────────────────
                if (_torqueLogger?.IsLogging == true)
                {
                    string p = _torqueLogger.OutputPath;
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                        await RunInferenceForCsvAsync(p, "torque", ct).ConfigureAwait(false);
                }
            }

            AppEvents.RaiseLog("[추론 서비스] 종료");
        }

        // ── CSV → 윈도우 추출 → 추론 ─────────────────────────────────────────
        private async Task RunInferenceForCsvAsync(
            string            csvPath,
            string            sensorType,
            CancellationToken ct)
        {
            const int WindowSize = 1024;

            try
            {
                var (window, nCh) = ReadLastWindow(csvPath, sensorType, WindowSize);
                if (window == null)
                    return;

                var result = await _client.PredictAsync(
                    window, WindowSize, nCh, sensorType, ct).ConfigureAwait(false);

                if (result.IsError)
                {
                    // 센서 타입별 동일 오류 반복 억제
                    _lastErrorByType.TryGetValue(sensorType, out string prev);
                    if (result.Error != prev)
                    {
                        _lastErrorByType[sensorType] = result.Error;
                        AppEvents.RaiseLog($"[추론 서비스] {sensorType} 오류: {result.Error}");
                    }
                }
                else
                {
                    _lastErrorByType.Remove(sensorType); // 정상 응답이면 리셋
                }

                AppEvents.RaiseInferenceResult(sensorType, result);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                // 예외도 센서 타입별 반복 억제
                string msg = ex.Message;
                _lastErrorByType.TryGetValue(sensorType, out string prev);
                if (msg != prev)
                {
                    _lastErrorByType[sensorType] = msg;
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
            int    windowSize)
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
                int[] signalCols = GetSignalColumnIndices(headers, sensorType);
                if (signalCols.Length == 0) return (null, 0);

                // 데이터 행 (헤더 제외)
                string[] dataLines = lines.Skip(1).ToArray();
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
        private static int[] GetSignalColumnIndices(string[] headers, string sensorType)
        {
            var result = new List<int>();

            if (sensorType == "accel")
            {
                // x, y, z 컬럼
                for (int i = 0; i < headers.Length; i++)
                {
                    string h = headers[i].Trim().ToLowerInvariant();
                    if (h == "x" || h == "y" || h == "z")
                        result.Add(i);
                }
            }
            else // torque
            {
                // Ax0_Trq(%), Ax1_Trq(%), ... — 연결된 모든 축 사용
                // 모델이 학습된 채널 수와 일치해야 하므로 축 번호 순서대로 전부 포함
                for (int i = 0; i < headers.Length; i++)
                {
                    string h = headers[i].Trim().ToLowerInvariant();
                    if (h.Contains("trq") || h.Contains("torque"))
                        result.Add(i);
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
