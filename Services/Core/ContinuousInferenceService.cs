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

        private const int IntervalMs = 128;   // stride(128) / sampleRate(1000Hz) × 1000
        private const int WindowSize = 256;   // 학습 window_size와 동일

        private int? _lastMovingAxis = null;

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

        private string GetCurrentOp()
        {
            if (_getAxisOperation == null || _axes == null)
                return "Pos";

            foreach (int ax in _axes)
                if (_getAxisOperation(ax) == "Pos")
                    return "Pos";

            return "Idle";
        }

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

        private async Task LoopAsync(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                string op = GetCurrentOp();

                if (op == "Idle")
                {
                    try { await Task.Delay(100, ct); }
                    catch { break; }
                    continue;
                }

                // Pos 상태
                try { await Task.Delay(IntervalMs, ct); }
                catch { break; }

                int? curAxis = GetMovingAxis();
                if (curAxis.HasValue)
                    _lastMovingAxis = curAxis;

                int? axis = _lastMovingAxis;

                // ── 통합 CSV (accel + torque 동시 수집) ──────────────
                if (_combinedLogger != null && _combinedLogger.IsLogging)
                {
                    string cp = _combinedLogger.OutputPath;
                    if (!string.IsNullOrEmpty(cp) && File.Exists(cp))
                    {
                        // 가속도 채널 추론
                        await RunInference(cp, "accel",  axis, ct);
                        // 토크 채널 추론
                        await RunInference(cp, "torque", axis, ct);
                    }
                }
                else
                {
                    // ── 단독 가속도 ──────────────────────────────────────
                    if (_accelLogger != null && _accelLogger.IsRunning)
                    {
                        string[] paths = _accelLogger.CsvPathByModule;
                        if (paths != null)
                        {
                            foreach (string p in paths)
                            {
                                if (string.IsNullOrEmpty(p) || !File.Exists(p)) continue;
                                await RunInference(p, "accel", axis, ct);
                                break;
                            }
                        }
                    }

                    // ── 단독 토크 ────────────────────────────────────────
                    if (_torqueLogger != null && _torqueLogger.IsLogging)
                    {
                        string p = _torqueLogger.OutputPath;
                        if (!string.IsNullOrEmpty(p) && File.Exists(p))
                            await RunInference(p, "torque", axis, ct);
                    }
                }
            }
        }

        private async Task RunInference(
            string csvPath,
            string sensorType,
            int? axis,
            CancellationToken ct)
        {
            int nCh;
            float[] window = ReadLastWindow(csvPath, sensorType, WindowSize, axis, out nCh);
            if (window == null) return;

            // ── /predict/combined 호출 (AE 이상탐지 + CLS 결함진단 동시) ────
            CombinedInferenceResult combined = await _client.PredictCombinedAsync(
                window, WindowSize, nCh, sensorType, axis, ct);

            if (combined.IsError)
            {
                // combined 엔드포인트 실패(구버전 서버 등) → /predict 폴백
                InferenceResult fallback = await _client.PredictAsync(
                    window, WindowSize, nCh, sensorType, axis, ct);
                AppEvents.RaiseInferenceResult(sensorType, fallback);
                return;
            }

            // AE 결과 발행 (기존 InferenceResultReceived 구독자용)
            AppEvents.RaiseInferenceResult(sensorType, combined.ToAeResult());

            // CLS 결과 발행 (ClsAvailable=false 이면 "모델 없음" 상태로 발행)
            AppEvents.RaiseClsInferenceResult(sensorType, combined);
        }

        /// <summary>
        /// CSV 끝 windowSize 행에서 신호 윈도우를 읽습니다.
        /// 성공 시 float 배열 반환, 실패 시 null.
        /// </summary>
        private static float[] ReadLastWindow(
            string csvPath,
            string sensorType,
            int windowSize,
            int? axis,
            out int nChannels)
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

                // Op 컬럼 인덱스 탐색
                int targetOpCol = -1;
                if (axis.HasValue)
                {
                    string opName = "Op_Ax" + axis.Value.ToString();
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (string.Equals(headers[i].Trim(), opName,
                                StringComparison.OrdinalIgnoreCase))
                        {
                            targetOpCol = i;
                            break;
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (string.Equals(headers[i].Trim(), "Op",
                                StringComparison.OrdinalIgnoreCase))
                        {
                            targetOpCol = i;
                            break;
                        }
                    }
                }

                // Pos 행만 필터
                List<string> dataLines = new List<string>();
                for (int li = 1; li < lines.Length; li++)
                {
                    string line = lines[li];
                    if (targetOpCol < 0)
                    {
                        dataLines.Add(line);
                        continue;
                    }
                    string[] cols = line.Split(',');
                    if (targetOpCol < cols.Length &&
                        string.Equals(cols[targetOpCol].Trim(), "Pos",
                            StringComparison.OrdinalIgnoreCase))
                    {
                        dataLines.Add(line);
                    }
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

        private static int[] GetSignalColumnIndices(
            string[] headers, string sensorType, int? axis)
        {
            List<int> result = new List<int>();

            if (sensorType == "accel")
            {
                // 통합 CSV: Ax{n}_x, Ax{n}_y, Ax{n}_z (축별 명시)
                if (axis.HasValue)
                {
                    string px = ("ax" + axis.Value + "_x");
                    string py = ("ax" + axis.Value + "_y");
                    string pz = ("ax" + axis.Value + "_z");
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].Trim().ToLower();
                        if (h == px || h == py || h == pz)
                            result.Add(i);
                    }
                }
                // 단독 가속도 CSV 폴백: x, y, z
                if (result.Count == 0)
                {
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].Trim().ToLower();
                        if (h == "x" || h == "y" || h == "z")
                            result.Add(i);
                    }
                }
            }
            else
            {
                if (axis.HasValue)
                {
                    string target = "ax" + axis.Value.ToString() + "_trq(%)";
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (headers[i].ToLower() == target)
                            result.Add(i);
                    }
                }

                if (result.Count == 0)
                {
                    for (int i = 0; i < headers.Length; i++)
                    {
                        string h = headers[i].ToLower();
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
