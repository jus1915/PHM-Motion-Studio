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
    public sealed class ContinuousInferenceService : IDisposable
    {
        private readonly InferenceServerClient _client;
        private readonly DaqAccelCsvLogger _accelLogger;
        private readonly AjinCsvLogger _torqueLogger;

        private readonly Func<int, string> _getAxisOperation;
        private readonly int[] _axes;

        private CancellationTokenSource _cts;
        private Task _loopTask;

        private const int IntervalMs = 250;
        private const int WindowSize = 256;

        private int? _lastMovingAxis = null;

        public bool IsRunning => _loopTask != null && !_loopTask.IsCompleted;

        public ContinuousInferenceService(
            string inferenceServerUrl,
            DaqAccelCsvLogger accelLogger,
            AjinCsvLogger torqueLogger,
            Func<int, string> getAxisOperation = null,
            int[] axes = null)
        {
            _client = new InferenceServerClient(inferenceServerUrl);
            _accelLogger = accelLogger;
            _torqueLogger = torqueLogger;
            _getAxisOperation = getAxisOperation;
            _axes = axes;
        }

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

                // ✅ Idle 상태 → 아무 것도 안 함
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

                // ── 가속도 ───────────────────────
                if (_accelLogger?.IsRunning == true)
                {
                    var paths = _accelLogger.CsvPathByModule;
                    if (paths != null)
                    {
                        foreach (var p in paths)
                        {
                            if (string.IsNullOrEmpty(p) || !File.Exists(p))
                                continue;

                            await RunInference(p, "accel", axis, ct);
                            break;
                        }
                    }
                }

                // ── 토크 ─────────────────────────
                if (_torqueLogger?.IsLogging == true)
                {
                    var p = _torqueLogger.OutputPath;
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                        await RunInference(p, "torque", axis, ct);
                }
            }
        }

        private async Task RunInference(
            string csvPath,
            string sensorType,
            int? axis,
            CancellationToken ct)
        {
            var (window, nCh) = ReadLastWindow(csvPath, sensorType, WindowSize, axis);
            if (window == null) return;

            var result = await _client.PredictAsync(
                window, WindowSize, nCh, sensorType, axis, ct);

            AppEvents.RaiseInferenceResult(sensorType, result);
        }

        private static (float[] window, int nChannels) ReadLastWindow(
    string csvPath,
    string sensorType,
    int windowSize,
    int? axis = null)
        {
            try
            {
                string[] lines;

                using (var fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs))
                    lines = sr.ReadToEnd()
                        .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

                if (lines.Length < 2) return (null, 0);

                string[] headers = lines[0].Split(',');

                int[] signalCols = GetSignalColumnIndices(headers, sensorType, axis);
                if (signalCols.Length == 0) return (null, 0);

                int targetOpCol = -1;

                if (axis.HasValue)
                {
                    string opName = $"Op_Ax{axis.Value}";

                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (string.Equals(headers[i].Trim(), opName, StringComparison.OrdinalIgnoreCase))
                        {
                            targetOpCol = i;
                            break;
                        }
                    }
                }
                else
                {
                    // fallback (거의 안씀)
                    for (int i = 0; i < headers.Length; i++)
                    {
                        if (string.Equals(headers[i].Trim(), "Op", StringComparison.OrdinalIgnoreCase))
                        {
                            targetOpCol = i;
                            break;
                        }
                    }
                }

                var dataLines = lines.Skip(1)
                    .Where(line =>
                    {
                        if (targetOpCol < 0) return true;

                        var cols = line.Split(',');
                        if (targetOpCol >= cols.Length) return false;

                        return string.Equals(
                            cols[targetOpCol].Trim(),
                            "Pos",
                            StringComparison.OrdinalIgnoreCase);
                    })
                    .ToArray();

                if (dataLines.Length < windowSize)
                    return (null, 0);

                var slice = dataLines.Skip(dataLines.Length - windowSize).ToArray();

                int nCh = signalCols.Length;
                float[] w = new float[windowSize * nCh];

                int idx = 0;

                foreach (var row in slice)
                {
                    var cols = row.Split(',');

                    foreach (var ci in signalCols)
                    {
                        if (ci < cols.Length &&
                            float.TryParse(cols[ci], NumberStyles.Float,
                                CultureInfo.InvariantCulture, out float v))
                        {
                            w[idx] = v;
                        }

                        idx++;
                    }
                }

                return (w, nCh);
            }
            catch
            {
                return (null, 0);
            }
        }

        private static int[] GetSignalColumnIndices(string[] headers, string sensorType, int? axis)
        {
            var result = new List<int>();

            if (sensorType == "accel")
            {
                for (int i = 0; i < headers.Length; i++)
                {
                    var h = headers[i].ToLower();
                    if (h == "x" || h == "y" || h == "z")
                        result.Add(i);
                }
            }
            else
            {
                if (axis.HasValue)
                {
                    string target = $"ax{axis}_trq(%)";
                    for (int i = 0; i < headers.Length; i++)
                        if (headers[i].ToLower() == target)
                            result.Add(i);
                }

                if (result.Count == 0)
                {
                    for (int i = 0; i < headers.Length; i++)
                    {
                        var h = headers[i].ToLower();
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