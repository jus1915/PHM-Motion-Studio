using Newtonsoft.Json;
using PHM_Project_DockPanel.Services;
using System;
using System.Collections.Concurrent;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.DAQ
{
    // =========================================================================
    //  InfluxDB 연결 설정 (JSON 파일로 저장/로드)
    // =========================================================================
    public sealed class InfluxConfig
    {
        public string Url    { get; set; } = "http://localhost:8086";
        public string Token  { get; set; } = "my-super-secret-token";
        public string Org    { get; set; } = "daq_org";
        public string Bucket { get; set; } = "vibration";

        public static InfluxConfig LoadOrDefault(string path)
        {
            try
            {
                if (File.Exists(path))
                    return JsonConvert.DeserializeObject<InfluxConfig>(File.ReadAllText(path))
                           ?? new InfluxConfig();
            }
            catch { }
            return new InfluxConfig();
        }

        public void Save(string path)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
                File.WriteAllText(path, JsonConvert.SerializeObject(this, Formatting.Indented), Encoding.UTF8);
            }
            catch { }
        }
    }

    // =========================================================================
    //  AccelInfluxPublisher
    //  - 자체 DAQ 태스크 없음 (리소스 충돌 없음)
    //  - DaqAccelCsvLogger.BlockReceived 또는 PassiveMonitorForm.BlockPublished
    //    콜백을 통해 block[channels, samples] (g 단위) 을 수신
    //  - RMS 계산 후 InfluxDB line protocol HTTP API 로 Write
    // =========================================================================
    public sealed class AccelInfluxPublisher : IDisposable
    {
        // ── 설정 ────────────────────────────────────────────────────────────
        public InfluxConfig Config { get; set; }
        public bool IsEnabled { get; private set; }

        // ── 세션 태그 (PHM_Motion RunAsync 에서 주입) ───────────────────────
        private volatile string _sessionId;
        private volatile bool   _sessionActive;

        // ── 런타임 ──────────────────────────────────────────────────────────
        private readonly BlockingCollection<string> _queue = new BlockingCollection<string>(500);
        private readonly HttpClient _http = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
        private CancellationTokenSource _cts;
        private Task _worker;
        private readonly Action<string> _log;

        public AccelInfluxPublisher(InfluxConfig config, Action<string> log = null)
        {
            Config = config ?? new InfluxConfig();
            _log   = log ?? (_ => { });
        }

        // ── 활성화/비활성화 ─────────────────────────────────────────────────
        public void Enable()
        {
            if (IsEnabled) return;
            IsEnabled = true;
            _cts = new CancellationTokenSource();
            _worker = Task.Run(() => WorkerLoop(_cts.Token));
            _log($"[InfluxDB] 실시간 게시 시작 → {Config.Url}  org={Config.Org}  bucket={Config.Bucket}");
        }

        public void Disable()
        {
            if (!IsEnabled) return;
            IsEnabled = false;
            _cts?.Cancel();
            _log("[InfluxDB] 실시간 게시 중지");
        }

        // ── 세션 (PHM_Motion 에서 호출) ─────────────────────────────────────
        public void BeginSession(string sessionId, string robotId = null, int[] axes = null)
        {
            _sessionId     = sessionId;
            _sessionActive = true;
        }

        public void EndSession()
        {
            _sessionActive = false;
            _sessionId     = null;
        }

        // ── 데이터 수신 콜백 ─────────────────────────────────────────────────
        /// <summary>
        /// DaqAccelCsvLogger.BlockReceived 또는 PassiveMonitorForm.BlockPublished 에서 호출됩니다.
        /// block[channels, samples] — 이미 g 단위.
        /// </summary>
        public void Feed(string module, double[,] block, DateTime timestampUtc)
        {
            if (!IsEnabled || block == null) return;

            int ch = block.GetLength(0);
            int n  = block.GetLength(1);
            if (n == 0) return;

            // 채널별 RMS
            var rms = new double[ch];
            for (int c = 0; c < ch; c++)
            {
                double sum2 = 0;
                for (int i = 0; i < n; i++) { double v = block[c, i]; sum2 += v * v; }
                rms[c] = Math.Sqrt(sum2 / n);
            }

            long tsMs = new DateTimeOffset(timestampUtc, TimeSpan.Zero).ToUnixTimeMilliseconds();

            // InfluxDB line protocol
            var sb = new StringBuilder();
            sb.Append("vibration_rms");
            sb.Append(",device=").Append(EscapeTag(module));
            if (_sessionActive && !string.IsNullOrEmpty(_sessionId))
                sb.Append(",session=").Append(EscapeTag(_sessionId));
            sb.Append(' ');

            bool first = true;
            if (ch >= 3)
            {
                sb.AppendFormat(CultureInfo.InvariantCulture,
                    "rms_x={0:G6},rms_y={1:G6},rms_z={2:G6}", rms[0], rms[1], rms[2]);
                first = false;
            }
            for (int c = 0; c < ch; c++)
            {
                if (!first) sb.Append(',');
                sb.AppendFormat(CultureInfo.InvariantCulture, "rms_{0}={1:G6}", c, rms[c]);
                first = false;
            }

            sb.Append(' ').Append(tsMs);

            try { _queue.TryAdd(sb.ToString()); } catch { }
        }

        // ── 백그라운드 워커 ─────────────────────────────────────────────────
        private async Task WorkerLoop(CancellationToken ct)
        {
            string writeUrl =
                $"{Config.Url.TrimEnd('/')}/api/v2/write" +
                $"?org={Uri.EscapeDataString(Config.Org)}" +
                $"&bucket={Uri.EscapeDataString(Config.Bucket)}" +
                $"&precision=ms";

            _http.DefaultRequestHeaders.Remove("Authorization");
            _http.DefaultRequestHeaders.Add("Authorization", $"Token {Config.Token}");

            int _errCount = 0;

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    string line;
                    if (_queue.TryTake(out line, 200, ct))
                    {
                        var content = new StringContent(line, Encoding.UTF8, "text/plain");
                        try
                        {
                            var resp = await _http.PostAsync(writeUrl, content, ct).ConfigureAwait(false);
                            if (resp.IsSuccessStatusCode)
                            {
                                if (_errCount > 0)
                                {
                                    _log("[InfluxDB] 연결 복구됨");
                                    _errCount = 0;
                                }
                            }
                            else
                            {
                                string body = "";
                                try { body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false); } catch { }
                                _errCount++;
                                if (_errCount <= 3 || _errCount % 20 == 0)
                                    _log($"[InfluxDB] Write 실패 ({(int)resp.StatusCode}): {body.Trim()}");
                            }
                        }
                        catch (Exception ex) when (!(ex is OperationCanceledException))
                        {
                            _errCount++;
                            // 첫 3회, 이후 20회마다만 로깅 (로그 스팸 방지)
                            if (_errCount <= 3 || _errCount % 20 == 0)
                            {
                                var inner = ex.InnerException?.Message ?? ex.Message;
                                _log($"[InfluxDB] 전송 오류 (#{_errCount}): {inner}  → {writeUrl}");
                            }

                            // 연결 실패 시 잠시 대기 (재시도 폭풍 방지)
                            if (_errCount >= 3)
                                await Task.Delay(2000, ct).ConfigureAwait(false);
                        }
                    }
                }
                catch (OperationCanceledException) { break; }
            }
        }

        private static string EscapeTag(string s)
            => (s ?? "unknown").Replace(" ", "\\ ").Replace(",", "\\,").Replace("=", "\\=");

        public void Dispose()
        {
            Disable();
            _http?.Dispose();
            try { _queue?.Dispose(); } catch { }
        }
    }
}
