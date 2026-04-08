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

        /// <summary>샘플 단위 타임스탬프 계산에 사용. ApplyDaqConfig 후 설정하세요.</summary>
        public double SampleRate { get; set; } = 1000.0;

        /// <summary>InfluxDB 태그 label=값. 빈 문자열이면 태그 미포함.</summary>
        public string Label { get; set; } = "";

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

            // 비동기 연결 확인 (결과가 나오면 로그로만 알림)
            Task.Run(async () =>
            {
                try
                {
                    var req = new HttpRequestMessage(HttpMethod.Get, Config.Url + "/health");
                    var resp = await _http.SendAsync(req).ConfigureAwait(false);
                    if (!resp.IsSuccessStatusCode)
                        _log($"[InfluxDB] 경고: 서버 응답 오류 (HTTP {(int)resp.StatusCode}) — 데이터가 전송되지 않을 수 있습니다.");
                    else
                        _log($"[InfluxDB] 서버 연결 확인 완료");
                }
                catch (Exception ex)
                {
                    _log($"[InfluxDB] 경고: 서버에 연결할 수 없습니다 ({Config.Url}) — {ex.Message}");
                    _log($"[InfluxDB]   데이터 수집은 계속되지만 InfluxDB 전송은 실패합니다.");
                }
            });
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
        /// timestampUtc 는 블록의 마지막 샘플 시각 기준.
        /// </summary>
        public void Feed(string module, double[,] block, DateTime timestampUtc)
        {
            if (!IsEnabled || block == null) return;

            int ch = block.GetLength(0);
            int n  = block.GetLength(1);
            if (ch < 3 || n == 0) return;

            // 블록 마지막 샘플 타임스탬프 (ms)
            long blockEndMs = new DateTimeOffset(timestampUtc, TimeSpan.Zero).ToUnixTimeMilliseconds();
            double intervalMs = 1000.0 / SampleRate;

            // 태그 공통 접두어 (measurement,tag=val )
            var prefix = new StringBuilder();
            prefix.Append("accel");
            prefix.Append(",device=").Append(EscapeTag(module));
            if (_sessionActive && !string.IsNullOrEmpty(_sessionId))
                prefix.Append(",session=").Append(EscapeTag(_sessionId));
            var lbl = Label;
            if (!string.IsNullOrEmpty(lbl))
                prefix.Append(",label=").Append(EscapeTag(lbl));
            string tagPrefix = prefix.ToString();

            // 샘플별 line protocol 조립 — 한 번에 큐에 넣기 (멀티라인 배치)
            var sb = new StringBuilder(n * 80);
            for (int i = 0; i < n; i++)
            {
                // 샘플 i의 타임스탬프: 블록 끝 기준으로 역산
                long tsMs = blockEndMs - (long)Math.Round((n - 1 - i) * intervalMs);

                sb.Append(tagPrefix);
                sb.Append(' ');
                sb.AppendFormat(CultureInfo.InvariantCulture,
                    "x={0:G8},y={1:G8},z={2:G8}",
                    block[0, i], block[1, i], block[2, i]);
                sb.Append(' ').Append(tsMs);
                sb.Append('\n');
            }

            // 멀티라인 배치를 큐에 단일 항목으로 추가
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
