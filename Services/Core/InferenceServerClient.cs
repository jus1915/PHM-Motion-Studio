using System;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace PHM_Project_DockPanel.Services.Core
{
    // =========================================================================
    //  InferenceServerClient — FastAPI PHM 추론 서버 HTTP 클라이언트
    //  POST /predict  : 신호 윈도우 → 이상탐지/분류 결과
    //  GET  /health   : 서버 상태 & 로드된 모델 목록
    // =========================================================================
    public sealed class InferenceServerClient : IDisposable
    {
        private readonly HttpClient _http;

        public InferenceServerClient(string baseUrl)
        {
            _http = new HttpClient
            {
                BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/"),
                Timeout     = TimeSpan.FromSeconds(10),
            };
        }

        // ── 헬스 체크 ─────────────────────────────────────────────────────────
        public async Task<bool> IsHealthyAsync()
        {
            try
            {
                var resp = await _http.GetAsync("health").ConfigureAwait(false);
                return resp.IsSuccessStatusCode;
            }
            catch { return false; }
        }

        // ── 모델 캐시 재로드 ──────────────────────────────────────────────────
        public async Task ReloadModelsAsync()
        {
            try { await _http.GetAsync("models/reload").ConfigureAwait(false); }
            catch { }
        }

        // ── 추론 요청 ─────────────────────────────────────────────────────────
        /// <summary>
        /// 신호 윈도우 하나를 서버에 보내 이상탐지/분류 결과를 받습니다.
        /// </summary>
        /// <param name="window">
        ///   평탄화된 float 배열, 길이 = windowSize × nChannels.
        ///   채널 순서: time 방향으로 interleaved — [t0c0, t0c1, … t1c0, t1c1, …]
        /// </param>
        /// <param name="windowSize">시간축 샘플 수 (예: 1024)</param>
        /// <param name="nChannels">채널 수 (예: accel=3, torque=1)</param>
        /// <param name="sensorType">"accel" 또는 "torque"</param>
        /// <param name="ct">취소 토큰</param>
        public async Task<InferenceResult> PredictAsync(
            float[]           window,
            int               windowSize,
            int               nChannels,
            string            sensorType = "accel",
            CancellationToken ct         = default)
        {
            var req = new
            {
                sensor_type = sensorType,
                window      = window,
                window_size = windowSize,
                n_channels  = nChannels,
            };

            string json     = JsonConvert.SerializeObject(req);
            var    content  = new StringContent(json, Encoding.UTF8, "application/json");

            HttpResponseMessage resp;
            try
            {
                resp = await _http.PostAsync("predict", content, ct).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                return InferenceResult.Fail($"연결 오류: {ex.Message}");
            }

            string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
                return InferenceResult.Fail($"HTTP {(int)resp.StatusCode}: {body}");

            try
            {
                return JsonConvert.DeserializeObject<InferenceResult>(body)
                       ?? InferenceResult.Fail("응답 역직렬화 실패");
            }
            catch (Exception ex)
            {
                return InferenceResult.Fail($"응답 파싱 오류: {ex.Message}");
            }
        }

        public void Dispose() => _http.Dispose();
    }

    // =========================================================================
    //  InferenceResult — /predict 응답 DTO
    // =========================================================================
    public sealed class InferenceResult
    {
        [JsonProperty("model_type")]    public string ModelType    { get; set; } = "";
        [JsonProperty("sensor_type")]   public string SensorType   { get; set; } = "";
        [JsonProperty("is_anomaly")]    public bool   IsAnomaly    { get; set; }
        [JsonProperty("anomaly_score")] public float  AnomalyScore { get; set; }
        [JsonProperty("threshold")]     public float  Threshold    { get; set; } = 1.0f;
        [JsonProperty("class_name")]    public string ClassName    { get; set; } = "";
        [JsonProperty("confidence")]    public float? Confidence   { get; set; }
        [JsonProperty("raw_mae")]       public float? RawMae       { get; set; }
        [JsonProperty("raw_threshold")] public float? RawThreshold { get; set; }

        /// <summary>네트워크/파싱 오류 시 설정되는 에러 메시지. null 이면 정상.</summary>
        [JsonIgnore] public string Error { get; set; }
        [JsonIgnore] public bool   IsError => Error != null;

        /// <summary>정규화 이상 점수 백분율 (0~100+). 100 초과 시 이상.</summary>
        [JsonIgnore] public float ScorePercent => AnomalyScore * 100f;

        public static InferenceResult Fail(string error) =>
            new InferenceResult { Error = error };

        public override string ToString()
        {
            if (IsError) return $"[오류] {Error}";
            string cls   = string.IsNullOrEmpty(ClassName) ? "" : $" ({ClassName})";
            string conf  = Confidence.HasValue ? $"  신뢰도={Confidence:P0}" : "";
            return $"{(IsAnomaly ? "⚠ 이상" : "✓ 정상")}{cls}  점수={AnomalyScore:F3}{conf}";
        }
    }
}
