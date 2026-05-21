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
        /// <param name="axis">
        ///   per-axis 모델 사용 시 축 인덱스 (0, 1, …). null 이면 레거시/전축 모델 사용.
        ///   ae_fd_ax{axis}.onnx 우선 로드, 없으면 ae_fd.onnx 로 폴백.
        /// </param>
        /// <param name="ct">취소 토큰</param>
        public async Task<InferenceResult> PredictAsync(
            float[]           window,
            int               windowSize,
            int               nChannels,
            string            sensorType = "accel",
            int?              axis       = null,
            CancellationToken ct         = default)
        {
            // axis null 이면 JSON 에 포함하지 않음 (서버 기본값 사용)
            object req = axis.HasValue
                ? (object)new
                  {
                      sensor_type = sensorType,
                      axis        = axis.Value,
                      window      = window,
                      window_size = windowSize,
                      n_channels  = nChannels,
                  }
                : new
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

        // ── /predict/combined 요청 ────────────────────────────────────────────
        /// <summary>
        /// AE 이상탐지 + CLS 결함진단을 동시에 요청합니다.
        /// AE 모델 없으면 IsError=true 반환.
        /// CLS 모델 없으면 IsError=false, ClsAvailable=false 반환.
        /// </summary>
        public async Task<CombinedInferenceResult> PredictCombinedAsync(
            float[]           window,
            int               windowSize,
            int               nChannels,
            string            sensorType = "accel",
            int?              axis       = null,
            CancellationToken ct         = default)
        {
            object req = axis.HasValue
                ? (object)new
                  {
                      sensor_type = sensorType,
                      axis        = axis.Value,
                      window      = window,
                      window_size = windowSize,
                      n_channels  = nChannels,
                  }
                : new
                  {
                      sensor_type = sensorType,
                      window      = window,
                      window_size = windowSize,
                      n_channels  = nChannels,
                  };

            string json    = JsonConvert.SerializeObject(req);
            var    content = new StringContent(json, Encoding.UTF8, "application/json");

            HttpResponseMessage resp;
            try
            {
                resp = await _http.PostAsync("predict/combined", content, ct).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                return CombinedInferenceResult.Fail($"연결 오류: {ex.Message}");
            }

            string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
                return CombinedInferenceResult.Fail($"HTTP {(int)resp.StatusCode}: {body}");

            try
            {
                return JsonConvert.DeserializeObject<CombinedInferenceResult>(body)
                       ?? CombinedInferenceResult.Fail("응답 역직렬화 실패");
            }
            catch (Exception ex)
            {
                return CombinedInferenceResult.Fail($"응답 파싱 오류: {ex.Message}");
            }
        }

        // ── 모델 윈도우 크기 조회 ─────────────────────────────────────────────
        /// <summary>
        /// /model_info 를 호출해 sensor_type → window_size 매핑을 반환합니다.
        /// 실패 시 null 반환 (호출 측에서 기본값 사용).
        /// </summary>
        public async Task<System.Collections.Generic.Dictionary<string, int>> GetModelInfoAsync()
        {
            try
            {
                var resp = await _http.GetAsync("model_info").ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode) return null;
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                var raw = JsonConvert.DeserializeObject<
                    System.Collections.Generic.Dictionary<string, ModelWindowInfo>>(body);
                if (raw == null) return null;
                var result = new System.Collections.Generic.Dictionary<string, int>();
                foreach (var kv in raw)
                    result[kv.Key] = kv.Value.WindowSize;
                return result;
            }
            catch { return null; }
        }

        // ── 프로파일 관리 ────────────────────────────────────────────────────
        /// <summary>사용 가능한 프로파일 목록과 현재 활성 프로파일을 반환합니다.</summary>
        public async Task<ProfileListResult> GetProfilesAsync()
        {
            try
            {
                var resp = await _http.GetAsync("profiles").ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode) return null;
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                return JsonConvert.DeserializeObject<ProfileListResult>(body);
            }
            catch { return null; }
        }

        /// <summary>활성 프로파일을 전환합니다. 성공 시 true 반환.</summary>
        public async Task<bool> ActivateProfileAsync(string profile)
        {
            try
            {
                string json    = JsonConvert.SerializeObject(new { profile });
                var    content = new StringContent(json, Encoding.UTF8, "application/json");
                var    resp    = await _http.PostAsync("profiles/activate", content).ConfigureAwait(false);
                return resp.IsSuccessStatusCode;
            }
            catch { return false; }
        }

        // ── 서버 스코어링 파라미터 조회/변경 ─────────────────────────────────

        /// <summary>GET /config — 현재 서버 스코어링 파라미터를 반환합니다.</summary>
        public async Task<ServerConfig> GetServerConfigAsync()
        {
            try
            {
                var resp = await _http.GetAsync("config").ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode) return null;
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                return JsonConvert.DeserializeObject<ServerConfig>(body);
            }
            catch { return null; }
        }

        /// <summary>POST /config — 서버 스코어링 파라미터를 변경합니다.</summary>
        public async Task<ServerConfig> SetServerConfigAsync(double? rmsWeight, double? anomalyThreshold)
        {
            try
            {
                var payload = new { rms_weight = rmsWeight, anomaly_threshold = anomalyThreshold };
                string json    = JsonConvert.SerializeObject(payload);
                var    content = new StringContent(json, Encoding.UTF8, "application/json");
                var    resp    = await _http.PostAsync("config", content).ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode) return null;
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                return JsonConvert.DeserializeObject<ServerConfig>(body);
            }
            catch { return null; }
        }

        public void Dispose() => _http.Dispose();
    }

    // =========================================================================
    //  InferenceResult — /predict 응답 DTO
    // =========================================================================
    public sealed class InferenceResult
    {
        [JsonProperty("model_type")]    public string  ModelType    { get; set; } = "";
        [JsonProperty("sensor_type")]   public string  SensorType   { get; set; } = "";
        [JsonProperty("axis")]          public int?    Axis         { get; set; }
        [JsonProperty("model_file")]    public string  ModelFile    { get; set; }
        [JsonProperty("is_anomaly")]    public bool    IsAnomaly    { get; set; }
        [JsonProperty("anomaly_score")] public float   AnomalyScore { get; set; }
        [JsonProperty("threshold")]     public float   Threshold    { get; set; } = 1.0f;
        [JsonProperty("class_name")]    public string  ClassName    { get; set; } = "";
        [JsonProperty("confidence")]    public float?  Confidence   { get; set; }
        [JsonProperty("raw_mae")]       public float?  RawMae       { get; set; }
        [JsonProperty("raw_threshold")] public float?  RawThreshold { get; set; }

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

    // =========================================================================
    //  CombinedInferenceResult — /predict/combined 응답 DTO
    //  AE 이상탐지 결과(필수) + CLS 결함진단 결과(선택)
    // =========================================================================
    public sealed class CombinedInferenceResult
    {
        // ── AE 이상탐지 ───────────────────────────────────────────────────────
        [JsonProperty("sensor_type")]   public string  SensorType    { get; set; } = "";
        [JsonProperty("axis")]          public int?    Axis          { get; set; }
        [JsonProperty("ae_model_file")] public string  AeModelFile   { get; set; }
        [JsonProperty("is_anomaly")]    public bool    IsAnomaly     { get; set; }
        [JsonProperty("anomaly_score")] public float   AnomalyScore  { get; set; }
        [JsonProperty("threshold")]     public float   Threshold     { get; set; } = 1.0f;
        [JsonProperty("raw_mae")]       public float?  RawMae        { get; set; }
        [JsonProperty("raw_threshold")] public float?  RawThreshold  { get; set; }
        // ── CLS 결함진단 ──────────────────────────────────────────────────────
        [JsonProperty("cls_available")]  public bool    ClsAvailable  { get; set; }
        [JsonProperty("cls_model_file")] public string  ClsModelFile  { get; set; }
        [JsonProperty("cls_class_name")] public string  ClsClassName  { get; set; }
        [JsonProperty("cls_confidence")] public float?  ClsConfidence { get; set; }
        [JsonProperty("cls_is_fault")]   public bool?   ClsIsFault    { get; set; }

        [JsonIgnore] public string Error    { get; set; }
        [JsonIgnore] public bool   IsError  => Error != null;

        /// <summary>AE 결과를 기존 InferenceResult 형식으로 변환합니다.</summary>
        public InferenceResult ToAeResult() => new InferenceResult
        {
            ModelType    = "AE-CNN1D",
            SensorType   = SensorType,
            Axis         = Axis,
            ModelFile    = AeModelFile,
            IsAnomaly    = IsAnomaly,
            AnomalyScore = AnomalyScore,
            Threshold    = Threshold,
            ClassName    = IsAnomaly ? "anomaly" : "normal",
            RawMae       = RawMae,
            RawThreshold = RawThreshold,
        };

        public static CombinedInferenceResult Fail(string error) =>
            new CombinedInferenceResult { Error = error };
    }

    // =========================================================================
    //  ModelWindowInfo — /model_info 응답 DTO (sensor_type별 윈도우 크기)
    // =========================================================================
    public sealed class ModelWindowInfo
    {
        [JsonProperty("window_size")] public int    WindowSize { get; set; } = 512;
        [JsonProperty("source")]      public string Source     { get; set; } = "";
    }

    // =========================================================================
    //  ProfileInfo / ProfileListResult — /profiles 응답 DTO
    // =========================================================================
    public sealed class ProfileInfo
    {
        [JsonProperty("name")]        public string Name        { get; set; } = "";
        [JsonProperty("label")]       public string Label       { get; set; } = "";
        [JsonProperty("description")] public string Description { get; set; } = "";
        [JsonProperty("model_count")] public int    ModelCount  { get; set; }
        [JsonProperty("created")]     public string Created     { get; set; } = "";
    }

    public sealed class ProfileListResult
    {
        [JsonProperty("profiles")] public ProfileInfo[] Profiles { get; set; } = new ProfileInfo[0];
        [JsonProperty("active")]   public string        Active   { get; set; } = "";
    }

    // =========================================================================
    //  ServerConfig — GET/POST /config 응답 DTO
    // =========================================================================
    public sealed class ServerConfig
    {
        [JsonProperty("rms_weight")]        public double RmsWeight        { get; set; } = 0.3;
        [JsonProperty("anomaly_threshold")] public double AnomalyThreshold { get; set; } = 1.0;
    }
}
