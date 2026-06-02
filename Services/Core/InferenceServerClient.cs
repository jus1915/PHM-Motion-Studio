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

        // ── "모델 없음" 캐시 ──────────────────────────────────────────────────
        // 서버가 404 ("sensor_type=X axis=N 모델 없음")로 응답한 (sensor_type, axis)
        // 조합을 기억해, 이후 같은 조합 호출은 네트워크 왕복 없이 즉시 실패 반환한다.
        // 학습되지 않은 축(예: Ax1, Ax2)에 매 사이클 추론 요청이 가는 것을 방지.
        // ReloadModelsAsync 또는 ResetModelAvailability 호출 시 비워진다 (재학습 후 갱신).
        private readonly System.Collections.Generic.HashSet<string> _missingModels
            = new System.Collections.Generic.HashSet<string>();
        private readonly object _missingLock = new object();

        private static string MakeKey(string sensorType, int? axis) =>
            $"{sensorType ?? ""}|{(axis.HasValue ? axis.Value.ToString() : "none")}";

        private bool IsModelKnownMissing(string sensorType, int? axis)
        {
            lock (_missingLock) return _missingModels.Contains(MakeKey(sensorType, axis));
        }

        private void MarkModelMissing(string sensorType, int? axis)
        {
            lock (_missingLock) _missingModels.Add(MakeKey(sensorType, axis));
        }

        private void ClearModelMissing(string sensorType, int? axis)
        {
            lock (_missingLock) _missingModels.Remove(MakeKey(sensorType, axis));
        }

        /// <summary>"모델 없음" 캐시를 비웁니다 (재학습 후 호출).</summary>
        public void ResetModelAvailability()
        {
            lock (_missingLock) _missingModels.Clear();
        }

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
            // 서버 캐시 재로드 시 클라이언트의 "모델 없음" 캐시도 비워야
            // 새로 추가된 축별 모델이 다시 시도된다.
            ResetModelAvailability();
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
            // ── 캐시 hit: 이전 호출에서 404 였던 (sensor_type, axis) 조합은
            //    네트워크 호출 없이 즉시 IsModelMissing=true 로 반환 ──────────
            if (IsModelKnownMissing(sensorType, axis))
            {
                return InferenceResult.MissingModel(sensorType, axis);
            }

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
            {
                // 404 = "모델 없음" — 캐시에 등록해 이후 호출 차단
                if ((int)resp.StatusCode == 404)
                {
                    MarkModelMissing(sensorType, axis);
                    return InferenceResult.MissingModel(sensorType, axis);
                }
                return InferenceResult.Fail($"HTTP {(int)resp.StatusCode}: {body}");
            }

            try
            {
                // 성공 응답이면 캐시에서 제거 (재학습 등으로 모델이 추가된 경우)
                ClearModelMissing(sensorType, axis);
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
            // 캐시 hit — 네트워크 호출 없이 즉시 반환 (학습되지 않은 축 차단)
            if (IsModelKnownMissing(sensorType, axis))
            {
                return CombinedInferenceResult.MissingModel(sensorType, axis);
            }

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
            {
                if ((int)resp.StatusCode == 404)
                {
                    MarkModelMissing(sensorType, axis);
                    return CombinedInferenceResult.MissingModel(sensorType, axis);
                }
                return CombinedInferenceResult.Fail($"HTTP {(int)resp.StatusCode}: {body}");
            }

            try
            {
                ClearModelMissing(sensorType, axis);
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
            var full = await GetModelInfoFullAsync().ConfigureAwait(false);
            if (full == null) return null;
            var result = new System.Collections.Generic.Dictionary<string, int>();
            foreach (var kv in full)
                result[kv.Key] = kv.Value.WindowSize;
            return result;
        }

        /// <summary>
        /// /model_info 를 호출해 sensor_type → ModelWindowInfo 전체(window_size + activity_threshold)를 반환합니다.
        /// 실패 시 null 반환.
        /// </summary>
        public async Task<System.Collections.Generic.Dictionary<string, ModelWindowInfo>> GetModelInfoFullAsync()
        {
            try
            {
                var resp = await _http.GetAsync("model_info").ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode) return null;
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                return JsonConvert.DeserializeObject<
                    System.Collections.Generic.Dictionary<string, ModelWindowInfo>>(body);
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

        /// <summary>
        /// 서버에 해당 (sensor_type, axis) 모델이 없는 경우 true.
        /// 호출 측은 IsError 일반 에러와 구분해 UI/로그 표시를 억제할 수 있다.
        /// </summary>
        [JsonIgnore] public bool   IsModelMissing { get; set; }

        /// <summary>정규화 이상 점수 백분율 (0~100+). 100 초과 시 이상.</summary>
        [JsonIgnore] public float ScorePercent => AnomalyScore * 100f;

        public static InferenceResult Fail(string error) =>
            new InferenceResult { Error = error };

        /// <summary>서버에 모델이 없는 경우 (HTTP 404 또는 캐시 hit) 반환되는 결과.</summary>
        public static InferenceResult MissingModel(string sensorType, int? axis)
        {
            string axisLabel = axis.HasValue ? $" Ax{axis.Value}" : "";
            return new InferenceResult
            {
                Error          = $"모델 없음: {sensorType}{axisLabel}",
                SensorType     = sensorType ?? "",
                Axis           = axis,
                IsModelMissing = true,
            };
        }

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

        /// <summary>
        /// 서버에 해당 (sensor_type, axis) 모델이 없는 경우 true.
        /// 호출 측은 IsError 일반 에러와 구분해 UI/로그 표시를 억제할 수 있다.
        /// </summary>
        [JsonIgnore] public bool   IsModelMissing { get; set; }

        /// <summary>AE 결과를 기존 InferenceResult 형식으로 변환합니다.</summary>
        public InferenceResult ToAeResult() => new InferenceResult
        {
            ModelType      = "AE-CNN1D",
            SensorType     = SensorType,
            Axis           = Axis,
            ModelFile      = AeModelFile,
            IsAnomaly      = IsAnomaly,
            AnomalyScore   = AnomalyScore,
            Threshold      = Threshold,
            ClassName      = IsAnomaly ? "anomaly" : "normal",
            RawMae         = RawMae,
            RawThreshold   = RawThreshold,
            Error          = Error,
            IsModelMissing = IsModelMissing,
        };

        public static CombinedInferenceResult Fail(string error) =>
            new CombinedInferenceResult { Error = error };

        /// <summary>서버에 모델이 없는 경우 (HTTP 404 또는 캐시 hit) 반환되는 결과.</summary>
        public static CombinedInferenceResult MissingModel(string sensorType, int? axis)
        {
            string axisLabel = axis.HasValue ? $" Ax{axis.Value}" : "";
            return new CombinedInferenceResult
            {
                Error          = $"모델 없음: {sensorType}{axisLabel}",
                SensorType     = sensorType ?? "",
                Axis           = axis,
                IsModelMissing = true,
            };
        }
    }

    // =========================================================================
    //  ModelWindowInfo — /model_info 응답 DTO (sensor_type별 윈도우 크기 + 활동성 임계값)
    // =========================================================================
    public sealed class ModelWindowInfo
    {
        [JsonProperty("window_size")]        public int    WindowSize        { get; set; } = 512;
        /// <summary>
        /// 활동성 게이팅 임계값 (채널별 std 최댓값 기준).
        /// 0.0 = 게이팅 미적용 (모델이 정지 구간도 포함해 학습된 경우).
        /// </summary>
        [JsonProperty("activity_threshold")] public double ActivityThreshold { get; set; } = 0.0;
        [JsonProperty("source")]             public string Source             { get; set; } = "";
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
