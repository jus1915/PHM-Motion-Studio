using System.Collections.Generic;
using Newtonsoft.Json;

namespace PHM_Project_DockPanel.Services.Core
{
    // =========================================================================
    //  채널 AE 추론 상태
    // =========================================================================
    public enum ChannelActiveState { Active, Inactive, Uncertain }

    // =========================================================================
    //  ChannelAeChannelInfo — ch_ae_meta.json 채널 항목 DTO
    // =========================================================================
    public sealed class ChannelAeChannelInfo
    {
        [JsonProperty("model_file")]           public string ModelFile             { get; set; }
        [JsonProperty("active_threshold")]     public double ActiveThreshold       { get; set; } = double.MaxValue;
        [JsonProperty("inactive_threshold")]   public double InactiveThreshold     { get; set; } = 0.0;
        [JsonProperty("recon_error_thr_95")]   public double ReconErrorThr95       { get; set; } = 1e-3;
        [JsonProperty("recon_error_thr_99")]   public double ReconErrorThr99       { get; set; } = 2e-3;
    }

    // =========================================================================
    //  ChannelAeMeta — GET /channel_ae_info 응답 DTO
    // =========================================================================
    public sealed class ChannelAeMeta
    {
        [JsonProperty("kind")]        public string Kind       { get; set; } = "ChannelActiveAE";
        [JsonProperty("window_size")] public int    WindowSize { get; set; } = 128;
        [JsonProperty("stride")]      public int    Stride     { get; set; } = 64;

        [JsonProperty("channels")]
        public Dictionary<string, ChannelAeChannelInfo> Channels { get; set; }
            = new Dictionary<string, ChannelAeChannelInfo>();
    }

    // =========================================================================
    //  ChannelAePredictResponse — POST /predict/channel_ae 응답 DTO
    // =========================================================================
    public sealed class ChannelAePredictResponse
    {
        [JsonProperty("channel")]       public string  Channel      { get; set; } = "";
        [JsonProperty("active_state")]  public string  ActiveState  { get; set; } = "uncertain";
        [JsonProperty("active_score")]  public double  ActiveScore  { get; set; }
        [JsonProperty("is_anomaly")]    public bool    IsAnomaly    { get; set; }
        [JsonProperty("anomaly_score")] public double  AnomalyScore { get; set; }
        [JsonProperty("threshold")]     public double  Threshold    { get; set; } = 1.0;
        [JsonProperty("recon_error")]   public double? ReconError   { get; set; }
        [JsonProperty("raw_threshold")] public double? RawThreshold { get; set; }

        // ── 클라이언트 전용 ─────────────────────────────────────────────────
        [JsonIgnore] public string Error         { get; set; }
        [JsonIgnore] public bool   IsError       => Error != null;
        [JsonIgnore] public bool   IsModelMissing{ get; set; }

        [JsonIgnore]
        public ChannelActiveState ChannelState
        {
            get
            {
                switch (ActiveState)
                {
                    case "active":   return ChannelActiveState.Active;
                    case "inactive": return ChannelActiveState.Inactive;
                    default:         return ChannelActiveState.Uncertain;
                }
            }
        }

        /// <summary>InferenceResult 형식으로 변환해 기존 이벤트 인프라를 재사용합니다.</summary>
        public InferenceResult ToInferenceResult(string sensorType)
        {
            return new InferenceResult
            {
                ModelType      = "ChannelAE",
                SensorType     = sensorType,
                Axis           = null,
                IsAnomaly      = IsAnomaly,
                AnomalyScore   = (float)AnomalyScore,
                Threshold      = (float)Threshold,
                ClassName      = IsAnomaly ? "anomaly" : "normal",
                RawMae         = ReconError.HasValue  ? (float?)ReconError.Value  : null,
                RawThreshold   = RawThreshold.HasValue? (float?)RawThreshold.Value: null,
                Error          = Error,
                IsModelMissing = IsModelMissing,
            };
        }

        public static ChannelAePredictResponse Fail(string error)
            => new ChannelAePredictResponse { Error = error };

        public static ChannelAePredictResponse MissingModel(string channel)
            => new ChannelAePredictResponse
               { Channel = channel, Error = $"모델 없음: {channel}", IsModelMissing = true };
    }

    // =========================================================================
    //  ChannelSafeName — 채널명 → sensorType 키 변환 헬퍼
    //  예: "Ax0_Trq(%)" → "ch_Ax0_Trq_pct_"
    // =========================================================================
    public static class ChannelSafeName
    {
        public static string Make(string channelName)
        {
            if (string.IsNullOrEmpty(channelName)) return "ch_unknown";
            return "ch_" + channelName
                .Replace("/",  "_")
                .Replace("\\", "_")
                .Replace(":",  "_")
                .Replace("%",  "pct")
                .Replace("(",  "")
                .Replace(")",  "")
                .Replace(" ",  "_");
        }
    }
}
