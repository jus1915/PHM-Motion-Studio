using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PHM_Project_DockPanel.Services.DAQ
{
    /// <summary>
    /// DAQ 가속도 센서 설정. daq_sensor_config.json 으로 영속화됩니다.
    /// </summary>
    public class DaqSensorConfig
    {
        // ── 하드웨어 ──────────────────────────────────────────────────────
        [JsonPropertyName("module")]
        public string Module { get; set; } = "cDAQ2Mod2";

        [JsonPropertyName("channel")]
        public string Channel { get; set; } = "ai0:2";

        // ── 민감도 (mV/g) ─────────────────────────────────────────────────
        [JsonPropertyName("sens_x_mvpg")]
        public double SensX { get; set; } = 1026.0;

        [JsonPropertyName("sens_y_mvpg")]
        public double SensY { get; set; } = 991.0;

        [JsonPropertyName("sens_z_mvpg")]
        public double SensZ { get; set; } = 985.0;

        // ── 수집 설정 ─────────────────────────────────────────────────────
        [JsonPropertyName("sample_rate_hz")]
        public double SampleRate { get; set; } = 1000.0;

        /// <summary>콜백 당 샘플 수. SampleRate / 10 을 기본값으로 사용합니다.</summary>
        [JsonPropertyName("read_block_samples")]
        public int ReadBlock { get; set; } = 100;

        /// <summary>측정 범위 단방향 (g). 실제 전압 범위 = GRange × max(Sens) / 1000 V.</summary>
        [JsonPropertyName("g_range")]
        public double GRange { get; set; } = 5.0;

        /// <summary>IEPE 전류 (A). DaqAccelHttpSender 에 전달됩니다.</summary>
        [JsonPropertyName("iepe_current_a")]
        public double IepeCurrentAmps { get; set; } = 0.004;

        // ── 계산 헬퍼 ────────────────────────────────────────────────────
        /// <summary>측정 범위(g) × 최대 민감도 / 1000 → CreateVoltageChannel min/max (V)</summary>
        public double VoltageRange =>
            GRange * Math.Max(SensX, Math.Max(SensY, SensZ)) / 1000.0;

        // ── JSON 직렬화 ───────────────────────────────────────────────────
        private static readonly JsonSerializerOptions _jsonOpts =
            new JsonSerializerOptions { WriteIndented = true };

        public static DaqSensorConfig LoadOrDefault(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    var cfg = JsonSerializer.Deserialize<DaqSensorConfig>(
                        File.ReadAllText(path), _jsonOpts);
                    if (cfg != null) return cfg;
                }
            }
            catch { }
            return new DaqSensorConfig();
        }

        public void Save(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllText(path, JsonSerializer.Serialize(this, _jsonOpts));
        }
    }
}
