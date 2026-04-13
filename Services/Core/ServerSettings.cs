using System;
using System.IO;
using Newtonsoft.Json;
using PHM_Project_DockPanel.Services.DAQ;

namespace PHM_Project_DockPanel.Services
{
    // =========================================================================
    //  ServerSettings — 모든 외부 서버 URL을 한 곳에서 관리
    //  파일: server_settings.json (influx_config.json 과 같은 폴더)
    // =========================================================================
    public sealed class ServerSettings
    {
        // ── InfluxDB ──────────────────────────────────────────────────────────
        public string InfluxUrl    { get; set; } = "http://localhost:8086";
        public string InfluxToken  { get; set; } = "my-super-secret-token";
        public string InfluxOrg    { get; set; } = "daq_org";
        public string InfluxBucket { get; set; } = "vibration";

        // ── MLflow ────────────────────────────────────────────────────────────
        public string MlflowUrl    { get; set; } = "http://localhost:5000";

        // ── Airflow ───────────────────────────────────────────────────────────
        public string AirflowUrl   { get; set; } = "http://localhost:8080";

        // ── 싱글톤 ────────────────────────────────────────────────────────────
        public static ServerSettings Current { get; private set; } = new ServerSettings();

        /// <summary>파일에서 로드. 없으면 기본값 사용. influx_config.json도 병합(하위 호환).</summary>
        public static void Load(string path)
        {
            if (File.Exists(path))
            {
                try
                {
                    Current = JsonConvert.DeserializeObject<ServerSettings>(
                        File.ReadAllText(path)) ?? new ServerSettings();
                    return;
                }
                catch { }
            }

            // server_settings.json 없으면 influx_config.json에서 InfluxDB 값 마이그레이션
            string influxPath = Path.Combine(Path.GetDirectoryName(path) ?? ".", "influx_config.json");
            var influxCfg = InfluxConfig.LoadOrDefault(influxPath);
            Current = new ServerSettings
            {
                InfluxUrl    = influxCfg.Url,
                InfluxToken  = influxCfg.Token,
                InfluxOrg    = influxCfg.Org,
                InfluxBucket = influxCfg.Bucket,
            };
        }

        /// <summary>현재 설정을 JSON 파일에 저장.</summary>
        public void Save(string path)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
                File.WriteAllText(path,
                    JsonConvert.SerializeObject(this, Formatting.Indented),
                    System.Text.Encoding.UTF8);
            }
            catch { }
        }

        /// <summary>InfluxConfig 변환 (AccelInfluxPublisher 호환).</summary>
        public InfluxConfig ToInfluxConfig() => new InfluxConfig
        {
            Url    = InfluxUrl,
            Token  = InfluxToken,
            Org    = InfluxOrg,
            Bucket = InfluxBucket,
        };

        /// <summary>모든 URL의 호스트(IP:Port)를 newHost로 일괄 교체.</summary>
        public void ReplaceAllHosts(string newHost)
        {
            InfluxUrl  = ReplaceHost(InfluxUrl,  newHost);
            MlflowUrl  = ReplaceHost(MlflowUrl,  newHost);
            AirflowUrl = ReplaceHost(AirflowUrl, newHost);
        }

        /// <summary>http://OLD_HOST:PORT/path → http://newHost:PORT/path</summary>
        public static string ReplaceHost(string url, string newHost)
        {
            if (string.IsNullOrWhiteSpace(url)) return url;
            try
            {
                var u = new Uri(url);
                // newHost에 포트가 포함된 경우 처리
                string host = newHost;
                int port = u.Port;
                if (newHost.Contains(":"))
                {
                    var parts = newHost.Split(new[] { ':' }, 2);
                    host = parts[0];
                    int.TryParse(parts[1], out port);
                }
                var builder = new UriBuilder(u) { Host = host, Port = port };
                return builder.Uri.ToString().TrimEnd('/');
            }
            catch
            {
                return url;
            }
        }
    }
}
