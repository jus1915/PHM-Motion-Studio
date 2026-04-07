using PHM_Project_DockPanel.Services.Core;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.DAQ
{
    /// <summary>
    /// InfluxDB Flux HTTP API 로 accel measurement 를 조회합니다.
    /// 별도 NuGet 없이 HttpClient + Flux 쿼리를 직접 사용합니다.
    /// </summary>
    public class InfluxDbDataSource : IDisposable
    {
        private readonly InfluxConfig _cfg;
        private readonly HttpClient _http;
        private bool _disposed;

        public InfluxDbDataSource(InfluxConfig cfg)
        {
            _cfg  = cfg ?? new InfluxConfig();
            _http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
            _http.DefaultRequestHeaders.Add("Authorization", $"Token {_cfg.Token}");
        }

        // ── 태그 목록 조회 ────────────────────────────────────────────────────
        public Task<List<string>> GetDevicesAsync(CancellationToken ct = default)
            => GetTagValuesAsync("device", ct);

        public Task<List<string>> GetLabelsAsync(CancellationToken ct = default)
            => GetTagValuesAsync("label", ct);

        private async Task<List<string>> GetTagValuesAsync(string tag, CancellationToken ct)
        {
            // schema.tagValues 로 tag 값 목록 조회
            var flux = $@"import ""influxdata/influxdb/schema""
schema.tagValues(
  bucket: ""{_cfg.Bucket}"",
  tag: ""{tag}"",
  predicate: (r) => r._measurement == ""accel"",
  start: -365d
)";
            var csv = await ExecuteFluxAsync(flux, ct).ConfigureAwait(false);
            return ParseSingleColumnCsv(csv);
        }

        // ── 데이터 시간 범위 ───────────────────────────────────────────────────
        public async Task<(DateTime first, DateTime last)> GetTimeRangeAsync(
            string device = null, string label = null, CancellationToken ct = default)
        {
            var filters = BuildFilters(device, label);
            string tmpl = $@"from(bucket: ""{_cfg.Bucket}"")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == ""accel"")
{filters}  |> filter(fn: (r) => r._field == ""x"")
  |> {{0}}()
  |> keep(columns: [""_time""])";

            string csvFirst = await ExecuteFluxAsync(string.Format(tmpl, "first"), ct).ConfigureAwait(false);
            string csvLast  = await ExecuteFluxAsync(string.Format(tmpl, "last"),  ct).ConfigureAwait(false);

            var first = ParseFirstTime(csvFirst);
            var last  = ParseFirstTime(csvLast);
            if (first == DateTime.MinValue) first = DateTime.UtcNow.AddHours(-1);
            if (last  == DateTime.MinValue) last  = DateTime.UtcNow;
            return (first, last);
        }

        // ── 데이터 조회 + 세그먼트 분할 ───────────────────────────────────────
        /// <summary>
        /// InfluxDB 에서 raw x/y/z 를 조회하고 segmentSeconds 단위로 잘라 반환.
        /// 각 SignalSegment 는 CSV 파일 1 개에 해당합니다.
        /// </summary>
        public async Task<List<SignalSegment>> QuerySegmentsAsync(
            string device, string label,
            DateTime from, DateTime to,
            double segmentSeconds = 1.0,
            IProgress<string> progress = null,
            CancellationToken ct = default)
        {
            if (segmentSeconds <= 0) segmentSeconds = 1.0;

            var filters = BuildFilters(device, label);
            string fromStr = from.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ");
            string toStr   = to.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ");

            progress?.Report("InfluxDB 조회 중...");

            var flux = $@"from(bucket: ""{_cfg.Bucket}"")
  |> range(start: {fromStr}, stop: {toStr})
  |> filter(fn: (r) => r._measurement == ""accel"")
{filters}  |> filter(fn: (r) => r._field == ""x"" or r._field == ""y"" or r._field == ""z"")
  |> pivot(rowKey: [""_time""], columnKey: [""_field""], valueColumn: ""_value"")
  |> keep(columns: [""_time"", ""x"", ""y"", ""z""])
  |> sort(columns: [""_time""])";

            string csv = await ExecuteFluxAsync(flux, ct).ConfigureAwait(false);
            progress?.Report("데이터 파싱 중...");

            var rows = ParsePivotCsv(csv);
            if (rows.Count == 0)
                return new List<SignalSegment>();

            progress?.Report($"총 {rows.Count:N0}개 샘플 → 세그먼트 분할 중...");

            string lbl = label ?? "unlabeled";
            string dev = device ?? "unknown";
            return Segmentize(rows, lbl, dev, segmentSeconds);
        }

        // ── 내부 Flux 실행 ────────────────────────────────────────────────────
        private async Task<string> ExecuteFluxAsync(string flux, CancellationToken ct)
        {
            string url = $"{_cfg.Url.TrimEnd('/')}/api/v2/query?org={Uri.EscapeDataString(_cfg.Org)}";

            var content = new StringContent(flux, Encoding.UTF8, "application/vnd.flux");
            content.Headers.Add("Accept", "application/csv");

            var resp = await _http.PostAsync(url, content, ct).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
            {
                string err = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                throw new InvalidOperationException(
                    $"InfluxDB 쿼리 실패 ({(int)resp.StatusCode}): {err.Trim()}");
            }
            return await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
        }

        // ── CSV 파싱 헬퍼 ─────────────────────────────────────────────────────
        private static List<string> ParseSingleColumnCsv(string csv)
        {
            var result = new List<string>();
            if (string.IsNullOrWhiteSpace(csv)) return result;

            bool headerPassed = false;
            foreach (var rawLine in csv.Split('\n'))
            {
                var line = rawLine.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;
                var cols = line.Split(',');
                if (!headerPassed)
                {
                    // 헤더 행 건너뜀 (result, table, _value 컬럼)
                    headerPassed = true;
                    continue;
                }
                // 마지막 컬럼이 값
                string val = cols[cols.Length - 1].Trim().Trim('"');
                if (!string.IsNullOrEmpty(val))
                    result.Add(val);
            }
            return result.Distinct().OrderBy(s => s).ToList();
        }

        private static DateTime ParseFirstTime(string csv)
        {
            if (string.IsNullOrWhiteSpace(csv)) return DateTime.MinValue;
            bool headerPassed = false;
            foreach (var rawLine in csv.Split('\n'))
            {
                var line = rawLine.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;
                if (!headerPassed) { headerPassed = true; continue; }
                foreach (var col in line.Split(','))
                {
                    var c = col.Trim();
                    DateTime dt;
                    if (DateTime.TryParse(c, null,
                        System.Globalization.DateTimeStyles.RoundtripKind, out dt))
                        return dt.ToUniversalTime();
                }
            }
            return DateTime.MinValue;
        }

        /// <summary>pivot 결과 CSV (_time, x, y, z) 파싱</summary>
        private static List<(DateTime t, double x, double y, double z)> ParsePivotCsv(string csv)
        {
            var rows = new List<(DateTime, double, double, double)>();
            if (string.IsNullOrWhiteSpace(csv)) return rows;

            int iTime = -1, ix = -1, iy = -1, iz = -1;
            bool headerFound = false;

            foreach (var rawLine in csv.Split('\n'))
            {
                var line = rawLine.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;

                var cols = line.Split(',');
                if (cols.Length == 0) continue;

                if (!headerFound)
                {
                    for (int i = 0; i < cols.Length; i++)
                    {
                        string h = cols[i].Trim();
                        if (h == "_time") iTime = i;
                        else if (h == "x")  ix = i;
                        else if (h == "y")  iy = i;
                        else if (h == "z")  iz = i;
                    }
                    if (iTime >= 0) { headerFound = true; }
                    continue;
                }

                if (iTime < 0 || iTime >= cols.Length) continue;
                DateTime ts;
                if (!DateTime.TryParse(cols[iTime].Trim(), null,
                    System.Globalization.DateTimeStyles.RoundtripKind, out ts)) continue;

                rows.Add((ts.ToUniversalTime(),
                          ParseCol(cols, ix),
                          ParseCol(cols, iy),
                          ParseCol(cols, iz)));
            }
            return rows;
        }

        private static double ParseCol(string[] cols, int idx)
        {
            if (idx < 0 || idx >= cols.Length) return 0.0;
            double v;
            double.TryParse(cols[idx].Trim(), NumberStyles.Float,
                            CultureInfo.InvariantCulture, out v);
            return v;
        }

        // ── 세그먼트 분할 ─────────────────────────────────────────────────────
        private static List<SignalSegment> Segmentize(
            List<(DateTime t, double x, double y, double z)> rows,
            string label, string device, double segmentSeconds)
        {
            var segments = new List<SignalSegment>();
            if (rows.Count == 0) return segments;

            var segSpan  = TimeSpan.FromSeconds(segmentSeconds);
            var segStart = rows[0].t;
            var buf      = new List<(DateTime, double, double, double)>();
            int segIdx   = 0;

            Action flush = () =>
            {
                if (buf.Count < 4) return;
                double firstMs = (double)new DateTimeOffset(buf[0].Item1).ToUnixTimeMilliseconds();
                int n = buf.Count;
                var time = new double[n];
                var xArr = new double[n];
                var yArr = new double[n];
                var zArr = new double[n];
                for (int i = 0; i < n; i++)
                {
                    double ms = (double)new DateTimeOffset(buf[i].Item1).ToUnixTimeMilliseconds();
                    time[i] = (ms - firstMs) / 1000.0;
                    xArr[i] = buf[i].Item2;
                    yArr[i] = buf[i].Item3;
                    zArr[i] = buf[i].Item4;
                }
                segments.Add(new SignalSegment
                {
                    Name      = $"{label}_{segIdx:D4}",
                    Label     = label,
                    Device    = device,
                    StartTime = buf[0].Item1,
                    Time      = time,
                    X         = xArr,
                    Y         = yArr,
                    Z         = zArr
                });
                segIdx++;
            };

            foreach (var row in rows)
            {
                if (row.t - segStart >= segSpan)
                {
                    flush();
                    buf.Clear();
                    segStart = row.t;
                }
                buf.Add(row);
            }
            flush();
            return segments;
        }

        // ── 필터 문자열 ───────────────────────────────────────────────────────
        private static string BuildFilters(string device, string label)
        {
            var sb = new StringBuilder();
            if (!string.IsNullOrEmpty(device))
                sb.AppendLine($"  |> filter(fn: (r) => r.device == \"{Esc(device)}\")");
            if (!string.IsNullOrEmpty(label))
                sb.AppendLine($"  |> filter(fn: (r) => r.label == \"{Esc(label)}\")");
            return sb.ToString();
        }

        private static string Esc(string s)
            => (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");

        public void Dispose()
        {
            if (_disposed) return;
            _http?.Dispose();
            _disposed = true;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    /// <summary>InfluxDB 에서 조회한 신호 세그먼트 (CSV 파일 1개에 해당)</summary>
    public class SignalSegment
    {
        public string   Name      { get; set; }
        public string   Label     { get; set; }
        public string   Device    { get; set; }
        public DateTime StartTime { get; set; }

        /// <summary>시간 배열 (초, 0-based)</summary>
        public double[] Time { get; set; }
        public double[] X    { get; set; }
        public double[] Y    { get; set; }
        public double[] Z    { get; set; }

        public int SampleCount => Time?.Length ?? 0;

        /// <summary>"x" | "y" | "z" | "time_s" → 배열 반환</summary>
        public double[] GetChannel(string name)
        {
            switch ((name ?? "x").ToLower())
            {
                case "y":      return Y;
                case "z":      return Z;
                case "time_s": return Time;
                default:       return X;
            }
        }
    }
}
