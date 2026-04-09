using PHM_Project_DockPanel.Services.Core;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
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
            // accel + torque 양쪽 measurement 에서 태그 값을 조회해 union
            var tasks = new[]
            {
                ExecuteFluxAsyncSafe($@"import ""influxdata/influxdb/schema""
schema.tagValues(
  bucket: ""{_cfg.Bucket}"",
  tag: ""{tag}"",
  predicate: (r) => r._measurement == ""accel"",
  start: -365d
)", ct),
                ExecuteFluxAsyncSafe($@"import ""influxdata/influxdb/schema""
schema.tagValues(
  bucket: ""{_cfg.Bucket}"",
  tag: ""{tag}"",
  predicate: (r) => r._measurement == ""torque"",
  start: -365d
)", ct)
            };
            await Task.WhenAll(tasks).ConfigureAwait(false);
            var result = new HashSet<string>();
            foreach (var t in tasks)
                foreach (var v in ParseSingleColumnCsv(t.Result))
                    result.Add(v);
            return result.OrderBy(s => s).ToList();
        }

        private async Task<string> ExecuteFluxAsyncSafe(string flux, CancellationToken ct)
        {
            try { return await ExecuteFluxAsync(flux, ct).ConfigureAwait(false); }
            catch { return ""; }
        }

        // ── 데이터 시간 범위 ───────────────────────────────────────────────────
        public async Task<(DateTime first, DateTime last)> GetTimeRangeAsync(
            string device = null, string label = null, CancellationToken ct = default)
        {
            var filters = BuildFilters(device, label);

            // accel 과 torque 양쪽에서 시간 범위 조회 후 합집합
            string accelTmpl = $@"from(bucket: ""{_cfg.Bucket}"")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == ""accel"")
{filters}  |> filter(fn: (r) => r._field == ""x"")
  |> {{0}}()
  |> keep(columns: [""_time""])";

            string torqueTmpl = $@"from(bucket: ""{_cfg.Bucket}"")
  |> range(start: -365d)
  |> filter(fn: (r) => r._measurement == ""torque"")
{filters}  |> filter(fn: (r) => r._field == ""fbtrq"")
  |> {{0}}()
  |> keep(columns: [""_time""])";

            var tasks = new[]
            {
                ExecuteFluxAsyncSafe(string.Format(accelTmpl, "first"), ct),
                ExecuteFluxAsyncSafe(string.Format(accelTmpl, "last"),  ct),
                ExecuteFluxAsyncSafe(string.Format(torqueTmpl, "first"), ct),
                ExecuteFluxAsyncSafe(string.Format(torqueTmpl, "last"),  ct),
            };
            await Task.WhenAll(tasks).ConfigureAwait(false);

            var times = tasks.Select(t => ParseFirstTime(t.Result))
                             .Where(d => d != DateTime.MinValue)
                             .ToList();

            DateTime first = times.Count > 0 ? times.Min() : DateTime.UtcNow.AddHours(-1);
            DateTime last  = times.Count > 0 ? times.Max() : DateTime.UtcNow;
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

            // ── 토크 쿼리 (실패해도 무시) ────────────────────────────────────
            Dictionary<long, double> torqueByMs = null;
            try
            {
                var torqueFlux = $@"from(bucket: ""{_cfg.Bucket}"")
  |> range(start: {fromStr}, stop: {toStr})
  |> filter(fn: (r) => r._measurement == ""torque"")
{filters}  |> filter(fn: (r) => r._field == ""fbtrq"")
  |> keep(columns: [""_time"", ""_value""])
  |> sort(columns: [""_time""])";
                string torqueCsv = await ExecuteFluxAsync(torqueFlux, ct).ConfigureAwait(false);
                torqueByMs = ParseTimeValueCsv(torqueCsv);
            }
            catch { /* 토크 데이터 없으면 빈 세그먼트로 계속 */ }

            progress?.Report($"총 {rows.Count:N0}개 샘플 → 세그먼트 분할 중...");

            string lbl = label ?? "";   // 빈 문자열 → AIForm에서 세션별 기본값(Normal 등) 적용
            string dev = device;        // null 유지 → DeleteAsync 에서 device 필터 생략됨
            var segments = Segmentize(rows, lbl, dev, segmentSeconds);

            // 세그먼트별 토크 배열 채우기
            if (torqueByMs != null && torqueByMs.Count > 0)
                FillTorque(segments, torqueByMs);

            return segments;
        }

        // ── 토크 전용 세그먼트 조회 ──────────────────────────────────────────────
        /// <summary>
        /// InfluxDB "torque" measurement 에서 fbtrq 를 직접 조회해 세그먼트로 반환합니다.
        /// 각 SignalSegment.Torque[] 에 데이터가 채워지고 X/Y/Z 는 null 입니다.
        /// </summary>
        public async Task<List<SignalSegment>> QueryTorqueSegmentsAsync(
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

            progress?.Report("InfluxDB 토크 조회 중...");

            var flux = $@"from(bucket: ""{_cfg.Bucket}"")
  |> range(start: {fromStr}, stop: {toStr})
  |> filter(fn: (r) => r._measurement == ""torque"")
{filters}  |> filter(fn: (r) => r._field == ""fbtrq"")
  |> keep(columns: [""_time"", ""_value""])
  |> sort(columns: [""_time""])";

            string csv = await ExecuteFluxAsync(flux, ct).ConfigureAwait(false);
            progress?.Report("토크 데이터 파싱 중...");

            var tvRows = ParseTimeValueCsv(csv);
            if (tvRows.Count == 0) return new List<SignalSegment>();

            // ms → sorted list for segmentizing
            var sortedMs = tvRows.Keys.OrderBy(k => k).ToList();
            progress?.Report($"총 {sortedMs.Count:N0}개 토크 샘플 → 세그먼트 분할 중...");

            string lbl = label ?? "";
            string dev = device;        // null 유지 → DeleteAsync 에서 device 필터 생략됨
            return SegmentizeTorque(sortedMs, tvRows, lbl, dev, segmentSeconds);
        }

        private static List<SignalSegment> SegmentizeTorque(
            List<long> sortedMs, Dictionary<long, double> tvRows,
            string label, string device, double segmentSeconds)
        {
            var segments = new List<SignalSegment>();
            if (sortedMs.Count == 0) return segments;

            long segSpanMs = (long)(segmentSeconds * 1000.0);
            long segStartMs = sortedMs[0];
            var buf = new List<long>();
            int segIdx = 0;

            Action flush = () =>
            {
                if (buf.Count < 4) return;
                double firstMs = buf[0];
                int n = buf.Count;
                var time = new double[n];
                var trq  = new double[n];
                for (int i = 0; i < n; i++)
                {
                    time[i] = (buf[i] - firstMs) / 1000.0;
                    trq[i]  = tvRows[buf[i]];
                }
                string segLabel = string.IsNullOrEmpty(label) ? "seg" : label;
                var startUtc = DateTimeOffset.FromUnixTimeMilliseconds(buf[0]).UtcDateTime;
                var seg = new SignalSegment
                {
                    Name      = $"{segLabel}_{segIdx:D4}",
                    Label     = label,
                    Device    = device,
                    StartTime = startUtc,
                    Time      = time,
                    Torque    = trq,
                };
                if (SegmentValidator.IsValid(seg, out _))
                    segments.Add(seg);
                segIdx++;
            };

            foreach (long ms in sortedMs)
            {
                if (ms - segStartMs >= segSpanMs)
                {
                    flush();
                    buf.Clear();
                    segStartMs = ms;
                }
                buf.Add(ms);
            }
            flush();
            return segments;
        }

        // ── 내부 Flux 실행 ────────────────────────────────────────────────────
        private async Task<string> ExecuteFluxAsync(string flux, CancellationToken ct)
        {
            string url = $"{_cfg.Url.TrimEnd('/')}/api/v2/query?org={Uri.EscapeDataString(_cfg.Org)}";

            var content = new StringContent(flux, Encoding.UTF8, "application/vnd.flux");

            // Accept 헤더는 content headers가 아닌 request message 헤더에 추가해야 함
            var request = new System.Net.Http.HttpRequestMessage(System.Net.Http.HttpMethod.Post, url)
            {
                Content = content
            };
            request.Headers.Accept.Add(
                new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("application/csv"));

            var resp = await _http.SendAsync(request, ct).ConfigureAwait(false);
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

        /// <summary>(_time, _value) CSV → ms 타임스탬프 → value 딕셔너리</summary>
        private static Dictionary<long, double> ParseTimeValueCsv(string csv)
        {
            var result = new Dictionary<long, double>();
            if (string.IsNullOrWhiteSpace(csv)) return result;

            int iTime = -1, iVal = -1;
            bool headerFound = false;

            foreach (var rawLine in csv.Split('\n'))
            {
                var line = rawLine.Trim();
                if (string.IsNullOrEmpty(line) || line.StartsWith("#")) continue;

                var cols = line.Split(',');
                if (!headerFound)
                {
                    for (int i = 0; i < cols.Length; i++)
                    {
                        string h = cols[i].Trim();
                        if (h == "_time")  iTime = i;
                        else if (h == "_value") iVal  = i;
                    }
                    if (iTime >= 0) headerFound = true;
                    continue;
                }

                if (iTime < 0 || iTime >= cols.Length) continue;
                DateTime ts;
                if (!DateTime.TryParse(cols[iTime].Trim(), null,
                    System.Globalization.DateTimeStyles.RoundtripKind, out ts)) continue;
                double val;
                if (iVal >= 0 && double.TryParse(cols[iVal].Trim(),
                    NumberStyles.Float, CultureInfo.InvariantCulture, out val))
                {
                    long ms = new DateTimeOffset(ts.ToUniversalTime()).ToUnixTimeMilliseconds();
                    result[ms] = val;
                }
            }
            return result;
        }

        /// <summary>각 세그먼트의 Torque[] 를 torqueByMs 딕셔너리에서 채웁니다.</summary>
        private static void FillTorque(List<SignalSegment> segments, Dictionary<long, double> torqueByMs)
        {
            foreach (var seg in segments)
            {
                if (seg.Time == null || seg.Time.Length == 0) continue;
                long segStartMs = new DateTimeOffset(seg.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds();
                var trq = new double[seg.Time.Length];
                for (int i = 0; i < seg.Time.Length; i++)
                {
                    long tMs = segStartMs + (long)(seg.Time[i] * 1000.0);
                    // 가장 가까운 토크 샘플 탐색 (±20ms 허용)
                    double best = double.NaN;
                    long bestDiff = 21;
                    foreach (long key in torqueByMs.Keys)
                    {
                        long diff = Math.Abs(key - tMs);
                        if (diff < bestDiff) { bestDiff = diff; best = torqueByMs[key]; }
                    }
                    trq[i] = double.IsNaN(best) ? 0.0 : best;
                }
                seg.Torque = trq;
            }
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
                string segLabel = string.IsNullOrEmpty(label) ? "seg" : label;
                var seg = new SignalSegment
                {
                    Name      = $"{segLabel}_{segIdx:D4}",
                    Label     = label,   // 원래 값 보존 (빈 문자열도 허용)
                    Device    = device,
                    StartTime = buf[0].Item1,
                    Time      = time,
                    X         = xArr,
                    Y         = yArr,
                    Z         = zArr
                };
                if (SegmentValidator.IsValid(seg, out _))
                    segments.Add(seg);
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

        // ── Delete API ────────────────────────────────────────────────────────
        /// <summary>
        /// 지정 조건의 데이터를 InfluxDB에서 삭제합니다.
        /// device=null/label=null 이면 해당 필터 생략 (전체).
        /// from/to=null 이면 전체 기간.
        /// </summary>
        public async Task DeleteAsync(
            string device, string label,
            DateTime? from = null, DateTime? to = null,
            CancellationToken ct = default)
        {
            // 밀리초 정밀도로 RFC3339 변환
            var start = (from ?? DateTime.UtcNow.AddYears(-10)).ToUniversalTime()
                            .ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
            var stop  = (to   ?? DateTime.UtcNow.AddYears(1)).ToUniversalTime()
                            .ToString("yyyy-MM-ddTHH:mm:ss.fffZ");

            // _measurement 를 predicate 에 포함하면 일부 InfluxDB 버전에서 무시됨.
            // device + label 태그만으로 필터링하고, 버킷 내 모든 measurement(accel/torque) 를 한 번에 삭제.
            var predicateParts = new List<string>();
            if (!string.IsNullOrEmpty(device))
                predicateParts.Add($"device=\"{Esc(device)}\"");
            if (!string.IsNullOrEmpty(label))
                predicateParts.Add($"label=\"{Esc(label)}\"");

            string url = $"{_cfg.Url.TrimEnd('/')}/api/v2/delete" +
                         $"?org={Uri.EscapeDataString(_cfg.Org)}" +
                         $"&bucket={Uri.EscapeDataString(_cfg.Bucket)}";

            string json;
            if (predicateParts.Count > 0)
            {
                string predicate = string.Join(" AND ", predicateParts);
                string escaped   = predicate.Replace("\"", "\\\"");
                json = $"{{\"start\":\"{start}\",\"stop\":\"{stop}\",\"predicate\":\"{escaped}\"}}";
            }
            else
            {
                json = $"{{\"start\":\"{start}\",\"stop\":\"{stop}\"}}";
            }

            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var resp = await _http.PostAsync(url, content, ct).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
            {
                string err = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                throw new InvalidOperationException($"InfluxDB 삭제 실패 ({(int)resp.StatusCode}): {err.Trim()}");
            }
        }

        // ── Write from CSV ────────────────────────────────────────────────────
        /// <summary>
        /// CSV 파일(time_s, x, y, z 헤더)을 읽어 InfluxDB에 씁니다.
        /// baseTimeUtc: 파일의 기준 시각 (null=현재 UTC). time_s 가 있으면 그만큼 오프셋.
        /// </summary>
        public async Task WriteCsvAsync(
            string csvPath, string device, string label,
            double sampleRateHz = 1000.0,
            DateTime? baseTimeUtc = null,
            IProgress<string> progress = null,
            CancellationToken ct = default)
        {
            var baseTime = (baseTimeUtc ?? DateTime.UtcNow).ToUniversalTime();
            var lines    = new List<string>(4096);

            string tagSet = $"device={EscapeTag(device ?? "unknown")}";
            if (!string.IsNullOrEmpty(label))
                tagSet += $",label={EscapeTag(label)}";

            progress?.Report($"CSV 읽는 중: {System.IO.Path.GetFileName(csvPath)}");

            using (var sr = new System.IO.StreamReader(csvPath, Encoding.UTF8, true))
            {
                string header = sr.ReadLine();
                if (header == null) return;

                var cols = header.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
                int iTime = -1, ix = -1, iy = -1, iz = -1;
                for (int c = 0; c < cols.Length; c++)
                {
                    string h = cols[c].Trim().ToLower();
                    if (h == "time_s" || h == "time") iTime = c;
                    else if (h == "x") ix = c;
                    else if (h == "y") iy = c;
                    else if (h == "z") iz = c;
                }
                if (ix < 0) throw new InvalidOperationException("CSV에 'x' 컬럼이 없습니다.");

                double sp = sampleRateHz > 0 ? 1.0 / sampleRateHz : 0.001;
                int idx = 0;
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    var parts = line.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
                    double xv = ParseField(parts, ix);
                    double yv = ParseField(parts, iy);
                    double zv = ParseField(parts, iz);

                    double tSec;
                    if (iTime >= 0 && iTime < parts.Length)
                        double.TryParse(parts[iTime].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out tSec);
                    else
                        tSec = idx * sp;

                    long tsMs = new DateTimeOffset(baseTime).ToUnixTimeMilliseconds()
                                + (long)(tSec * 1000);

                    lines.Add(string.Format(CultureInfo.InvariantCulture,
                        $"accel,{tagSet} x={{0:G8}},y={{1:G8}},z={{2:G8}} {tsMs}", xv, yv, zv));
                    idx++;

                    // 배치 1000개마다 쓰기
                    if (lines.Count >= 1000)
                    {
                        await FlushLinesAsync(lines, ct).ConfigureAwait(false);
                        progress?.Report($"  {idx:N0}개 샘플 업로드 중...");
                        lines.Clear();
                    }
                }
            }
            if (lines.Count > 0)
                await FlushLinesAsync(lines, ct).ConfigureAwait(false);

            progress?.Report($"업로드 완료");
        }

        private async Task FlushLinesAsync(List<string> lines, CancellationToken ct)
        {
            string url = $"{_cfg.Url.TrimEnd('/')}/api/v2/write" +
                         $"?org={Uri.EscapeDataString(_cfg.Org)}" +
                         $"&bucket={Uri.EscapeDataString(_cfg.Bucket)}" +
                         $"&precision=ms";
            string body = string.Join("\n", lines);
            var content = new StringContent(body, Encoding.UTF8, "text/plain");
            var resp = await _http.PostAsync(url, content, ct).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
            {
                string err = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                throw new InvalidOperationException($"InfluxDB Write 실패 ({(int)resp.StatusCode}): {err.Trim()}");
            }
        }

        private static double ParseField(string[] parts, int idx)
        {
            if (idx < 0 || idx >= parts.Length) return 0.0;
            double v;
            double.TryParse(parts[idx].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out v);
            return v;
        }

        private static string EscapeTag(string s)
            => (s ?? "unknown").Replace(" ", "\\ ").Replace(",", "\\,").Replace("=", "\\=");

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
        public double[] Time   { get; set; }
        public double[] X      { get; set; }
        public double[] Y      { get; set; }
        public double[] Z      { get; set; }
        /// <summary>토크 배열 (%). AjinCsvLogger/WmxTorqueLogger 에서 게시된 경우에만 non-null.</summary>
        public double[] Torque { get; set; }

        public int SampleCount => Time?.Length ?? 0;

        /// <summary>"x" | "y" | "z" | "torque" | "fbtrq" | "time_s" → 배열 반환</summary>
        public double[] GetChannel(string name)
        {
            switch ((name ?? "x").ToLower())
            {
                case "y":              return Y;
                case "z":              return Z;
                case "time_s":         return Time;
                case "torque":
                case "fbtrq":
                case "trq":            return Torque;
                default:               return X;
            }
        }
    }
}
