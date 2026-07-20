using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.Core
{
    /// <summary>
    /// Airflow REST API 2.x 클라이언트.
    /// Basic Auth 기반으로 DAG 트리거 / 실행 상태 조회를 수행합니다.
    /// </summary>
    public sealed class AirflowClient : IDisposable
    {
        private readonly HttpClient _http;
        private readonly string     _baseUrl;

        public AirflowClient(string baseUrl, string username, string password)
        {
            _baseUrl = (baseUrl ?? "http://localhost:8080").TrimEnd('/');
            _http    = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
            var creds = Convert.ToBase64String(
                Encoding.UTF8.GetBytes($"{username}:{password}"));
            _http.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Basic", creds);
            _http.DefaultRequestHeaders.Accept.ParseAdd("application/json");
        }

        /// <summary>
        /// DAG를 즉시 한 번 실행합니다.
        /// </summary>
        /// <param name="dagId">Airflow DAG ID (예: "phm_retrain")</param>
        /// <param name="conf">dag_run.conf 로 전달할 학습 파라미터</param>
        /// <returns>(성공 여부, dag_run_id, 오류 메시지)</returns>
        public async Task<(bool Ok, string RunId, string Error)> TriggerDagAsync(
            string dagId, Dictionary<string, object> conf = null)
        {
            try
            {
                string url  = $"{_baseUrl}/api/v1/dags/{Uri.EscapeDataString(dagId)}/dagRuns";
                var    body = new { conf = conf ?? new Dictionary<string, object>() };
                var    json = JsonSerializer.Serialize(body);
                using (var content = new StringContent(json, Encoding.UTF8, "application/json"))
                {
                    var resp = await _http.PostAsync(url, content).ConfigureAwait(false);
                    string respBody = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);

                    if (!resp.IsSuccessStatusCode)
                        return (false, null, $"HTTP {(int)resp.StatusCode}: {respBody.Trim()}");

                    using (var doc = JsonDocument.Parse(respBody))
                    {
                        var runId = doc.RootElement.GetProperty("dag_run_id").GetString();
                        return (true, runId, null);
                    }
                }
            }
            catch (Exception ex) { return (false, null, ex.Message); }
        }

        /// <summary>
        /// DAG 실행의 현재 상태를 반환합니다.
        /// 상태: queued / running / success / failed / upstream_failed
        /// </summary>
        public async Task<(string State, string Error)> GetDagRunStatusAsync(
            string dagId, string runId)
        {
            try
            {
                string url = $"{_baseUrl}/api/v1/dags/{Uri.EscapeDataString(dagId)}" +
                             $"/dagRuns/{Uri.EscapeDataString(runId)}";
                var resp = await _http.GetAsync(url).ConfigureAwait(false);
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);

                if (!resp.IsSuccessStatusCode)
                    return (null, $"HTTP {(int)resp.StatusCode}");

                using (var doc = JsonDocument.Parse(body))
                    return (doc.RootElement.GetProperty("state").GetString(), null);
            }
            catch (Exception ex) { return (null, ex.Message); }
        }

        /// <summary>
        /// DAG의 가장 최근 실행 상태를 반환합니다 (run_id 없이 조회할 때).
        /// </summary>
        public async Task<(string State, string RunId, string Error)> GetLatestDagRunAsync(
            string dagId, int limit = 1)
        {
            try
            {
                string url = $"{_baseUrl}/api/v1/dags/{Uri.EscapeDataString(dagId)}" +
                             $"/dagRuns?limit={limit}&order_by=-execution_date";
                var resp = await _http.GetAsync(url).ConfigureAwait(false);
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);

                if (!resp.IsSuccessStatusCode)
                    return (null, null, $"HTTP {(int)resp.StatusCode}");

                using (var doc = JsonDocument.Parse(body))
                {
                    var runs = doc.RootElement.GetProperty("dag_runs");
                    if (runs.GetArrayLength() == 0) return ("none", null, null);
                    var first = runs[0];
                    string state = first.GetProperty("state").GetString();
                    string runId = first.GetProperty("dag_run_id").GetString();
                    return (state, runId, null);
                }
            }
            catch (Exception ex) { return (null, null, ex.Message); }
        }

        /// <summary>
        /// DAG의 일시정지(is_paused) 상태와 스케줄 문자열을 조회합니다.
        /// DAG가 서버에 배포되지 않은 경우 IsPaused=null, Error에 HTTP 404가 담깁니다.
        /// </summary>
        public async Task<(bool? IsPaused, string Schedule, string Error)> GetDagInfoAsync(string dagId)
        {
            try
            {
                string url = $"{_baseUrl}/api/v1/dags/{Uri.EscapeDataString(dagId)}";
                var resp = await _http.GetAsync(url).ConfigureAwait(false);
                string body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);

                if (!resp.IsSuccessStatusCode)
                    return (null, null, $"HTTP {(int)resp.StatusCode}: {body.Trim()}");

                using (var doc = JsonDocument.Parse(body))
                {
                    var root = doc.RootElement;
                    bool? isPaused = root.TryGetProperty("is_paused", out var pEl) && pEl.ValueKind != JsonValueKind.Null
                        ? pEl.GetBoolean() : (bool?)null;

                    string schedule = null;
                    if (root.TryGetProperty("schedule_interval", out var sEl) && sEl.ValueKind != JsonValueKind.Null)
                    {
                        if (sEl.ValueKind == JsonValueKind.String)
                            schedule = sEl.GetString();
                        else if (sEl.ValueKind == JsonValueKind.Object && sEl.TryGetProperty("value", out var vEl))
                            schedule = vEl.GetString();
                    }
                    return (isPaused, schedule, null);
                }
            }
            catch (Exception ex) { return (null, null, ex.Message); }
        }

        /// <summary>
        /// DAG를 일시정지(true)하거나 재개(false)합니다.
        /// Airflow 스케줄러가 관리하는 주기 실행의 On/Off 전환에 사용합니다.
        /// </summary>
        public async Task<(bool Ok, string Error)> SetDagPausedAsync(string dagId, bool paused)
        {
            try
            {
                string url = $"{_baseUrl}/api/v1/dags/{Uri.EscapeDataString(dagId)}?update_mask=is_paused";
                var    body = new { is_paused = paused };
                var    json = JsonSerializer.Serialize(body);
                using (var content = new StringContent(json, Encoding.UTF8, "application/json"))
                using (var req = new HttpRequestMessage(new HttpMethod("PATCH"), url) { Content = content })
                {
                    var resp = await _http.SendAsync(req).ConfigureAwait(false);
                    string respBody = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
                    if (!resp.IsSuccessStatusCode)
                        return (false, $"HTTP {(int)resp.StatusCode}: {respBody.Trim()}");
                    return (true, null);
                }
            }
            catch (Exception ex) { return (false, ex.Message); }
        }

        public void Dispose() { _http?.Dispose(); }
    }
}
