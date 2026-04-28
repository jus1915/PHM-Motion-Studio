using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace PHM_Project_DockPanel.Services.LLM
{
    public class ToolUseBlock
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public JObject Input { get; set; }
    }

    public class ClaudeResponse
    {
        public string StopReason { get; set; }
        public string Text { get; set; }
        public List<ToolUseBlock> ToolUses { get; set; } = new List<ToolUseBlock>();
    }

    /// <summary>
    /// Claude Messages API 클라이언트 (스트리밍 + Tool Use 지원).
    /// </summary>
    public class ClaudeApiClient
    {
        private static readonly HttpClient _http;

        static ClaudeApiClient()
        {
            _http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        }

        public string ApiKey { get; set; }
        public string Model { get; set; } = "claude-sonnet-4-6";
        public int MaxTokens { get; set; } = 4096;

        /// <summary>
        /// Claude에 메시지를 보내고 스트리밍 응답을 받습니다.
        /// onToken 콜백으로 텍스트 토큰이 실시간 전달되며,
        /// 최종 응답(텍스트 + 도구 호출 목록)을 반환합니다.
        /// </summary>
        public async Task<ClaudeResponse> SendAsync(
            JArray messages,
            string systemPrompt,
            JArray tools,
            Action<string> onToken,
            CancellationToken ct)
        {
            var body = new JObject
            {
                ["model"] = Model,
                ["max_tokens"] = MaxTokens,
                ["stream"] = true,
                ["messages"] = messages
            };

            if (!string.IsNullOrEmpty(systemPrompt))
                body["system"] = systemPrompt;

            if (tools != null && tools.Count > 0)
                body["tools"] = tools;

            var content = new StringContent(
                body.ToString(Formatting.None), Encoding.UTF8, "application/json");

            var request = new HttpRequestMessage(HttpMethod.Post,
                "https://api.anthropic.com/v1/messages") { Content = content };
            request.Headers.Add("x-api-key", ApiKey);
            request.Headers.Add("anthropic-version", "2023-06-01");

            using (var resp = await _http.SendAsync(
                request, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                if (!resp.IsSuccessStatusCode)
                {
                    string errBody = await resp.Content.ReadAsStringAsync();
                    throw new InvalidOperationException(
                        $"Claude API 오류 {(int)resp.StatusCode}: {errBody}");
                }

                return await ParseSseStreamAsync(resp, onToken, ct);
            }
        }

        private static async Task<ClaudeResponse> ParseSseStreamAsync(
            HttpResponseMessage resp,
            Action<string> onToken,
            CancellationToken ct)
        {
            var result = new ClaudeResponse();
            var textBuilder = new StringBuilder();

            // 블록 인덱스별 타입과 누적 데이터 추적
            var blockTypes = new Dictionary<int, string>();
            var toolMeta = new Dictionary<int, JObject>();       // index → {id, name}
            var toolJsonBuilders = new Dictionary<int, StringBuilder>(); // index → partial JSON

            using (var stream = await resp.Content.ReadAsStreamAsync())
            using (var reader = new StreamReader(stream, Encoding.UTF8))
            {
                string line;
                while ((line = await reader.ReadLineAsync()) != null)
                {
                    ct.ThrowIfCancellationRequested();

                    if (!line.StartsWith("data: ")) continue;
                    string json = line.Substring(6).Trim();
                    if (string.IsNullOrEmpty(json) || json == "[DONE]") continue;

                    JObject evt;
                    try { evt = JObject.Parse(json); }
                    catch { continue; }

                    string type = evt["type"]?.ToString();
                    if (type == null) continue;

                    switch (type)
                    {
                        case "content_block_start":
                        {
                            int idx = evt["index"]?.Value<int>() ?? 0;
                            var block = evt["content_block"] as JObject;
                            string bType = block?["type"]?.ToString() ?? "text";
                            blockTypes[idx] = bType;

                            if (bType == "tool_use")
                            {
                                toolMeta[idx] = new JObject
                                {
                                    ["id"] = block["id"],
                                    ["name"] = block["name"]
                                };
                                toolJsonBuilders[idx] = new StringBuilder();
                            }
                            break;
                        }

                        case "content_block_delta":
                        {
                            int idx = evt["index"]?.Value<int>() ?? 0;
                            var delta = evt["delta"] as JObject;
                            string dType = delta?["type"]?.ToString();

                            if (dType == "text_delta")
                            {
                                string text = delta["text"]?.ToString() ?? "";
                                textBuilder.Append(text);
                                onToken?.Invoke(text);
                            }
                            else if (dType == "input_json_delta")
                            {
                                string partial = delta["partial_json"]?.ToString() ?? "";
                                if (toolJsonBuilders.TryGetValue(idx, out var sb))
                                    sb.Append(partial);
                            }
                            break;
                        }

                        case "content_block_stop":
                        {
                            int idx = evt["index"]?.Value<int>() ?? 0;
                            if (blockTypes.TryGetValue(idx, out string bt) &&
                                bt == "tool_use" &&
                                toolMeta.TryGetValue(idx, out var meta) &&
                                toolJsonBuilders.TryGetValue(idx, out var jsonSb))
                            {
                                JObject inputObj;
                                try { inputObj = JObject.Parse(jsonSb.ToString()); }
                                catch { inputObj = new JObject(); }

                                result.ToolUses.Add(new ToolUseBlock
                                {
                                    Id = meta["id"]?.ToString(),
                                    Name = meta["name"]?.ToString(),
                                    Input = inputObj
                                });
                            }
                            break;
                        }

                        case "message_delta":
                        {
                            var delta = evt["delta"] as JObject;
                            result.StopReason = delta?["stop_reason"]?.ToString();
                            break;
                        }
                    }
                }
            }

            result.Text = textBuilder.ToString();
            if (result.StopReason == null)
                result.StopReason = result.ToolUses.Count > 0 ? "tool_use" : "end_turn";

            return result;
        }
    }
}
