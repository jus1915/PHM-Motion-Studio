using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Newtonsoft.Json.Linq;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services.LLM;

namespace PHM_Project_DockPanel.UI.Chat
{
    /// <summary>
    /// Claude 기반 AI 관제 보조원 채팅 패널.
    /// 자연어로 서보 제어, 축 이동, 파라미터 변경, 시퀀스 생성 등을 수행합니다.
    /// </summary>
    public class LlmChatPanel : DockContent
    {
        // ──────────────────────────────────────────────────────────────────────
        // Win32 — TextBox 워터마크
        // ──────────────────────────────────────────────────────────────────────
        private const int EM_SETCUEBANNER = 0x1501;
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, string lParam);

        private static void SetWatermark(TextBox tb, string text)
        {
            if (tb == null) return;
            void apply() => SendMessage(tb.Handle, EM_SETCUEBANNER, (IntPtr)1, text);
            if (tb.IsHandleCreated) apply();
            else tb.HandleCreated += (s, e) => apply();
        }

        // ──────────────────────────────────────────────────────────────────────
        // 상수 / 설정
        // ──────────────────────────────────────────────────────────────────────
        private static readonly string ApiKeyFile = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "PHMMotionStudio", "llm_api_key.txt");

        private const int MaxTurnsPerRequest = 8;   // 한 사용자 메시지당 최대 agent 턴 수

        // ──────────────────────────────────────────────────────────────────────
        // 필드
        // ──────────────────────────────────────────────────────────────────────
        private readonly LlmAppContext _ctx;
        private readonly ClaudeApiClient _client;
        private readonly JArray _tools;
        private readonly JArray _messages = new JArray();

        // UI 컨트롤
        private RichTextBox _rtbChat;
        private TextBox _txtInput;
        private TextBox _txtApiKey;
        private Button _btnSend;
        private Button _btnCancel;
        private Button _btnClear;
        private Label _lblStatus;

        // 상태
        private CancellationTokenSource _cts;
        private bool _isBusy;

        // ──────────────────────────────────────────────────────────────────────
        // 생성자
        // ──────────────────────────────────────────────────────────────────────
        public LlmChatPanel(LlmAppContext ctx)
        {
            _ctx = ctx;
            _client = new ClaudeApiClient();
            _tools = AppToolProvider.GetToolDefinitions();

            Text = "AI 관제 보조원";
            HideOnClose = true;

            // 확인 다이얼로그 콜백 설정
            // RunAgentLoopAsync는 UI 스레드의 await 체인에서 실행되므로
            // MessageBox.Show를 직접 호출해도 안전합니다.
            _ctx.ConfirmAction = msg =>
                MessageBox.Show(msg, "AI 작업 확인",
                    MessageBoxButtons.YesNo, MessageBoxIcon.Question,
                    MessageBoxDefaultButton.Button2) == DialogResult.Yes;

            BuildUI();
            LoadApiKey();
        }

        // ──────────────────────────────────────────────────────────────────────
        // UI 구성
        // ──────────────────────────────────────────────────────────────────────
        private void BuildUI()
        {
            // ── 상단: API Key 바 ──────────────────────────────────────────────
            var topPanel = new Panel
            {
                Dock = DockStyle.Top, Height = 30,
                BackColor = Color.FromArgb(240, 242, 246),
                Padding = new Padding(6, 4, 6, 0)
            };

            var lblKey = new Label
            {
                Text = "API Key:", AutoSize = true,
                Left = 6, Top = 7, ForeColor = Color.FromArgb(80, 80, 80)
            };
            _txtApiKey = new TextBox
            {
                Left = 60, Top = 4, Width = 240, Height = 22,
                UseSystemPasswordChar = true,
                Font = new Font("Consolas", 9f),
                BorderStyle = BorderStyle.FixedSingle
            };
            _txtApiKey.TextChanged += (s, e) =>
            {
                _client.ApiKey = _txtApiKey.Text.Trim();
                SaveApiKey();
            };

            var lblModel = new Label
            {
                Text = "  |  " + _client.Model,
                AutoSize = true, Left = 308, Top = 7,
                ForeColor = Color.FromArgb(120, 120, 140),
                Font = new Font("맑은 고딕", 8f)
            };

            topPanel.Controls.Add(lblKey);
            topPanel.Controls.Add(_txtApiKey);
            topPanel.Controls.Add(lblModel);

            // ── 하단: 입력 영역 ───────────────────────────────────────────────
            var bottomPanel = new Panel
            {
                Dock = DockStyle.Bottom, Height = 60,
                Padding = new Padding(6, 4, 6, 4)
            };

            _lblStatus = new Label
            {
                Text = "준비됨",
                Dock = DockStyle.Bottom, Height = 16,
                ForeColor = Color.Gray,
                Font = new Font("맑은 고딕", 8f),
                TextAlign = ContentAlignment.MiddleLeft
            };

            var inputRow = new Panel { Dock = DockStyle.Fill };

            _btnSend = new Button
            {
                Text = "전송",
                Width = 54, Dock = DockStyle.Right,
                BackColor = Color.FromArgb(0, 112, 203),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font = new Font("맑은 고딕", 9f, FontStyle.Bold),
                Cursor = Cursors.Hand
            };
            _btnSend.FlatAppearance.BorderSize = 0;
            _btnSend.Click += async (s, e) => await SendMessageAsync();

            _btnCancel = new Button
            {
                Text = "취소",
                Width = 48, Dock = DockStyle.Right,
                Enabled = false,
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            _btnCancel.Click += (s, e) => _cts?.Cancel();

            _btnClear = new Button
            {
                Text = "지우기",
                Width = 54, Dock = DockStyle.Right,
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            _btnClear.Click += (s, e) =>
            {
                _messages.Clear();
                _rtbChat.Clear();
                PrintWelcome();
            };

            _txtInput = new TextBox
            {
                Dock = DockStyle.Fill,
                Font = new Font("맑은 고딕", 10f),
                BorderStyle = BorderStyle.FixedSingle
            };
            _txtInput.KeyDown += async (s, e) =>
            {
                if (e.KeyCode == Keys.Enter && !e.Shift)
                {
                    e.SuppressKeyPress = true;
                    await SendMessageAsync();
                }
            };

            inputRow.Controls.Add(_txtInput);
            inputRow.Controls.Add(_btnClear);
            inputRow.Controls.Add(_btnCancel);
            inputRow.Controls.Add(_btnSend);

            bottomPanel.Controls.Add(inputRow);
            bottomPanel.Controls.Add(_lblStatus);

            // ── 중앙: 채팅 히스토리 ───────────────────────────────────────────
            _rtbChat = new RichTextBox
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                BackColor = Color.FromArgb(252, 252, 254),
                Font = new Font("맑은 고딕", 9.5f),
                BorderStyle = BorderStyle.None,
                ScrollBars = RichTextBoxScrollBars.Vertical,
                WordWrap = true,
                Padding = new Padding(4)
            };

            Controls.Add(_rtbChat);
            Controls.Add(bottomPanel);
            Controls.Add(topPanel);

            // 워터마크
            SetWatermark(_txtInput, "메시지를 입력하세요... (Enter: 전송,  Shift+Enter: 줄바꿈)");

            PrintWelcome();
        }

        private void PrintWelcome()
        {
            AppendLine("─────────────────────────────────────────",
                Color.FromArgb(200, 200, 210));
            AppendLine("  AI 관제 보조원 (PHM Motion Studio)",
                Color.FromArgb(0, 80, 160), bold: true);
            AppendLine("─────────────────────────────────────────",
                Color.FromArgb(200, 200, 210));
            AppendLine("서보 제어, 축 이동, 파라미터 변경, 시퀀스 생성 등을 자연어로 요청하세요.",
                Color.FromArgb(80, 80, 80), italic: true);
            AppendLine("");
            AppendLine("예시 명령어:", Color.FromArgb(100, 100, 100), bold: true);
            AppendLine("  • 서보 전체 On 해줘", Color.FromArgb(60, 120, 60));
            AppendLine("  • 1번 축을 제외하고 50 mm씩 왕복하는 시퀀스 저장해줘", Color.FromArgb(60, 120, 60));
            AppendLine("  • 2번 축 최대 속도 500으로 변경해줘", Color.FromArgb(60, 120, 60));
            AppendLine("  • 현재 설비 상태 알려줘", Color.FromArgb(60, 120, 60));
            AppendLine("");
        }

        // ──────────────────────────────────────────────────────────────────────
        // 메시지 전송
        // ──────────────────────────────────────────────────────────────────────
        private async Task SendMessageAsync()
        {
            string text = _txtInput.Text.Trim();
            if (string.IsNullOrEmpty(text) || _isBusy) return;

            if (string.IsNullOrEmpty(_client.ApiKey))
            {
                AppendLine("[오류] 상단의 API Key 필드에 Anthropic API Key를 입력하세요.",
                    Color.Red, bold: true);
                return;
            }

            _txtInput.Clear();
            SetBusy(true);

            // 사용자 메시지 표시
            AppendLine("");
            AppendLine("┌ 나", Color.FromArgb(20, 80, 160), bold: true);
            AppendLine($"│  {text}", Color.FromArgb(20, 60, 140));
            AppendLine("└─", Color.FromArgb(20, 80, 160));

            _messages.Add(new JObject { ["role"] = "user", ["content"] = text });

            _cts = new CancellationTokenSource();
            try
            {
                await RunAgentLoopAsync(_cts.Token);
            }
            catch (OperationCanceledException)
            {
                AppendLine("[취소됨]", Color.Gray, italic: true);
            }
            catch (Exception ex)
            {
                AppendLine($"[오류] {ex.Message}", Color.Red);
            }
            finally
            {
                SetBusy(false);
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        // Agent 루프 (Tool Use 처리)
        // ──────────────────────────────────────────────────────────────────────
        private async Task RunAgentLoopAsync(CancellationToken ct)
        {
            for (int turn = 0; turn < MaxTurnsPerRequest; turn++)
            {
                SetStatus(turn == 0 ? "AI 응답 생성 중..." : "후속 처리 중...");

                bool firstToken = true;
                var aiTextSb = new StringBuilder();

                ClaudeResponse resp = await _client.SendAsync(
                    _messages,
                    AppToolProvider.BuildSystemPrompt(_ctx),
                    _tools,
                    token =>
                    {
                        // 스트리밍 토큰 — UI 스레드에서 호출됨
                        if (firstToken)
                        {
                            AppendLine("");
                            AppendLine("┌ AI", Color.FromArgb(20, 130, 60), bold: true);
                            Print("│  ", Color.FromArgb(20, 130, 60));
                            firstToken = false;
                        }
                        // 줄바꿈 처리
                        foreach (char ch in token)
                        {
                            if (ch == '\n')
                                Print("\n│  ", Color.FromArgb(20, 130, 60));
                            else
                                Print(ch.ToString(), Color.FromArgb(20, 20, 20));
                        }
                        aiTextSb.Append(token);
                    },
                    ct);

                if (!firstToken)
                {
                    AppendLine("");
                    AppendLine("└─", Color.FromArgb(20, 130, 60));
                }

                // assistant 메시지 히스토리에 추가
                var assistantContent = new JArray();
                if (aiTextSb.Length > 0)
                    assistantContent.Add(new JObject { ["type"] = "text", ["text"] = aiTextSb.ToString() });

                foreach (var tu in resp.ToolUses)
                    assistantContent.Add(new JObject
                    {
                        ["type"] = "tool_use",
                        ["id"] = tu.Id,
                        ["name"] = tu.Name,
                        ["input"] = tu.Input
                    });

                _messages.Add(new JObject { ["role"] = "assistant", ["content"] = assistantContent });

                // 도구 호출 없으면 루프 종료
                if (resp.StopReason != "tool_use" || resp.ToolUses.Count == 0)
                    break;

                // ── 도구 실행 ─────────────────────────────────────────────────
                var toolResults = new JArray();
                foreach (var tu in resp.ToolUses)
                {
                    SetStatus($"도구 실행: {tu.Name}…");

                    string friendlyName = GetFriendlyToolName(tu.Name);
                    AppendLine($"  ▶ {friendlyName}", Color.FromArgb(100, 60, 180), italic: true);

                    string toolResult = await AppToolProvider.ExecuteToolAsync(tu.Name, tu.Input, _ctx);
                    AppendLine($"  ← {toolResult}", Color.FromArgb(90, 90, 100), italic: true);

                    toolResults.Add(new JObject
                    {
                        ["type"] = "tool_result",
                        ["tool_use_id"] = tu.Id,
                        ["content"] = toolResult
                    });
                }

                _messages.Add(new JObject { ["role"] = "user", ["content"] = toolResults });
            }

            AppendLine("");
            SetStatus("준비됨");
        }

        private static string GetFriendlyToolName(string name)
        {
            switch (name)
            {
                case "get_system_status":   return "시스템 상태 조회";
                case "get_axes_status":     return "축 상태 조회";
                case "get_axes_config":     return "축 파라미터 조회";
                case "set_servo":           return "서보 ON/OFF";
                case "move_axes":           return "축 이동";
                case "set_axis_param":      return "파라미터 변경";
                case "set_label":           return "레이블 변경";
                case "save_teaching_sequence": return "시퀀스 저장";
                default:                    return name;
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        // UI 헬퍼
        // ──────────────────────────────────────────────────────────────────────
        private void SetBusy(bool busy)
        {
            _isBusy = busy;
            _btnSend.Enabled = !busy;
            _btnCancel.Enabled = busy;
            _txtInput.Enabled = !busy;
        }

        private void SetStatus(string msg)
        {
            if (InvokeRequired) BeginInvoke(new Action(() => _lblStatus.Text = msg));
            else _lblStatus.Text = msg;
        }

        private void Print(string text, Color color, bool bold = false, bool italic = false)
        {
            if (_rtbChat.InvokeRequired)
            {
                _rtbChat.BeginInvoke(new Action(() => Print(text, color, bold, italic)));
                return;
            }

            _rtbChat.SelectionStart = _rtbChat.TextLength;
            _rtbChat.SelectionLength = 0;
            _rtbChat.SelectionColor = color;

            FontStyle style = FontStyle.Regular;
            if (bold) style |= FontStyle.Bold;
            if (italic) style |= FontStyle.Italic;

            if (style != FontStyle.Regular)
                _rtbChat.SelectionFont = new Font(_rtbChat.Font, style);

            _rtbChat.AppendText(text);

            _rtbChat.SelectionColor = _rtbChat.ForeColor;
            _rtbChat.SelectionFont = _rtbChat.Font;
            _rtbChat.ScrollToCaret();
        }

        private void AppendLine(string text = "",
            Color? color = null, bool bold = false, bool italic = false)
        {
            Print(text + "\n", color ?? Color.Black, bold, italic);
        }

        // ──────────────────────────────────────────────────────────────────────
        // API Key 로드 / 저장
        // ──────────────────────────────────────────────────────────────────────
        private void LoadApiKey()
        {
            try
            {
                if (!File.Exists(ApiKeyFile)) return;
                string key = File.ReadAllText(ApiKeyFile).Trim();
                if (!string.IsNullOrEmpty(key))
                {
                    _txtApiKey.Text = key;
                    _client.ApiKey = key;
                }
            }
            catch { /* 무시 */ }
        }

        private void SaveApiKey()
        {
            try
            {
                string dir = Path.GetDirectoryName(ApiKeyFile);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(ApiKeyFile, _txtApiKey.Text.Trim());
            }
            catch { /* 무시 */ }
        }
    }
}
