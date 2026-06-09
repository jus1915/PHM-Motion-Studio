using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using PHM_Project_DockPanel.Services;
using PHM_Project_DockPanel.Services.Core;
using WeifenLuo.WinFormsUI.Docking;

namespace PHM_Project_DockPanel.UI.Dashboard
{
    // =========================================================================
    //  ChannelAeDashboardForm — 채널 AE 전용 미니멀 실시간 추론 대시보드
    //
    //  ▶ 설계 원칙: 레거시(KNN/CLS/SKL/DB모니터링/토크전역AE) 전부 무시.
    //    서버에 학습된 "채널 AE" 모델만 추론하고, active 여부와 관계없이
    //    모든 채널의 상태/점수를 그리드에 항상 표시한다.
    //
    //  ▶ 데이터 소스: ServerSettings.Current.ContinuousDataPath 아래에서
    //    가장 최근 "*_Combined.csv" 를 찾아 각 채널 컬럼의 마지막 window_size
    //    샘플을 읽어 POST /predict/channel_ae 로 전송한다.
    //
    //  ▶ 외부 의존: ServerSettings(URL/경로), InferenceServerClient(HTTP),
    //    ChannelAeMeta/ChannelAePredictResponse(DTO) 만 사용. 모터/로거 내부에
    //    직접 의존하지 않아 완전히 독립적으로 동작한다.
    // =========================================================================
    public sealed class ChannelAeDashboardForm : DockContent
    {
        // ── UI ───────────────────────────────────────────────────────────────
        private readonly DataGridView _grid;
        private readonly Button       _btnStart;
        private readonly Button       _btnStop;
        private readonly Label        _lblStatus;
        private readonly Label        _lblInfo;
        private readonly NumericUpDown _numInterval;
        private readonly ListBox      _events;

        // 채널별 직전 이상 여부 — 정상↔이상 전이 시에만 이벤트 로그 (스팸 방지)
        private readonly System.Collections.Generic.Dictionary<string, bool> _lastAnomaly
            = new System.Collections.Generic.Dictionary<string, bool>();

        // 이벤트 로그 항목 (OwnerDraw로 위험은 빨간색)
        private sealed class Ev
        {
            public string Text;
            public bool   Danger;
            public override string ToString() => Text;
        }

        // ── 루프 상태 ─────────────────────────────────────────────────────────
        private InferenceServerClient _client;
        private CancellationTokenSource _cts;
        private Task _loop;

        private ChannelAeMeta _meta;          // 채널 목록 + window_size
        private string _csvPath;              // 현재 사용 중인 Combined CSV
        private int _intervalMs = 500;

        public ChannelAeDashboardForm()
        {
            Text     = "채널 AE 추론";
            DockAreas = DockAreas.Document | DockAreas.Float;

            // ── 상단 툴바 ──────────────────────────────────────────────────────
            _btnStart = new Button { Text = "▶ 진단 시작", Width = 100, Height = 28 };
            _btnStop  = new Button { Text = "■ 중지", Width = 70, Height = 28, Enabled = false };
            _btnStart.Click += (s, e) => StartLoop();
            _btnStop.Click  += (s, e) => StopLoop();

            _numInterval = new NumericUpDown
            {
                Minimum = 100, Maximum = 5000, Increment = 100,
                Value = _intervalMs, Width = 70, Height = 28,
            };
            _numInterval.ValueChanged += (s, e) => _intervalMs = (int)_numInterval.Value;

            var toolbar = new FlowLayoutPanel
            {
                Dock = DockStyle.Top, Height = 38, Padding = new Padding(6, 5, 6, 5),
                FlowDirection = FlowDirection.LeftToRight, WrapContents = false,
            };
            toolbar.Controls.Add(_btnStart);
            toolbar.Controls.Add(_btnStop);
            toolbar.Controls.Add(new Label { Text = "주기(ms)", AutoSize = true, Padding = new Padding(8, 8, 2, 0) });
            toolbar.Controls.Add(_numInterval);

            // ── 정보/상태 라벨 ─────────────────────────────────────────────────
            _lblInfo = new Label
            {
                Dock = DockStyle.Top, Height = 22, ForeColor = Color.DimGray,
                Padding = new Padding(8, 4, 0, 0), Text = "대기 중",
            };
            _lblStatus = new Label
            {
                Dock = DockStyle.Bottom, Height = 24, ForeColor = Color.DimGray,
                Padding = new Padding(8, 4, 0, 0), Text = "상태: 대기",
            };

            // ── 그리드 ─────────────────────────────────────────────────────────
            _grid = new DataGridView
            {
                Dock = DockStyle.Fill, ReadOnly = true, AllowUserToAddRows = false,
                AllowUserToDeleteRows = false, RowHeadersVisible = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                MultiSelect = false, BackgroundColor = Color.White,
                ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing,
            };
            _grid.Columns.Add("ch",   "채널");
            _grid.Columns.Add("st",   "상태");
            _grid.Columns.Add("act",  "active_score");
            _grid.Columns.Add("ano",  "anomaly_score");
            _grid.Columns.Add("recon","recon_error");
            _grid.Columns.Add("verdict", "판정");
            _grid.Columns.Add("ts",   "갱신");
            _grid.Columns["verdict"].DefaultCellStyle.Font =
                new Font(_grid.Font, FontStyle.Bold);

            // ── 이벤트 로그 ────────────────────────────────────────────────────
            _events = new ListBox
            {
                Dock = DockStyle.Bottom, Height = 140, IntegralHeight = false,
                DrawMode = DrawMode.OwnerDrawFixed, BorderStyle = BorderStyle.FixedSingle,
                Font = new Font("Consolas", 9f),
            };
            _events.DrawItem += (s, e) =>
            {
                if (e.Index < 0) return;
                e.DrawBackground();
                var ev = _events.Items[e.Index] as Ev;
                Color c = (ev != null && ev.Danger) ? Color.Red : Color.Black;
                using (var br = new SolidBrush(c))
                    e.Graphics.DrawString(_events.Items[e.Index].ToString(),
                        e.Font, br, e.Bounds);
                e.DrawFocusRectangle();
            };
            var lblEv = new Label
            {
                Dock = DockStyle.Bottom, Height = 20, ForeColor = Color.DimGray,
                Padding = new Padding(8, 3, 0, 0), Text = "── 이상 감지 이벤트 ──",
            };

            // 도킹 순서: grid(Fill) → 하단(status, events, lblEv) → 상단(toolbar, info)
            Controls.Add(_grid);
            Controls.Add(_lblStatus);
            Controls.Add(_events);
            Controls.Add(lblEv);
            Controls.Add(toolbar);
            Controls.Add(_lblInfo);
        }

        // ── 시작/중지 ──────────────────────────────────────────────────────────
        private void StartLoop()
        {
            if (_loop != null && !_loop.IsCompleted) return;

            string url = ServerSettings.Current.InferenceServerUrl;
            if (string.IsNullOrWhiteSpace(url))
            {
                MessageBox.Show("추론 서버 URL이 설정되지 않았습니다. (연결 설정)");
                return;
            }

            _client = new InferenceServerClient(url);
            _cts    = new CancellationTokenSource();
            _btnStart.Enabled = false;
            _btnStop.Enabled  = true;
            SetStatus("상태: 시작 중...");
            _loop = Task.Run(() => LoopAsync(_cts.Token));
        }

        private void StopLoop()
        {
            try { _cts?.Cancel(); } catch { }
            _btnStart.Enabled = true;
            _btnStop.Enabled  = false;
            SetStatus("상태: 중지됨");
        }

        // ── 메인 루프 ──────────────────────────────────────────────────────────
        private async Task LoopAsync(CancellationToken ct)
        {
            // 1) 채널 AE 메타 조회 — 없으면 종료
            _meta = await _client.GetChannelAeInfoAsync().ConfigureAwait(false);
            if (_meta == null || _meta.Channels == null || _meta.Channels.Count == 0)
            {
                Ui(() =>
                {
                    SetStatus("상태: 채널 AE 모델 없음 — Airflow에서 채널 AE 학습을 먼저 실행하세요.");
                    _btnStart.Enabled = true;
                    _btnStop.Enabled  = false;
                });
                return;
            }
            Ui(() =>
            {
                SeedRows();
                _lblInfo.Text = $"채널 {_meta.Channels.Count}개  ·  window_size={_meta.WindowSize}  ·  서버={ServerSettings.Current.InferenceServerUrl}";
            });

            int cycle = 0;
            while (!ct.IsCancellationRequested)
            {
                try { await Task.Delay(_intervalMs, ct).ConfigureAwait(false); }
                catch { break; }

                // CSV 경로 주기적 재탐색 (시작 시 + 20사이클마다 + 캐시 유실 시)
                if (_csvPath == null || !File.Exists(_csvPath) || (++cycle % 20 == 0))
                    _csvPath = FindLatestCombinedCsv();

                if (string.IsNullOrEmpty(_csvPath))
                {
                    Ui(() => SetStatus("상태: Combined CSV 없음 — 연속 수집을 시작하세요. "
                        + $"(탐색 경로: {ServerSettings.Current.ContinuousDataPath})"));
                    continue;
                }

                int ws = _meta.WindowSize > 0 ? _meta.WindowSize : 128;
                int updated = 0;

                foreach (var kv in _meta.Channels)
                {
                    if (ct.IsCancellationRequested) break;
                    string channel = kv.Key;

                    float[] window = ReadLastWindow(_csvPath, channel, ws);
                    if (window == null)
                    {
                        Ui(() => SetRow(channel, "데이터부족", null, null, null, "-", Color.Gray));
                        continue;
                    }

                    ChannelAePredictResponse r =
                        await _client.PredictChannelAeAsync(channel, window, ws, ct)
                                     .ConfigureAwait(false);

                    if (r.IsModelMissing) { Ui(() => SetRow(channel, "모델없음", null, null, null, "-", Color.Gray)); continue; }
                    if (r.IsError)        { Ui(() => SetRow(channel, "오류", null, null, null, r.Error, Color.OrangeRed)); continue; }

                    bool isActive = r.ChannelState == ChannelActiveState.Active;
                    string verdict;
                    Color color;
                    if (!isActive)             { verdict = "-";    color = Color.Gray; }
                    else if (r.IsAnomaly)      { verdict = "⚠ 이상"; color = Color.Red; }
                    else                       { verdict = "✓ 정상"; color = Color.ForestGreen; }

                    double? ano   = isActive ? (double?)r.AnomalyScore : null;
                    double? recon = isActive ? r.ReconError : null;

                    // 정상↔이상 전이 시에만 이벤트 로그 (매 주기 스팸 방지)
                    bool nowAnomaly = isActive && r.IsAnomaly;
                    bool wasAnomaly;
                    _lastAnomaly.TryGetValue(channel, out wasAnomaly);
                    if (nowAnomaly && !wasAnomaly)
                        Ui(() => AddEvent($"⚠ {channel} 이상 감지  anomaly={r.AnomalyScore:F3}  recon={r.ReconError:E2}", true));
                    else if (!nowAnomaly && wasAnomaly)
                        Ui(() => AddEvent($"✓ {channel} 정상 복귀", false));
                    _lastAnomaly[channel] = nowAnomaly;

                    bool rowAnomaly = nowAnomaly;
                    Ui(() => SetRow(channel, r.ActiveState, r.ActiveScore, ano, recon, verdict, color, rowAnomaly));
                    updated++;
                }

                int u = updated;
                Ui(() => SetStatus($"상태: 추론 중...  ({u}/{_meta.Channels.Count} 채널)  ·  {Path.GetFileName(_csvPath)}"));
            }
        }

        // ── 그리드 헬퍼 ────────────────────────────────────────────────────────
        private void SeedRows()
        {
            _grid.Rows.Clear();
            _lastAnomaly.Clear();
            foreach (var kv in _meta.Channels)
            {
                int i = _grid.Rows.Add(kv.Key, "-", "-", "-", "-", "-", "-");
                _grid.Rows[i].Tag = kv.Key;
            }
        }

        private void SetRow(string channel, string state, double? act, double? ano,
                            double? recon, string verdict, Color verdictColor,
                            bool anomalyRow = false)
        {
            foreach (DataGridViewRow row in _grid.Rows)
            {
                if (!Equals(row.Tag, channel)) continue;
                row.Cells["st"].Value      = state;
                row.Cells["act"].Value     = act.HasValue   ? act.Value.ToString("F3", CultureInfo.InvariantCulture) : "-";
                row.Cells["ano"].Value     = ano.HasValue   ? ano.Value.ToString("F3", CultureInfo.InvariantCulture) : "-";
                row.Cells["recon"].Value   = recon.HasValue ? recon.Value.ToString("E2", CultureInfo.InvariantCulture) : "-";
                row.Cells["verdict"].Value = verdict;
                row.Cells["verdict"].Style.ForeColor = verdictColor;
                row.Cells["ts"].Value      = DateTime.Now.ToString("HH:mm:ss");
                row.DefaultCellStyle.BackColor = anomalyRow ? Color.MistyRose : Color.White;
                return;
            }
        }

        private void AddEvent(string msg, bool danger)
        {
            string line = $"[{DateTime.Now:HH:mm:ss}] {msg}";
            _events.Items.Insert(0, new Ev { Text = line, Danger = danger });
            while (_events.Items.Count > 200)
                _events.Items.RemoveAt(_events.Items.Count - 1);
        }

        private void SetStatus(string text) => _lblStatus.Text = text;

        private void Ui(Action a)
        {
            if (IsDisposed || Disposing) return;
            try { if (InvokeRequired) BeginInvoke(a); else a(); } catch { }
        }

        // ── CSV 탐색/읽기 ──────────────────────────────────────────────────────
        /// <summary>ContinuousDataPath 아래에서 가장 최근 수정된 *_Combined.csv 경로.</summary>
        private static string FindLatestCombinedCsv()
        {
            try
            {
                string root = ServerSettings.Current.ContinuousDataPath;
                if (string.IsNullOrWhiteSpace(root)) root = @"C:\Data\PHM_Logs\Signals";
                if (!Directory.Exists(root)) return null;

                string best = null;
                DateTime bestTime = DateTime.MinValue;
                foreach (string f in Directory.EnumerateFiles(root, "*_Combined.csv", SearchOption.AllDirectories))
                {
                    DateTime t = File.GetLastWriteTime(f);
                    if (t > bestTime) { bestTime = t; best = f; }
                }
                return best;
            }
            catch { return null; }
        }

        /// <summary>CSV에서 channel 컬럼의 마지막 windowSize 샘플을 읽음. 부족하면 null.</summary>
        private static float[] ReadLastWindow(string csvPath, string channel, int windowSize)
        {
            try
            {
                string[] lines;
                using (var fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs))
                    lines = sr.ReadToEnd().Split(
                        new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

                if (lines.Length < windowSize + 1) return null;

                string[] headers = lines[0].Split(',');
                int col = -1;
                for (int i = 0; i < headers.Length; i++)
                    if (string.Equals(headers[i].Trim(), channel, StringComparison.OrdinalIgnoreCase))
                    { col = i; break; }
                if (col < 0) return null;

                int dataCount = lines.Length - 1;
                var window = new float[windowSize];
                int start = dataCount - windowSize;
                for (int i = 0; i < windowSize; i++)
                {
                    string[] cols = lines[start + i + 1].Split(',');
                    float v = 0f;
                    if (col < cols.Length)
                        float.TryParse(cols[col], NumberStyles.Float, CultureInfo.InvariantCulture, out v);
                    window[i] = v;
                }
                return window;
            }
            catch { return null; }
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            StopLoop();
            try { _client?.Dispose(); } catch { }
            base.OnFormClosed(e);
        }
    }
}
