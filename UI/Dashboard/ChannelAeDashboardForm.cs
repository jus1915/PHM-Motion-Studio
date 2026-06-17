using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
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
        private readonly Chart        _chartActive;
        private readonly Chart        _chartAnomaly;
        private readonly ComboBox     _cmbProfile;     // 모델(프로파일) 선택
        private bool                  _suppressProfileEvent;
        private const string AutoProfileLabel = "(자동)";

        // 차트 롤링 윈도우 길이 (채널별 시리즈당 최대 점 수)
        private const int MaxChartPoints = 300;

        // 채널 → 색상 (active/anomaly 차트에서 동일 색 사용)
        private readonly System.Collections.Generic.Dictionary<string, Color> _channelColors
            = new System.Collections.Generic.Dictionary<string, Color>();
        private static readonly Color[] Palette =
        {
            Color.RoyalBlue, Color.Crimson, Color.ForestGreen, Color.DarkOrange,
            Color.MediumPurple, Color.Teal, Color.Sienna, Color.DeepPink,
        };

        // 가속도 x/y/z 합성 magnitude 가상 채널 (학습 스크립트와 동일 규약).
        // 이 채널은 CSV의 x/y/z 컬럼을 읽어 √(x²+y²+z²) 로 변환해 전송한다.
        private const string AccelMagChannel = "accel_mag";

        // 채널별 직전 이상 여부 — 정상↔이상 전이 시에만 이벤트 로그 (스팸 방지)
        private readonly System.Collections.Generic.Dictionary<string, bool> _lastAnomaly
            = new System.Collections.Generic.Dictionary<string, bool>();

        // 채널별 누적 판정: active 판정 횟수 / 그 중 이상 횟수 (진단 시작 이후)
        private readonly System.Collections.Generic.Dictionary<string, int> _cumActive
            = new System.Collections.Generic.Dictionary<string, int>();
        private readonly System.Collections.Generic.Dictionary<string, int> _cumAnomaly
            = new System.Collections.Generic.Dictionary<string, int>();

        // 이동 이상률 판정: 최근 _windowN 개 active 윈도우 중 raw 이상 비율이
        // _ratioPct% 이상이면 "확정 이상". 부하처럼 산발적(윈도우의 일부만 thr 초과)
        // 인 이상도 잡는다. (연속 판정으로는 산발 이상이 안 잡힘)
        // 각 표본에 시각을 기록해 RecentMaxAgeSec 지난 것은 만료 → 부하 제거 후
        // 해당 축이 정지(inactive)해도 옛 이상 기록이 사라져 정상 복귀한다.
        private struct RawSample { public bool Raw; public DateTime T; }
        private readonly System.Collections.Generic.Dictionary<string, System.Collections.Generic.Queue<RawSample>> _recentRaw
            = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.Queue<RawSample>>();
        private volatile int _windowN  = 40;   // 최근 active 윈도우 수
        private volatile int _ratioPct = 12;   // 이상률 임계 % (thr95 정상 5% 오탐을 흡수)
        // 표본 만료 시간(초). 간헐적으로만 운동하는 축(예: Ax1)도 표본이 모이도록
        // 충분히 길게. 단 이 시간만큼 부하 제거 후 정상 복귀가 지연됨.
        private const double RecentMaxAgeSec = 90.0;
        // 판정에 필요한 최소 표본 수. 너무 작으면 표본이 적을 때 비율이 폭발해
        // 오탐(예: 4창 중 1 이상 = 25%). active 비율 ~21%면 90초 큐에 충분히 모임.
        private const int    RecentMinSamples = 10;
        private readonly NumericUpDown _numWindow;
        private readonly NumericUpDown _numRatio;

        // anomaly 차트 y축 상한 (데이터 글리치 스파이크가 차트를 망치지 않게)
        private const double AnomalyChartMax = 5.0;

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

            _cmbProfile = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList, Width = 120, Height = 28,
            };
            _cmbProfile.SelectedIndexChanged += OnProfileChanged;
            toolbar.Controls.Add(new Label { Text = "모델", AutoSize = true, Padding = new Padding(10, 8, 2, 0) });
            toolbar.Controls.Add(_cmbProfile);

            toolbar.Controls.Add(new Label { Text = "주기(ms)", AutoSize = true, Padding = new Padding(8, 8, 2, 0) });
            toolbar.Controls.Add(_numInterval);

            _numWindow = new NumericUpDown
            {
                Minimum = 5, Maximum = 200, Increment = 5, Value = _windowN, Width = 55, Height = 28,
            };
            _numWindow.ValueChanged += (s, e) => _windowN = (int)_numWindow.Value;
            toolbar.Controls.Add(new Label { Text = "이상률: 최근", AutoSize = true, Padding = new Padding(10, 8, 2, 0) });
            toolbar.Controls.Add(_numWindow);

            _numRatio = new NumericUpDown
            {
                Minimum = 1, Maximum = 100, Value = _ratioPct, Width = 50, Height = 28,
            };
            _numRatio.ValueChanged += (s, e) => _ratioPct = (int)_numRatio.Value;
            toolbar.Controls.Add(new Label { Text = "중", AutoSize = true, Padding = new Padding(2, 8, 2, 0) });
            toolbar.Controls.Add(_numRatio);
            toolbar.Controls.Add(new Label { Text = "% 이상", AutoSize = true, Padding = new Padding(2, 8, 2, 0) });

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
            _grid.Columns.Add("thr",  "임계(act/inact)");
            _grid.Columns.Add("ano",  "anomaly_score");
            _grid.Columns.Add("recon","recon_error");
            _grid.Columns.Add("verdict", "판정");
            _grid.Columns.Add("cum",  "누적(정상/이상)");
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

            // ── 차트 (active_score / anomaly_score 시계열) ─────────────────────
            _chartActive  = BuildScoreChart("채널별 active_score", "active_score (robust z·무차원)", anomalyLine: false);
            _chartAnomaly = BuildScoreChart("채널별 anomaly_score", "anomaly_score (= recon/thr95)", anomalyLine: true);

            // 차트 2개를 가로로 나란히 (각 50%) — 화면 폭을 최대 활용
            var chartTable = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1,
            };
            chartTable.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            chartTable.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            chartTable.Controls.Add(_chartActive,  0, 0);
            chartTable.Controls.Add(_chartAnomaly, 1, 0);

            // 그리드는 위 가로 전체에 작게(채널 수가 적음), 차트는 아래 넓게
            _grid.Dock   = DockStyle.Top;
            _grid.Height = 150;

            // 도킹 순서: Fill 먼저, 이어서 가장자리는 바깥쪽부터 추가
            //   상단(위→아래): toolbar, info, grid   하단(아래→위): status, events, lblEv
            Controls.Add(chartTable);   // Fill (중앙)
            Controls.Add(toolbar);      // Top - 최상
            Controls.Add(_lblInfo);     // Top
            Controls.Add(_grid);        // Top
            Controls.Add(_lblStatus);   // Bottom - 최하
            Controls.Add(_events);      // Bottom
            Controls.Add(lblEv);        // Bottom
        }

        // ── 프로파일(모델) 선택 ────────────────────────────────────────────────
        /// <summary>서버 프로파일 목록으로 콤보박스를 채우고 현재 활성 항목을 선택합니다.</summary>
        private void PopulateProfiles(ChannelAeProfileList pl)
        {
            _suppressProfileEvent = true;
            _cmbProfile.Items.Clear();
            _cmbProfile.Items.Add(AutoProfileLabel);
            if (pl?.Profiles != null)
                foreach (var p in pl.Profiles)
                    _cmbProfile.Items.Add(p);

            // active(null=자동)에 맞춰 선택. 자동이면 실제 로드 폴더(resolved)를 정보에 표시.
            string sel = string.IsNullOrEmpty(pl?.Active) ? AutoProfileLabel : pl.Active;
            int idx = _cmbProfile.Items.IndexOf(sel);
            _cmbProfile.SelectedIndex = idx >= 0 ? idx : 0;
            _suppressProfileEvent = false;

            if (!string.IsNullOrEmpty(pl?.Resolved))
                AppendResolvedInfo(pl.Resolved, string.IsNullOrEmpty(pl.Active));
        }

        private void AppendResolvedInfo(string resolved, bool isAuto)
        {
            string tag = isAuto ? $"  ·  모델=자동({resolved})" : $"  ·  모델={resolved}";
            // _lblInfo 끝에 모델 정보 갱신 (기존 텍스트 유지가 어려우므로 상태줄에 병기)
            SetStatus($"상태: 모델 '{resolved}' 로드됨");
        }

        private async void OnProfileChanged(object sender, EventArgs e)
        {
            if (_suppressProfileEvent || _client == null) return;
            string sel = _cmbProfile.SelectedItem as string;
            string profile = (sel == AutoProfileLabel) ? null : sel;

            SetStatus($"상태: 모델 전환 중… ({sel})");
            bool ok = await _client.ActivateChannelAeProfileAsync(profile).ConfigureAwait(true);
            if (!ok)
            {
                SetStatus($"상태: 모델 전환 실패 — '{sel}' 메타 없음");
                return;
            }
            // 새 메타로 그리드/차트/baseline 초기화
            var meta = await _client.GetChannelAeInfoAsync().ConfigureAwait(true);
            if (meta != null && meta.Channels != null && meta.Channels.Count > 0)
            {
                _meta = meta;
                SeedRows();
                _lblInfo.Text = $"채널 {meta.Channels.Count}개  ·  window_size={meta.WindowSize}  ·  모델={sel}  ·  서버={ServerSettings.Current.InferenceServerUrl}";
                SetStatus($"상태: 모델 '{sel}' 적용 완료");
            }
        }

        // ── 차트 빌드 ──────────────────────────────────────────────────────────
        private static Chart BuildScoreChart(string title, string yTitle, bool anomalyLine)
        {
            var chart = new Chart { Dock = DockStyle.Fill, BackColor = Color.White };
            var ca = new ChartArea("a") { BackColor = Color.White };
            // 플롯 영역 수동 배치: 위(제목) 9% / 플롯 80% / 아래(범례) · 우측 4% 여백
            // → 자동배치 시 마지막 시간 레이블이 오른쪽에서 잘리던 문제 해결
            ca.Position = new ElementPosition(1f, 9f, 95f, 80f);
            ca.AxisX.LabelStyle.Format     = "HH:mm:ss";
            ca.AxisX.MajorGrid.LineColor   = Color.Gainsboro;
            ca.AxisX.IntervalAutoMode      = IntervalAutoMode.VariableCount;
            ca.AxisY.MajorGrid.LineColor   = Color.Gainsboro;
            ca.AxisY.Minimum               = 0;
            ca.AxisY.Title                 = yTitle;
            ca.AxisY.TitleFont             = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            ca.AxisY.TitleForeColor        = Color.DimGray;
            chart.ChartAreas.Add(ca);

            // anomaly 차트: 임계선 y=1.0 (이 위면 이상) + y축 상한 고정(글리치 스파이크 방지)
            if (anomalyLine)
            {
                ca.AxisY.Maximum = AnomalyChartMax;
                var strip = new StripLine
                {
                    IntervalOffset = 1.0, StripWidth = 0,
                    BorderColor = Color.Red, BorderDashStyle = ChartDashStyle.Dash,
                    BorderWidth = 1,
                };
                ca.AxisY.StripLines.Add(strip);
            }

            chart.Legends.Add(new Legend("lg")
            {
                Docking = Docking.Bottom, Alignment = StringAlignment.Center,
                Font = new Font("Segoe UI", 8f),
            });
            chart.Titles.Add(new Title(title, Docking.Top)
            {
                Font = new Font("Segoe UI", 9f, FontStyle.Bold),
                ForeColor = Color.DimGray,
            });
            return chart;
        }

        /// <summary>차트에서 채널 시리즈를 찾거나 생성합니다.</summary>
        private Series EnsureChartSeries(Chart chart, string channel)
        {
            var s = chart.Series.FindByName(channel);
            if (s != null) return s;

            Color color;
            if (!_channelColors.TryGetValue(channel, out color))
            {
                color = Palette[_channelColors.Count % Palette.Length];
                _channelColors[channel] = color;
            }
            s = new Series(channel)
            {
                ChartType = SeriesChartType.FastLine, XValueType = ChartValueType.DateTime,
                Color = color, BorderWidth = 2, LegendText = channel,
            };
            chart.Series.Add(s);
            return s;
        }

        /// <summary>채널 점수를 차트에 추가하고 롤링 윈도우를 유지합니다.</summary>
        private void PushChart(string channel, double? activeScore, double? anomalyScore)
        {
            DateTime now = DateTime.Now;
            if (activeScore.HasValue)
                AddChartPoint(_chartActive, channel, now, activeScore.Value);
            if (anomalyScore.HasValue)
                AddChartPoint(_chartAnomaly, channel, now, anomalyScore.Value);
        }

        private void AddChartPoint(Chart chart, string channel, DateTime t, double v)
        {
            var s = EnsureChartSeries(chart, channel);
            s.Points.AddXY(t.ToOADate(), v);
            while (s.Points.Count > MaxChartPoints)
                s.Points.RemoveAt(0);
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
            // 0) 프로파일(모델) 목록 로드 → 콤보박스 채움
            var profs = await _client.GetChannelAeProfilesAsync().ConfigureAwait(false);
            Ui(() => PopulateProfiles(profs));

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

                // 각 채널 독립 추론 (가속도는 게이팅 없이 항상 추론 — 정지/운동을
                // 신호만으로 구분할 수 없어 전체 베이스라인으로 학습됨)
                foreach (var kv in _meta.Channels)
                {
                    if (ct.IsCancellationRequested) break;
                    await InferChannelAsync(kv.Key, _csvPath, ws, ct).ConfigureAwait(false);
                }

                Ui(() => SetStatus($"상태: 추론 중...  ({_meta.Channels.Count} 채널)  ·  {Path.GetFileName(_csvPath)}"));
            }
        }

        /// <summary>단일 채널 추론 → 그리드/이벤트 갱신.</summary>
        private async Task InferChannelAsync(
            string channel, string csvPath, int ws, CancellationToken ct)
        {
            float[] window = ReadLastWindow(csvPath, channel, ws);
            if (window == null)
            {
                _lastAnomaly[channel] = false;
                Ui(() => SetRow(channel, "데이터부족", null, null, null, "-", Color.Gray));
                return;
            }

            ChannelAePredictResponse r =
                await _client.PredictChannelAeAsync(channel, window, ws, ct).ConfigureAwait(false);

            if (r.IsModelMissing)
            {
                _lastAnomaly[channel] = false;
                Ui(() => SetRow(channel, "모델없음", null, null, null, "-", Color.Gray));
                return;
            }
            if (r.IsError)
            {
                _lastAnomaly[channel] = false;
                Ui(() => SetRow(channel, "오류", null, null, null, r.Error, Color.OrangeRed));
                return;
            }

            bool isActive = r.ChannelState == ChannelActiveState.Active;

            // ── 이동 이상률 판정: 최근 _windowN active 윈도우 중 raw 이상 비율 ──
            // 부하는 산발적(윈도우의 일부만 thr 초과)이라 "연속"으로는 안 잡힘 →
            // 최근 active 윈도우 중 이상 비율이 _ratioPct% 이상이면 "확정 이상".
            bool rawAnomaly = isActive && r.IsAnomaly;   // 서버 thr99 판정(score>1)

            System.Collections.Generic.Queue<RawSample> q;
            if (!_recentRaw.TryGetValue(channel, out q))
            { q = new System.Collections.Generic.Queue<RawSample>(); _recentRaw[channel] = q; }

            int wn = _windowN;
            DateTime now = DateTime.Now;
            // active 윈도우만 표본 추가
            if (isActive)
                q.Enqueue(new RawSample { Raw = rawAnomaly, T = now });
            // 만료(시간 초과)·크기 초과 표본 제거 — 매 사이클 적용(정지해도 옛 기록 만료)
            DateTime cut = now.AddSeconds(-RecentMaxAgeSec);
            while (q.Count > 0 && (q.Peek().T < cut || q.Count > wn))
                q.Dequeue();

            int anom = 0;
            foreach (var smp in q) if (smp.Raw) anom++;
            double ratioPct = q.Count > 0 ? 100.0 * anom / q.Count : 0;
            int minN = System.Math.Min(wn, RecentMinSamples);
            bool confirmed = q.Count >= minN && ratioPct >= _ratioPct;

            // confirmed 우선: 정지(inactive)여도 최근 이상률이 높으면 이상 유지
            string verdict;
            Color  color;
            if (confirmed)      { verdict = $"⚠ 이상(이상률 {ratioPct:F0}%)"; color = Color.Red; }
            else if (!isActive) { verdict = "-";                          color = Color.Gray; }
            else                { verdict = $"✓ 정상({ratioPct:F0}%)";      color = Color.ForestGreen; }

            double? ano   = isActive ? (double?)r.AnomalyScore : null;
            double? recon = isActive ? r.ReconError : null;

            // 확정 이상 전이 시에만 이벤트 로그
            bool wasAnomaly;
            _lastAnomaly.TryGetValue(channel, out wasAnomaly);
            if (confirmed && !wasAnomaly)
                Ui(() => AddEvent($"⚠ {channel} 이상 확정  이상률={ratioPct:F0}% (최근 {q.Count}창)  anomaly={r.AnomalyScore:F3}", true));
            else if (!confirmed && wasAnomaly)
                Ui(() => AddEvent($"✓ {channel} 정상 복귀  (이상률 {ratioPct:F0}%)", false));
            _lastAnomaly[channel] = confirmed;

            bool active = isActive;
            Ui(() => SetRow(channel, r.ActiveState, r.ActiveScore, ano, recon, verdict, color, confirmed, active));
        }

        // ── 그리드 헬퍼 ────────────────────────────────────────────────────────
        private void SeedRows()
        {
            _grid.Rows.Clear();
            _lastAnomaly.Clear();
            _cumActive.Clear();
            _cumAnomaly.Clear();
            _recentRaw.Clear();
            _channelColors.Clear();
            _chartActive.Series.Clear();
            _chartAnomaly.Series.Clear();
            foreach (var kv in _meta.Channels)
            {
                var info = kv.Value;
                string thr = info == null ? "-"
                    : $"{info.ActiveThreshold.ToString("F2", CultureInfo.InvariantCulture)} / "
                    + $"{info.InactiveThreshold.ToString("F2", CultureInfo.InvariantCulture)}";
                int i = _grid.Rows.Add(kv.Key, "-", "-", thr, "-", "-", "-", "-", "-");
                _grid.Rows[i].Tag = kv.Key;
            }
        }

        private void SetRow(string channel, string state, double? act, double? ano,
                            double? recon, string verdict, Color verdictColor,
                            bool anomalyRow = false, bool countActive = false)
        {
            // 누적 판정 집계 (active 판정일 때만 카운트)
            if (countActive)
            {
                int a; _cumActive.TryGetValue(channel, out a);
                _cumActive[channel] = a + 1;
                if (anomalyRow)
                {
                    int an; _cumAnomaly.TryGetValue(channel, out an);
                    _cumAnomaly[channel] = an + 1;
                }
            }

            foreach (DataGridViewRow row in _grid.Rows)
            {
                if (!Equals(row.Tag, channel)) continue;
                row.Cells["st"].Value      = state;
                row.Cells["act"].Value     = act.HasValue   ? act.Value.ToString("F3", CultureInfo.InvariantCulture) : "-";
                row.Cells["ano"].Value     = ano.HasValue   ? ano.Value.ToString("F3", CultureInfo.InvariantCulture) : "-";
                row.Cells["recon"].Value   = recon.HasValue ? recon.Value.ToString("E2", CultureInfo.InvariantCulture) : "-";
                row.Cells["verdict"].Value = verdict;
                row.Cells["verdict"].Style.ForeColor = verdictColor;
                row.Cells["cum"].Value     = FormatCum(channel);
                row.Cells["ts"].Value      = DateTime.Now.ToString("HH:mm:ss");
                row.DefaultCellStyle.BackColor = anomalyRow ? Color.MistyRose : Color.White;
                break;
            }

            // 차트 갱신: active_score 는 항상, anomaly_score 는 active 일 때만 값 존재
            PushChart(channel, act, ano);
        }

        /// <summary>채널 누적 판정 문자열: "정상수 / 이상수 (이상률%)".</summary>
        private string FormatCum(string channel)
        {
            int a; _cumActive.TryGetValue(channel, out a);
            if (a == 0) return "-";
            int an; _cumAnomaly.TryGetValue(channel, out an);
            double rate = (double)an / a;
            return $"{a - an} / {an} ({rate.ToString("P1", CultureInfo.InvariantCulture)})";
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

        /// <summary>
        /// CSV에서 channel 컬럼의 마지막 windowSize 샘플을 읽음. 부족하면 null.
        /// channel == "accel_mag" 이면 x/y/z 를 읽어 √(x²+y²+z²) 로 합성한다.
        /// </summary>
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
                bool isMag = string.Equals(channel, AccelMagChannel, StringComparison.OrdinalIgnoreCase);

                int[] cols;
                if (isMag)
                {
                    int ix = FindCol(headers, "x"), iy = FindCol(headers, "y"), iz = FindCol(headers, "z");
                    if (ix < 0 || iy < 0 || iz < 0) return null;
                    cols = new[] { ix, iy, iz };
                }
                else
                {
                    int c = FindCol(headers, channel);
                    if (c < 0) return null;
                    cols = new[] { c };
                }

                int dataCount = lines.Length - 1;
                var window = new float[windowSize];
                int start = dataCount - windowSize;
                for (int i = 0; i < windowSize; i++)
                {
                    string[] parts = lines[start + i + 1].Split(',');
                    if (isMag)
                    {
                        float sx = ParseAt(parts, cols[0]);
                        float sy = ParseAt(parts, cols[1]);
                        float sz = ParseAt(parts, cols[2]);
                        window[i] = (float)Math.Sqrt(sx * sx + sy * sy + sz * sz);
                    }
                    else
                    {
                        window[i] = ParseAt(parts, cols[0]);
                    }
                }
                return window;
            }
            catch { return null; }
        }

        private static int FindCol(string[] headers, string name)
        {
            for (int i = 0; i < headers.Length; i++)
                if (string.Equals(headers[i].Trim(), name, StringComparison.OrdinalIgnoreCase))
                    return i;
            return -1;
        }

        private static float ParseAt(string[] cols, int idx)
        {
            float v = 0f;
            if (idx >= 0 && idx < cols.Length)
                float.TryParse(cols[idx], NumberStyles.Float, CultureInfo.InvariantCulture, out v);
            return v;
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            StopLoop();
            try { _client?.Dispose(); } catch { }
            base.OnFormClosed(e);
        }
    }
}
