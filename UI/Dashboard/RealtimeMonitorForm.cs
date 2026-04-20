using System;
using System.Collections.Concurrent;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services.Core;

namespace PHM_Project_DockPanel.UI.Dashboard
{
    /// <summary>
    /// 서버 실시간 이상탐지 모니터.
    /// ContinuousInferenceService → AppEvents.InferenceResultReceived 를 구독하여
    /// 센서별 상태 카드, Anomaly Score 추이 차트, 이상 이벤트 로그를 표시합니다.
    /// </summary>
    public class RealtimeMonitorForm : DockContent
    {
        // ────────────────────────────────────────────────────────────────────
        // 내부 타입
        // ────────────────────────────────────────────────────────────────────

        /// <summary>이벤트 로그 그리드 한 행</summary>
        private class EventItem
        {
            public string Time   { get; set; }
            public string Sensor { get; set; }
            public string Status { get; set; }
            public string Score  { get; set; }
            public string Class  { get; set; }
        }

        // ────────────────────────────────────────────────────────────────────
        // Anomaly Score 게이지 바 (커스텀 컨트롤)
        // ────────────────────────────────────────────────────────────────────
        private sealed class ScoreBar : Control
        {
            private double _score;

            public ScoreBar()
            {
                SetStyle(ControlStyles.OptimizedDoubleBuffer |
                         ControlStyles.ResizeRedraw |
                         ControlStyles.AllPaintingInWmPaint, true);
                MinimumSize = new Size(60, 18);
            }

            public void SetScore(double score) { _score = score; Invalidate(); }

            protected override void OnPaint(PaintEventArgs e)
            {
                var g = e.Graphics;
                g.SmoothingMode = SmoothingMode.None;
                int w = ClientRectangle.Width, h = ClientRectangle.Height;

                // 배경
                using (var br = new SolidBrush(Color.FromArgb(235, 235, 240)))
                    g.FillRectangle(br, 0, 0, w, h);

                // 채움 (표시 최대 = 2.0)
                double ratio  = Math.Min(Math.Max(_score, 0) / 2.0, 1.0);
                int    fillW  = (int)(w * ratio);
                if (fillW > 0)
                {
                    Color c;
                    if (_score < 0.5)       c = Color.FromArgb(34,  170,  80);
                    else if (_score < 0.85) c = Color.FromArgb(210, 150,   0);
                    else if (_score < 1.0)  c = Color.FromArgb(220,  80,  20);
                    else                    c = Color.FromArgb(200,  30,  30);

                    using (var br = new SolidBrush(c))
                        g.FillRectangle(br, 0, 0, fillW, h);
                }

                // threshold 기준선 (score=1.0 → x=w/2)
                int thrX = w / 2;
                using (var pen = new Pen(Color.FromArgb(180, Color.Red), 2))
                    g.DrawLine(pen, thrX, 0, thrX, h);

                // 텍스트
                string txt = $"{_score:F3}  /  thr 1.00";
                using (var br = new SolidBrush(Color.FromArgb(40, 40, 40)))
                using (var sf = new StringFormat { Alignment = StringAlignment.Center, LineAlignment = StringAlignment.Center })
                using (var f  = new Font("Segoe UI", 7.5f))
                    g.DrawString(txt, f, br, new RectangleF(0, 0, w, h), sf);

                // 테두리
                using (var pen = new Pen(Color.FromArgb(190, 190, 200)))
                    g.DrawRectangle(pen, 0, 0, w - 1, h - 1);
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // 센서 상태 카드 (커스텀 Panel)
        // ────────────────────────────────────────────────────────────────────
        private sealed class SensorCard : Panel
        {
            private readonly string _sensor;
            private Label    _lblName, _lblStatus, _lblInfo;
            private ScoreBar _bar;
            private Color    _borderColor = Color.FromArgb(200, 200, 210);

            public SensorCard(string sensor)
            {
                _sensor   = sensor;
                BackColor = Color.White;
                Padding   = new Padding(14, 10, 14, 10);
                SetStyle(ControlStyles.OptimizedDoubleBuffer |
                         ControlStyles.ResizeRedraw |
                         ControlStyles.AllPaintingInWmPaint, true);
                Build();
            }

            private void Build()
            {
                var tl = new TableLayoutPanel
                {
                    Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4,
                    Margin = Padding.Empty, Padding = Padding.Empty
                };
                tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 20));  // 센서명
                tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 46));  // 상태
                tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));  // 게이지
                tl.RowStyles.Add(new RowStyle(SizeType.Percent, 100));  // 정보

                _lblName = new Label
                {
                    Text      = _sensor == "accel" ? "가속도 센서" : "토크 센서",
                    Dock      = DockStyle.Fill,
                    Font      = new Font("Segoe UI", 8.5f),
                    ForeColor = Color.Gray,
                    TextAlign = ContentAlignment.BottomLeft
                };
                _lblStatus = new Label
                {
                    Text      = "대기 중",
                    Dock      = DockStyle.Fill,
                    Font      = new Font("Segoe UI", 20, FontStyle.Bold),
                    ForeColor = Color.Gray,
                    TextAlign = ContentAlignment.MiddleLeft
                };
                _bar = new ScoreBar { Dock = DockStyle.Fill };
                _lblInfo = new Label
                {
                    Text      = "",
                    Dock      = DockStyle.Fill,
                    Font      = new Font("Segoe UI", 8.5f),
                    ForeColor = Color.DimGray,
                    TextAlign = ContentAlignment.TopLeft,
                    Padding   = new Padding(0, 4, 0, 0)
                };

                tl.Controls.Add(_lblName,   0, 0);
                tl.Controls.Add(_lblStatus, 0, 1);
                tl.Controls.Add(_bar,       0, 2);
                tl.Controls.Add(_lblInfo,   0, 3);
                Controls.Add(tl);
            }

            public void Update(InferenceResult r)
            {
                if (r == null) return;

                _lblStatus.Text      = r.IsAnomaly ? "⚠  이상 감지" : "✓  정상";
                _lblStatus.ForeColor = r.IsAnomaly
                    ? Color.FromArgb(200, 30, 30)
                    : Color.FromArgb(0, 140, 70);

                BackColor    = r.IsAnomaly
                    ? Color.FromArgb(255, 243, 243)
                    : Color.FromArgb(243, 255, 247);
                _borderColor = r.IsAnomaly
                    ? Color.FromArgb(220, 100, 100)
                    : Color.FromArgb(100, 200, 130);

                _bar.SetScore(r.AnomalyScore);

                string cls  = string.IsNullOrEmpty(r.ClassName) ? "" : $"  [{r.ClassName}]";
                string mae  = r.RawMae.HasValue ? $"  MAE {r.RawMae:F4}" : "";
                _lblInfo.Text = $"Score {r.AnomalyScore:F4}{cls}{mae}\n{r.ModelType}  {DateTime.Now:HH:mm:ss}";

                Invalidate();
            }

            protected override void OnPaint(PaintEventArgs e)
            {
                base.OnPaint(e);
                using (var pen = new Pen(_borderColor, 1.5f))
                    e.Graphics.DrawRectangle(pen, 0, 0, Width - 1, Height - 1);
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // 필드
        // ────────────────────────────────────────────────────────────────────
        private SensorCard _cardAccel, _cardTorque;
        private Chart      _chart;
        private DataGridView _grid;
        private Label      _lblStatus;
        private System.Windows.Forms.Timer _timer;

        // (sensorType, time, result) 스레드-안전 큐
        private readonly ConcurrentQueue<Tuple<string, DateTime, InferenceResult>>
            _queue = new ConcurrentQueue<Tuple<string, DateTime, InferenceResult>>();

        private readonly BindingList<EventItem> _events = new BindingList<EventItem>();
        private const int MaxEvents   = 300;
        private const int ChartPoints = 150;    // 2s × 150 ≈ 5분

        private const string SeriesAccel  = "가속도";
        private const string SeriesTorque = "토크";

        // ────────────────────────────────────────────────────────────────────
        // 생성자 / 소멸자
        // ────────────────────────────────────────────────────────────────────
        public RealtimeMonitorForm()
        {
            Text        = "실시간 이상탐지 모니터";
            MinimumSize = new Size(860, 540);
            BackColor   = Color.White;

            BuildUI();

            _timer = new System.Windows.Forms.Timer { Interval = 300 };
            _timer.Tick += (s, e) => FlushQueue();
            _timer.Start();

            AppEvents.InferenceResultReceived += OnInferenceResult;
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            AppEvents.InferenceResultReceived -= OnInferenceResult;
            _timer?.Stop();
            base.OnFormClosing(e);
        }

        // ────────────────────────────────────────────────────────────────────
        // UI 구성
        // ────────────────────────────────────────────────────────────────────
        private void BuildUI()
        {
            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4,
                Padding = new Padding(14, 10, 14, 10), BackColor = Color.White
            };
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.Absolute,  40));   // 헤더
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 142));   // 카드
            root.RowStyles.Add(new RowStyle(SizeType.Percent,   58));   // 차트
            root.RowStyles.Add(new RowStyle(SizeType.Percent,   42));   // 이벤트

            // ── 헤더 ──────────────────────────────────────────────────────────
            var header = new Panel { Dock = DockStyle.Fill, Margin = new Padding(0, 0, 0, 4) };
            var lblTitle = new Label
            {
                Text      = "실시간 이상탐지 모니터",
                Font      = new Font("Segoe UI", 14, FontStyle.Bold),
                ForeColor = Color.FromArgb(25, 25, 30),
                Dock      = DockStyle.Left,
                AutoSize  = true,
                Padding   = new Padding(0, 6, 0, 0)
            };
            _lblStatus = new Label
            {
                Text      = "● 대기 중",
                Font      = new Font("Segoe UI", 9.5f),
                ForeColor = Color.Gray,
                Dock      = DockStyle.Right,
                AutoSize  = true,
                Padding   = new Padding(0, 10, 4, 0)
            };
            header.Controls.Add(_lblStatus);
            header.Controls.Add(lblTitle);

            // ── 센서 카드 ─────────────────────────────────────────────────────
            var cardRow = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1,
                Margin = new Padding(0, 4, 0, 8)
            };
            cardRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            cardRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            cardRow.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            _cardAccel  = new SensorCard("accel")  { Dock = DockStyle.Fill, Margin = new Padding(0, 0, 8, 0) };
            _cardTorque = new SensorCard("torque") { Dock = DockStyle.Fill, Margin = new Padding(8, 0, 0, 0) };
            cardRow.Controls.Add(_cardAccel,  0, 0);
            cardRow.Controls.Add(_cardTorque, 1, 0);

            // ── 차트 ─────────────────────────────────────────────────────────
            var chartWrap = new Panel { Dock = DockStyle.Fill };
            var lblChart = new Label
            {
                Text    = "Anomaly Score 실시간 추이",
                Font    = new Font("Segoe UI", 10, FontStyle.Bold),
                Dock    = DockStyle.Top, Height = 26,
                Padding = new Padding(0, 6, 0, 0),
                ForeColor = Color.FromArgb(40, 40, 50)
            };
            _chart = BuildChart();
            _chart.Dock = DockStyle.Fill;
            chartWrap.Controls.Add(_chart);
            chartWrap.Controls.Add(lblChart);

            // ── 이벤트 로그 ──────────────────────────────────────────────────
            var evtWrap = new Panel { Dock = DockStyle.Fill };
            var lblEvt = new Label
            {
                Text    = "이상 이벤트 로그",
                Font    = new Font("Segoe UI", 10, FontStyle.Bold),
                Dock    = DockStyle.Top, Height = 26,
                Padding = new Padding(0, 4, 0, 0),
                ForeColor = Color.FromArgb(40, 40, 50)
            };
            _grid = BuildGrid();
            _grid.Dock = DockStyle.Fill;
            evtWrap.Controls.Add(_grid);
            evtWrap.Controls.Add(lblEvt);

            root.Controls.Add(header,    0, 0);
            root.Controls.Add(cardRow,   0, 1);
            root.Controls.Add(chartWrap, 0, 2);
            root.Controls.Add(evtWrap,   0, 3);
            Controls.Add(root);
        }

        private Chart BuildChart()
        {
            var chart = new Chart { BackColor = Color.White };

            var area = new ChartArea("a")
            {
                BackColor   = Color.FromArgb(248, 249, 252),
                BorderColor = Color.FromArgb(200, 200, 215),
                BorderWidth = 1
            };

            // X축 (시간)
            area.AxisX.LabelStyle.Format   = "HH:mm:ss";
            area.AxisX.IntervalAutoMode    = IntervalAutoMode.VariableCount;
            area.AxisX.MajorGrid.LineColor = Color.FromArgb(225, 225, 232);
            area.AxisX.LineColor           = Color.FromArgb(180, 180, 195);

            // Y축 (score)
            area.AxisY.Minimum    = 0;
            area.AxisY.Maximum    = 2.0;
            area.AxisY.Interval   = 0.25;
            area.AxisY.MajorGrid.LineColor = Color.FromArgb(225, 225, 232);
            area.AxisY.LabelStyle.Format   = "0.00";
            area.AxisY.LineColor           = Color.FromArgb(180, 180, 195);
            area.AxisY.Title     = "Anomaly Score";
            area.AxisY.TitleFont = new Font("Segoe UI", 8f);
            area.AxisY.TitleForeColor = Color.Gray;

            // 이상 영역 배경 (score >= 1.0)
            area.AxisY.StripLines.Add(new StripLine
            {
                IntervalOffset = 1.0,
                Interval       = 0,
                StripWidth     = 1.0,
                BackColor      = Color.FromArgb(22, 255, 80, 80)
            });

            // threshold 기준선
            area.AxisY.StripLines.Add(new StripLine
            {
                IntervalOffset    = 1.0,
                Interval          = 0,
                StripWidth        = 0.004,
                BackColor         = Color.FromArgb(180, Color.OrangeRed),
                Text              = "▶ Threshold 1.0",
                ForeColor         = Color.OrangeRed,
                Font              = new Font("Segoe UI", 7.5f),
                TextAlignment     = StringAlignment.Far
            });

            chart.ChartAreas.Add(area);

            // 범례
            chart.Legends.Clear();
            chart.Legends.Add(new Legend
            {
                Docking     = Docking.Top,
                Alignment   = StringAlignment.Near,
                BackColor   = Color.Transparent,
                Font        = new Font("Segoe UI", 9f),
                LegendStyle = LegendStyle.Row
            });

            // 스켈레톤 (X축 초기 범위 잡기용)
            var now = DateTime.Now;
            var skeleton = new Series("_sk_")
            {
                ChartType        = SeriesChartType.FastLine,
                XValueType       = ChartValueType.DateTime,
                IsVisibleInLegend = false,
                Color            = Color.Transparent
            };
            skeleton.Points.AddXY(now.AddMinutes(-5), 0);
            skeleton.Points.AddXY(now, 0);
            chart.Series.Add(skeleton);

            // 가속도 시리즈
            chart.Series.Add(new Series(SeriesAccel)
            {
                ChartType   = SeriesChartType.FastLine,
                XValueType  = ChartValueType.DateTime,
                BorderWidth = 2,
                Color       = Color.FromArgb(41, 128, 220),
                LegendText  = "가속도"
            });

            // 토크 시리즈
            chart.Series.Add(new Series(SeriesTorque)
            {
                ChartType   = SeriesChartType.FastLine,
                XValueType  = ChartValueType.DateTime,
                BorderWidth = 2,
                Color       = Color.FromArgb(220, 100, 20),
                LegendText  = "토크"
            });

            return chart;
        }

        private DataGridView BuildGrid()
        {
            var grid = new DataGridView
            {
                ReadOnly              = true,
                AllowUserToAddRows    = false,
                AllowUserToDeleteRows = false,
                AutoSizeColumnsMode   = DataGridViewAutoSizeColumnsMode.Fill,
                RowHeadersVisible     = false,
                BorderStyle           = BorderStyle.None,
                BackgroundColor       = Color.White,
                GridColor             = Color.FromArgb(220, 220, 228),
                ColumnHeadersHeight   = 26,
                RowTemplate           = { Height = 22 },
                DataSource            = _events
            };
            grid.EnableHeadersVisualStyles = false;
            grid.ColumnHeadersDefaultCellStyle.BackColor  = Color.FromArgb(244, 244, 248);
            grid.ColumnHeadersDefaultCellStyle.Font       = new Font("Segoe UI", 8.5f, FontStyle.Bold);
            grid.ColumnHeadersDefaultCellStyle.ForeColor  = Color.FromArgb(50, 50, 60);
            grid.DefaultCellStyle.Font                    = new Font("Consolas", 8.5f);
            grid.DefaultCellStyle.SelectionBackColor      = Color.FromArgb(210, 228, 255);
            grid.DefaultCellStyle.SelectionForeColor      = Color.Black;

            grid.RowPrePaint += (s, e) =>
            {
                if (e.RowIndex < 0 || e.RowIndex >= _events.Count) return;
                if (_events[e.RowIndex].Status?.Contains("이상") == true)
                    grid.Rows[e.RowIndex].DefaultCellStyle.BackColor = Color.FromArgb(255, 242, 242);
            };

            return grid;
        }

        // ────────────────────────────────────────────────────────────────────
        // 이벤트 수신 (백그라운드 스레드)
        // ────────────────────────────────────────────────────────────────────
        private void OnInferenceResult(string sensorType, InferenceResult result)
        {
            if (result == null) return;
            _queue.Enqueue(Tuple.Create(sensorType, DateTime.Now, result));
            while (_queue.Count > 600)
            {
                Tuple<string, DateTime, InferenceResult> _;
                _queue.TryDequeue(out _);
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // UI 갱신 (UI 스레드 — 300ms 타이머)
        // ────────────────────────────────────────────────────────────────────
        private bool _skeletonRemoved;

        private void FlushQueue()
        {
            if (_chart == null || _chart.IsDisposed) return;
            if (_queue.IsEmpty) return;

            // 첫 번째 실제 데이터 수신 시 스켈레톤 제거
            if (!_skeletonRemoved)
            {
                var sk = _chart.Series.FindByName("_sk_");
                if (sk != null) _chart.Series.Remove(sk);
                var area = _chart.ChartAreas["a"];
                area.AxisX.Minimum = double.NaN;
                area.AxisX.Maximum = double.NaN;
                _skeletonRemoved = true;
            }

            Tuple<string, DateTime, InferenceResult> item;
            while (_queue.TryDequeue(out item))
            {
                string         sensor = item.Item1;
                DateTime       t      = item.Item2;
                InferenceResult r     = item.Item3;

                bool isAccel = string.Equals(sensor, "accel", StringComparison.OrdinalIgnoreCase);

                // ── 카드 업데이트 ────────────────────────────────────────────
                (isAccel ? _cardAccel : _cardTorque).Update(r);

                // ── 차트 포인트 추가 ─────────────────────────────────────────
                string sName = isAccel ? SeriesAccel : SeriesTorque;
                var s = _chart.Series.FindByName(sName);
                if (s != null)
                {
                    s.Points.AddXY(t.ToOADate(), (double)r.AnomalyScore);
                    while (s.Points.Count > ChartPoints) s.Points.RemoveAt(0);
                }

                // ── 이상 이벤트 로그 ─────────────────────────────────────────
                if (r.IsAnomaly)
                {
                    string cls = string.IsNullOrEmpty(r.ClassName) ? "—" : r.ClassName;
                    _events.Insert(0, new EventItem
                    {
                        Time   = t.ToString("HH:mm:ss"),
                        Sensor = isAccel ? "가속도" : "토크",
                        Status = "⚠ 이상",
                        Score  = r.AnomalyScore.ToString("F4"),
                        Class  = cls
                    });
                    if (_events.Count > MaxEvents) _events.RemoveAt(_events.Count - 1);
                }
            }

            // 연결 상태 라벨 갱신 (마지막 수신 기준)
            _lblStatus.Text      = "● 수신 중";
            _lblStatus.ForeColor = Color.FromArgb(0, 150, 60);

            _chart.ChartAreas["a"].RecalculateAxesScale();
        }
    }
}
