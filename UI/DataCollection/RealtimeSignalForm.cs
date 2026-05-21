using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services;

namespace PHM_Project_DockPanel.UI.DataCollection
{
    /// <summary>
    /// 실시간 가속도(X/Y/Z) + 토크(Ax0/Ax1/Ax2) 파형 모니터.
    /// AppEvents.AccelBlockReceived / TorqueSampleReceived 이벤트를 구독합니다.
    /// </summary>
    public sealed class RealtimeSignalForm : DockContent
    {
        // ── 설정 ──────────────────────────────────────────────────────────────
        private int _windowSec = 3;               // 롤링 창 크기 (초)
        private const int TimerMs    = 80;        // UI 갱신 주기 (ms)
        private const int MaxPts     = 6000;      // 시리즈당 최대 포인트
        private const int AccelSkip  = 2;         // 가속도 데시메이션 (1=전체)
        private const int TorqueSkip = 4;         // 토크 데시메이션
        private bool _paused;

        // ── UI ────────────────────────────────────────────────────────────────
        private Chart _chartAccel;
        private Chart _chartTorque;
        private Label _lblRms;
        private Label _lblStatus;
        private System.Windows.Forms.Timer _timer;

        // ── 데이터 큐 (비UI 스레드 → UI 스레드) ─────────────────────────────
        private readonly ConcurrentQueue<AccelSample>  _accelQ  = new ConcurrentQueue<AccelSample>();
        private readonly ConcurrentQueue<TorquePoint>  _torqueQ = new ConcurrentQueue<TorquePoint>();

        // ── 데시메이션 카운터 ─────────────────────────────────────────────────
        private int _accelSkipCnt;
        private int _torqueSkipCnt;
        private bool _hasAccel;
        private bool _hasTorque;

        private struct AccelSample  { public double T; public float X, Y, Z; }
        private struct TorquePoint  { public double T; public float V; public int Ax; }

        // ── 생성자 ────────────────────────────────────────────────────────────
        public RealtimeSignalForm()
        {
            Text      = "실시간 신호 모니터";
            DockAreas = DockAreas.Document | DockAreas.Float;
            MinimumSize = new Size(480, 320);
            BuildUI();
            Subscribe();
        }

        // ── 이벤트 구독 ───────────────────────────────────────────────────────
        private void Subscribe()
        {
            AppEvents.AccelBlockReceived   += OnAccelBlock;
            AppEvents.TorqueSampleReceived += OnTorqueSample;
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            AppEvents.AccelBlockReceived   -= OnAccelBlock;
            AppEvents.TorqueSampleReceived -= OnTorqueSample;
            _timer?.Stop();
            base.OnFormClosed(e);
        }

        // ── 데이터 수신 (비UI 스레드) ─────────────────────────────────────────

        private void OnAccelBlock(string module, double[,] block, DateTime ts, double sampleRate)
        {
            if (_paused) return;
            int n = block.GetLength(1);
            if (n == 0) return;

            double sr = sampleRate > 0 ? sampleRate : 1000.0;
            double dtOA = 1.0 / (sr * 86400.0); // 1샘플 = ? OADate
            double endOA = ts.ToLocalTime().ToOADate();

            for (int i = 0; i < n; i++)
            {
                _accelSkipCnt++;
                if (_accelSkipCnt % AccelSkip != 0) continue;

                float x = block.GetLength(0) > 0 ? (float)block[0, i] : 0f;
                float y = block.GetLength(0) > 1 ? (float)block[1, i] : 0f;
                float z = block.GetLength(0) > 2 ? (float)block[2, i] : 0f;
                double t = endOA - (n - 1 - i) * dtOA;
                _accelQ.Enqueue(new AccelSample { T = t, X = x, Y = y, Z = z });
            }
            _hasAccel = true;
        }

        private void OnTorqueSample(int axis, double value, DateTime ts)
        {
            if (_paused || axis < 0 || axis > 2) return;
            _torqueSkipCnt++;
            if (_torqueSkipCnt % TorqueSkip != 0) return;
            _torqueQ.Enqueue(new TorquePoint
            {
                T  = ts.ToLocalTime().ToOADate(),
                V  = (float)value,
                Ax = axis
            });
            _hasTorque = true;
        }

        // ── 타이머 갱신 ───────────────────────────────────────────────────────

        private void FlushCharts()
        {
            if (!IsHandleCreated || IsDisposed) return;

            double nowOA = DateTime.Now.ToOADate();
            double cutOA = nowOA - _windowSec / 86400.0;

            FlushAccel(cutOA, nowOA);
            FlushTorque(cutOA, nowOA);

            // 상태 라벨
            if (_lblStatus != null)
            {
                bool accelOk  = _hasAccel;
                bool torqueOk = _hasTorque;
                _lblStatus.Text =
                    (accelOk  ? "● 가속도" : "○ 가속도") + "  " +
                    (torqueOk ? "● 토크"   : "○ 토크");
                _lblStatus.ForeColor = (accelOk || torqueOk)
                    ? Color.FromArgb(20, 140, 60) : Color.FromArgb(140, 80, 20);
            }
        }

        // 가속도 시리즈: X/Y/Z 각각 Points.AddXY + 창 밖 제거
        private void FlushAccel(double cutOA, double nowOA)
        {
            if (_chartAccel == null || _chartAccel.IsDisposed) return;

            var sx = _chartAccel.Series["X"];
            var sy = _chartAccel.Series["Y"];
            var sz = _chartAccel.Series["Z"];

            AccelSample s;
            while (_accelQ.TryDequeue(out s))
            {
                sx.Points.AddXY(s.T, s.X);
                sy.Points.AddXY(s.T, s.Y);
                sz.Points.AddXY(s.T, s.Z);
            }

            // 창 밖 포인트 제거 (앞에서부터)
            TrimFront(sx, cutOA);
            TrimFront(sy, cutOA);
            TrimFront(sz, cutOA);

            // 과다 포인트 제거 (안전망)
            while (sx.Points.Count > MaxPts) sx.Points.RemoveAt(0);
            while (sy.Points.Count > MaxPts) sy.Points.RemoveAt(0);
            while (sz.Points.Count > MaxPts) sz.Points.RemoveAt(0);

            int cnt = sx.Points.Count;
            if (cnt > 0)
            {
                var area = _chartAccel.ChartAreas[0];
                // 토크와 동일하게 cutOA 기준 → 배치 도착 시 X축 점프 방지
                area.AxisX.Minimum = cutOA;
                area.AxisX.Maximum = nowOA + 0.5 / 86400.0;

                // Y 자동 범위
                area.AxisY.Maximum = double.NaN;
                area.AxisY.Minimum = double.NaN;
                area.RecalculateAxesScale();

                // RMS (가시 창 내) — O(n) 인덱스 순회
                double sumSq = 0; int rCnt = 0;
                for (int ri = 0; ri < sx.Points.Count; ri++)
                {
                    if (sx.Points[ri].XValue < cutOA) continue;
                    double xv = sx.Points[ri].YValues[0];
                    double yv = ri < sy.Points.Count ? sy.Points[ri].YValues[0] : 0;
                    double zv = ri < sz.Points.Count ? sz.Points[ri].YValues[0] : 0;
                    sumSq += xv * xv + yv * yv + zv * zv;
                    rCnt++;
                }
                double rms = rCnt > 0 ? Math.Sqrt(sumSq / rCnt) : 0;
                if (_lblRms != null) _lblRms.Text = $"RMS  {rms:0.000} g";
            }
        }

        // 토크 시리즈: Ax0/1/2 각각 Points.AddXY + 창 밖 제거
        private void FlushTorque(double cutOA, double nowOA)
        {
            if (_chartTorque == null || _chartTorque.IsDisposed) return;

            var s0 = _chartTorque.Series["Ax0"];
            var s1 = _chartTorque.Series["Ax1"];
            var s2 = _chartTorque.Series["Ax2"];

            TorquePoint tp;
            while (_torqueQ.TryDequeue(out tp))
            {
                if      (tp.Ax == 0) s0.Points.AddXY(tp.T, tp.V);
                else if (tp.Ax == 1) s1.Points.AddXY(tp.T, tp.V);
                else if (tp.Ax == 2) s2.Points.AddXY(tp.T, tp.V);
            }

            TrimFront(s0, cutOA); TrimFront(s1, cutOA); TrimFront(s2, cutOA);
            while (s0.Points.Count > MaxPts) s0.Points.RemoveAt(0);
            while (s1.Points.Count > MaxPts) s1.Points.RemoveAt(0);
            while (s2.Points.Count > MaxPts) s2.Points.RemoveAt(0);

            bool hasData = s0.Points.Count > 0 || s1.Points.Count > 0 || s2.Points.Count > 0;
            if (hasData)
            {
                var area = _chartTorque.ChartAreas[0];
                area.AxisX.Minimum = cutOA;
                area.AxisX.Maximum = nowOA + 0.5 / 86400.0;
                area.AxisY.Maximum = double.NaN;
                area.AxisY.Minimum = double.NaN;
                area.RecalculateAxesScale();
            }
        }

        private static void TrimFront(Series s, double cutOA)
        {
            while (s.Points.Count > 0 && s.Points[0].XValue < cutOA)
                s.Points.RemoveAt(0);
        }

        // ── UI 구성 ───────────────────────────────────────────────────────────

        private void BuildUI()
        {
            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3,
                BackColor = Color.White, Padding = new Padding(4, 4, 4, 4)
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));   // 툴바
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 50));    // 가속도
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 50));    // 토크
            Controls.Add(root);

            root.Controls.Add(BuildToolbar(), 0, 0);

            _chartAccel = BuildSignalChart(
                new[] {
                    ("X",  Color.FromArgb(30, 120, 215)),
                    ("Y",  Color.FromArgb(25, 160, 70)),
                    ("Z",  Color.FromArgb(215, 100, 25))
                },
                isAccel: true);
            root.Controls.Add(WrapChart("가속도  [g]", _chartAccel, Color.FromArgb(0, 84, 166)), 0, 1);

            _chartTorque = BuildSignalChart(
                new[] {
                    ("Ax0", Color.FromArgb(205, 50, 20)),
                    ("Ax1", Color.FromArgb(230, 140, 20)),
                    ("Ax2", Color.FromArgb(60, 160, 80))
                },
                isAccel: false);
            root.Controls.Add(WrapChart("토크  [%]", _chartTorque, Color.FromArgb(140, 40, 10)), 0, 2);

            _timer = new System.Windows.Forms.Timer { Interval = TimerMs };
            _timer.Tick += (s, e) => FlushCharts();
            _timer.Start();
        }

        private Panel BuildToolbar()
        {
            var bar = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill, AutoSize = false,
                FlowDirection = FlowDirection.LeftToRight,
                BackColor = Color.FromArgb(245, 245, 250),
                Padding = new Padding(6, 5, 0, 0)
            };

            // 창 크기 버튼
            bar.Controls.Add(MakeLabel("창 크기:"));
            var winBtns = new List<Button>();
            foreach (int sec in new[] { 1, 3, 5, 10 })
            {
                int s = sec;
                var b = MakeSmallButton($"{sec}s",
                    sec == _windowSec ? Color.FromArgb(0, 84, 166) : Color.FromArgb(225, 225, 235),
                    sec == _windowSec ? Color.White                 : Color.FromArgb(50, 50, 65));
                winBtns.Add(b);
                b.Click += (sender, e) =>
                {
                    _windowSec = s;
                    foreach (var wb in winBtns)
                    {
                        bool active = (int)wb.Tag == s;
                        wb.BackColor = active ? Color.FromArgb(0, 84, 166) : Color.FromArgb(225, 225, 235);
                        wb.ForeColor = active ? Color.White : Color.FromArgb(50, 50, 65);
                    }
                };
                b.Tag = sec;
                bar.Controls.Add(b);
            }

            // 일시정지 버튼
            var btnPause = MakeSmallButton("⏸", Color.FromArgb(225, 225, 235), Color.FromArgb(50, 50, 65));
            btnPause.Margin = new Padding(10, 0, 4, 0);
            btnPause.Click += (sender, e) =>
            {
                _paused = !_paused;
                btnPause.Text     = _paused ? "▶" : "⏸";
                btnPause.BackColor= _paused ? Color.FromArgb(190, 30, 30) : Color.FromArgb(225, 225, 235);
                btnPause.ForeColor= _paused ? Color.White : Color.FromArgb(50, 50, 65);
            };
            bar.Controls.Add(btnPause);

            // 구분선
            bar.Controls.Add(new Label
            {
                Text = "|", AutoSize = true, ForeColor = Color.FromArgb(190, 190, 210),
                Font = new Font("Segoe UI", 10f), Margin = new Padding(6, 1, 6, 0)
            });

            // RMS 라벨
            _lblRms = new Label
            {
                Text = "RMS  ---", AutoSize = true,
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 80, 160), Margin = new Padding(0, 3, 12, 0)
            };
            bar.Controls.Add(_lblRms);

            // 상태 라벨
            _lblStatus = new Label
            {
                Text = "○ 가속도  ○ 토크", AutoSize = true,
                Font = new Font("Segoe UI", 8f), ForeColor = Color.FromArgb(140, 80, 20),
                Margin = new Padding(0, 3, 0, 0)
            };
            bar.Controls.Add(_lblStatus);

            return bar;
        }

        private static Label MakeLabel(string text)
            => new Label
            {
                Text = text, AutoSize = true,
                Font = new Font("Segoe UI", 8.5f), ForeColor = Color.FromArgb(55, 55, 70),
                Margin = new Padding(0, 4, 4, 0)
            };

        private static Button MakeSmallButton(string text, Color back, Color fore)
        {
            var b = new Button
            {
                Text = text, AutoSize = false, Width = 34, Height = 24,
                Font = new Font("Segoe UI", 8f), FlatStyle = FlatStyle.Flat,
                BackColor = back, ForeColor = fore, Margin = new Padding(0, 0, 3, 0)
            };
            b.FlatAppearance.BorderSize  = 1;
            b.FlatAppearance.BorderColor = Color.FromArgb(180, 180, 200);
            return b;
        }

        private static Chart BuildSignalChart((string name, Color color)[] series, bool isAccel)
        {
            var chart = new Chart { Dock = DockStyle.Fill, BackColor = Color.White };
            var ca = new ChartArea("a") { BackColor = Color.White };
            ca.Position          = new ElementPosition(0, 0, 100, 100);
            ca.InnerPlotPosition = new ElementPosition(9, 3, 89, 87);

            // X 축 (시간)
            ca.AxisX.LabelStyle.Format      = "HH:mm:ss";
            ca.AxisX.LabelStyle.Font        = new Font("Segoe UI", 7.5f);
            ca.AxisX.LabelStyle.ForeColor   = Color.FromArgb(70, 70, 85);
            ca.AxisX.IntervalAutoMode       = IntervalAutoMode.VariableCount;
            ca.AxisX.MajorGrid.LineColor    = Color.FromArgb(215, 215, 225);
            ca.AxisX.MajorGrid.LineDashStyle= ChartDashStyle.Dot;
            ca.AxisX.MajorTickMark.LineColor= Color.FromArgb(155, 155, 170);
            ca.AxisX.MajorTickMark.Size     = 3;
            ca.AxisX.LineColor              = Color.FromArgb(175, 175, 195);

            // Y 축
            ca.AxisY.LabelStyle.Font        = new Font("Segoe UI", 8f);
            ca.AxisY.LabelStyle.ForeColor   = Color.FromArgb(55, 55, 70);
            ca.AxisY.LabelStyle.Format      = "0.0##";
            ca.AxisY.MajorGrid.LineColor    = Color.FromArgb(215, 215, 225);
            ca.AxisY.MajorGrid.LineDashStyle= ChartDashStyle.Dot;
            ca.AxisY.MajorTickMark.LineColor= Color.FromArgb(145, 145, 165);
            ca.AxisY.MajorTickMark.Size     = 4;
            ca.AxisY.LineColor              = Color.FromArgb(175, 175, 195);
            ca.AxisY.IsStartedFromZero      = false;

            chart.ChartAreas.Add(ca);
            chart.Legends.Clear();
            chart.Legends.Add(new Legend
            {
                Docking   = Docking.Top,
                Alignment = StringAlignment.Near,
                Font      = new Font("Segoe UI", 7.5f),
                BackColor = Color.Transparent
            });

            foreach (var (name, color) in series)
            {
                chart.Series.Add(new Series(name)
                {
                    ChartType   = SeriesChartType.FastLine,
                    XValueType  = ChartValueType.DateTime,
                    Color       = color,
                    BorderWidth = 1
                });
            }
            return chart;
        }

        private static Panel WrapChart(string title, Chart chart, Color titleColor)
        {
            var panel = new Panel
            {
                Dock = DockStyle.Fill, BackColor = Color.White,
                Margin = new Padding(0, 2, 0, 2)
            };
            panel.Paint += (s, e) =>
            {
                using (var pen = new Pen(Color.FromArgb(205, 205, 218)))
                    e.Graphics.DrawRectangle(pen, 0, 0, panel.Width - 1, panel.Height - 1);
            };
            var lbl = new Label
            {
                Text      = title, Dock = DockStyle.Top, Height = 20,
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                ForeColor = titleColor, BackColor = Color.White,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding   = new Padding(8, 0, 0, 0)
            };
            panel.Controls.Add(chart);
            panel.Controls.Add(lbl);
            return panel;
        }
    }
}
