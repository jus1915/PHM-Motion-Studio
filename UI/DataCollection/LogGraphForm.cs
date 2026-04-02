using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics; // FFT
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using MathNet.Numerics.IntegralTransforms; // NuGet: MathNet.Numerics
using PHM_Project_DockPanel.Services;
using WeifenLuo.WinFormsUI.Docking;

namespace PHM_Project_DockPanel.Windows
{
    public class LogGraphForm : DockContent
    {
        public enum LogKind { Unknown, Torque, Accel }
        private Panel _toolbarSpacer;

        private ComboBox _cmbKind;
        private ComboBox _cmbAccelFile;                 // ★ 모듈별 Accel CSV 선택 콤보
        private Label _lblAccelFile;
        private bool _suppressAccelComboEvent;      // ★ 콤보 바인딩 중 이벤트 억제
        private string _lastTorquePath, _lastAccelPath; // ★ 최근 경로 저장
        private TableLayoutPanel _root;   // 폼 루트 컨테이너
        private Panel _headerSpacer;      // 헤더와 차트 사이 간격

        private TableLayoutPanel _table;
        private FlowLayoutPanel _buttonPanel;
        private Button _btnFeedback;
        private Button _btnResidual;

        // 차트 컨테이너: [row, col=0(Time)/1(Freq)]
        private Chart[,] _charts;

        // 공통
        private LogKind _kind = LogKind.Unknown;
        private List<double> _time;  // torque: seconds(환산/보정), accel: seconds

        // Torque 데이터
        private List<double> _pos, _vel, _trq;
        private List<double> _cmdPos, _cmdVel, _cmdTrq;

        // Accel 데이터
        private List<double> _ax, _ay, _az;

        private static readonly Font AxisFont = new Font("Segoe UI", 9f, FontStyle.Regular);
        private static readonly Font TitleFont = new Font("Segoe UI Semibold", 10f, FontStyle.Bold);

        public LogGraphForm()
        {
            Text = "Log Graph Viewer";

            BuildRootLayout();     // ★ 먼저
            BuildHeaderButtons();
            BuildKindSelector();
            BuildEmptyLayout();    // ★ 마지막: 차트 컨테이너
        }

        // ======================= UI 헤더 =======================
        private void BuildRootLayout()
        {
            _root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3
            };
            _root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            _root.RowStyles.Add(new RowStyle(SizeType.AutoSize));          // Row 0: 헤더
            _root.RowStyles.Add(new RowStyle(SizeType.Absolute, 8f));      // Row 1: 스페이서(간격)
            _root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));      // Row 2: 차트
            Controls.Add(_root);
        }

        private void BuildHeaderButtons()
        {
            _buttonPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                Height = 42,
                Padding = new Padding(6, 6, 6, 6),
                AutoSize = false,
                WrapContents = false,
                FlowDirection = FlowDirection.LeftToRight
            };

            _btnFeedback = new Button { Text = "Feedback View", Enabled = false };
            _btnResidual = new Button { Text = "Residual View", Enabled = false };
            _btnFeedback.Click += (s, e) => { if (_kind == LogKind.Torque) DrawFeedbackView(); };
            _btnResidual.Click += (s, e) => { if (_kind == LogKind.Torque) DrawResidualView(); };

            _buttonPanel.Controls.Add(_btnFeedback);
            _buttonPanel.Controls.Add(_btnResidual);
            Controls.Add(_buttonPanel);

            _root.Controls.Add(_buttonPanel, 0, 0);

            // Row 1: 스페이서(간격 + 하단 1px 라인)
            _headerSpacer = new Panel { Dock = DockStyle.Fill, Margin = Padding.Empty };
            _headerSpacer.Paint += (s, e) =>
            {
                e.Graphics.DrawLine(SystemPens.ControlDark, 0, _headerSpacer.Height - 1, _headerSpacer.Width, _headerSpacer.Height - 1);
            };
            _root.Controls.Add(_headerSpacer, 0, 1);
        }

        private void BuildKindSelector()
        {
            // Source(Kind)
            _buttonPanel.Controls.Add(new Label { Text = "Source:", AutoSize = true, Margin = new Padding(12, 10, 3, 0) });

            _cmbKind = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                Width = 80
            };
            _cmbKind.Items.AddRange(new object[] { "Torque", "Accel" });
            _cmbKind.SelectedIndex = AppState.LogGraphPreferredKind == LogKind.Accel ? 1 : 0;
            _cmbKind.SelectedIndexChanged += (s, e) =>
            {
                AppState.LogGraphPreferredKind = _cmbKind.SelectedIndex == 1 ? LogKind.Accel : LogKind.Torque;

                UpdateAccelFileComboVisibility();

                bool loaded = false;

                if (AppState.LogGraphPreferredKind == LogKind.Torque)
                {
                    if (!string.IsNullOrEmpty(_lastTorquePath))
                    {
                        LoadCsv(_lastTorquePath, LogKind.Torque);
                        loaded = true;
                    }
                }
                else // Accel
                {
                    if (AppState.LastAccelCsvs != null && AppState.LastAccelCsvs.Count > 0)
                    {
                        BindAccelFileCombo(AppState.LastAccelCsvs, _lastAccelPath);
                        var sel = _cmbAccelFile.SelectedItem as AccelItem;
                        if (sel != null && File.Exists(sel.Path))
                        {
                            LoadCsv(sel.Path, LogKind.Accel);
                            loaded = true;
                        }
                    }
                    else if (!string.IsNullOrEmpty(_lastAccelPath))
                    {
                        LoadCsv(_lastAccelPath, LogKind.Accel);
                        loaded = true;
                    }
                }

                // ★ 파일 로드가 없거나 실패했어도 타이틀은 즉시 갱신
                if (!loaded)
                {
                    string p = null;
                    if (AppState.LogGraphPreferredKind == LogKind.Accel)
                    {
                        var sel = _cmbAccelFile.SelectedItem as AccelItem;
                        if (sel != null && File.Exists(sel.Path)) p = sel.Path;
                    }
                    else
                    {
                        if (!string.IsNullOrEmpty(_lastTorquePath)) p = _lastTorquePath;
                    }
                    UpdateFormTitle(AppState.LogGraphPreferredKind, p);
                }
            };
            _buttonPanel.Controls.Add(_cmbKind);

            // Accel 파일 선택 콤보 라벨
            _lblAccelFile = new Label
            {
                Text = "Accel File:",
                AutoSize = true,
                Margin = new Padding(12, 10, 3, 0),
                Visible = (_cmbKind.SelectedIndex == 1) // ★ 초기 상태 동기화
            };
            _buttonPanel.Controls.Add(_lblAccelFile);

            _cmbAccelFile = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                Width = 100,
                Visible = (_cmbKind.SelectedIndex == 1)
            };
            _cmbAccelFile.SelectedIndexChanged += (s, e) =>
            {
                if (_suppressAccelComboEvent) return; // ★ 바인딩 중 재진입 방지
                if (_kind == LogKind.Accel && _cmbAccelFile.SelectedItem is AccelItem item && File.Exists(item.Path))
                {
                    // 동일 파일이면 재로딩 불필요
                    if (string.Equals(item.Path, _lastAccelPath, StringComparison.OrdinalIgnoreCase)) return;
                    LoadCsv(item.Path, LogKind.Accel);
                    AppState.LastAccelSelectedPath = item.Path;
                }
            };
            _buttonPanel.Controls.Add(_cmbAccelFile);
        }

        private void BuildEmptyLayout()
        {
            _table = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                RowCount = 1,
                ColumnCount = 1
            };
            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            _table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            _root.Controls.Add(_table, 0, 2);
        }

        private void BuildTorqueLayout()
        {
            // 3행 2열: Pos/Vel/Trq × (Time/Freq)
            _table.Controls.Clear();
            _table.RowCount = 3;
            _table.ColumnCount = 2;
            _table.RowStyles.Clear();
            _table.ColumnStyles.Clear();

            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 33));
            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 33));
            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 34));
            _table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            _table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

            _charts = new Chart[3, 2];

            string[] titles = { "Pos", "Vel", "Trq" };
            Color[] colors = { Color.Blue, Color.Orange, Color.Green };

            for (int row = 0; row < 3; row++)
            {
                _charts[row, 0] = CreateChart($"{titles[row]} (Time)", colors[row]);
                _charts[row, 1] = CreateChart($"{titles[row]} (Freq)", colors[row]);

                // ★ 레이아웃을 강제로 자동으로 돌려놓기 (Time/Freq 모두)
                NormalizeAreaLayout(_charts[row, 0]);
                NormalizeAreaLayout(_charts[row, 1]);

                // ★ Freq 차트: 제목은 차트영역 '밖'에, 해당 영역에 도킹(공간 예약됨)
                var freqAreaName = _charts[row, 1].ChartAreas[0].Name; // 보통 "Main"
                var freqTitle = _charts[row, 1].Titles[0];
                freqTitle.IsDockedInsideChartArea = false;
                freqTitle.DockedToChartArea = freqAreaName; // 해당 영역 위에 '밖으로' 도킹
                freqTitle.Docking = Docking.Top;
                freqTitle.DockingOffset = 6;

                // Freq 축 보정(음수 방지 + 라벨 한 줄)
                var freqArea = _charts[row, 1].ChartAreas[0];
                freqArea.AxisX.Title = "Freq (Hz)";
                freqArea.AxisX.Minimum = 0;
                freqArea.AxisX.IsLabelAutoFit = false;
                freqArea.AxisX.LabelStyle.IsStaggered = false;
                freqArea.AxisX.LabelStyle.Angle = 0;

                _table.Controls.Add(_charts[row, 0], 0, row);
                _table.Controls.Add(_charts[row, 1], 1, row);
            }

            _btnFeedback.Enabled = true;
            _btnResidual.Enabled = true;
        }

        private void BuildAccelLayout()
        {
            // 3행 2열: X/Y/Z × (Time/Freq)
            _table.Controls.Clear();
            _table.RowCount = 3;
            _table.ColumnCount = 2;
            _table.RowStyles.Clear();
            _table.ColumnStyles.Clear();

            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 33));
            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 33));
            _table.RowStyles.Add(new RowStyle(SizeType.Percent, 34));
            _table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            _table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

            _charts = new Chart[3, 2];

            string[] titles = { "Accel X", "Accel Y", "Accel Z" };
            Color[] colors = { Color.Blue, Color.Orange, Color.Green };

            for (int row = 0; row < 3; row++)
            {
                _charts[row, 0] = CreateChart($"{titles[row]} (Time)", colors[row]);
                _charts[row, 1] = CreateChart($"{titles[row]} (Freq)", colors[row]);

                // ★ 레이아웃을 강제로 자동으로 돌려놓기 (Time/Freq 모두)
                NormalizeAreaLayout(_charts[row, 0]);
                NormalizeAreaLayout(_charts[row, 1]);

                // ★ Freq 차트: 제목은 차트영역 '밖'에, 해당 영역에 도킹(공간 예약됨)
                var freqAreaName = _charts[row, 1].ChartAreas[0].Name; // 보통 "Main"
                var freqTitle = _charts[row, 1].Titles[0];
                freqTitle.IsDockedInsideChartArea = false;
                freqTitle.DockedToChartArea = freqAreaName; // 해당 영역 위에 '밖으로' 도킹
                freqTitle.Docking = Docking.Top;
                freqTitle.DockingOffset = 6;

                // Freq 축 보정(음수 방지 + 라벨 한 줄)
                var freqArea = _charts[row, 1].ChartAreas[0];
                freqArea.AxisX.Title = "Freq (Hz)";
                freqArea.AxisX.Minimum = 0;
                freqArea.AxisX.IsLabelAutoFit = false;
                freqArea.AxisX.LabelStyle.IsStaggered = false;
                freqArea.AxisX.LabelStyle.Angle = 0;

                _table.Controls.Add(_charts[row, 0], 0, row);
                _table.Controls.Add(_charts[row, 1], 1, row);
            }

            // 가속도는 Residual 개념이 없으니 비활성화
            _btnFeedback.Enabled = false;
            _btnResidual.Enabled = false;
        }

        private static void NormalizeAreaLayout(Chart c)
        {
            var a = c.ChartAreas[0];
            a.Position.Auto = true;          // ★ 자동 레이아웃 복귀
            a.InnerPlotPosition.Auto = true; // ★ 내부 플롯도 자동
        }

        protected override void OnTextChanged(EventArgs e)
        {
            base.OnTextChanged(e);
            // 항상 탭 텍스트를 폼 Text와 동일하게
            if (DockHandler != null && DockHandler.TabText != this.Text)
                DockHandler.TabText = this.Text;
        }

        private void UpdateFormTitle(LogKind kind, string filePath = null)
        {
            string suffix =
                !string.IsNullOrEmpty(filePath) ? Path.GetFileName(filePath)
              : (kind == LogKind.Accel ? "(Accel)"
              : kind == LogKind.Torque ? "(Torque)" : "");

            this.Text = string.IsNullOrEmpty(suffix) ? "Log Graph Viewer" : $"Log Graph - {suffix}";
        }
        // ======================= CSV 로드 =======================
        public void LoadCsv(string filePath)
        {
            if (!TryDetectAndParse(filePath))
            {
                MessageBox.Show("CSV 파싱 실패 또는 지원하지 않는 포맷입니다.", "오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            UpdateFormTitle(_kind, filePath);

            if (_kind == LogKind.Torque)
            {
                _lastTorquePath = filePath;
                BuildTorqueLayout();
                DrawFeedbackView(); // 기본 뷰
                UpdateAccelFileComboVisibility();
            }
            else if (_kind == LogKind.Accel)
            {
                _lastAccelPath = filePath;
                BuildAccelLayout();
                DrawAccelView();

                if (AppState.LastAccelCsvs != null && AppState.LastAccelCsvs.Count > 0)
                    BindAccelFileCombo(AppState.LastAccelCsvs, _lastAccelPath);
                UpdateAccelFileComboVisibility();
            }
        }

        public void LoadCsv(string filePath, LogKind? forceKind)
        {
            if (!TryDetectAndParse(filePath, forceKind))
            {
                MessageBox.Show("CSV 파싱 실패 또는 지원하지 않는 포맷입니다.", "오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            UpdateFormTitle(_kind, filePath);

            if (forceKind.HasValue)
            {
                _cmbKind.SelectedIndex = (forceKind.Value == LogKind.Accel) ? 1 : 0;
                AppState.LogGraphPreferredKind = forceKind.Value;
            }

            if (_kind == LogKind.Torque)
            {
                _lastTorquePath = filePath;   // 경로 저장
                BuildTorqueLayout();
                DrawFeedbackView();
                UpdateAccelFileComboVisibility();
            }
            else if (_kind == LogKind.Accel)
            {
                _lastAccelPath = filePath;    // 경로 저장
                BuildAccelLayout();
                DrawAccelView();

                if (AppState.LastAccelCsvs != null && AppState.LastAccelCsvs.Count > 0)
                    BindAccelFileCombo(AppState.LastAccelCsvs, _lastAccelPath);
                UpdateAccelFileComboVisibility();
            }
        }

        private bool TryDetectAndParse(string filePath, LogKind? forceKind = null)
        {
            string[] lines;
            try { lines = File.ReadAllLines(filePath); } catch { return false; }
            if (lines.Length < 2) return false;

            var headers = lines[0].Split(',').Select(h => h.Trim()).ToArray();
            var lower = headers.Select(h => h.ToLowerInvariant()).ToArray();

            bool looksTorque =
                lower.Any(h => h.Contains("cycle") || h == "time" || h == "time_ms") &&
                (lower.Any(h => h.Contains("cmdpos")) || lower.Any(h => h.Contains("commandpos"))) &&
                (lower.Any(h => h.Contains("fbpos")) || lower.Any(h => h.Contains("feedbackpos")));

            bool looksAccel =
                lower.Contains("time_s") && lower.Contains("x") && lower.Contains("y") && lower.Contains("z");

            // 강제 타입 우선
            if (forceKind == LogKind.Torque && TryParseTorqueCsv(lines, headers)) { _kind = LogKind.Torque; return true; }
            if (forceKind == LogKind.Accel && TryParseAccelCsv(lines, headers)) { _kind = LogKind.Accel; return true; }

            // 자동 탐지
            if (looksTorque && TryParseTorqueCsv(lines, headers)) { _kind = LogKind.Torque; return true; }
            if (looksAccel && TryParseAccelCsv(lines, headers)) { _kind = LogKind.Accel; return true; }

            _kind = LogKind.Unknown;
            return false;
        }

        private bool TryDetectAndParse(string filePath)
        {
            string[] lines;
            try { lines = File.ReadAllLines(filePath); } catch { return false; }
            if (lines.Length < 2) return false;

            var headers = lines[0].Split(',').Select(h => h.Trim()).ToArray();
            var lower = headers.Select(h => h.ToLowerInvariant()).ToArray();

            bool looksTorque =
                lower.Any(h => h.Contains("cycle") || h == "time" || h == "time_ms") &&
                (lower.Any(h => h.Contains("cmdpos")) || lower.Any(h => h.Contains("commandpos"))) &&
                (lower.Any(h => h.Contains("fbpos")) || lower.Any(h => h.Contains("feedbackpos")));

            bool looksAccel = lower.Contains("time_s") && lower.Contains("x") && lower.Contains("y") && lower.Contains("z");

            if (looksTorque && TryParseTorqueCsv(lines, headers)) { _kind = LogKind.Torque; return true; }
            if (looksAccel && TryParseAccelCsv(lines, headers)) { _kind = LogKind.Accel; return true; }

            _kind = LogKind.Unknown;
            return false;
        }

        // ======================= Torque =======================
        private bool TryParseTorqueCsv(string[] lines, string[] headers)
        {
            _time = new List<double>();
            _pos = new List<double>();
            _vel = new List<double>();
            _trq = new List<double>();
            _cmdPos = new List<double>();
            _cmdVel = new List<double>();
            _cmdTrq = new List<double>();

            int FindIndex(params string[] candidates)
            {
                var lower = headers.Select(h => h.ToLowerInvariant()).ToArray();
                for (int i = 0; i < lower.Length; i++)
                {
                    foreach (var c in candidates)
                    {
                        var lc = c.ToLowerInvariant();
                        if (lower[i] == lc || lower[i].Contains(lc))
                            return i;
                    }
                }
                return -1;
            }

            int timeIdx = FindIndex("cycle", "time", "time_ms");
            int posIdx = FindIndex("feedbackpos0", "fbpos0", "feedbackpos", "fbpos");
            int velIdx = FindIndex("feedbackvelocity0", "fbvel0", "feedbackvelocity", "fbvel");
            int trqIdx = FindIndex("feedbacktrq0", "fbtrq0", "feedbacktrq", "fbtrq");

            int cmdPosIdx = FindIndex("commandpos0", "cmdpos0", "commandpos", "cmdpos");
            int cmdVelIdx = FindIndex("commandvelocity0", "cmdvel0", "commandvelocity", "cmdvel");
            int cmdTrqIdx = FindIndex("commandtrq0", "cmdtrq0", "commandtrq", "cmdtrq");

            if (timeIdx < 0 || posIdx < 0 || velIdx < 0 || trqIdx < 0 ||
                cmdPosIdx < 0 || cmdVelIdx < 0 || cmdTrqIdx < 0)
                return false;

            bool timeIsCycle = headers[timeIdx].ToLowerInvariant().Contains("cycle");
            bool timeIsMs = headers[timeIdx].ToLowerInvariant().Contains("ms");

            // 틱 → 초 환산계수 (AppState.TorqueCycleSeconds, 기본 1ms)
            double tickToSeconds =
                (AppState.TorqueCycleSeconds > 0)
                ? AppState.TorqueCycleSeconds
                : AppState.GetPeriodForColumn("torque");   // 토크 기본 주기

            for (int i = 1; i < lines.Length; i++)
            {
                var cols = lines[i].Split(',');
                if (cols.Length <= trqIdx || cols.Length <= cmdTrqIdx) continue;

                if (double.TryParse(cols[timeIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double t) &&
                    double.TryParse(cols[posIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double p) &&
                    double.TryParse(cols[velIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double v) &&
                    double.TryParse(cols[trqIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double tq) &&
                    double.TryParse(cols[cmdPosIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double cp) &&
                    double.TryParse(cols[cmdVelIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double cv) &&
                    double.TryParse(cols[cmdTrqIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double ct))
                {
                    if (timeIsCycle)
                        _time.Add(t * tickToSeconds);   // cycle → seconds
                    else if (timeIsMs)
                        _time.Add(t / 1000.0);          // ms → s
                    else
                        _time.Add(t);                    // already seconds

                    _pos.Add(p); _vel.Add(v); _trq.Add(tq);
                    _cmdPos.Add(cp); _cmdVel.Add(cv); _cmdTrq.Add(ct);
                }
            }

            // 시간축 보정: 첫 샘플 기준 + SamplePeriod부터 시작
            if (_time.Count > 0)
            {
                double t0 = _time[0];
                double sp = AppState.GetPeriodForColumn("torque");
                for (int i = 0; i < _time.Count; i++)
                    _time[i] = (_time[i] - t0) + sp;
            }

            return _time.Count > 1;
        }

        private void DrawFeedbackView()
        {
            if (_kind != LogKind.Torque) return;
            DrawTimeAndFreq(_time, _pos, 0);
            DrawTimeAndFreq(_time, _vel, 1);
            DrawTimeAndFreq(_time, _trq, 2);
        }

        private void DrawResidualView()
        {
            if (_kind != LogKind.Torque) return;
            var posRes = ZipResidual(_cmdPos, _pos);
            var velRes = ZipResidual(_cmdVel, _vel);
            var trqRes = ZipResidual(_cmdTrq, _trq);
            DrawTimeAndFreq(_time, posRes, 0);
            DrawTimeAndFreq(_time, velRes, 1);
            DrawTimeAndFreq(_time, trqRes, 2);
        }

        private static List<double> ZipResidual(List<double> cmd, List<double> fb)
            => Enumerable.Zip(cmd, fb, (c, f) => c - f).ToList();

        // ======================= Accel =======================
        private bool TryParseAccelCsv(string[] lines, string[] headers)
        {
            _time = new List<double>();
            _ax = new List<double>();
            _ay = new List<double>();
            _az = new List<double>();

            int timeIdx = Array.FindIndex(headers, h => h.Equals("time_s", StringComparison.OrdinalIgnoreCase));
            int xIdx = Array.FindIndex(headers, h => h.Equals("x", StringComparison.OrdinalIgnoreCase));
            int yIdx = Array.FindIndex(headers, h => h.Equals("y", StringComparison.OrdinalIgnoreCase));
            int zIdx = Array.FindIndex(headers, h => h.Equals("z", StringComparison.OrdinalIgnoreCase));

            if (timeIdx < 0 || xIdx < 0 || yIdx < 0 || zIdx < 0) return false;

            for (int i = 1; i < lines.Length; i++)
            {
                var cols = lines[i].Split(',');
                if (cols.Length <= zIdx) continue;

                if (double.TryParse(cols[timeIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double t) &&
                    double.TryParse(cols[xIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double x) &&
                    double.TryParse(cols[yIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double y) &&
                    double.TryParse(cols[zIdx], NumberStyles.Any, CultureInfo.InvariantCulture, out double z))
                {
                    _time.Add(t);  // seconds
                    _ax.Add(x);
                    _ay.Add(y);
                    _az.Add(z);
                }
            }

            // 시간축 보정: 첫 샘플 기준 + SamplePeriod부터 시작
            if (_time.Count > 0)
            {
                double t0 = _time[0];
                double sp = AppState.GetPeriodForColumn("x"); // 가속도 기본 주기
                for (int i = 0; i < _time.Count; i++)
                    _time[i] = (_time[i] - t0) + sp;
            }

            return _time.Count > 1;
        }

        private void DrawAccelView()
        {
            if (_kind != LogKind.Accel) return;
            DrawTimeAndFreq(_time, _ax, 0);
            DrawTimeAndFreq(_time, _ay, 1);
            DrawTimeAndFreq(_time, _az, 2);
        }

        // ======================= 공통 렌더링 =======================
        private Chart CreateChart(string title, Color color)
        {
            var chart = new Chart { Dock = DockStyle.Fill };

            // 컨트롤 외곽/내부 여백: 상단 잘림 방지
            chart.Margin = new Padding(6, 10, 6, 6);
            chart.Padding = new Padding(6, 6, 6, 4);

            chart.AntiAliasing = AntiAliasingStyles.All;
            chart.TextAntiAliasingQuality = TextAntiAliasingQuality.High;

            var area = new ChartArea("Main");

            // ★ 축 라벨: 자동 맞춤/줄바꿈/번갈아 배치 금지
            area.AxisX.IsLabelAutoFit = false;
            area.AxisY.IsLabelAutoFit = false;
            area.AxisX.LabelStyle.IsStaggered = false;
            area.AxisY.LabelStyle.IsStaggered = false;
            area.AxisX.LabelStyle.Angle = 0;
            area.AxisX.IntervalAutoMode = IntervalAutoMode.VariableCount; // 간격은 자동으로 적당히

            // 보기 좋은 포맷(시간 기본은 짧게)
            area.AxisX.LabelStyle.Format = "0.###";
            area.AxisY.LabelStyle.Format = "0.###";

            chart.ChartAreas.Add(area);

            // 제목: 차트 영역 밖 + 간격
            var t = chart.Titles.Add(title);
            t.Docking = Docking.Top;
            t.IsDockedInsideChartArea = false;   // 제목은 PlotArea 바깥
            t.Font = TitleFont;
            t.Alignment = ContentAlignment.MiddleLeft;
            t.DockingOffset = 6;                 // 제목과 플롯 사이 간격

            chart.Series.Add(new Series
            {
                ChartType = SeriesChartType.Line,
                BorderWidth = 2,
                Color = color
            });

            return chart;
        }

        private void ApplyYPadding(Chart chart, IList<double> y)
        {
            if (y == null || y.Count == 0) return;
            double min = y.Min(), max = y.Max();
            if (double.IsNaN(min) || double.IsNaN(max)) return;

            if (min == max)
            {
                double pad = Math.Max(1e-6, Math.Abs(max) * 0.1 + 1);
                chart.ChartAreas[0].AxisY.Minimum = min - pad;
                chart.ChartAreas[0].AxisY.Maximum = max + pad;
            }
            else
            {
                double span = max - min;
                double pad = span * 0.05; // 위/아래 5% 여백
                chart.ChartAreas[0].AxisY.Minimum = min - pad;
                chart.ChartAreas[0].AxisY.Maximum = max + pad;
            }
        }

        private void UpdateChartData(Chart chart, IList<double> xData, IList<double> yData)
        {
            var s = chart.Series[0];
            s.Points.Clear();
            int n = Math.Min(xData.Count, yData.Count);
            for (int i = 0; i < n; i++) s.Points.AddXY(xData[i], yData[i]);

            chart.ChartAreas[0].RecalculateAxesScale();
        }

        private void DrawTimeAndFreq(List<double> xTime, List<double> signal, int rowIndex)
        {
            if (_charts == null) return;

            // Time
            UpdateChartData(_charts[rowIndex, 0], xTime, signal);
            ApplyYPadding(_charts[rowIndex, 0], signal);

            // Freq
            var fft = CalculateFFT(xTime, signal, out var freqAxis);
            UpdateChartData(_charts[rowIndex, 1], freqAxis, fft);
            ApplyYPadding(_charts[rowIndex, 1], fft);
        }

        private List<double> CalculateFFT(List<double> timeData, List<double> signal, out List<double> freqAxis)
        {
            int n = Math.Min(timeData.Count, signal.Count);
            if (n < 2) { freqAxis = new List<double>(); return new List<double>(); }

            // 샘플 간격(평균) 추정: 마지막-첫번째 / (n-1)
            double sampleInterval = (timeData[n - 1] - timeData[0]) / Math.Max(1, (n - 1));
            if (sampleInterval <= 0) sampleInterval = 1.0; // 가드
            double sampleRate = 1.0 / sampleInterval;

            // 원본 길이 사용
            Complex[] samples = signal.Take(n).Select(v => new Complex(v, 0)).ToArray();
            Fourier.Forward(samples, FourierOptions.Matlab);

            int half = n / 2;
            var magnitudes = new List<double>(half);
            for (int i = 0; i < half; i++)
            {
                // 진폭 정규화(단면 스펙트럼)
                magnitudes.Add(samples[i].Magnitude / n * 2.0);
            }
            freqAxis = Enumerable.Range(0, half).Select(i => i * sampleRate / n).ToList();
            return magnitudes;
        }

        // ======================= Accel Combo Helpers =======================
        private void UpdateAccelFileComboVisibility()
        {
            bool show = (AppState.LogGraphPreferredKind == LogKind.Accel);
            if (_cmbAccelFile != null) _cmbAccelFile.Visible = show;
            if (_lblAccelFile != null) _lblAccelFile.Visible = show;
        }

        private sealed class AccelItem
        {
            public string Display { get; set; }
            public string Path { get; set; }
            public override string ToString() => Display;
        }

        private void BindAccelFileCombo(IEnumerable<string> paths, string preferPath = null)
        {
            if (_cmbAccelFile == null) return;

            var items = (paths ?? Enumerable.Empty<string>())
                .Where(p => !string.IsNullOrWhiteSpace(p) && File.Exists(p))
                .Select(p =>
                {
                    var mod = new DirectoryInfo(Path.GetDirectoryName(p)).Name; // ex) cDAQ3Mod1
                    var name = Path.GetFileName(p);
                    return new AccelItem { Display = $"{mod}", Path = p };
                })
                .ToList();

            _suppressAccelComboEvent = true; // ★ 이벤트 억제 시작
            try
            {
                _cmbAccelFile.BeginUpdate();
                _cmbAccelFile.Items.Clear();
                foreach (var it in items) _cmbAccelFile.Items.Add(it);
                _cmbAccelFile.EndUpdate();

                if (items.Count == 0)
                {
                    _cmbAccelFile.SelectedIndex = -1;
                    return;
                }

                int sel = 0;
                string prefer = preferPath ?? AppState.LastAccelSelectedPath;
                if (!string.IsNullOrEmpty(prefer))
                {
                    sel = items.FindIndex(i => string.Equals(i.Path, prefer, StringComparison.OrdinalIgnoreCase));
                    if (sel < 0) sel = 0;
                }
                _cmbAccelFile.SelectedIndex = sel; // 프로그램적 선택 (이벤트 억제됨)
            }
            finally
            {
                _suppressAccelComboEvent = false; // ★ 이벤트 억제 해제
            }
        }
    }
}