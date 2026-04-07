using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services.Core; // SignalFeatures

namespace PHM_Project_DockPanel.UI.DataAnalysis
{
    public class AIForm : DockContent
    {
        // ===== 세션 분리 =====
        public enum SessionType { AnomalyDetection, FaultDiagnosis } // 이상 탐지 / 결함 진단
        private SessionType _session = SessionType.AnomalyDetection;

        // ===== 외부에서 주입받는 정의 =====
        public struct FeatureDef { public string Key; public string Title; }

        // ===== 내부 상태 =====
        private List<object> _featureRowsRaw = new List<object>();
        private FeatureDef[] _featureList = new FeatureDef[0];

        private class Sample
        {
            public string FileName;
            public double[] X;
            public int Label;          // 내부 학습용(정수)
            public string LabelName;   // 표시/입력용(문자열)
        }

        // kNN 공통 파라미터/모델
        private int _k = 5;
        private bool _useStandardize = true;
        private double[] _mean = null;
        private double[] _std = null;
        private string[] _selectedKeys = new string[0];
        private string _yColumnName = null;

        // 라벨 인코더 (FaultDiagnosis에서 사용)
        private class LabelEncoder
        {
            private readonly Dictionary<string, int> _toId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            private readonly List<string> _toName = new List<string>();

            public void Fit(IEnumerable<string> names)
            {
                _toId.Clear();
                _toName.Clear();
                foreach (var n in names.Where(s => !string.IsNullOrWhiteSpace(s)).Distinct(StringComparer.OrdinalIgnoreCase))
                {
                    _toId[n] = _toName.Count;
                    _toName.Add(n);
                }
                if (!_toId.ContainsKey("UNK")) { _toId["UNK"] = _toName.Count; _toName.Add("UNK"); }
            }
            public int ToId(string name)
            {
                name = string.IsNullOrWhiteSpace(name) ? "UNK" : name;
                if (!_toId.TryGetValue(name, out var id)) { id = _toName.Count; _toId[name] = id; _toName.Add(name); }
                return id;
            }
            public string ToName(int id) => (id >= 0 && id < _toName.Count) ? _toName[id] : $"cls{id}";
            public string[] ClassNames() => _toName.ToArray();
        }
        private LabelEncoder _enc = new LabelEncoder();
        private string[] _classNames = new string[0];

        // 모델 인터페이스 & 구현
        private IModel _model = null;                   // 세션별로 할당
        private double[][] _trainVectors = null;        // AD에서 정상 학습 벡터 저장

        // ===== UI =====
        private TabControl tabs;

        // 공통/학습 탭
        private ComboBox cmbSession;                    // ★ 세션 전환
        private CheckedListBox clbTimeFeatures;
        private CheckedListBox clbFreqFeatures;
        private Button btnTimeSelectAll, btnTimeClearAll;
        private Button btnFreqSelectAll, btnFreqClearAll;
        private Button btnAllSelectBoth, btnAllClearBoth;
        private NumericUpDown numK;
        private CheckBox chkStd;
        private Button btnMarkAllNormal, btnMarkAllAnomaly, btnTrain, btnSaveModel;
        private DataGridView gridTrain;
        private Label lblYColumn;

        // 검증 탭
        private NumericUpDown numHoldout;
        private Button btnValidate;
        private TrackBar trThreshold;
        private NumericUpDown numThreshold;
        private Label lblMetrics;
        private Chart chartROC;
        private DataGridView gridScores;                // AD: 점수테이블 / CLS: 혼동행렬 테이블 재사용
        private Panel pnlClsMetrics;                    // ★ CLS 지표 패널(정확도/매크로F1 등) - 간단 레이블로 구성
        private Label lblClsMore;

        // 평가 탭
        private Button btnScoreAll;
        private DataGridView gridEval;                  // AD/CLS 공용 평가 테이블

        // 검증 상태
        private List<Sample> _samples = new List<Sample>();
        private List<Sample> _trainSet = new List<Sample>();
        private List<Sample> _valSet = new List<Sample>();
        private List<Tuple<double, int, string>> _valScores = new List<Tuple<double, int, string>>();
        private double _optThresh = double.NaN;
        private bool _updatingThreshold = false;

        public AIForm()
        {
            Text = "AI";
            BuildUI();
        }

        // =============== 외부 데이터 주입 ===============
        public void SetFeatureData(IEnumerable<object> rows, (string Key, string Title)[] featureList)
        {
            if (this.InvokeRequired) { this.BeginInvoke(new Action(() => SetFeatureData(rows, featureList))); return; }

            _featureRowsRaw = (rows != null) ? rows.ToList() : new List<object>();

            if (featureList != null)
            {
                _featureList = featureList.Select(f => new FeatureDef { Key = f.Key, Title = f.Title }).ToArray();
            }
            else _featureList = new FeatureDef[0];

            PopulateFeaturePickers();   // 시간/주파수 체크리스트 채움
            RebuildDatasetTable();      // 학습 그리드 갱신
        }
        public void SetYColumnName(string ycol)
        {
            _yColumnName = string.IsNullOrWhiteSpace(ycol) ? null : ycol;
            UpdateYLabelUI();
        }

        private void UpdateYLabelUI()
        {
            if (lblYColumn == null) return;
            if (this.InvokeRequired) { try { this.BeginInvoke(new Action(UpdateYLabelUI)); } catch { } return; }
            lblYColumn.Text = "Y 컬럼: " + (_yColumnName ?? "(미지정)");
        }

        // =============== UI 구성 ===============
        private void BuildUI()
        {
            tabs = new TabControl { Dock = DockStyle.Fill };
            var tabTrain = new TabPage("학습");
            var tabVal = new TabPage("검증");
            var tabEval = new TabPage("평가");

            // ---- 학습 탭 상단 (3열) ----
            var trainRoot = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2 };
            trainRoot.RowStyles.Add(new RowStyle(SizeType.Absolute, 240)); // ↑ 세션 콤보 추가로 20 늘림
            trainRoot.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            var top3 = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, Margin = new Padding(6) };
            top3.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 45));
            top3.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 45));
            top3.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 520)); // 약간 늘림

            // (1) 시간 영역
            clbTimeFeatures = new CheckedListBox { Dock = DockStyle.Fill, CheckOnClick = true, BorderStyle = BorderStyle.FixedSingle };
            btnTimeSelectAll = new Button { Text = "전체 선택", Width = 90 };
            btnTimeClearAll = new Button { Text = "전체 해제", Width = 90 };
            btnTimeSelectAll.Click += (s, e) => CheckAll(clbTimeFeatures, true);
            btnTimeClearAll.Click += (s, e) => CheckAll(clbTimeFeatures, false);
            var pnlTimeBtns = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 34, FlowDirection = FlowDirection.LeftToRight };
            pnlTimeBtns.Controls.Add(btnTimeSelectAll);
            pnlTimeBtns.Controls.Add(btnTimeClearAll);
            var grpTime = new GroupBox { Text = "시간 영역 특징", Dock = DockStyle.Fill, Padding = new Padding(6) };
            grpTime.Controls.Add(clbTimeFeatures);
            grpTime.Controls.Add(pnlTimeBtns);

            // (2) 주파수 영역
            clbFreqFeatures = new CheckedListBox { Dock = DockStyle.Fill, CheckOnClick = true, BorderStyle = BorderStyle.FixedSingle };
            btnFreqSelectAll = new Button { Text = "전체 선택", Width = 90 };
            btnFreqClearAll = new Button { Text = "전체 해제", Width = 90 };
            btnFreqSelectAll.Click += (s, e) => CheckAll(clbFreqFeatures, true);
            btnFreqClearAll.Click += (s, e) => CheckAll(clbFreqFeatures, false);
            var pnlFreqBtns = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 34, FlowDirection = FlowDirection.LeftToRight };
            pnlFreqBtns.Controls.Add(btnFreqSelectAll);
            pnlFreqBtns.Controls.Add(btnFreqClearAll);
            var grpFreq = new GroupBox { Text = "주파수 영역 특징", Dock = DockStyle.Fill, Padding = new Padding(6) };
            grpFreq.Controls.Add(clbFreqFeatures);
            grpFreq.Controls.Add(pnlFreqBtns);

            // (3) 오른쪽 스택
            var rightScroll = new Panel { Dock = DockStyle.Fill, AutoScroll = true };
            var rightStack = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                ColumnCount = 1,
                Padding = new Padding(0, 0, 6, 0)
            };
            rightStack.RowStyles.Clear();
            rightStack.RowStyles.Add(new RowStyle());                       // (0) 세션 콤보 ★
            rightStack.RowStyles.Add(new RowStyle());                       // (1) Y 라벨
            rightStack.RowStyles.Add(new RowStyle());                       // (2) k 라벨
            rightStack.RowStyles.Add(new RowStyle());                       // (3) k numeric
            rightStack.RowStyles.Add(new RowStyle());                       // (4) 표준화
            rightStack.RowStyles.Add(new RowStyle());                       // (5) 전체선택/해제(모두)
            rightStack.RowStyles.Add(new RowStyle());                       // (6) 학습/저장 버튼들
            rightStack.RowStyles.Add(new RowStyle(SizeType.Percent, 100));  // (7) filler

            // 세션 전환 콤보
            var lblSess = new Label { Text = "세션:", AutoSize = true, Margin = new Padding(0, 6, 0, 0) };
            cmbSession = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 220 };
            cmbSession.Items.Add("이상 탐지 (Anomaly Detection)");
            cmbSession.Items.Add("결함 진단 (Fault Diagnosis)");
            cmbSession.SelectedIndex = 0;
            cmbSession.SelectedIndexChanged += (s, e) =>
            {
                _session = (cmbSession.SelectedIndex == 0) ? SessionType.AnomalyDetection : SessionType.FaultDiagnosis;
                ApplySessionUI();
            };
            var sessPanel = new FlowLayoutPanel { FlowDirection = FlowDirection.LeftToRight, AutoSize = true, Dock = DockStyle.Top };
            sessPanel.Controls.Add(lblSess);
            sessPanel.Controls.Add(cmbSession);

            lblYColumn = new Label { AutoSize = true, Text = "Y 컬럼: (미지정)", Margin = new Padding(0, 6, 0, 0) };
            var lblK = new Label { Text = "k:", AutoSize = true, Margin = new Padding(0, 6, 0, 0) };
            numK = new NumericUpDown { Minimum = 1, Maximum = 50, Value = 5, Width = 80, Anchor = AnchorStyles.Left | AnchorStyles.Top };
            chkStd = new CheckBox { Text = "표준화", Checked = true, AutoSize = true, Margin = new Padding(0, 6, 0, 0) };

            var pnlBoth = new FlowLayoutPanel { FlowDirection = FlowDirection.LeftToRight, WrapContents = true, AutoSize = true, Dock = DockStyle.Top };
            btnAllSelectBoth = new Button { Text = "전체 선택(모두)", Width = 130 };
            btnAllClearBoth = new Button { Text = "전체 해제(모두)", Width = 130 };
            btnAllSelectBoth.Click += (s, e) => { CheckAll(clbTimeFeatures, true); CheckAll(clbFreqFeatures, true); };
            btnAllClearBoth.Click += (s, e) => { CheckAll(clbTimeFeatures, false); CheckAll(clbFreqFeatures, false); };
            pnlBoth.Controls.AddRange(new Control[] { btnAllSelectBoth, btnAllClearBoth });

            var pnlTrainBtns = new FlowLayoutPanel { FlowDirection = FlowDirection.LeftToRight, WrapContents = true, AutoSize = true, Dock = DockStyle.Top };
            btnMarkAllNormal = new Button { Text = "전체 정상", Width = 90 };
            btnMarkAllAnomaly = new Button { Text = "전체 이상", Width = 90 };
            btnTrain = new Button { Text = "학습", Width = 80 };
            btnSaveModel = new Button { Text = "모델 저장", Width = 90 };
            btnMarkAllNormal.Click += (s, e) => MarkAll("Normal");
            btnMarkAllAnomaly.Click += (s, e) => MarkAll("Anomaly");
            btnTrain.Click += (s, e) => TrainModel();
            btnSaveModel.Click += (s, e) => SaveModelBySession();
            pnlTrainBtns.Controls.AddRange(new Control[] { btnMarkAllNormal, btnMarkAllAnomaly, btnTrain, btnSaveModel });

            rightStack.Controls.Add(sessPanel, 0, 0);
            rightStack.Controls.Add(lblYColumn, 0, 1);
            rightStack.Controls.Add(lblK, 0, 2);
            rightStack.Controls.Add(numK, 0, 3);
            rightStack.Controls.Add(chkStd, 0, 4);
            rightStack.Controls.Add(pnlBoth, 0, 5);
            rightStack.Controls.Add(pnlTrainBtns, 0, 6);
            rightScroll.Controls.Add(rightStack);

            top3.Controls.Add(grpTime, 0, 0);
            top3.Controls.Add(grpFreq, 1, 0);
            top3.Controls.Add(rightScroll, 2, 0);

            // 하단 그리드
            gridTrain = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = false,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };

            var trainRootGrid = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2 };
            trainRootGrid.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            trainRootGrid.Controls.Add(gridTrain, 0, 0);

            trainRoot.Controls.Add(top3, 0, 0);
            trainRoot.Controls.Add(trainRootGrid, 0, 1);
            tabTrain.Controls.Add(trainRoot);

            // ---- 검증 탭 ----
            var vLayout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4 };
            vLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 52));
            vLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));   // CLS 지표 라인
            vLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            vLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 45));

            var vBar = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            vBar.Controls.Add(new Label { Text = "홀드아웃(%):", AutoSize = true, Margin = new Padding(4, 12, 4, 0) });
            numHoldout = new NumericUpDown { Minimum = 10, Maximum = 90, Value = 30, Width = 60 };
            vBar.Controls.Add(numHoldout);
            btnValidate = new Button { Text = "검증 실행", Width = 100 };
            btnValidate.Click += (s, e) => ValidateModel();
            vBar.Controls.Add(btnValidate);
            vBar.Controls.Add(new Label { Text = "임계값:", AutoSize = true, Margin = new Padding(16, 12, 4, 0) });
            trThreshold = new TrackBar { Minimum = 0, Maximum = 1000, TickFrequency = 100, Width = 300 };
            trThreshold.Scroll += ThresholdChangedFromTrackbar;
            numThreshold = new NumericUpDown { DecimalPlaces = 3, Maximum = 1000000, Width = 100 };
            numThreshold.ValueChanged += ThresholdChangedFromNumeric;
            vBar.Controls.Add(trThreshold);
            vBar.Controls.Add(numThreshold);
            lblMetrics = new Label { AutoSize = true, Margin = new Padding(16, 12, 0, 0) };
            vBar.Controls.Add(lblMetrics);

            // CLS 지표 라인
            pnlClsMetrics = new Panel { Dock = DockStyle.Fill, Height = 24 };
            lblClsMore = new Label { AutoSize = true, Text = "", Margin = new Padding(6, 4, 0, 0) };
            pnlClsMetrics.Controls.Add(lblClsMore);

            chartROC = new Chart { Dock = DockStyle.Fill };
            var rocArea = new ChartArea("ROC");
            rocArea.AxisX.Title = "FPR"; rocArea.AxisY.Title = "TPR";
            rocArea.AxisX.Minimum = 0; rocArea.AxisX.Maximum = 1; rocArea.AxisY.Minimum = 0; rocArea.AxisY.Maximum = 1;
            chartROC.ChartAreas.Add(rocArea); chartROC.Legends.Clear();

            gridScores = new DataGridView { Dock = DockStyle.Fill, ReadOnly = true, AllowUserToAddRows = false, AllowUserToDeleteRows = false, AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill };

            vLayout.Controls.Add(vBar, 0, 0);
            vLayout.Controls.Add(pnlClsMetrics, 0, 1);
            vLayout.Controls.Add(chartROC, 0, 2);
            vLayout.Controls.Add(gridScores, 0, 3);
            tabVal.Controls.Add(vLayout);

            // ---- 평가 탭 ----
            var eLayout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2 };
            eLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));
            eLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            var eBar = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            btnScoreAll = new Button { Text = "전체 점수/예측", Width = 120 };
            btnScoreAll.Click += (s, e) => ScoreAll();
            eBar.Controls.Add(btnScoreAll);
            gridEval = new DataGridView { Dock = DockStyle.Fill, ReadOnly = true, AllowUserToAddRows = false, AllowUserToDeleteRows = false, AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill };
            eLayout.Controls.Add(eBar, 0, 0);
            eLayout.Controls.Add(gridEval, 0, 1);
            tabEval.Controls.Add(eLayout);

            tabs.TabPages.Add(tabTrain);
            tabs.TabPages.Add(tabVal);
            tabs.TabPages.Add(tabEval);
            Controls.Add(tabs);

            ClientSize = new Size(1180, 740);
            ApplySessionUI();
        }

        private void ApplySessionUI()
        {
            bool isAD = (_session == SessionType.AnomalyDetection);
            // AD 전용
            trThreshold.Enabled = isAD;
            numThreshold.Enabled = isAD;
            chartROC.Visible = isAD;

            // CLS 전용
            pnlClsMetrics.Visible = !isAD;

            // Label 열 에디터를 세션에 맞게 구성 (AD=콤보 Normal/Anomaly, CLS=텍스트)
            var dt = gridTrain.DataSource as DataTable;
            if (dt != null && dt.Columns.Contains("Label"))
            {
                if (gridTrain.Columns.Contains("Label"))
                {
                    int idx = gridTrain.Columns["Label"].DisplayIndex;
                    gridTrain.Columns.Remove("Label");

                    DataGridViewColumn col;
                    if (isAD)
                    {
                        var combo = new DataGridViewComboBoxColumn
                        {
                            Name = "Label",
                            HeaderText = "Label",
                            DataPropertyName = "Label",
                            FlatStyle = FlatStyle.Flat
                        };
                        combo.Items.AddRange("Normal", "Anomaly");
                        col = combo;
                    }
                    else
                    {
                        col = new DataGridViewTextBoxColumn
                        {
                            Name = "Label",
                            HeaderText = "Label",
                            DataPropertyName = "Label"
                        };
                    }
                    gridTrain.Columns.Insert(idx, col);
                }
            }
        }

        private void CheckAll(CheckedListBox clb, bool check)
        {
            for (int i = 0; i < clb.Items.Count; i++) clb.SetItemChecked(i, check);
        }

        // 시간/주파수 피커 채우기 + 기본 체크
        private void PopulateFeaturePickers()
        {
            clbTimeFeatures.Items.Clear();
            clbFreqFeatures.Items.Clear();
            for (int i = 0; i < _featureList.Length; i++)
            {
                var f = _featureList[i];
                bool defaultTime = (f.Key == "RMS") || (f.Key == "AbsMean") || (f.Key == "AbsMax") || (f.Key == "P2P") || (f.Key == "Skewness") || (f.Key == "Kurtosis") || (f.Key == "Crest") || (f.Key == "Shape") || (f.Key == "Impulse");
                bool defaultFreq = (f.Key == "Peak1Freq") || (f.Key == "Peak1Amp") || (f.Key == "Peak2Freq") || (f.Key == "Peak2Amp") || (f.Key == "Peak3Freq") || (f.Key == "Peak3Amp") || (f.Key == "Peak4Freq") || (f.Key == "Peak4Amp");
                if (IsFreqKey(f.Key)) clbFreqFeatures.Items.Add(f.Title, defaultFreq);
                else clbTimeFeatures.Items.Add(f.Title, defaultTime);
            }
        }
        private static bool IsFreqKey(string key)
        {
            if (string.IsNullOrEmpty(key)) return false;
            return key.StartsWith("Peak", StringComparison.OrdinalIgnoreCase) || key.IndexOf("Freq", StringComparison.OrdinalIgnoreCase) >= 0 || key.IndexOf("Amp", StringComparison.OrdinalIgnoreCase) >= 0;
        }
        private string KeyToTitle(string key)
        {
            for (int i = 0; i < _featureList.Length; i++) if (_featureList[i].Key == key) return _featureList[i].Title; return key;
        }
        private string TitleToKey(string title)
        {
            for (int i = 0; i < _featureList.Length; i++) if (_featureList[i].Title == title) return _featureList[i].Key; return title;
        }
        private static void RightAlignNumeric(DataGridView grid)
        {
            foreach (DataGridViewColumn col in grid.Columns)
            {
                if (col.ValueType == typeof(double) || col.ValueType == typeof(float) || col.ValueType == typeof(decimal))
                {
                    col.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
                    col.DefaultCellStyle.Format = "F3";
                }
            }
        }

        // ===== 학습 그리드 재구성 =====
        private void RebuildDatasetTable()
        {
            var dt = new DataTable();
            dt.Columns.Add("FileName", typeof(string));
            dt.Columns.Add("Label", typeof(string)); // 문자열 라벨
            for (int i = 0; i < _featureList.Length; i++) dt.Columns.Add(_featureList[i].Title, typeof(double));

            for (int r = 0; r < _featureRowsRaw.Count; r++)
            {
                object row = _featureRowsRaw[r];
                var dr = dt.NewRow();
                dr["FileName"] = GetProp<string>(row, "FileName") ?? "";
                // InfluxDB 세그먼트는 Label 프로퍼티가 설정됨 → 그대로 사용, 없으면 세션별 기본값
                string rowLabel = GetProp<string>(row, "Label") ?? "";
                if (string.IsNullOrEmpty(rowLabel))
                {
                    rowLabel = (_session == SessionType.AnomalyDetection) ? "Normal" : "";
                }
                else if (_session == SessionType.AnomalyDetection)
                {
                    // AD 모드: 콤보 항목은 정확히 "Normal" / "Anomaly" 이어야 함 (대소문자 정규화)
                    if (rowLabel.Equals("Anomaly", StringComparison.OrdinalIgnoreCase))
                        rowLabel = "Anomaly";
                    else
                        rowLabel = "Normal"; // normal, fault 등 그 외 모두 Normal로
                }
                dr["Label"] = rowLabel;
                for (int i = 0; i < _featureList.Length; i++)
                {
                    string key = _featureList[i].Key;
                    double? v = GetFeatureValue(row, key);
                    dr[_featureList[i].Title] = (v.HasValue && !double.IsNaN(v.Value) && !double.IsInfinity(v.Value)) ? (object)v.Value : DBNull.Value;
                }
                dt.Rows.Add(dr);
            }
            gridTrain.DataSource = dt;
            RightAlignNumeric(gridTrain);
            gridTrain.AutoResizeColumns(DataGridViewAutoSizeColumnsMode.AllCells);
            gridTrain.Refresh();

            // 세션별 라벨 에디터 적용
            ApplySessionUI();
        }

        private static T GetProp<T>(object o, string name)
        {
            if (o == null || string.IsNullOrEmpty(name)) return default(T);
            var type = o.GetType();
            var p = type.GetProperty(name);
            if (p != null) { object v = p.GetValue(o, null); if (v is T) return (T)v; return default(T); }
            var f = type.GetField(name);
            if (f != null) { object v = f.GetValue(o); if (v is T) return (T)v; }
            return default(T);
        }
        private static double? GetFeatureValue(object r, string key)
        {
            if (r == null || string.IsNullOrEmpty(key)) return null;
            var type = r.GetType();
            var p = type.GetProperty(key);
            if (p != null) { object v = p.GetValue(r, null); if (v is double) return (double)v; return null; }
            var f = type.GetField(key);
            if (f != null) { object v = f.GetValue(r); if (v is double) return (double)v; }
            return null;
        }

        private string[] SelectedKeys()
        {
            var list = new List<string>();
            for (int i = 0; i < clbTimeFeatures.Items.Count; i++) if (clbTimeFeatures.GetItemChecked(i)) list.Add(TitleToKey(Convert.ToString(clbTimeFeatures.Items[i])));
            for (int i = 0; i < clbFreqFeatures.Items.Count; i++) if (clbFreqFeatures.GetItemChecked(i)) list.Add(TitleToKey(Convert.ToString(clbFreqFeatures.Items[i])));
            return list.ToArray();
        }

        private List<Sample> BuildSamplesFromGrid(string[] keys)
        {
            var list = new List<Sample>();
            var dt = gridTrain.DataSource as DataTable; if (dt == null) return list;
            for (int r = 0; r < dt.Rows.Count; r++)
            {
                var dr = dt.Rows[r]; var s = new Sample();
                s.FileName = Convert.ToString(dr["FileName"]);
                s.LabelName = Convert.ToString(dr["Label"]);
                s.X = new double[keys.Length];
                for (int j = 0; j < keys.Length; j++)
                {
                    string title = KeyToTitle(keys[j]); object val = dr[title];
                    s.X[j] = (val == DBNull.Value) ? double.NaN : Convert.ToDouble(val, CultureInfo.InvariantCulture);
                }
                bool bad = false; for (int j = 0; j < s.X.Length; j++) { double x = s.X[j]; if (double.IsNaN(x) || double.IsInfinity(x)) { bad = true; break; } }
                if (!bad) list.Add(s);
            }
            return list;
        }

        // ====== 학습 ======
        private void MarkAll(string label)
        {
            var dt = gridTrain.DataSource as DataTable; if (dt == null) return; for (int i = 0; i < dt.Rows.Count; i++) dt.Rows[i]["Label"] = label; gridTrain.Refresh();
        }

        private void TrainModel()
        {
            _selectedKeys = SelectedKeys(); if (_selectedKeys.Length == 0) { MessageBox.Show("학습에 사용할 특징을 선택하세요."); return; }
            _samples = BuildSamplesFromGrid(_selectedKeys); if (_samples.Count == 0) { MessageBox.Show("유효한 샘플이 없습니다."); return; }
            _k = (int)numK.Value; _useStandardize = chkStd.Checked;

            // 표준화 파라미터: 학습셋 기준 (누설 방지)
            if (_useStandardize)
            {
                var Xall = _samples.Select(s => s.X).ToArray();
                (_mean, _std) = Standardizer.Fit(Xall);
                //Standardizer.TransformInPlace(Xall, _mean, _std);
                for (int i = 0; i < _samples.Count; i++) _samples[i].X = Xall[i];
            }
            else { _mean = null; _std = null; }

            if (_session == SessionType.AnomalyDetection)
            {
                // 문자열 → 0/1 변환 규칙: "Anomaly"(대소문자 무시)이면 1, 그 외는 0
                foreach (var s in _samples)
                {
                    var name = (s.LabelName ?? "").Trim();
                    s.Label = name.Equals("anomaly", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
                }
                _trainSet = _samples.Where(s => s.Label == 0).ToList();
                if (_trainSet.Count == 0) { MessageBox.Show("정상으로 표시된 샘플이 없습니다."); return; }
                _model = new KNNAnomalyModel(_k, _useStandardize, _mean, _std);
                _model.Fit(_trainSet.Select(s => s.X).ToList(), _trainSet.Select(_ => 0).ToList());
                _trainVectors = _trainSet.Select(s => s.X).ToArray();
                _classNames = new[] { "Normal", "Anomaly" };
            }
            else
            {
                // FaultDiagnosis: 문자열 라벨 인코딩
                var nameSet = _samples.Select(s => string.IsNullOrWhiteSpace(s.LabelName) ? "UNK" : s.LabelName.Trim());
                _enc.Fit(nameSet);
                foreach (var s in _samples) s.Label = _enc.ToId(s.LabelName?.Trim());

                _trainSet = _samples.ToList();
                _model = new KNNClassifier(_k, _useStandardize, _mean, _std);
                _model.Fit(_trainSet.Select(s => s.X).ToList(), _trainSet.Select(s => s.Label).ToList());
                _trainVectors = null; // CLS는 보관 불필요
                _classNames = _enc.ClassNames();
            }

            MessageBox.Show($"학습 완료\n세션={_session} / 샘플={_trainSet.Count} / 특징={_selectedKeys.Length}\nk={_k}, 표준화={_useStandardize}");
        }

        // ====== 검증 ======
        private void ValidateModel()
        {
            if (_samples == null || _samples.Count == 0) { MessageBox.Show("먼저 학습 탭에서 데이터/레이블을 준비하세요."); return; }
            if (_model == null) { MessageBox.Show("먼저 학습을 수행하세요."); return; }

            // 홀드아웃 샘플링
            var rnd = new Random(1234);
            var shuffled = _samples.OrderBy(_ => rnd.Next()).ToList();
            int nVal = (int)Math.Round(_samples.Count * (double)numHoldout.Value / 100.0);
            nVal = Math.Max(1, Math.Min(_samples.Count - 1, nVal));
            _valSet = shuffled.Take(nVal).ToList();

            if (_session == SessionType.AnomalyDetection)
            {
                _valScores.Clear();
                foreach (var s in _valSet)
                {
                    double score = (_model as KNNAnomalyModel).Score(s.X);
                    _valScores.Add(Tuple.Create(score, s.Label, s.FileName));
                }

                List<double> rocX; List<double> rocY; double auc; double best;
                ComputeRocAndBest(_valScores, out rocX, out rocY, out auc, out best);
                _optThresh = best; DrawRoc(rocX, rocY);
                double minS = _valScores.Min(v => v.Item1); double maxS = _valScores.Max(v => v.Item1); if (minS == maxS) maxS = minS + 1;
                trThreshold.Tag = new double[] { minS, maxS }; SetThreshold(best, minS, maxS);
                FillScoreTableAD(); UpdateMetricsLabel(_optThresh);
            }
            else // FaultDiagnosis
            {
                var labels = _valSet.Select(s => s.Label).Union(_trainSet.Select(t => t.Label)).Distinct().OrderBy(x => x).ToArray();
                var index = labels.Select((c, i) => (c, i)).ToDictionary(t => t.c, t => t.i);
                int C = labels.Length; int[,] cm = new int[C, C]; // [true, pred]

                int correct = 0; var rows = new DataTable();
                rows.Columns.Add("File", typeof(string)); rows.Columns.Add("True", typeof(string)); rows.Columns.Add("Pred", typeof(string));

                foreach (var s in _valSet)
                {
                    int pred = _model.Predict(s.X);
                    if (pred == s.Label) correct++;
                    int ti = index[s.Label]; int pi = index.ContainsKey(pred) ? index[pred] : 0;
                    cm[ti, pi]++;
                    rows.Rows.Add(s.FileName, _enc.ToName(s.Label), _enc.ToName(pred));
                }
                gridScores.DataSource = BuildConfusionMatrixTable(cm, labels);
                RightAlignNumeric(gridScores);

                double acc = (double)correct / _valSet.Count;
                var (precMacro, recMacro, f1Macro) = MacroPRF(cm);
                lblMetrics.Text = $"정확도: {acc:F3}";
                lblClsMore.Text = $"Macro P/R/F1 = {precMacro:F3} / {recMacro:F3} / {f1Macro:F3} | 클래스={C} ({string.Join(", ", _classNames)})";

                // 행별 예측은 하단 평가 그리드에 표시
                gridEval.DataSource = rows; gridEval.AutoResizeColumns(DataGridViewAutoSizeColumnsMode.AllCells);
            }
        }

        private void DrawRoc(List<double> xs, List<double> ys)
        {
            chartROC.Series.Clear();
            var sROC = new Series("ROC") { ChartType = SeriesChartType.Line, BorderWidth = 2, IsVisibleInLegend = false };
            for (int i = 0; i < xs.Count; i++) sROC.Points.AddXY(xs[i], ys[i]);
            chartROC.Series.Add(sROC);
        }

        private DataTable BuildConfusionMatrixTable(int[,] cm, int[] labels)
        {
            var dt = new DataTable();
            dt.Columns.Add("True\\Pred", typeof(string));
            for (int j = 0; j < labels.Length; j++) dt.Columns.Add((_session == SessionType.FaultDiagnosis ? _enc.ToName(labels[j]) : labels[j].ToString()), typeof(int));
            for (int i = 0; i < labels.Length; i++)
            {
                var row = dt.NewRow(); row[0] = (_session == SessionType.FaultDiagnosis) ? _enc.ToName(labels[i]) : labels[i].ToString();
                for (int j = 0; j < labels.Length; j++) row[j + 1] = cm[i, j];
                dt.Rows.Add(row);
            }
            return dt;
        }

        private (double prec, double rec, double f1) MacroPRF(int[,] cm)
        {
            int C = cm.GetLength(0);
            double pSum = 0, rSum = 0, fSum = 0; int cnt = 0;
            for (int c = 0; c < C; c++)
            {
                int tp = cm[c, c];
                int fp = 0, fn = 0;
                for (int j = 0; j < C; j++) if (j != c) fp += cm[j, c];
                for (int i = 0; i < C; i++) if (i != c) fn += cm[c, i];
                double prec = (tp + fp) == 0 ? 0 : (double)tp / (tp + fp);
                double rec = (tp + fn) == 0 ? 0 : (double)tp / (tp + fn);
                double f1 = (prec + rec) == 0 ? 0 : 2 * prec * rec / (prec + rec);
                pSum += prec; rSum += rec; fSum += f1; cnt++;
            }
            return (pSum / cnt, rSum / cnt, fSum / cnt);
        }

        private void ComputeRocAndBest(List<Tuple<double, int, string>> pairs, out List<double> xs, out List<double> ys, out double auc, out double best)
        {
            var arr = pairs.OrderByDescending(t => t.Item1).ToArray();
            int P = arr.Count(t => t.Item2 == 1); int N = arr.Length - P;
            if (P == 0 || N == 0)
            {
                xs = new List<double> { 0, 1 }; ys = new List<double> { 0, 1 }; auc = double.NaN; best = arr.Length > 0 ? arr[0].Item1 : 0; return;
            }
            xs = new List<double>(); ys = new List<double>();
            int tp = 0, fp = 0; double prev = double.PositiveInfinity, lastFpr = 0, lastTpr = 0; auc = 0; best = arr[arr.Length - 1].Item1; double bestYouden = double.NegativeInfinity;
            for (int i = 0; i < arr.Length; i++)
            {
                double s = arr[i].Item1; int y = arr[i].Item2;
                if (s != prev)
                {
                    double fpr = (double)fp / N; double tpr = (double)tp / P;
                    xs.Add(fpr); ys.Add(tpr);
                    auc += (fpr - lastFpr) * (tpr + lastTpr) / 2.0; lastFpr = fpr; lastTpr = tpr;
                    double youden = tpr - fpr; if (youden > bestYouden) { bestYouden = youden; best = prev; }
                    prev = s;
                }
                if (y == 1) tp++; else fp++;
            }
            xs.Add(1); ys.Add(1); auc += (1 - lastFpr) * (1 + lastTpr) / 2.0; if (double.IsPositiveInfinity(best)) best = arr[0].Item1;
        }

        private void SetThreshold(double thr, double minS, double maxS)
        {
            _updatingThreshold = true;
            if (thr < (double)numThreshold.Minimum) thr = (double)numThreshold.Minimum;
            numThreshold.Value = (decimal)thr;
            double t = (thr - minS) / (maxS - minS); if (t < 0) t = 0; if (t > 1) t = 1;
            trThreshold.Value = (int)Math.Round(t * trThreshold.Maximum);
            _updatingThreshold = false; UpdateMetricsLabel(thr);
        }
        private void ThresholdChangedFromTrackbar(object sender, EventArgs e)
        {
            if (_updatingThreshold) return; var tag = trThreshold.Tag as double[]; if (tag == null || tag.Length < 2) return; double minS = tag[0], maxS = tag[1];
            double thr = minS + (maxS - minS) * trThreshold.Value / (double)trThreshold.Maximum;
            _updatingThreshold = true; if ((double)numThreshold.Value != thr) numThreshold.Value = (decimal)thr; _updatingThreshold = false; UpdateMetricsLabel(thr);
        }
        private void ThresholdChangedFromNumeric(object sender, EventArgs e)
        {
            if (_updatingThreshold) return; var tag = trThreshold.Tag as double[]; if (tag == null || tag.Length < 2) return; double minS = tag[0], maxS = tag[1];
            double thr = (double)numThreshold.Value; double t = (thr - minS) / (maxS - minS); if (t < 0) t = 0; if (t > 1) t = 1;
            _updatingThreshold = true; trThreshold.Value = (int)Math.Round(t * trThreshold.Maximum); _updatingThreshold = false; UpdateMetricsLabel(thr);
        }

        private void UpdateMetricsLabel(double thr)
        {
            if (_session != SessionType.AnomalyDetection) return;
            if (_valScores == null || _valScores.Count == 0) { lblMetrics.Text = ""; return; }
            int tp = 0, fp = 0, tn = 0, fn = 0;
            for (int i = 0; i < _valScores.Count; i++)
            {
                double s = _valScores[i].Item1; int y = _valScores[i].Item2; int pred = (s >= thr) ? 1 : 0;
                if (pred == 1 && y == 1) tp++; else if (pred == 1 && y == 0) fp++; else if (pred == 0 && y == 0) tn++; else fn++;
            }
            double prec = (tp + fp) == 0 ? 0 : (double)tp / (tp + fp);
            double rec = (tp + fn) == 0 ? 0 : (double)tp / (tp + fn);
            double f1 = (prec + rec) == 0 ? 0 : 2 * prec * rec / (prec + rec);
            lblMetrics.Text = $"TP:{tp} FP:{fp} TN:{tn} FN:{fn} | Precision:{prec:F3} Recall:{rec:F3} F1:{f1:F3}";
        }

        private void FillScoreTableAD()
        {
            var dt = new DataTable();
            dt.Columns.Add("File", typeof(string)); dt.Columns.Add("Label", typeof(string)); dt.Columns.Add("Score", typeof(double)); dt.Columns.Add("Pred", typeof(int));
            for (int i = 0; i < _valScores.Count; i++)
            {
                double sc = _valScores[i].Item1; int la = _valScores[i].Item2; string fi = _valScores[i].Item3; int pred = (sc >= _optThresh) ? 1 : 0;
                dt.Rows.Add(fi, la == 1 ? "Anomaly" : "Normal", sc, pred);
            }
            gridScores.DataSource = dt; RightAlignNumeric(gridScores);
        }

        // ====== 평가 ======
        private void ScoreAll()
        {
            if (_samples == null || _samples.Count == 0) return; if (_model == null) return;
            if (_session == SessionType.AnomalyDetection)
            {
                var list = new DataTable(); 
                list.Columns.Add("File", typeof(string)); 
                list.Columns.Add("Label", typeof(string)); 
                list.Columns.Add("Score", typeof(double)); 
                list.Columns.Add("Pred", typeof(int));
                double thr = (double)numThreshold.Value;
                foreach (var s in _samples)
                {
                    double sc = (_model as KNNAnomalyModel).Score(s.X); 
                    int pred = (sc >= thr) ? 1 : 0; 
                    list.Rows.Add(s.FileName, s.Label == 1 ? "Anomaly" : "Normal", sc, pred);
                }
                gridEval.DataSource = list; RightAlignNumeric(gridEval);
            }
            else
            {
                var list = new DataTable(); list.Columns.Add("File", typeof(string)); list.Columns.Add("True", typeof(string)); list.Columns.Add("Pred", typeof(string));
                foreach (var s in _samples) { int pred = _model.Predict(s.X); list.Rows.Add(s.FileName, _enc.ToName(s.Label), _enc.ToName(pred)); }
                gridEval.DataSource = list; RightAlignNumeric(gridEval);
            }
        }

        private void SaveModelBySession()
        {
            if (_model == null) { MessageBox.Show("먼저 학습을 수행하세요."); return; }
            if (_selectedKeys == null || _selectedKeys.Length == 0) { MessageBox.Show("특징 선택이 비었습니다."); return; }

            double thrToSave = (double)numThreshold.Value;
            if (_session == SessionType.AnomalyDetection)
            {
                // 임계값 자동 산출 (필요시)
                if (thrToSave <= 0.0)
                {
                    var loo = ComputeTrainScoresLOO();
                    if (loo.Length >= 2)
                    {
                        thrToSave = Percentile(loo, 0.99);
                        try { numThreshold.Value = (decimal)thrToSave; } catch { }
                        MessageBox.Show($"검증 없이 자동 임계값 적용\nLOO 99% = {thrToSave:F3}", "자동 임계값", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        MessageBox.Show("학습 샘플 수가 너무 적어(≤1) 자동 임계값을 계산할 수 없습니다.", "임계값 미설정", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }
            }

            var payload = new
            {
                Session = _session.ToString(),
                ModelType = (_session == SessionType.AnomalyDetection ? "KNN_AD" : "KNN_CLS"),
                K = _k,
                Standardize = _useStandardize,
                Features = _selectedKeys,
                Threshold = (_session == SessionType.AnomalyDetection ? thrToSave : (double?)null),
                Mean = _mean,
                Std = _std,
                Train = (_session == SessionType.AnomalyDetection ? _trainVectors : null),
                YColumn = _yColumnName,
                ClassNames = (_session == SessionType.FaultDiagnosis ? _classNames : null) // ← 문자열 클래스명 보존
            };

            using (var sfd = new SaveFileDialog { Filter = "PHM Model (*.json)|*.json", FileName = (_session == SessionType.AnomalyDetection ? "knn_ad_model.json" : "knn_cls_model.json") })
            {
                if (sfd.ShowDialog() == DialogResult.OK)
                {
                    var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
                    System.IO.File.WriteAllText(sfd.FileName, json);
                    MessageBox.Show("모델 저장 완료");
                }
            }
        }

        private double[] ComputeTrainScoresLOO()
        {
            if (_session != SessionType.AnomalyDetection) return Array.Empty<double>();
            if (!(_model is KNNAnomalyModel)) return Array.Empty<double>();
            if (_trainVectors == null || _trainVectors.Length == 0) return Array.Empty<double>();
            int n = _trainVectors.Length; var scores = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (n <= 1) { scores[i] = 0.0; continue; }
                var tmp = new double[n - 1][]; int idx = 0; for (int j = 0; j < n; j++) if (j != i) tmp[idx++] = _trainVectors[j];
                int kEff = Math.Min(_k, tmp.Length); if (kEff <= 0) { scores[i] = 0.0; continue; }
                scores[i] = SignalFeatures.ScoreKnn(_trainVectors[i], tmp, kEff, _useStandardize, _mean, _std);
            }
            return scores;
        }

        private static double Percentile(double[] arr, double p)
        {
            if (arr == null || arr.Length == 0) return 0.0; if (p <= 0) return arr.Min(); if (p >= 1) return arr.Max();
            var copy = (double[])arr.Clone(); Array.Sort(copy); double pos = (copy.Length - 1) * p; int i = (int)Math.Floor(pos); double frac = pos - i;
            if (i + 1 < copy.Length) return copy[i] * (1.0 - frac) + copy[i + 1] * frac; return copy[i];
        }

        // ===== 내부 거리(표준화 고려) =====
        private static double EuclidDist(double[] a, double[] b, bool useStd, double[] mean, double[] std)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double ax = a[i], bx = b[i];
                if (useStd && std != null && mean != null)
                {
                    double denom = (std[i] == 0 ? 1 : std[i]);
                    ax = (ax - mean[i]) / denom;
                    bx = (bx - mean[i]) / denom;
                }
                double d = ax - bx;
                s += d * d;
            }
            return Math.Sqrt(s);
        }

        // ====== 모델 인터페이스 & 구현 ======
        private interface IModel
        {
            void Fit(List<double[]> X, List<int> y);           // AD: y는 전부 0으로 전달
            double Score(double[] x);                           // AD: 이상도; CLS: 선택사항(0)
            int Predict(double[] x);                            // AD: 외부 임계값으로 판단, CLS: 클래스 예측
        }

        private class KNNAnomalyModel : IModel
        {
            private int _k; private bool _useStd; private double[] _mean; private double[] _std; private double[][] _train;
            public KNNAnomalyModel(int k, bool useStd, double[] mean, double[] std) { _k = k; _useStd = useStd; _mean = mean; _std = std; }
            public void Fit(List<double[]> X, List<int> y) { _train = X.Select(a => a.ToArray()).ToArray(); }
            public double Score(double[] x)
            {
                // 거리 작은 k개 평균 거리(=이상도)
                var dists = new List<double>(_train.Length);
                for (int i = 0; i < _train.Length; i++) dists.Add(EuclidDist(x, _train[i], _useStd, _mean, _std));
                dists.Sort(); int kEff = Math.Min(_k, dists.Count); if (kEff <= 0) return 0;
                double sum = 0; for (int i = 0; i < kEff; i++) sum += dists[i]; return sum / kEff;
            }
            public int Predict(double[] x) { throw new NotSupportedException(); }
        }

        private class KNNClassifier : IModel
        {
            private int _k; private bool _useStd; private double[] _mean; private double[] _std; private double[][] _train; private int[] _y;
            public KNNClassifier(int k, bool useStd, double[] mean, double[] std) { _k = k; _useStd = useStd; _mean = mean; _std = std; }
            public void Fit(List<double[]> X, List<int> y) { _train = X.Select(a => a.ToArray()).ToArray(); _y = y.ToArray(); }
            public double Score(double[] x) { return 0.0; }
            public int Predict(double[] x)
            {
                var dists = new List<(double d, int y)>(_train.Length);
                for (int i = 0; i < _train.Length; i++) dists.Add((EuclidDist(x, _train[i], _useStd, _mean, _std), _y[i]));
                var top = dists.OrderBy(t => t.d).Take(Math.Min(_k, dists.Count)).ToArray();
                var grp = top.GroupBy(t => t.y).Select(g => new { Cls = g.Key, Cnt = g.Count() }).OrderByDescending(g => g.Cnt).ThenBy(g => g.Cls).First();
                return grp.Cls;
            }
        }

        private static class Standardizer
        {
            public static (double[] Mean, double[] Std) Fit(double[][] X)
            {
                int d = X[0].Length; double[] mean = new double[d]; double[] std = new double[d]; int n = X.Length;
                for (int j = 0; j < d; j++)
                {
                    double mu = 0; for (int i = 0; i < n; i++) mu += X[i][j]; mu /= Math.Max(1, n);
                    double var = 0; for (int i = 0; i < n; i++) { double t = X[i][j] - mu; var += t * t; }
                    var /= Math.Max(1, n - 1); double sd = Math.Sqrt(var); mean[j] = mu; std[j] = (sd <= 1e-12) ? 1.0 : sd;
                }
                return (mean, std);
            }
            public static void TransformInPlace(double[][] X, double[] mean, double[] std)
            {
                if (mean == null || std == null) return; int n = X.Length, d = X[0].Length;
                for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) X[i][j] = (X[i][j] - mean[j]) / (std[j] == 0 ? 1 : std[j]);
            }
        }
    }
}
