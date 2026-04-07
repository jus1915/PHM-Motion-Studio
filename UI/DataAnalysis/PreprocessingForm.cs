using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using System.Numerics;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using PHM_Project_DockPanel.Services.Core;
using PHM_Project_DockPanel.Services;

namespace PHM_Project_DockPanel.UI.DataAnalysis
{
    public partial class PreprocessingForm : DockContent
    {
        private const int AutoCheckLimit = 100;
        private const int MaxDisplayPointsPerSeries = 4000; // 화면 표시용 상한
        private const int MaxFftSamples = 16384;            // FFT 사용 샘플 상한
        private const int MaxSeriesForFFT = 8;              // FFT 동시 표시 시리즈 상한
        private const int DebounceIntervalMs = 300;

        // === Bulk/배치 렌더링 제어 ===
        private bool _bulkToggle = false;     // 전체 체크/해제 등 일괄 토글 중 플래그
        private const int MaxSeriesOnChart = 300;   // 한 화면에 표시할 최대 시리즈 수(필요시 조정)
        private const int BulkBatchSize = 40;       // 배치 크기(한 번에 추가할 시리즈 수)

        private const int WM_SETREDRAW = 0x000B;
        [DllImport("user32.dll")] private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, IntPtr lParam);

        private const int EM_SETCUEBANNER = 0x1501;
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wParam, string lParam);
        private static void SetWatermark(TextBox tb, string text)
        {
            if (tb == null) return;
            if (tb.IsHandleCreated) SendMessage(tb.Handle, EM_SETCUEBANNER, (IntPtr)1, text);
            else tb.HandleCreated += (s, e) => SendMessage(tb.Handle, EM_SETCUEBANNER, (IntPtr)1, text);
        }

        private sealed class DisposeAction : IDisposable
        {
            private readonly Action _a;
            public DisposeAction(Action a) { _a = a; }
            public void Dispose() { _a(); }
        }

        private static IDisposable SuspendPainting(Control c)
        {
            if (c.IsHandleCreated) SendMessage(c.Handle, WM_SETREDRAW, (IntPtr)0, IntPtr.Zero);
            return new DisposeAction(() =>
            {
                if (c.IsHandleCreated)
                {
                    SendMessage(c.Handle, WM_SETREDRAW, (IntPtr)1, IntPtr.Zero);
                    c.Invalidate(true);
                }
            });
        }

        private TabControl tabControl;

        private AIForm _aiForm;          // AI 창 핸들
        private Button btnOpenAI;        // "AI 열기" 버튼

        // 가상화된 파일 리스트
        private ListView fileList;       // ← CheckedListBox 대체
        private TextBox txtFilter;

        // 데이터 분포 분석용 컨트롤
        private Chart chart;
        private Chart chartFreq;
        private Button btnSelectFolder;
        private Button btnCheckAll;
        private Button btnUncheckAll;
        private ComboBox cmbYColumn;
        private Button btnResetZoom;
        private Button btnSavePng;
        private string currentFolder;
        private readonly object chartSync = new object();

        // 실시간 감시
        private FileSystemWatcher _watcher;
        private System.Windows.Forms.Timer _watchDebounce;

        // 디바운스
        private System.Windows.Forms.Timer _freqDebounce;

        // Feature Tab
        private ComboBox cmbFeature;
        private Button btnComputeFeatures;
        private Chart chartFeatureDist;
        private DataGridView gridFeatures;
        private Label lblFeatureInfo;
        private CheckBox chkFeaturesAll;
        private readonly List<SignalFeatures.FeatureRow> _featureTable = new List<SignalFeatures.FeatureRow>();

        private static readonly (string Key, string Title)[] FeatureList = new[]
        {
            ("AbsMax","AbsMax"), ("AbsMean","AbsMean"), ("P2P","P2P"), ("RMS","RMS"),
            ("Skewness","Skewness"), ("Kurtosis","Kurtosis"),
            ("Crest","Crest"), ("Shape","Shape"), ("Impulse","Impulse"),
            ("Peak1Freq","1st Freq"), ("Peak1Amp","1st Amp"),
            ("Peak2Freq","2nd Freq"), ("Peak2Amp","2nd Amp"),
            ("Peak3Freq","3rd Freq"), ("Peak3Amp","3rd Amp"),
            ("Peak4Freq","4th Freq"), ("Peak4Amp","4th Amp")
        };
        private class FeatureItem { public string Key { get; set; } public string Title { get; set; } }
        private ComboBox cmbDistType;

        // Correlation Tab
        private CheckedListBox clbCorrFeatures;
        private ComboBox cmbCorrMethod;
        private ComboBox cmbXFeatureCorr;
        private ComboBox cmbYFeatureCorr;
        private CheckBox chkAbsCorr;
        private CheckBox chkTrendLine;
        private NumericUpDown numTopPairs;
        private DataGridView gridCorr;
        private DataGridView gridTopCorr;
        private Chart chartPair;
        private string[] _corrSelectedKeys = new string[0];

        // Other state
        private readonly string defaultFolder = @"D:\Data\";
        private bool _splitterInitialized = false;
        private readonly HashSet<string> _loadingSeries = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        private static readonly string[] TimeColumnCandidates = { "time_s", "cycle" };

        // 가상화 리스트 상태
        private List<string> _allFiles = new List<string>(); // 전체 파일명(정렬)
        private List<int> _viewIndex = new List<int>();      // 화면에 보이는 인덱스(원본 인덱스)
        private BitArray _checked = new BitArray(0);         // 체크 상태

        // 헤더 캐시 (지연 로딩 + LRU)
        private readonly Dictionary<string, string[]> _headerCache =
            new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
        private readonly Queue<string> _headerOrder = new Queue<string>();
        private const int HeaderCacheMax = 500;

        // 대량 동작 상한
        private const int MaxAutoCheckInView = 1000;  // 왼쪽 뷰에서 "전체 체크" 상한
        private const int MaxFeatureFiles = 2000;     // 특징계산 "모든 파일" 상한

        private double CurrentSampleRate() => AppState.GetForColumn(cmbYColumn?.SelectedItem as string);

        private double CurrentSamplePeriod() => AppState.GetPeriodForColumn(cmbYColumn?.SelectedItem as string);

        private static double SampleRateFor(string yColumn) => AppState.GetForColumn(yColumn);

        private static double SamplePeriodFor(string yColumn) => AppState.GetPeriodForColumn(yColumn);

        // ===========================
        // Ctor
        // ===========================
        public PreprocessingForm()
        {
            InitializeComponent();
            Text = "데이터 전처리";

            this.SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint, true);
            this.UpdateStyles();

            currentFolder = defaultFolder;

            // 디바운스 타이머
            _watchDebounce = new System.Windows.Forms.Timer { Interval = DebounceIntervalMs };
            _watchDebounce.Tick += (s, e) => { _watchDebounce.Stop(); SafeRefreshListPreservingChecks(); };

            _freqDebounce = new System.Windows.Forms.Timer { Interval = DebounceIntervalMs };
            _freqDebounce.Tick += (s, e) => { _freqDebounce.Stop(); UpdateFrequencyChart(); };

            if (Directory.Exists(currentFolder))
            {
                LoadCsvFileListInitial_Virtual();
                if (_allFiles.Count > 30000) StopWatching(); else StartWatching(currentFolder);
            }
        }

        private void EnsureAIForm()
        {
            if (this.DockPanel == null)
            {
                if (_aiForm == null || _aiForm.IsDisposed) _aiForm = new AIForm();
                if (!_aiForm.Visible) _aiForm.Show();
                return;
            }

            var existing = this.DockPanel.Contents.OfType<AIForm>().FirstOrDefault();
            if (existing == null || existing.IsDisposed)
            {
                _aiForm = new AIForm();
            }
            else
            {
                _aiForm = existing;
            }
        }

        private void InitializeComponent()
        {
            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Vertical,
                FixedPanel = FixedPanel.Panel1,
                IsSplitterFixed = true
            };

            // ----- Left -----
            var leftPanel = new Panel { Dock = DockStyle.Fill };

            var buttonPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                Height = 35,
                FlowDirection = FlowDirection.LeftToRight
            };

            btnSelectFolder = new Button { Text = "폴더 선택", Width = 80 };
            btnSelectFolder.Click += BtnSelectFolder_Click;

            btnCheckAll = new Button { Text = "전체 체크", Width = 80 };
            btnCheckAll.Click += (s, e) =>
            {
                _bulkToggle = true; // ← ItemCheck에서 차트 갱신 막기
                try
                {
                    int limit = Math.Min(MaxAutoCheckInView, _viewIndex.Count);
                    for (int i = 0; i < limit; i++) _checked[_viewIndex[i]] = true;
                    fileList.Invalidate();
                }
                finally { _bulkToggle = false; }

                // 한 번만 배치 렌더링
                StartBulkRenderFromChecked();

                if (_viewIndex.Count > MaxAutoCheckInView)
                    MessageBox.Show(
                        $"표시 중인 파일이 많아 현재 뷰 상위 {MaxAutoCheckInView}개만 체크했습니다. (표시 {_viewIndex.Count}개 / 전체 {_allFiles.Count}개)",
                        "알림", MessageBoxButtons.OK, MessageBoxIcon.Information);
            };

            btnUncheckAll = new Button { Text = "전체 해제", Width = 80 };
            btnUncheckAll.Click += (s, e) =>
            {
                _bulkToggle = true;
                try
                {
                    for (int i = 0; i < _viewIndex.Count; i++) _checked[_viewIndex[i]] = false;
                    fileList.Invalidate();
                }
                finally { _bulkToggle = false; }

                // 차트 한 번에 정리
                lock (chartSync) chart.Series.Clear();
                AutoAdjustYAxis();
                ClearFrequencyChart();
            };

            buttonPanel.Controls.AddRange(new Control[] { btnSelectFolder, btnCheckAll, btnUncheckAll });

            cmbYColumn = new ComboBox
            {
                Dock = DockStyle.Top,
                DropDownStyle = ComboBoxStyle.DropDownList,
                Height = 28,
                Margin = new Padding(3)
            };
            cmbYColumn.SelectedIndexChanged += CmbYColumn_SelectedIndexChanged;

            txtFilter = new TextBox { Dock = DockStyle.Top };
            SetWatermark(txtFilter, "파일명 필터...");
            txtFilter.TextChanged += (s, e) => ApplyFilter();

            // === ListView 생성/속성 ===
            fileList = new ListView
            {
                Dock = DockStyle.Fill,
                View = View.Details,                 // Details 필수
                FullRowSelect = true,
                HeaderStyle = ColumnHeaderStyle.None,
                UseCompatibleStateImageBehavior = false,
                OwnerDraw = false,                   // 반드시 false
                ShowGroups = false,
                CheckBoxes = true,                   // VirtualMode=true 이전에!
                VirtualMode = true,                  // 그 다음
                VirtualListSize = 0,
                BorderStyle = BorderStyle.FixedSingle
            };
            // 컬럼 반드시 추가 (폭 0 금지)
            fileList.Columns.Add("File", 100);

            // 체크박스를 덮는 이미지리스트 제거
            fileList.StateImageList = null;
            fileList.SmallImageList = null;
            fileList.LargeImageList = null;

            // 리사이즈 시 컬럼폭 맞춤
            fileList.Resize += (s, e) => ResizeFileColumn();

            // 가상 아이템 공급
            fileList.RetrieveVirtualItem += (s, e) =>
            {
                int srcIdx = _viewIndex[e.ItemIndex];
                string name = _allFiles[srcIdx];

                var item = new ListViewItem(name);      // 텍스트 설정 필수
                item.Checked = _checked[srcIdx];        // 체크 상태 반영
                e.Item = item;
            };

            // 체크 토글 시 차트 갱신 (e.Index 사용)
            fileList.ItemCheck += (s, e) =>
            {
                int srcIdx = _viewIndex[e.Index];
                bool willChecked = (e.NewValue == CheckState.Checked);
                _checked[srcIdx] = willChecked;

                if (_bulkToggle) return; // ← 전체체크 중엔 끝!

                string name = _allFiles[srcIdx];
                string path = System.IO.Path.Combine(currentFolder, name);
                string ycol = (cmbYColumn.SelectedItem != null) ? cmbYColumn.SelectedItem.ToString() : null;

                if (willChecked) { AddCsvSeriesToChart(name, path, ycol); }
                else
                {
                    lock (chartSync)
                    {
                        if (chart.Series.IndexOf(name) >= 0)
                            chart.Series.Remove(chart.Series[name]);
                    }
                }
                AutoAdjustYAxis();
                ScheduleFreqUpdate();
            };

            // 더블클릭으로도 토글
            fileList.Activation = ItemActivation.Standard;
            fileList.ItemActivate += (s, e) =>
            {
                if (fileList.SelectedIndices.Count == 0) return;
                int viewIdx = fileList.SelectedIndices[0];
                int srcIdx = _viewIndex[viewIdx];

                bool next = !_checked[srcIdx];
                _checked[srcIdx] = next;
                fileList.RedrawItems(viewIdx, viewIdx, true);

                string name = _allFiles[srcIdx];
                string path = System.IO.Path.Combine(currentFolder, name);
                string ycol = (cmbYColumn.SelectedItem != null) ? cmbYColumn.SelectedItem.ToString() : null;

                if (next) AddCsvSeriesToChart(name, path, ycol);
                else
                {
                    lock (chartSync)
                    {
                        if (chart.Series.IndexOf(name) >= 0)
                            chart.Series.Remove(chart.Series[name]);
                    }
                    AutoAdjustYAxis();
                    ScheduleFreqUpdate();
                }
            };

            leftPanel.Controls.Add(fileList);
            leftPanel.Controls.Add(txtFilter);
            leftPanel.Controls.Add(cmbYColumn);
            leftPanel.Controls.Add(buttonPanel);
            split.Panel1.Controls.Add(leftPanel);

            // ----- Right -----
            tabControl = new TabControl { Dock = DockStyle.Fill };
            var tabDistribution = new TabPage("데이터 분포 분석");
            var tabFeature = new TabPage("특징 추출");
            var tabCorrelation = new TabPage("상관관계 분석");

            var rightContainer = new Panel { Dock = DockStyle.Fill };
            var rightTop = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                Height = 32,
                FlowDirection = FlowDirection.LeftToRight
            };
            btnResetZoom = new Button { Text = "줌 리셋", Width = 80 };
            btnResetZoom.Click += (s, e) => { ResetZoom(); ResetFreqZoom(); };
            rightTop.Controls.Add(btnResetZoom);

            btnSavePng = new Button { Text = "PNG 저장", Width = 80 };
            btnSavePng.Click += (s, e) => SaveChartsAsPng();
            rightTop.Controls.Add(btnSavePng);

            var rightSplit = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Horizontal,
                SplitterDistance = 2 * this.ClientSize.Height / 3
            };

            chart = new Chart { Dock = DockStyle.Fill };
            var chartArea = new ChartArea("MainArea");
            chartArea.AxisX.Title = "Time (s)";
            chartArea.AxisY.Title = "Y";
            chart.ChartAreas.Add(chartArea);
            chart.Legends.Clear();

            var area = chart.ChartAreas["MainArea"];
            area.CursorX.IsUserEnabled = true;
            area.CursorX.IsUserSelectionEnabled = true;
            area.CursorY.IsUserEnabled = true;
            area.CursorY.IsUserSelectionEnabled = true;
            area.AxisX.ScaleView.Zoomable = true;
            area.AxisY.ScaleView.Zoomable = true;
            area.AxisX.ScrollBar.Enabled = true;
            area.AxisY.ScrollBar.Enabled = true;
            chart.MouseWheel += Chart_MouseWheel;

            chartFreq = new Chart { Dock = DockStyle.Fill };
            var freqArea = new ChartArea("FreqArea");
            freqArea.AxisX.Title = "Frequency (Hz)";
            freqArea.AxisY.Title = "Magnitude";
            chartFreq.ChartAreas.Add(freqArea);
            chartFreq.Legends.Clear();

            var farea = chartFreq.ChartAreas["FreqArea"];
            farea.CursorX.IsUserEnabled = true;
            farea.CursorX.IsUserSelectionEnabled = true;
            farea.CursorY.IsUserEnabled = true;
            farea.CursorY.IsUserSelectionEnabled = true;
            farea.AxisX.ScaleView.Zoomable = true;
            farea.AxisY.ScaleView.Zoomable = true;
            farea.AxisX.ScrollBar.Enabled = true;
            farea.AxisY.ScrollBar.Enabled = true;
            chartFreq.MouseWheel += ChartFreq_MouseWheel;

            rightSplit.Panel1.Controls.Add(chart);
            rightSplit.Panel2.Controls.Add(chartFreq);

            rightContainer.Controls.Add(rightSplit);
            rightContainer.Controls.Add(rightTop);
            tabDistribution.Controls.Add(rightContainer);

            // Feature Tab
            var featureContainer = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3
            };
            featureContainer.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            featureContainer.RowStyles.Add(new RowStyle(SizeType.Percent, 55));
            featureContainer.RowStyles.Add(new RowStyle(SizeType.Percent, 45));

            var topBar = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false
            };

            cmbFeature = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 220 };
            cmbFeature.DataSource = FeatureList.Select(f => new FeatureItem { Key = f.Key, Title = f.Title }).ToList();
            cmbFeature.DisplayMember = "Title";
            cmbFeature.ValueMember = "Key";
            cmbFeature.SelectedIndexChanged += (s, e) => RenderFeatureDistribution();

            btnComputeFeatures = new Button { Text = "특징 계산/갱신", Width = 120 };
            btnComputeFeatures.Click += (s, e) => ComputeAndRenderFeatures();

            chkFeaturesAll = new CheckBox { Text = "모든 파일(체크 무시)", Checked = false, AutoSize = true, Margin = new Padding(12, 8, 0, 0) };

            lblFeatureInfo = new Label { AutoSize = true, Margin = new Padding(12, 8, 0, 0) };

            btnOpenAI = new Button { Text = "AI 열기", Width = 80 };
            btnOpenAI.Click += (s, e) =>
            {
                if (_featureTable == null || _featureTable.Count == 0)
                {
                    MessageBox.Show("먼저 '특징 계산/갱신'을 실행해 특징 값을 만들어 주세요.");
                    return;
                }
                if (_aiForm == null || _aiForm.IsDisposed) _aiForm = new AIForm();
                if (this.DockPanel == null) { if (!_aiForm.Visible) _aiForm.Show(this); _aiForm.Activate(); }
                else { if (!_aiForm.Visible) _aiForm.Show(this.DockPanel, DockState.Document); _aiForm.DockHandler.Activate(); }
                if (cmbYColumn.SelectedItem != null) _aiForm.SetYColumnName(cmbYColumn.SelectedItem.ToString());
                _aiForm.SetFeatureData(_featureTable.Cast<object>(), FeatureList);
            };

            topBar.Controls.Add(new Label { Text = "특징:", AutoSize = true, Margin = new Padding(0, 8, 4, 0) });
            topBar.Controls.Add(cmbFeature);
            topBar.Controls.Add(btnComputeFeatures);
            topBar.Controls.Add(chkFeaturesAll);
            topBar.Controls.Add(lblFeatureInfo);
            topBar.Controls.Add(btnOpenAI);

            cmbDistType = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 140 };
            cmbDistType.Items.AddRange(new object[] { "Histogram", "BoxPlot", "ECDF", "Density(KDE)", "Strip" });
            cmbDistType.SelectedIndex = 0;
            cmbDistType.SelectedIndexChanged += (s, e) => RenderFeatureDistribution();
            topBar.Controls.Add(new Label { Text = "표현:", AutoSize = true, Margin = new Padding(12, 8, 4, 0) });
            topBar.Controls.Add(cmbDistType);

            chartFeatureDist = new Chart { Dock = DockStyle.Fill };
            var fa = new ChartArea("FeatureArea");
            fa.AxisX.Title = "Value";
            fa.AxisY.Title = "Count";
            chartFeatureDist.ChartAreas.Add(fa);
            chartFeatureDist.Legends.Clear();

            gridFeatures = new DataGridView
            {
                Dock = DockStyle.Fill,
                AutoGenerateColumns = true,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AllowUserToResizeColumns = false,
                AllowUserToResizeRows = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None,
                ScrollBars = ScrollBars.Both
            };
            gridFeatures.CellFormatting += (s, e) =>
            {
                if (e.Value is double)
                {
                    var d = (double)e.Value;
                    if (double.IsNaN(d) || double.IsInfinity(d)) { e.Value = ""; e.FormattingApplied = true; }
                }
            };
            gridFeatures.DataBindingComplete += (s, e) => { ApplyNumericFormat(gridFeatures); FitFeatureGridColumns(); };

            featureContainer.Controls.Add(topBar, 0, 0);
            featureContainer.Controls.Add(chartFeatureDist, 0, 1);
            featureContainer.Controls.Add(gridFeatures, 0, 2);

            tabFeature.Controls.Clear();
            tabFeature.Controls.Add(featureContainer);

            // Correlation Tab
            var corrLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3
            };
            corrLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            corrLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 60));
            corrLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 40));

            var topBarC = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };

            clbCorrFeatures = new CheckedListBox { Width = 420, Height = 40, CheckOnClick = true, BorderStyle = BorderStyle.FixedSingle };
            for (int i = 0; i < FeatureList.Length; i++) clbCorrFeatures.Items.Add(FeatureList[i].Title);

            var btnCorrSelectAll = new Button { Text = "전체선택", Width = 80 };
            btnCorrSelectAll.Click += (s, e) => { for (int i = 0; i < clbCorrFeatures.Items.Count; i++) clbCorrFeatures.SetItemChecked(i, true); };
            var btnCorrClear = new Button { Text = "해제", Width = 60 };
            btnCorrClear.Click += (s, e) => { for (int i = 0; i < clbCorrFeatures.Items.Count; i++) clbCorrFeatures.SetItemChecked(i, false); };

            cmbCorrMethod = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 100 };
            cmbCorrMethod.Items.AddRange(new object[] { "Pearson", "Spearman" });
            cmbCorrMethod.SelectedIndex = 0;

            chkAbsCorr = new CheckBox { Text = "|corr| 정렬", Checked = true, AutoSize = true, Margin = new Padding(8, 12, 0, 0) };
            chkTrendLine = new CheckBox { Text = "추세선", Checked = true, AutoSize = true, Margin = new Padding(8, 12, 0, 0) };
            numTopPairs = new NumericUpDown { Minimum = 3, Maximum = 100, Value = 10, Width = 60 };

            var btnCorrCompute = new Button { Text = "계산/갱신", Width = 90 };
            btnCorrCompute.Click += (s, e) => ComputeAndRenderCorrelation();

            cmbXFeatureCorr = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 180 };
            cmbYFeatureCorr = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 180 };
            cmbXFeatureCorr.SelectedIndexChanged += (s, e) => RenderPairScatter();
            cmbYFeatureCorr.SelectedIndexChanged += (s, e) => RenderPairScatter();
            chkTrendLine.CheckedChanged += (s, e) => RenderPairScatter();

            topBarC.Controls.Add(new Label { Text = "특징 선택:", AutoSize = true, Margin = new Padding(4, 12, 4, 0) });
            topBarC.Controls.Add(clbCorrFeatures);
            topBarC.Controls.Add(btnCorrSelectAll);
            topBarC.Controls.Add(btnCorrClear);
            topBarC.Controls.Add(new Label { Text = "방법:", AutoSize = true, Margin = new Padding(12, 12, 4, 0) });
            topBarC.Controls.Add(cmbCorrMethod);
            topBarC.Controls.Add(chkAbsCorr);
            topBarC.Controls.Add(new Label { Text = "Top", AutoSize = true, Margin = new Padding(12, 12, 4, 0) });
            topBarC.Controls.Add(numTopPairs);
            topBarC.Controls.Add(btnCorrCompute);
            topBarC.Controls.Add(new Label { Text = "X:", AutoSize = true, Margin = new Padding(12, 12, 4, 0) });
            topBarC.Controls.Add(cmbXFeatureCorr);
            topBarC.Controls.Add(new Label { Text = "Y:", AutoSize = true, Margin = new Padding(6, 12, 4, 0) });
            topBarC.Controls.Add(cmbYFeatureCorr);
            topBarC.Controls.Add(chkTrendLine);

            var midSplit = new SplitContainer { Dock = DockStyle.Fill, Orientation = Orientation.Vertical, SplitterDistance = this.ClientSize.Width * 2 / 3 };

            gridCorr = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                RowHeadersVisible = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };
            gridCorr.DataBindingComplete += (s, e) => { ApplyNumericFormat(gridCorr); ColorizeCorrGrid(); };
            gridCorr.CellClick += (s, e) =>
            {
                if (e.RowIndex < 0 || e.ColumnIndex <= 0) return;
                var xKey = _corrSelectedKeys[e.ColumnIndex - 1];
                var yKey = _corrSelectedKeys[e.RowIndex];
                cmbXFeatureCorr.SelectedValue = xKey;
                cmbYFeatureCorr.SelectedValue = yKey;
                RenderPairScatter();
            };
            midSplit.Panel1.Controls.Add(gridCorr);

            chartPair = new Chart { Dock = DockStyle.Fill };
            var pairArea = new ChartArea("PairArea");
            pairArea.AxisX.Title = "X";
            pairArea.AxisY.Title = "Y";
            chartPair.ChartAreas.Add(pairArea);
            chartPair.Legends.Clear();
            midSplit.Panel2.Controls.Add(chartPair);

            gridTopCorr = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                RowHeadersVisible = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };
            gridTopCorr.DataBindingComplete += (s, e) => ApplyNumericFormat(gridTopCorr);

            corrLayout.Controls.Add(topBarC, 0, 0);
            corrLayout.Controls.Add(midSplit, 0, 1);
            corrLayout.Controls.Add(gridTopCorr, 0, 2);
            tabCorrelation.Controls.Add(corrLayout);

            var defaultKeys = new[] { "RMS", "AbsMax", "P2P", "Crest", "Peak1Freq", "Peak1Amp" };
            for (int i = 0; i < FeatureList.Length; i++)
                if (defaultKeys.Contains(FeatureList[i].Key))
                    clbCorrFeatures.SetItemChecked(i, true);

            tabControl.TabPages.Add(tabDistribution);
            tabControl.TabPages.Add(tabFeature);
            tabControl.TabPages.Add(tabCorrelation);
            split.Panel2.Controls.Add(tabControl);

            this.Controls.Add(split);
            this.ClientSize = new Size(1000, 700);
            this.Text = "데이터 전처리";
            this.SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint, true);
            this.UpdateStyles();

            _splitterInitialized = false;
            this.Shown += (s, e) =>
            {
                if (_splitterInitialized) return;
                int leftPanelWidth = btnSelectFolder.Width + btnCheckAll.Width + btnUncheckAll.Width + 20;
                split.SplitterDistance = leftPanelWidth;
                _splitterInitialized = true;
            };
        }

        private async void StartBulkRenderFromChecked()
        {
            try { _freqDebounce.Stop(); } catch { }

            var ycol = (cmbYColumn.SelectedItem != null) ? cmbYColumn.SelectedItem.ToString() : null;

            // 목표 집합(체크된 파일)
            var targets = new HashSet<string>(GetCheckedFileNames(), StringComparer.OrdinalIgnoreCase);

            // 현재 차트에 있는 시리즈
            var current = new HashSet<string>(chart.Series.Cast<Series>().Select(s => s.Name),
                                              StringComparer.OrdinalIgnoreCase);

            // 제거 목록: 체크 해제되었거나 대상 아닌 것
            var toRemove = current.Where(n => !targets.Contains(n)).ToList();

            // 추가 후보: 체크됨 & 아직 없음
            var toAddAll = targets.Where(n => !current.Contains(n)).ToList();

            // 상한 적용
            int remainCapacity = Math.Max(0, MaxSeriesOnChart - (current.Count - toRemove.Count));
            if (toAddAll.Count > remainCapacity)
            {
                toAddAll = toAddAll.Take(remainCapacity).ToList();
                MessageBox.Show(
                    $"성능을 위해 최대 {MaxSeriesOnChart}개만 차트에 표시합니다.\n" +
                    $"추가 표시는 필터로 범위를 줄이거나 일부만 체크해 주세요.",
                    "표시 상한", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }

            // 1) 제거 먼저 (UI 깜빡임 최소화)
            if (toRemove.Count > 0)
            {
                using (SuspendPainting(chart))
                {
                    lock (chartSync)
                    {
                        foreach (var name in toRemove)
                            if (chart.Series.IndexOf(name) >= 0)
                                chart.Series.Remove(chart.Series[name]);
                    }
                }
            }

            // 2) 추가는 배치로
            for (int i = 0; i < toAddAll.Count; i += BulkBatchSize)
            {
                var batch = toAddAll.Skip(i).Take(BulkBatchSize).ToList();

                using (SuspendPainting(chart))
                {
                    foreach (var name in batch)
                    {
                        string path = System.IO.Path.Combine(currentFolder, name);
                        AddCsvSeriesToChart(name, path, ycol); // 내부에서 Task.Run + BeginInvoke → OK
                    }
                }

                // UI 살리기 & 디스크/스레드풀 숨 고르기
                await Task.Yield();
            }

            AutoAdjustYAxis();
            ScheduleFreqUpdate();
        }

        private void ResizeFileColumn()
        {
            if (fileList != null && fileList.Columns.Count > 0)
            {
                int pad = 4 + System.Windows.Forms.SystemInformation.VerticalScrollBarWidth;
                int width = Math.Max(60, fileList.ClientSize.Width - pad);
                fileList.Columns[0].Width = width;
            }
        }

        // ===========================
        // Folder Watch
        // ===========================
        private void StartWatching(string folder)
        {
            StopWatching();
            _watcher = new FileSystemWatcher(folder, "*.csv")
            {
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite,
                IncludeSubdirectories = false,
                InternalBufferSize = 64 * 1024,
                EnableRaisingEvents = true
            };
            _watcher.Created += OnFolderChanged;
            _watcher.Deleted += OnFolderChanged;
            _watcher.Changed += OnFolderChanged;
            _watcher.Renamed += OnFolderRenamed;
            _watcher.Error += (s, e) => DebounceRefresh();
        }
        private void StopWatching()
        {
            if (_watcher == null) return;
            try
            {
                _watcher.EnableRaisingEvents = false;
                _watcher.Created -= OnFolderChanged;
                _watcher.Deleted -= OnFolderChanged;
                _watcher.Changed -= OnFolderChanged;
                _watcher.Renamed -= OnFolderRenamed;
                _watcher.Dispose();
            }
            catch { }
            finally { _watcher = null; }
        }
        private void OnFolderChanged(object s, FileSystemEventArgs e) { DebounceRefresh(); }
        private void OnFolderRenamed(object s, RenamedEventArgs e) { DebounceRefresh(); }
        private void DebounceRefresh()
        {
            if (IsDisposed) return;
            try
            {
                BeginInvoke(new Action(() =>
                {
                    _watchDebounce.Stop();
                    _watchDebounce.Start();
                }));
            }
            catch { }
        }

        private void LoadCsvFileListInitial_Virtual()
        {
            _allFiles.Clear();
            _viewIndex.Clear();

            if (!Directory.Exists(currentFolder)) return;

            _allFiles = Directory.EnumerateFiles(currentFolder, "*.csv", SearchOption.TopDirectoryOnly)
                                 .Select(Path.GetFileName)
                                 .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                                 .ToList();

            _checked = new BitArray(_allFiles.Count, false);

            if (_allFiles.Count > 0)
            {
                var headers = GetCachedHeaders(Path.Combine(currentFolder, _allFiles[0]));
                if (headers != null) LoadYColumnCombo(headers);
            }
            ApplyFilter();
        }

        private string[] GetCachedHeaders(string filePath)
        {
            var name = System.IO.Path.GetFileName(filePath);
            string[] hs;
            if (_headerCache.TryGetValue(name, out hs)) return hs;

            hs = GetCsvHeaders(filePath); // 파일 첫 줄만
            if (hs != null)
            {
                _headerCache[name] = hs;
                _headerOrder.Enqueue(name);
                if (_headerOrder.Count > HeaderCacheMax)
                {
                    var old = _headerOrder.Dequeue();
                    _headerCache.Remove(old);
                }
            }
            return hs;
        }

        private void ApplyFilter()
        {
            string q = (txtFilter.Text ?? string.Empty).Trim();
            _viewIndex.Clear();

            if (string.IsNullOrEmpty(q))
            {
                for (int i = 0; i < _allFiles.Count; i++) _viewIndex.Add(i);
            }
            else
            {
                for (int i = 0; i < _allFiles.Count; i++)
                    if (_allFiles[i].IndexOf(q, StringComparison.OrdinalIgnoreCase) >= 0)
                        _viewIndex.Add(i);
            }

            fileList.VirtualListSize = _viewIndex.Count;
            fileList.Invalidate();
            ResizeFileColumn(); // ★ 텍스트/체크박스 보이도록 폭 맞춤
        }

        private void RefreshCsvFileListDiff_Virtual()
        {
            if (!Directory.Exists(currentFolder)) return;

            var disk = Directory.EnumerateFiles(currentFolder, "*.csv", SearchOption.TopDirectoryOnly)
                                .Select(Path.GetFileName)
                                .OrderBy(n => n, StringComparer.OrdinalIgnoreCase)
                                .ToList();

            var cur = _allFiles;
            var toAdd = disk.Except(cur, StringComparer.OrdinalIgnoreCase).ToList();
            var toRemove = cur.Except(disk, StringComparer.OrdinalIgnoreCase).ToList();

            if (toAdd.Count == 0 && toRemove.Count == 0) return;

            _allFiles = disk;

            // 체크 보존
            var map = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < cur.Count; i++) map[cur[i]] = i;

            var newChecked = new BitArray(_allFiles.Count, false);
            for (int i = 0; i < _allFiles.Count; i++)
            {
                int j;
                if (map.TryGetValue(_allFiles[i], out j))
                {
                    if (j >= 0 && j < _checked.Length) newChecked[i] = _checked[j];
                }
            }
            _checked = newChecked;

            ApplyFilter();

            // 차트에서 사라진 파일 제거
            var now = new HashSet<string>(_allFiles, StringComparer.OrdinalIgnoreCase);
            lock (chartSync)
            {
                var gone = chart.Series.Cast<Series>().Select(s => s.Name)
                             .Where(n => !now.Contains(n)).ToList();
                for (int k = 0; k < gone.Count; k++)
                {
                    var name = gone[k];
                    if (chart.Series.IndexOf(name) >= 0) chart.Series.Remove(chart.Series[name]);
                }
            }

            AutoAdjustYAxis();
            ScheduleFreqUpdate();
        }

        private void SafeRefreshListPreservingChecks() { RefreshCsvFileListDiff_Virtual(); }

        // ===========================
        // UI Handlers
        // ===========================
        private void BtnSelectFolder_Click(object sender, EventArgs e)
        {
            using (var dialog = new OpenFileDialog
            {
                ValidateNames = false,
                CheckFileExists = false,
                CheckPathExists = true,
                FileName = "폴더 선택"
            })
            {
                if (dialog.ShowDialog() != DialogResult.OK) return;

                string path = Path.GetDirectoryName(dialog.FileName);
                currentFolder = path;

                LoadCsvFileListInitial_Virtual();
                if (_allFiles.Count > 30000) StopWatching(); else StartWatching(currentFolder);
            }
        }

        private void CmbYColumn_SelectedIndexChanged(object sender, EventArgs e)
        {
            lock (chartSync) chart.Series.Clear();

            var yColumn = cmbYColumn.SelectedItem != null ? cmbYColumn.SelectedItem.ToString() : null;
            foreach (var fileName in GetCheckedFileNames())
            {
                string filePath = Path.Combine(currentFolder, fileName);
                AddCsvSeriesToChart(fileName, filePath, yColumn);
            }

            if (cmbYColumn.SelectedItem != null)
                chart.ChartAreas["MainArea"].AxisY.Title = cmbYColumn.SelectedItem.ToString();

            AutoAdjustYAxis();
        }

        // ===========================
        // Chart (Time/Freq)
        // ===========================
        private void ResetZoom()
        {
            var area = chart.ChartAreas["MainArea"];
            area.AxisX.ScaleView.ZoomReset(0);
            area.AxisY.ScaleView.ZoomReset(0);
        }
        private void ResetFreqZoom()
        {
            if (chartFreq?.ChartAreas.Count > 0)
            {
                var a = chartFreq.ChartAreas["FreqArea"];
                a.AxisX.ScaleView.ZoomReset(0);
                a.AxisY.ScaleView.ZoomReset(0);
            }
        }
        private void Chart_MouseWheel(object sender, MouseEventArgs e)
        {
            var area = chart.ChartAreas["MainArea"];
            try
            {
                if (e.Delta > 0)
                {
                    double xMin = area.AxisX.ScaleView.ViewMinimum;
                    double xMax = area.AxisX.ScaleView.ViewMaximum;
                    if (double.IsNaN(xMin) || double.IsNaN(xMax)) { xMin = area.AxisX.Minimum; xMax = area.AxisX.Maximum; }
                    double xPos = area.AxisX.PixelPositionToValue(e.Location.X);
                    double span = (xMax - xMin) / 2.0;
                    double start = xPos - span / 2.0;
                    double end = xPos + span / 2.0;
                    area.AxisX.ScaleView.Zoom(start, end);

                    if ((ModifierKeys & Keys.Shift) == Keys.Shift)
                    {
                        double yMin = area.AxisY.ScaleView.ViewMinimum;
                        double yMax = area.AxisY.ScaleView.ViewMaximum;
                        if (double.IsNaN(yMin) || double.IsNaN(yMax)) { yMin = area.AxisY.Minimum; yMax = area.AxisY.Maximum; }
                        double yPos = area.AxisY.PixelPositionToValue(e.Location.Y);
                        double ySpan = (yMax - yMin) / 2.0;
                        area.AxisY.ScaleView.Zoom(yPos - ySpan / 2.0, yPos + ySpan / 2.0);
                    }
                }
                else
                {
                    area.AxisX.ScaleView.ZoomReset(1);
                    if ((ModifierKeys & Keys.Shift) == Keys.Shift) area.AxisY.ScaleView.ZoomReset(1);
                }
            }
            catch { }
        }
        private void ChartFreq_MouseWheel(object sender, MouseEventArgs e)
        {
            var area = chartFreq.ChartAreas["FreqArea"];
            try
            {
                if (e.Delta > 0)
                {
                    double xMin = area.AxisX.ScaleView.ViewMinimum;
                    double xMax = area.AxisX.ScaleView.ViewMaximum;
                    if (double.IsNaN(xMin) || double.IsNaN(xMax)) { xMin = area.AxisX.Minimum; xMax = area.AxisX.Maximum; }
                    double xPos = area.AxisX.PixelPositionToValue(e.Location.X);
                    double span = (xMax - xMin) / 2.0;
                    area.AxisX.ScaleView.Zoom(xPos - span / 2.0, xPos + span / 2.0);

                    if ((ModifierKeys & Keys.Shift) == Keys.Shift)
                    {
                        double yMin = area.AxisY.ScaleView.ViewMinimum;
                        double yMax = area.AxisY.ScaleView.ViewMaximum;
                        if (double.IsNaN(yMin) || double.IsNaN(yMax)) { yMin = area.AxisY.Minimum; yMax = area.AxisY.Maximum; }
                        double yPos = area.AxisY.PixelPositionToValue(e.Location.Y);
                        double ySpan = (yMax - yMin) / 2.0;
                        area.AxisY.ScaleView.Zoom(yPos - ySpan / 2.0, yPos + ySpan / 2.0);
                    }
                }
                else
                {
                    area.AxisX.ScaleView.ZoomReset(1);
                    if ((ModifierKeys & Keys.Shift) == Keys.Shift) area.AxisY.ScaleView.ZoomReset(1);
                }
            }
            catch { }
        }

        private void ClearFrequencyChart()
        {
            try { _freqDebounce?.Stop(); } catch { }
            if (chartFreq == null || chartFreq.ChartAreas.Count == 0) return;

            chartFreq.BeginInit();
            try
            {
                chartFreq.Series.Clear();
                var a = chartFreq.ChartAreas["FreqArea"];
                a.AxisX.Minimum = double.NaN; a.AxisX.Maximum = double.NaN;
                a.AxisY.Minimum = double.NaN; a.AxisY.Maximum = double.NaN;
                ResetFreqZoom();
                a.RecalculateAxesScale();
            }
            finally { chartFreq.EndInit(); }
            chartFreq.Invalidate();
        }

        private void AutoAdjustYAxis(bool addMargin = true)
        {
            var area = chart.ChartAreas["MainArea"];
            area.AxisY.Minimum = double.NaN;
            area.AxisY.Maximum = double.NaN;
            area.RecalculateAxesScale();

            if (!addMargin || chart.Series.Count == 0) return;

            double minY = double.MaxValue, maxY = double.MinValue;
            foreach (var s in chart.Series)
                foreach (var pt in s.Points)
                    if (pt.YValues.Length > 0)
                    {
                        double y = pt.YValues[0];
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }

            if (minY < maxY && minY != double.MaxValue)
            {
                double margin = (maxY - minY) * 0.05;
                area.AxisY.Minimum = minY - margin;
                area.AxisY.Maximum = maxY + margin;
            }
        }

        // ---------------------------
        // Time Series add/bind
        // ---------------------------
        private IEnumerable<string> GetCheckedFileNames()
        {
            for (int i = 0; i < _allFiles.Count; i++)
                if (_checked[i]) yield return _allFiles[i];
        }

        private void RebuildChartFromCheckedFiles(bool forceReload = false)
        {
            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action(() => RebuildChartFromCheckedFiles(forceReload)));
                return;
            }

            string yColumn = cmbYColumn.SelectedItem != null ? cmbYColumn.SelectedItem.ToString() : null;

            var desired = new HashSet<string>(GetCheckedFileNames(), StringComparer.OrdinalIgnoreCase);

            var current = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (Series s in chart.Series) current.Add(s.Name);

            if (forceReload)
            {
                using (SuspendPainting(chart))
                {
                    lock (chartSync) chart.Series.Clear();
                }
                foreach (var name in desired)
                {
                    string path = Path.Combine(currentFolder, name);
                    AddCsvSeriesToChart(name, path, yColumn);
                }
                AutoAdjustYAxis();
                ScheduleFreqUpdate();
                return;
            }

            var toRemove = new List<string>();
            foreach (var name in current)
                if (!desired.Contains(name)) toRemove.Add(name);

            var toAdd = new List<string>();
            foreach (var name in desired)
                if (!current.Contains(name)) toAdd.Add(name);

            if (toRemove.Count == 0 && toAdd.Count == 0) return;

            if (toRemove.Count > 0)
            {
                using (SuspendPainting(chart))
                {
                    lock (chartSync)
                    {
                        for (int i = 0; i < toRemove.Count; i++)
                        {
                            string name = toRemove[i];
                            if (chart.Series.IndexOf(name) >= 0)
                                chart.Series.Remove(chart.Series[name]);
                        }
                    }
                }
            }

            for (int i = 0; i < toAdd.Count; i++)
            {
                string name = toAdd[i];
                string path = Path.Combine(currentFolder, name);
                AddCsvSeriesToChart(name, path, yColumn);
            }

            AutoAdjustYAxis();
            ScheduleFreqUpdate();
        }

        private void AddCsvSeriesToChart(string seriesName, string filePath, string yColumn)
        {
            lock (_loadingSeries)
            {
                if (_loadingSeries.Contains(seriesName)) return;
                _loadingSeries.Add(seriesName);
            }

            Task.Run(new Action(() =>
            {
                try
                {
                    string[] headers = GetCachedHeaders(filePath);
                    if (headers == null || headers.Length == 0) return;

                    int xIndex = FindTimeColumnIndex(headers);
                    string yHeader = !string.IsNullOrEmpty(yColumn) ? yColumn : GetYHeaders(headers).FirstOrDefault();

                    int yIndex = -1;
                    if (!string.IsNullOrEmpty(yHeader))
                    {
                        for (int i = 0; i < headers.Length; i++)
                            if (string.Equals(headers[i], yHeader, StringComparison.OrdinalIgnoreCase)) { yIndex = i; break; }
                    }
                    if (yIndex < 0)
                    {
                        for (int i = 0; i < headers.Length; i++)
                            if (!IsTimeColumn(headers[i])) { yIndex = i; break; }
                    }
                    if (yIndex < 0) return;

                    bool hasTime = xIndex >= 0 && xIndex < headers.Length && IsTimeColumn(headers[xIndex]);
                    string xName = hasTime ? headers[xIndex] : null;

                    var culture = CultureInfo.InvariantCulture;
                    var ys = new List<double>(8192);
                    var xs = new List<double>(8192);

                    using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                    using (var sr = new StreamReader(fs, Encoding.UTF8, true))
                    {
                        sr.ReadLine(); // header skip
                        string line; int i = 0; double? firstX = null;

                        while ((line = sr.ReadLine()) != null)
                        {
                            if (string.IsNullOrWhiteSpace(line)) continue;
                            var parts = line.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
                            int need = Math.Max(yIndex, hasTime ? xIndex : 0);
                            if (parts.Length <= need) continue;

                            double yv;
                            if (!double.TryParse(parts[yIndex], NumberStyles.Float, culture, out yv)) continue;
                            if (double.IsNaN(yv) || double.IsInfinity(yv)) continue;
                            ys.Add(yv);

                            double xv;
                            if (hasTime)
                            {
                                if (!double.TryParse(parts[xIndex], NumberStyles.Float, culture, out xv))
                                    xv = i;
                            }
                            else
                            {
                                xv = i;
                            }

                            if (!firstX.HasValue) firstX = xv;

                            double tSec;
                            // 선택된 Y열 기준 샘플 주기
                            double sp = SamplePeriodFor(yColumn);
                             if (hasTime && string.Equals(xName, "cycle", StringComparison.OrdinalIgnoreCase))
                                tSec = (xv - firstX.Value) * sp;   // cycle -> time
                             else if (hasTime)
                                tSec = xv - firstX.Value;          // time_s는 그대로 초(sec)
                             else
                                tSec = i * sp;                     // 시간열 없으면 샘플 인덱스 기반

                            xs.Add(tSec);
                            i++;
                        }
                    }

                    if (ys.Count == 0) return;

                    double[] dx, dy;
                    DownsampleMinMax(xs, ys, MaxDisplayPointsPerSeries, out dx, out dy);

                    this.BeginInvoke(new Action(() =>
                    {
                        chart.BeginInit();
                        try
                        {
                            chart.AntiAliasing = AntiAliasingStyles.None;
                            chart.TextAntiAliasingQuality = TextAntiAliasingQuality.Normal;

                            var series = chart.Series.FindByName(seriesName);
                            if (series == null)
                            {
                                series = new Series(seriesName)
                                {
                                    ChartType = SeriesChartType.FastLine,
                                    BorderWidth = 1,
                                    IsVisibleInLegend = false
                                };
                                series.SmartLabelStyle.Enabled = false;
                                chart.Series.Add(series);
                            }

                            series.Points.DataBindXY(dx, dy);
                            AutoAdjustYAxis();
                            ScheduleFreqUpdate();
                        }
                        finally { chart.EndInit(); }
                    }));
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine("CSV load error: " + ex.Message);
                }
                finally
                {
                    lock (_loadingSeries) _loadingSeries.Remove(seriesName);
                }
            }));
        }

        private void ScheduleFreqUpdate()
        {
            _freqDebounce.Stop();
            _freqDebounce.Start();
        }

        private void UpdateFrequencyChart()
        {
            if (chartFreq == null) return;

            var seriesList = chart.Series.Cast<Series>().ToList();
            if (seriesList.Count == 0)
            {
                ClearFrequencyChart();
                return;
            }

            var targets = seriesList.Take(MaxSeriesForFFT)
                                    .Select(s => new
                                    {
                                        s.Name,
                                        Y = s.Points.Take(MaxFftSamples).Select(p => p.YValues[0]).ToArray()
                                    })
                                    .ToList();

            // UI 컨트롤 값은 Task.Run 진입 전 UI 스레드에서 미리 캡처
            double capturedSampleRate = CurrentSampleRate();

            Task.Run(new Action(() =>
            {
                var results = new List<(string Name, double[] F, double[] S)>(targets.Count);
                for (int i = 0; i < targets.Count; i++)
                {
                    var t = targets[i];
                    if (t.Y.Length < 4) continue;
                    double[] freq;
                    double sr = capturedSampleRate;
                    var spec = SignalFeatures.ComputeMagnitudeSpectrum(t.Y, sr, out freq);
                    results.Add((t.Name, freq, spec));
                }

                this.BeginInvoke(new Action(() =>
                {
                    chartFreq.BeginInit();
                    try
                    {
                        chartFreq.Series.Clear();
                        double minFreq = double.MaxValue, maxFreq = double.MinValue;

                        for (int i = 0; i < results.Count; i++)
                        {
                            var r = results[i];
                            var fseries = new Series(r.Name)
                            {
                                ChartType = SeriesChartType.FastLine,
                                BorderWidth = 2,
                                IsVisibleInLegend = false
                            };
                            for (int k = 0; k < r.F.Length; k++)
                            {
                                fseries.Points.AddXY(r.F[k], r.S[k]);
                            }
                            if (r.F.Length > 0)
                            {
                                if (r.F.First() < minFreq) minFreq = r.F.First();
                                if (r.F.Last() > maxFreq) maxFreq = r.F.Last();
                            }
                            chartFreq.Series.Add(fseries);
                        }

                        var area2 = chartFreq.ChartAreas["FreqArea"];
                        if (results.Count > 0 && minFreq < double.MaxValue && maxFreq > double.MinValue)
                        {
                            area2.AxisX.Minimum = minFreq;
                            area2.AxisX.Maximum = maxFreq;
                        }
                        else
                        {
                            area2.AxisX.Minimum = double.NaN;
                            area2.AxisX.Maximum = double.NaN;
                        }
                        area2.AxisY.Minimum = double.NaN;
                        area2.AxisY.Maximum = double.NaN;
                        area2.RecalculateAxesScale();
                    }
                    finally { chartFreq.EndInit(); }
                }));
            }));
        }

        private void SaveChartsAsPng()
        {
            try
            {
                using (var sfd = new SaveFileDialog
                {
                    Title = "그래프 PNG 저장",
                    Filter = "PNG Image (*.png)|*.png",
                    FileName = "distribution.png"
                })
                {
                    if (sfd.ShowDialog() != DialogResult.OK) return;

                    string dir = Path.GetDirectoryName(sfd.FileName);
                    string baseName = Path.GetFileNameWithoutExtension(sfd.FileName);
                    const int side = 1024;

                    string timePath = Path.Combine(dir, baseName + "_time.png");
                    SaveChartAsSquarePng(chart, timePath, side);

                    string freqPath = Path.Combine(dir, baseName + "_freq.png");
                    SaveChartAsSquarePng(chartFreq, freqPath, side);

                    MessageBox.Show(
                        "저장 완료\n- " + timePath + "\n- " + freqPath,
                        "완료", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("PNG 저장 중 오류: " + ex.Message,
                    "오류", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        private void SaveChartAsSquarePng(Chart target, string path, int side)
        {
            if (target == null || side <= 0) return;

            var parent = target.Parent;
            var oldSize = target.Size;

            try
            {
                target.SuspendLayout();
                if (parent != null) parent.SuspendLayout();

                target.Width = side;
                target.Height = side;

                using (var bmp = new Bitmap(side, side))
                {
                    target.DrawToBitmap(bmp, new Rectangle(0, 0, side, side));
                    bmp.Save(path, ImageFormat.Png);
                }
            }
            finally
            {
                target.Width = oldSize.Width;
                target.Height = oldSize.Height;

                target.ResumeLayout();
                if (parent != null) parent.ResumeLayout();
            }
        }

        private void ComputeAndRenderFeatures()
        {
            if (cmbYColumn.SelectedItem == null)
            {
                MessageBox.Show("Y 컬럼을 먼저 선택하세요.");
                return;
            }
            var yColumn = cmbYColumn.SelectedItem.ToString();

            List<string> targets;
            if (chkFeaturesAll != null && chkFeaturesAll.Checked)
            {
                targets = _allFiles.ToList();
                if (targets.Count > MaxFeatureFiles)
                {
                    // Fisher–Yates shuffle sampling
                    var rnd = new Random(1234);
                    for (int i = targets.Count - 1; i > 0; i--)
                    {
                        int j = rnd.Next(i + 1);
                        var tmp = targets[i]; targets[i] = targets[j]; targets[j] = tmp;
                    }
                    targets = targets.Take(MaxFeatureFiles).ToList();
                    MessageBox.Show(
                        string.Format("파일이 너무 많아 {0}개(무작위)만 사용합니다.", MaxFeatureFiles),
                        "알림", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            else
            {
                targets = GetCheckedFileNames().ToList();
                if (targets.Count == 0)
                {
                    MessageBox.Show("특징을 계산할 CSV를 체크하거나 '모든 파일' 옵션을 사용하세요.");
                    return;
                }
            }

            _featureTable.Clear();

            Task.Run(new Action(() =>
            {
                for (int idx = 0; idx < targets.Count; idx++)
                {
                    var fileName = targets[idx];
                    var path = Path.Combine(currentFolder, fileName);
                    var headers = GetCachedHeaders(path);
                    if (headers == null) continue;

                    int xIndex = FindTimeColumnIndex(headers);
                    int yIndex = Array.FindIndex(headers, delegate (string h) { return h.Equals(yColumn, StringComparison.OrdinalIgnoreCase); });
                    if (yIndex < 0) continue;

                    var culture = CultureInfo.InvariantCulture;
                    var ys = new List<double>(4096);
                    try
                    {
                        using (var st = new StreamReader(path, Encoding.UTF8, true))
                        {
                            st.ReadLine();
                            string line;
                            while ((line = st.ReadLine()) != null)
                            {
                                var parts = line.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
                                if (parts.Length <= Math.Max(xIndex, yIndex)) continue;
                                double yv;
                                if (!double.TryParse(parts[yIndex], NumberStyles.Float, culture, out yv)) continue;
                                if (double.IsNaN(yv) || double.IsInfinity(yv)) continue;
                                ys.Add(yv);
                            }
                        }
                    }
                    catch { continue; }

                    if (ys.Count < 4) continue;

                    var row = new SignalFeatures.FeatureRow { FileName = fileName };
                    SignalFeatures.FillTimeDomainFeatures(ys, row);

                    int avail = Math.Min(ys.Count, MaxFftSamples);
                    var yarr = ys.Take(avail).ToArray();
                    double[] freq;
                    double sr = SampleRateFor(yColumn);
                    var spec = SignalFeatures.ComputeMagnitudeSpectrum(yarr, sr, out freq);
                    SignalFeatures.FillPeakFeatures(freq, spec, row);

                    _featureTable.Add(row);
                }

                this.BeginInvoke(new Action(() =>
                {
                    gridFeatures.DataSource = null;
                    gridFeatures.DataSource = _featureTable;
                    ApplyNumericFormat(gridFeatures);
                    RenderFeatureDistribution();
                    lblFeatureInfo.Text = string.Format("샘플 파일: {0}개", _featureTable.Count);
                    EnsureAIForm();
                    _aiForm.SetYColumnName(yColumn);
                    _aiForm.SetFeatureData(_featureTable.Cast<object>(), FeatureList);
                }));
            }));
        }

        private IEnumerable<double> GetFeatureValuesByKey(string key)
        {
            switch (key)
            {
                case "AbsMax": return _featureTable.Select(r => r.AbsMax);
                case "AbsMean": return _featureTable.Select(r => r.AbsMean);
                case "P2P": return _featureTable.Select(r => r.P2P);
                case "RMS": return _featureTable.Select(r => r.RMS);
                case "Skewness": return _featureTable.Select(r => r.Skewness);
                case "Kurtosis": return _featureTable.Select(r => r.Kurtosis);
                case "Crest": return _featureTable.Select(r => r.Crest);
                case "Shape": return _featureTable.Select(r => r.Shape);
                case "Impulse": return _featureTable.Select(r => r.Impulse);
                case "Peak1Freq": return _featureTable.Select(r => r.Peak1Freq);
                case "Peak1Amp": return _featureTable.Select(r => r.Peak1Amp);
                case "Peak2Freq": return _featureTable.Select(r => r.Peak2Freq);
                case "Peak2Amp": return _featureTable.Select(r => r.Peak2Amp);
                case "Peak3Freq": return _featureTable.Select(r => r.Peak3Freq);
                case "Peak3Amp": return _featureTable.Select(r => r.Peak3Amp);
                case "Peak4Freq": return _featureTable.Select(r => r.Peak4Freq);
                case "Peak4Amp": return _featureTable.Select(r => r.Peak4Amp);
                default: return Enumerable.Empty<double>();
            }
        }

        private static (double[] binCenters, int[] counts) BuildHistogram(List<double> values, int bins)
        {
            double min = values.Min();
            double max = values.Max();
            if (max <= min) return (new[] { min }, new[] { values.Count });

            double width = (max - min) / bins;
            var counts = new int[bins];
            for (int i = 0; i < values.Count; i++)
            {
                int idx = (int)Math.Floor((values[i] - min) / width);
                if (idx == bins) idx--;
                idx = Math.Max(0, Math.Min(bins - 1, idx));
                counts[idx]++;
            }
            var centers = new double[bins];
            for (int b = 0; b < bins; b++)
                centers[b] = min + (b + 0.5) * width;

            return (centers, counts);
        }

        private void RenderFeatureDistribution(int binCount = 20)
        {
            chartFeatureDist.Series.Clear();
            if (_featureTable.Count == 0 || cmbFeature.SelectedValue == null) return;

            string key = cmbFeature.SelectedValue.ToString();
            var values = GetFeatureValuesByKey(key).Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToList();
            if (values.Count == 0) return;

            switch ((cmbDistType.SelectedItem as string) ?? "Histogram")
            {
                case "BoxPlot": RenderBoxPlot(values, key); break;
                case "ECDF": RenderECDF(values, key); break;
                case "Density(KDE)": RenderKDE(values, key); break;
                case "Strip": RenderStrip(values, key); break;
                default: RenderHistogram(values, key, binCount); break;
            }
        }

        private void RenderHistogram(List<double> values, string key, int bins)
        {
            var (xs, counts) = BuildHistogram(values, bins);
            var s = new Series("Histogram") { ChartType = SeriesChartType.Column, IsVisibleInLegend = false };
            for (int i = 0; i < xs.Length; i++) s.Points.AddXY(xs[i], counts[i]);
            chartFeatureDist.Series.Add(s);

            var a = chartFeatureDist.ChartAreas["FeatureArea"];
            a.AxisX.Title = FeatureList.First(f => f.Key == key).Title;
            a.AxisY.Title = "Count";
            a.AxisY.Minimum = double.NaN; a.AxisY.Maximum = double.NaN;
            a.RecalculateAxesScale();
        }

        private void RenderBoxPlot(List<double> values, string key)
        {
            var raw = new Series("raw") { ChartType = SeriesChartType.Point, IsVisibleInLegend = false, Enabled = false };
            foreach (var v in values) raw.Points.AddY(v);

            var box = new Series("box") { ChartType = SeriesChartType.BoxPlot, IsVisibleInLegend = false };
            box["BoxPlotSeries"] = "raw";
            box["BoxPlotShowAverage"] = "true";
            box["BoxPlotWhiskerPercentile"] = "10";

            chartFeatureDist.Series.Add(raw);
            chartFeatureDist.Series.Add(box);

            var a = chartFeatureDist.ChartAreas["FeatureArea"];
            a.AxisX.LabelStyle.Enabled = false;
            a.AxisX.Title = FeatureList.First(f => f.Key == key).Title;
            a.AxisY.Title = "Value";
            a.RecalculateAxesScale();
        }

        private void RenderECDF(List<double> values, string key)
        {
            values.Sort();
            int n = values.Count;
            var s = new Series("ECDF") { ChartType = SeriesChartType.StepLine, BorderWidth = 2, IsVisibleInLegend = false };
            for (int i = 0; i < n; i++) s.Points.AddXY(values[i], (i + 1.0) / n);
            chartFeatureDist.Series.Add(s);

            var a = chartFeatureDist.ChartAreas["FeatureArea"];
            a.AxisX.Title = FeatureList.First(f => f.Key == key).Title;
            a.AxisY.Title = "Cumulative Probability";
            a.AxisY.Minimum = 0; a.AxisY.Maximum = 1;
        }

        private void RenderKDE(List<double> values, string key)
        {
            int n = values.Count;
            double mean = values.Average();
            double var = values.Sum(v => { var d = v - mean; return d * d; }) / Math.Max(1, n - 1);
            double std = Math.Sqrt(var);
            if (std <= 0) { RenderStrip(values, key); return; }

            double h = 1.06 * std * Math.Pow(n, -0.2);

            double min = values.Min(), max = values.Max();
            double pad = (max - min) * 0.1; if (pad == 0) pad = Math.Abs(min) * 0.1 + 1e-6;
            min -= pad; max += pad;

            int m = 200;
            var s = new Series("Density") { ChartType = SeriesChartType.Line, BorderWidth = 2, IsVisibleInLegend = false };
            double inv = 1.0 / (n * h * Math.Sqrt(2 * Math.PI));
            for (int i = 0; i < m; i++)
            {
                double x = min + (max - min) * i / (m - 1);
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    double u = (x - values[j]) / h;
                    sum += Math.Exp(-0.5 * u * u);
                }
                s.Points.AddXY(x, sum * inv);
            }
            chartFeatureDist.Series.Add(s);

            var a = chartFeatureDist.ChartAreas["FeatureArea"];
            a.AxisX.Title = FeatureList.First(f => f.Key == key).Title;
            a.AxisY.Title = "Density";
            a.AxisY.Minimum = double.NaN; a.AxisY.Maximum = double.NaN;
            a.RecalculateAxesScale();
        }

        private void RenderStrip(List<double> values, string key)
        {
            var s = new Series("Strip") { ChartType = SeriesChartType.Point, IsVisibleInLegend = false };
            var rng = new Random(1234);
            foreach (var v in values)
            {
                double y = 1.0 + (rng.NextDouble() - 0.5) * 0.25;
                s.Points.AddXY(v, y);
            }
            chartFeatureDist.Series.Add(s);

            var a = chartFeatureDist.ChartAreas["FeatureArea"];
            a.AxisX.Title = FeatureList.First(f => f.Key == key).Title;
            a.AxisY.Title = "";
            a.AxisY.Minimum = 0.5; a.AxisY.Maximum = 1.5;
        }

        private void ComputeAndRenderCorrelation()
        {
            if (_featureTable == null || _featureTable.Count == 0)
            {
                MessageBox.Show("먼저 '특징 계산/갱신'을 실행해 특징 값을 만들어 주세요.");
                return;
            }

            _corrSelectedKeys = clbCorrFeatures.CheckedIndices
                .Cast<int>()
                .Select(i => FeatureList[i].Key)
                .ToArray();

            if (_corrSelectedKeys.Length < 2)
            {
                MessageBox.Show("최소 2개 이상의 특징을 선택하세요.");
                return;
            }

            var series = _corrSelectedKeys
                .Select(k => _featureTable.Select(r => GetFeatureValueByKey(r, k)).ToList())
                .ToArray();

            int klen = _corrSelectedKeys.Length;
            var corr = new double[klen, klen];
            string method = (cmbCorrMethod.SelectedItem as string) ?? "Pearson";

            for (int i = 0; i < klen; i++)
                for (int j = 0; j < klen; j++)
                    corr[i, j] = (i == j) ? 1.0 : ComputeCorr(series[i], series[j], method);

            var dt = new System.Data.DataTable();
            dt.Columns.Add("Feature", typeof(string));
            foreach (var key in _corrSelectedKeys) dt.Columns.Add(GetTitleByKey(key), typeof(double));

            for (int i = 0; i < klen; i++)
            {
                var row = dt.NewRow();
                row[0] = GetTitleByKey(_corrSelectedKeys[i]);
                for (int j = 0; j < klen; j++)
                {
                    double v = corr[i, j];
                    row[j + 1] = (double.IsNaN(v) || double.IsInfinity(v)) ? (object)DBNull.Value : v;
                }
                dt.Rows.Add(row);
            }
            gridCorr.DataSource = dt;

            var topPairs = new List<(string X, string Y, double Corr, double Abs)>();
            for (int i = 0; i < klen; i++)
                for (int j = i + 1; j < klen; j++)
                {
                    var v = corr[i, j];
                    topPairs.Add((GetTitleByKey(_corrSelectedKeys[i]), GetTitleByKey(_corrSelectedKeys[j]), v, Math.Abs(v)));
                }
            bool useAbs = chkAbsCorr.Checked;
            var top = topPairs
                .OrderByDescending(p => useAbs ? p.Abs : p.Corr)
                .Take((int)numTopPairs.Value)
                .Select(p => new { X = p.X, Y = p.Y, Corr = p.Corr })
                .ToList();
            gridTopCorr.DataSource = top;

            var items = _corrSelectedKeys.Select(k => new { Key = k, Title = GetTitleByKey(k) }).ToList();
            cmbXFeatureCorr.DataSource = items.ToList();
            cmbXFeatureCorr.DisplayMember = "Title";
            cmbXFeatureCorr.ValueMember = "Key";
            cmbYFeatureCorr.DataSource = items.ToList();
            cmbYFeatureCorr.DisplayMember = "Title";
            cmbYFeatureCorr.ValueMember = "Key";

            if (cmbXFeatureCorr.SelectedIndex < 0) cmbXFeatureCorr.SelectedIndex = 0;
            if (cmbYFeatureCorr.SelectedIndex < 0) cmbYFeatureCorr.SelectedIndex = Math.Min(1, items.Count - 1);

            if (top.Count > 0)
            {
                string findKey(string title) => FeatureList.First(f => f.Title == title).Key;
                cmbXFeatureCorr.SelectedValue = findKey(top[0].X);
                cmbYFeatureCorr.SelectedValue = findKey(top[0].Y);
            }
            RenderPairScatter();
        }

        private double? GetFeatureValueByKey(SignalFeatures.FeatureRow r, string key)
        {
            switch (key)
            {
                case "AbsMax": return r.AbsMax;
                case "AbsMean": return r.AbsMean;
                case "P2P": return r.P2P;
                case "RMS": return r.RMS;
                case "Skewness": return r.Skewness;
                case "Kurtosis": return r.Kurtosis;
                case "Crest": return r.Crest;
                case "Shape": return r.Shape;
                case "Impulse": return r.Impulse;
                case "Peak1Freq": return r.Peak1Freq;
                case "Peak1Amp": return r.Peak1Amp;
                case "Peak2Freq": return r.Peak2Freq;
                case "Peak2Amp": return r.Peak2Amp;
                case "Peak3Freq": return r.Peak3Freq;
                case "Peak3Amp": return r.Peak3Amp;
                case "Peak4Freq": return r.Peak4Freq;
                case "Peak4Amp": return r.Peak4Amp;
                default: return null;
            }
        }

        private string GetTitleByKey(string key)
        {
            var f = FeatureList.FirstOrDefault(t => t.Key == key);
            return string.IsNullOrEmpty(f.Title) ? key : f.Title;
        }

        private double ComputeCorr(List<double?> xa, List<double?> ya, string method)
        {
            var xs = new List<double>();
            var ys = new List<double>();
            int n = Math.Min(xa.Count, ya.Count);
            for (int i = 0; i < n; i++)
            {
                var xv = xa[i]; var yv = ya[i];
                if (!xv.HasValue || !yv.HasValue) continue;
                double x = xv.Value, y = yv.Value;
                if (double.IsNaN(x) || double.IsInfinity(x)) continue;
                if (double.IsNaN(y) || double.IsInfinity(y)) continue;
                xs.Add(x); ys.Add(y);
            }
            if (xs.Count < 3) return double.NaN;

            if (method == "Spearman")
            {
                var xr = Rank(xs.ToArray());
                var yr = Rank(ys.ToArray());
                return Pearson(xr, yr);
            }
            else { return Pearson(xs.ToArray(), ys.ToArray()); }
        }

        private static double Pearson(double[] x, double[] y)
        {
            int n = x.Length;
            double mx = x.Average();
            double my = y.Average();
            double sxx = 0, syy = 0, sxy = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = x[i] - mx;
                double dy = y[i] - my;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
            if (sxx <= 0 || syy <= 0) return double.NaN;
            return sxy / Math.Sqrt(sxx * syy);
        }

        private static double[] Rank(double[] a)
        {
            int n = a.Length;
            var idx = Enumerable.Range(0, n).OrderBy(i => a[i]).ToArray();
            var ranks = new double[n];
            int i0 = 0;
            while (i0 < n)
            {
                int i1 = i0 + 1;
                while (i1 < n && a[idx[i1]].Equals(a[idx[i0]])) i1++;
                double r = (i0 + 1 + i1) / 2.0;
                for (int k = i0; k < i1; k++) ranks[idx[k]] = r;
                i0 = i1;
            }
            return ranks;
        }

        private void RenderPairScatter()
        {
            chartPair.Series.Clear();
            if (_featureTable == null || _featureTable.Count == 0) return;
            if (cmbXFeatureCorr.SelectedValue == null || cmbYFeatureCorr.SelectedValue == null) return;

            string xk = cmbXFeatureCorr.SelectedValue.ToString();
            string yk = cmbYFeatureCorr.SelectedValue.ToString();

            var xs = new List<double>();
            var ys = new List<double>();

            foreach (var r in _featureTable)
            {
                var xv = GetFeatureValueByKey(r, xk);
                var yv = GetFeatureValueByKey(r, yk);
                if (!xv.HasValue || !yv.HasValue) continue;
                double x = xv.Value, y = yv.Value;
                if (double.IsNaN(x) || double.IsNaN(y) || double.IsInfinity(x) || double.IsInfinity(y)) continue;
                xs.Add(x); ys.Add(y);
            }
            if (xs.Count == 0) return;

            var s = new Series("points")
            {
                ChartType = SeriesChartType.Point,
                MarkerStyle = MarkerStyle.Circle,
                MarkerSize = 6,
                IsVisibleInLegend = false
            };
            for (int i = 0; i < xs.Count; i++) s.Points.AddXY(xs[i], ys[i]);
            chartPair.Series.Add(s);

            var area = chartPair.ChartAreas["PairArea"];
            area.AxisX.Title = GetTitleByKey(xk);
            area.AxisY.Title = GetTitleByKey(yk);
            area.AxisX.Minimum = double.NaN; area.AxisX.Maximum = double.NaN;
            area.AxisY.Minimum = double.NaN; area.AxisY.Maximum = double.NaN;
            area.RecalculateAxesScale();

            if (chkTrendLine.Checked && xs.Count >= 2)
            {
                double mx = xs.Average(), my = ys.Average();
                double sxx = 0, sxy = 0;
                for (int i = 0; i < xs.Count; i++)
                {
                    double dx = xs[i] - mx;
                    sxx += dx * dx;
                    sxy += dx * (ys[i] - my);
                }
                if (sxx > 0)
                {
                    double b = sxy / sxx;
                    double a0 = my - b * mx;
                    double minX = xs.Min(), maxX = xs.Max();
                    var line = new Series("trend") { ChartType = SeriesChartType.Line, BorderWidth = 2, IsVisibleInLegend = false };
                    line.Points.AddXY(minX, a0 + b * minX);
                    line.Points.AddXY(maxX, a0 + b * maxX);
                    chartPair.Series.Add(line);
                }
            }
        }

        // ===========================
        // Helpers / CSV / Columns
        // ===========================
        private string[] GetCsvHeaders(string filePath)
        {
            try
            {
                var firstLine = File.ReadLines(filePath).FirstOrDefault();
                if (!string.IsNullOrEmpty(firstLine))
                    return firstLine.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
            }
            catch { }
            return null;
        }
        private void LoadYColumnCombo(string[] headers)
        {
            cmbYColumn.Items.Clear();
            var yHeaders = GetYHeaders(headers).ToList();
            foreach (var h in yHeaders) cmbYColumn.Items.Add(h);
            if (cmbYColumn.Items.Count > 0) cmbYColumn.SelectedIndex = 0;
        }
        private static bool IsTimeColumn(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return false;
            var n = name.Trim();
            return TimeColumnCandidates.Any(tc => n.Equals(tc, StringComparison.OrdinalIgnoreCase));
        }
        private static int FindTimeColumnIndex(string[] headers)
        {
            if (headers == null) return -1;
            for (int i = 0; i < headers.Length; i++)
                if (IsTimeColumn(headers[i])) return i;
            return -1;
        }
        private static IEnumerable<string> GetYHeaders(string[] headers)
        {
            if (headers == null) yield break;
            foreach (var h in headers)
                if (!IsTimeColumn(h)) yield return h;
        }
        private int FindItemIndexByText(string text)
        {
            if (string.IsNullOrEmpty(text)) return -1;
            for (int i = 0; i < fileList.Items.Count; i++)
                if (string.Equals(fileList.Items[i].ToString(), text, StringComparison.OrdinalIgnoreCase))
                    return i;
            return -1;
        }
        private static void ApplyNumericFormat(DataGridView grid)
        {
            foreach (DataGridViewColumn col in grid.Columns)
            {
                if (col.ValueType == typeof(double) ||
                    col.ValueType == typeof(float) ||
                    col.ValueType == typeof(decimal))
                {
                    col.DefaultCellStyle.Format = "F3";
                    col.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
                }
            }
        }
        private void FitFeatureGridColumns()
        {
            if (gridFeatures == null || gridFeatures.Columns.Count == 0) return;

            gridFeatures.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.DisplayedCells;
            gridFeatures.AutoResizeColumns(DataGridViewAutoSizeColumnsMode.DisplayedCells);

            foreach (DataGridViewColumn col in gridFeatures.Columns)
            {
                int w = col.Width;
                col.AutoSizeMode = DataGridViewAutoSizeColumnMode.None;
                col.Width = w;
                col.Resizable = DataGridViewTriState.False;
            }

            gridFeatures.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
            gridFeatures.ScrollBars = ScrollBars.Both;
            gridFeatures.ColumnHeadersDefaultCellStyle.WrapMode = DataGridViewTriState.False;
        }

        // ===========================
        // Lifecycle
        // ===========================
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            StopWatching();
            if (_watchDebounce != null)
            {
                try { _watchDebounce.Stop(); _watchDebounce.Dispose(); } catch { }
                _watchDebounce = null;
            }
            base.OnFormClosing(e);
        }

        // ===========================
        // Downsample
        // ===========================
        private static void DownsampleMinMax(IReadOnlyList<double> xs, IReadOnlyList<double> ys, int maxPoints,
            out double[] dx, out double[] dy)
        {
            int n = Math.Min(xs.Count, ys.Count);
            if (n <= maxPoints || maxPoints < 4)
            {
                dx = xs.Take(n).ToArray();
                dy = ys.Take(n).ToArray();
                return;
            }

            int buckets = maxPoints / 2; // min+max 두 점
            double bucketSize = (double)n / buckets;

            var xList = new List<double>(maxPoints);
            var yList = new List<double>(maxPoints);

            for (int b = 0; b < buckets; b++)
            {
                int start = (int)Math.Floor(b * bucketSize);
                int end = (int)Math.Min(n, Math.Floor((b + 1) * bucketSize));
                if (end <= start) continue;

                int minIdx = start, maxIdx = start;
                double minV = ys[start], maxV = ys[start];

                for (int i = start + 1; i < end; i++)
                {
                    double v = ys[i];
                    if (v < minV) { minV = v; minIdx = i; }
                    if (v > maxV) { maxV = v; maxIdx = i; }
                }

                if (minIdx <= maxIdx)
                {
                    xList.Add(xs[minIdx]); yList.Add(ys[minIdx]);
                    xList.Add(xs[maxIdx]); yList.Add(ys[maxIdx]);
                }
                else
                {
                    xList.Add(xs[maxIdx]); yList.Add(ys[maxIdx]);
                    xList.Add(xs[minIdx]); yList.Add(ys[minIdx]);
                }
            }

            dx = xList.ToArray();
            dy = yList.ToArray();
        }

        // ===========================
        // Corr grid coloring
        // ===========================
        private void ColorizeCorrGrid()
        {
            if (gridCorr.DataSource is System.Data.DataTable dt)
            {
                for (int r = 0; r < dt.Rows.Count; r++)
                    for (int c = 1; c < dt.Columns.Count; c++)
                    {
                        var cell = gridCorr.Rows[r].Cells[c];
                        if (dt.Rows[r][c] == DBNull.Value)
                        {
                            cell.Style.BackColor = Color.Gainsboro;
                            cell.Style.ForeColor = Color.DimGray;
                            cell.Style.Alignment = DataGridViewContentAlignment.MiddleCenter;
                            continue;
                        }
                        double d = (double)dt.Rows[r][c];
                        cell.Style.BackColor = CorrColor(d);
                        cell.Style.ForeColor = Color.Black;
                        cell.Style.Format = "F3";
                        cell.Style.Alignment = DataGridViewContentAlignment.MiddleCenter;
                    }
            }
        }
        private Color CorrColor(double v)
        {
            v = Math.Max(-1, Math.Min(1, v));
            if (v >= 0) return LerpColor(Color.White, Color.FromArgb(255, 80, 80), v);
            else return LerpColor(Color.FromArgb(80, 120, 255), Color.White, v + 1.0);
        }
        private Color LerpColor(Color a, Color b, double t)
        {
            int r = (int)Math.Round(a.R + (b.R - a.R) * t);
            int g = (int)Math.Round(a.G + (b.G - a.G) * t);
            int bch = (int)Math.Round(a.B + (b.B - a.B) * t);
            return Color.FromArgb(255, r, g, bch);
        }
    }
}
