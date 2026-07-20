using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Text.Json;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace PHM_Project_DockPanel.UI.DataAnalysis
{
    /// <summary>
    /// AI 탭 — 딥러닝(AE-CNN/CLS-CNN) + IsolationForest 모델을 로컬 또는 Airflow로 학습합니다.
    /// (2026-xx 개정: 단일축 고전 ML(kNN/IsoForest/OCSVM 등) 학습·검증·평가 탭은 제거됨.
    ///  단일축 통계 모델이 필요하면 server/phm_scripts/train_model.py 를 직접 사용하세요.)
    /// </summary>
    public class AIForm : DockContent
    {
        // ── 데이터 소스 / 클래스 ─────────────────────────────────────────────
        private TextBox          _dlDataDir, _dlOutputPath, _dlLabelColumn, _dlLr;
        private RadioButton      _dlRdoAccel, _dlRdoTorque;
        private CheckBox         _dlChX, _dlChY, _dlChZ;         // Accel 채널
        private CheckBox[]       _dlChTrq;                        // Torque 채널 (Trq(%))
        private Panel            _dlChPanel;                      // 채널 선택 컨테이너
        private CheckBox         _dlChkNormal;
        private CheckBox         _dlChkOverload;
        private CheckBox         _dlChkLooseness;
        private CheckBox         _dlChkOverspeed;

        // ── 모델 설정 / 출력 ─────────────────────────────────────────────────
        private NumericUpDown    _dlWindowSize, _dlStride, _dlEpochs, _dlBatch, _dlValSplit, _dlThresholdPct, _dlBaseFilters;
        private RadioButton      _dlRdoCls, _dlRdoAe;   // DL 모델 유형: 분류(CLS) / AE
        private Label            _dlClassListLbl;        // "결함 클래스:" ↔ "정상 클래스:"
        private TextBox          _dlPythonPath;           // Python 실행 경로 (로컬 학습용)

        // ── 실행 / 로그 ──────────────────────────────────────────────────────
        private Button           _dlBtnTrain, _dlBtnStop, _dlBtnVenv, _dlBtnBatch;
        private RichTextBox      _dlLog;
        private ProgressBar      _dlProgress;
        private Label            _dlStatus;
        private System.Diagnostics.Process _dlProc;

        // ── 설정 영속화 ──────────────────────────────────────────────────────
        private ComboBox         _dlPresetCombo;
        private static readonly string DlSettingsFile = @"C:\Data\PHM_Logs\dl_settings.json";
        private static readonly string DlPresetsDir   = @"C:\Data\PHM_Logs\dl_presets";

        // ── Airflow 패널 ─────────────────────────────────────────────────────
        private TextBox  _aflUrl, _aflDagId;
        private Button   _aflBtnTrigger, _aflBtnStatus;
        private Label    _aflStatusLbl;
        private TextBox  _aflProfile;     // 저장 프로파일명 (예: default, v2026-05-11)
        private TextBox  _aflProfileLabel;// 프로파일 표시 이름 (선택)
        private string   _aflLastRunId;

        private FlowLayoutPanel _aflModelFlow;
        private CheckBox _aflChkAll;                                      // 전체 선택
        private CheckBox _aflChkAeAccelG, _aflChkAeTorqueG,              // AE 전역
                         _aflChkAeTorqueAx,                              // AE 토크 축별
                         _aflChkAeCombG,   _aflChkAeCombAx;             // AE 결합
        private CheckBox _aflChkIsoAccel;                                // Accel IsolationForest (AE의 선택 가능한 대안)
        private CheckBox _aflChkClsAccel,  _aflChkClsTorque,            // CLS
                         _aflChkClsComb;
        private bool     _aflUpdatingAll;

        // 축별 학습 — 단일 축만 학습하려면 ComboBox 에서 "Ax{N}" 선택
        // (전체 선택 시 DAG 가 axis_count 를 자동 감지해 모든 축 학습)
        private ComboBox _aflAxisCombo;

        // ── 자동 재학습 (별도 DAG: phm_auto_retrain — On/Off + 주기(스케줄) 모두 여기서 제어) ──
        private const string AutoRetrainDagId       = "phm_auto_retrain";
        private const string AutoRetrainScheduleVar = "phm_auto_retrain_schedule";
        private CheckBox _aflChkAutoRetrain;
        private Label    _aflAutoStatusLbl;
        private bool     _aflSyncingAutoRetrain;   // 상태 조회로 체크박스를 동기화할 때 CheckedChanged 재호출 방지
        private ComboBox _aflSchedulePreset;       // 자주 쓰는 주기 프리셋 → cron 텍스트박스 채움
        private TextBox  _aflScheduleCron;         // 실제 적용되는 cron 표현식 (직접 수정 가능)
        private Button   _aflBtnScheduleApply;
        private bool     _aflSyncingSchedulePreset;

        private static readonly (string Label, string Cron)[] AutoRetrainSchedulePresets =
        {
            ("매일 새벽 2시",        "0 2 * * *"),
            ("매일 새벽 4시",        "0 4 * * *"),
            ("6시간마다",            "0 */6 * * *"),
            ("12시간마다",           "0 */12 * * *"),
            ("매주 월요일 새벽 2시", "0 2 * * 1"),
            ("사용자 지정 (cron)",   null),
        };

        public AIForm()
        {
            Text = "AI";
            BuildUI();
        }

        // =============== UI 구성 ===============
        private void BuildUI()
        {
            var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3 };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 340));  // 설정 패널
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 212));  // Airflow 패널 (+26px 자동 재학습 On/Off, +26px 주기 지정)
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));   // 로그 영역

            // ── 상단: 설정 2열 ────────────────────────────────────────────
            var top2 = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, Padding = new Padding(6, 4, 6, 0) };
            top2.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            top2.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));

            top2.Controls.Add(BuildDlLeftPanel(),  0, 0);
            top2.Controls.Add(BuildDlRightPanel(), 1, 0);

            // ── 하단: 로그 + 진행 ─────────────────────────────────────────
            var botLayout = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3, Padding = new Padding(6, 0, 6, 6) };
            botLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 26)); // 상태 + 프로그레스바
            botLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 26)); // 버튼
            botLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); // 로그

            // 프로그레스 행
            var progRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            _dlStatus   = new Label { AutoSize = true, Text = "대기 중", Margin = new Padding(0, 4, 8, 0) };
            _dlProgress = new ProgressBar { Width = 300, Height = 20, Minimum = 0, Maximum = 100 };
            progRow.Controls.Add(_dlStatus);
            progRow.Controls.Add(_dlProgress);

            // 버튼 행
            var btnRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            _dlBtnTrain = new Button { Text = "▶ 학습 시작", Width = 110, Height = 24, BackColor = Color.FromArgb(0, 120, 212), ForeColor = Color.White, FlatStyle = FlatStyle.Flat };
            _dlBtnBatch = new Button { Text = "⚡ 전체 축 일괄", Width = 115, Height = 24, BackColor = Color.FromArgb(16, 110, 60), ForeColor = Color.White, FlatStyle = FlatStyle.Flat };
            _dlBtnStop  = new Button { Text = "■ 중지",          Width = 70,  Height = 24, Enabled = false };
            _dlBtnVenv  = new Button { Text = "🐍 가상환경",     Width = 100, Height = 24 };
            _dlBtnTrain.Click += (s, e) => StartDlTrainingAsync();
            _dlBtnBatch.Click += (s, e) => StartBatchTrainingAsync();
            _dlBtnStop.Click  += (s, e) => StopDlTraining();
            _dlBtnVenv.Click  += (s, e) => RunSetupVenv();

            // ── 프리셋 저장 / 불러오기 ─────────────────────────────────────
            var sep = new Label { Text = "|", AutoSize = true, ForeColor = Color.FromArgb(190, 190, 210),
                Font = new Font("Segoe UI", 10f), Margin = new Padding(6, 2, 6, 0) };
            var btnPresetSave = new Button { Text = "💾 프리셋 저장", Width = 100, Height = 24 };
            btnPresetSave.Click += (s, e) => ShowSavePresetDialog();
            _dlPresetCombo = new ComboBox { Width = 130, Height = 24, DropDownStyle = ComboBoxStyle.DropDownList };
            var btnPresetLoad = new Button { Text = "📂 불러오기", Width = 85, Height = 24 };
            btnPresetLoad.Click += (s, e) =>
            {
                if (_dlPresetCombo.SelectedItem is string name && !string.IsNullOrEmpty(name))
                    LoadDlSettings(System.IO.Path.Combine(DlPresetsDir, name + ".json"));
            };

            btnRow.Controls.Add(_dlBtnTrain);
            btnRow.Controls.Add(_dlBtnBatch);
            btnRow.Controls.Add(_dlBtnStop);
            btnRow.Controls.Add(_dlBtnVenv);
            btnRow.Controls.Add(sep);
            btnRow.Controls.Add(btnPresetSave);
            btnRow.Controls.Add(_dlPresetCombo);
            btnRow.Controls.Add(btnPresetLoad);

            // 로그
            _dlLog = new RichTextBox
            {
                Dock = DockStyle.Fill, ReadOnly = true, BackColor = Color.FromArgb(30, 30, 30),
                ForeColor = Color.LightGreen, Font = new Font("Consolas", 9f),
                ScrollBars = RichTextBoxScrollBars.Vertical
            };

            botLayout.Controls.Add(progRow,  0, 0);
            botLayout.Controls.Add(btnRow,   0, 1);
            botLayout.Controls.Add(_dlLog,   0, 2);

            root.Controls.Add(top2,               0, 0);
            root.Controls.Add(BuildAirflowPanel(), 0, 1);
            root.Controls.Add(botLayout,          0, 2);

            // 서버 설정 변경 시 Airflow 패널 URL·DagId 동기화
            Services.AppEvents.ServerSettingsChanged += s =>
            {
                if (IsDisposed) return;
                if (InvokeRequired) { BeginInvoke(new Action(() => SyncAirflowPanel(s))); return; }
                SyncAirflowPanel(s);
            };

            Controls.Add(root);
            ClientSize = new Size(1180, 800);

            // 컨트롤 생성 완료 후 마지막 설정 복원 (자동 저장된 dl_settings.json)
            LoadDlSettings(DlSettingsFile);
            RefreshPresetCombo();

            // 자동 재학습(phm_auto_retrain) DAG의 현재 On/Off 상태를 서버에서 조회해 동기화
            _ = RefreshAutoRetrainStatusAsync();
        }

        private GroupBox BuildDlLeftPanel()
        {
            var grp = new GroupBox { Text = "데이터 소스 / 클래스", Dock = DockStyle.Fill, Padding = new Padding(8) };
            var tl  = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 6 };
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 90));
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            for (int i = 0; i < 5; i++) tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));
            tl.RowStyles.Add(new RowStyle(SizeType.Percent, 100));   // 클래스 리스트

            int row = 0;

            // 데이터 폴더
            tl.Controls.Add(Lbl("데이터 폴더:"), 0, row);
            _dlDataDir = new TextBox { Dock = DockStyle.Fill, Text = @"C:\Data\PHM_Logs\Signals" };
            var dirRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            _dlDataDir.Width = 150;
            dirRow.Controls.Add(_dlDataDir);
            var btnBrowseDir = new Button { Text = "…", Width = 26, Height = 22 };
            btnBrowseDir.Click += (s, e) => {
                using (var fbd = new FolderBrowserDialog { SelectedPath = _dlDataDir.Text })
                    if (fbd.ShowDialog() == DialogResult.OK) { _dlDataDir.Text = fbd.SelectedPath; ScanDataFolder(); }
            };
            var btnScan = new Button { Text = "🔍", Width = 28, Height = 22, Font = new Font(Font.FontFamily, 8.5f) };
            btnScan.Click += (s, e) => ScanDataFolder();
            // 연속 수집 폴더 바로가기
            var btnContDir = new Button { Text = "📁수집", Width = 52, Height = 22, Font = new Font(Font.FontFamily, 8f),
                BackColor = Color.FromArgb(0, 100, 160), ForeColor = Color.White, FlatStyle = FlatStyle.Flat };
            btnContDir.Click += (s, e) => {
                _dlDataDir.Text = @"C:\Data\PHM_Logs\Signals";
                if (System.IO.Directory.Exists(_dlDataDir.Text)) ScanDataFolder();
            };
            dirRow.Controls.AddRange(new Control[] { btnBrowseDir, btnScan, btnContDir });
            tl.Controls.Add(dirRow, 1, row++);

            // 신호 타입 (Accel / Torque)
            tl.Controls.Add(Lbl("신호 타입:"), 0, row);
            var sensorRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight };
            _dlRdoAccel  = new RadioButton { Text = "가속도계",  Checked = true, AutoSize = true };
            _dlRdoTorque = new RadioButton { Text = "토크",      Checked = false, AutoSize = true, Margin = new Padding(8,0,0,0) };
            sensorRow.Controls.AddRange(new Control[] { _dlRdoAccel, _dlRdoTorque });
            tl.Controls.Add(sensorRow, 1, row++);

            // 입력 채널 (동적 패널)
            tl.Controls.Add(Lbl("입력 채널:"), 0, row);
            _dlChPanel = new Panel { Dock = DockStyle.Fill };

            // Accel 채널 패널
            var accelFlow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, Name = "accelFlow" };
            _dlChX = new CheckBox { Text = "x", Checked = true, AutoSize = true };
            _dlChY = new CheckBox { Text = "y", Checked = true, AutoSize = true };
            _dlChZ = new CheckBox { Text = "z", Checked = true, AutoSize = true };
            accelFlow.Controls.AddRange(new Control[] { _dlChX, _dlChY, _dlChZ });

            // Torque 채널 패널 (토크 CSV는 Trq(%) 만 저장)
            var trqFlow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, Name = "trqFlow", Visible = false };
            var trqCols = new[] { "Trq(%)" };
            _dlChTrq = new CheckBox[trqCols.Length];
            for (int i = 0; i < trqCols.Length; i++)
            {
                _dlChTrq[i] = new CheckBox { Text = trqCols[i], Checked = true, AutoSize = true, Margin = new Padding(0,2,6,0) };
                trqFlow.Controls.Add(_dlChTrq[i]);
            }

            _dlChPanel.Controls.AddRange(new Control[] { accelFlow, trqFlow });
            tl.Controls.Add(_dlChPanel, 1, row++);

            // 신호 타입 전환 이벤트
            _dlRdoAccel.CheckedChanged += (s, e) =>
            {
                if (!_dlRdoAccel.Checked) return;
                accelFlow.Visible = true; trqFlow.Visible = false;
                // 출력 파일명 힌트 업데이트
                if (_dlOutputPath != null && _dlOutputPath.Text.Contains("torque"))
                    _dlOutputPath.Text = _dlOutputPath.Text.Replace("torque", "accel");
            };
            _dlRdoTorque.CheckedChanged += (s, e) =>
            {
                if (!_dlRdoTorque.Checked) return;
                accelFlow.Visible = false; trqFlow.Visible = true;
                if (_dlOutputPath != null && !_dlOutputPath.Text.Contains("torque"))
                    _dlOutputPath.Text = System.IO.Path.Combine(
                        System.IO.Path.GetDirectoryName(_dlOutputPath.Text) ?? "",
                        "cnn1d_torque.onnx");
            };

            // Label 컬럼
            tl.Controls.Add(Lbl("Label 컬럼:"), 0, row);
            _dlLabelColumn = new TextBox { Dock = DockStyle.Fill, Text = "Label" };
            tl.Controls.Add(_dlLabelColumn, 1, row++);


            // ── 결함 클래스 체크박스 (채널과 동일한 스타일) ───────────────────
            _dlClassListLbl = Lbl("결함 클래스:");
            tl.Controls.Add(_dlClassListLbl, 0, row);

            var clsFlow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false,
                AutoSize = true
            };
            _dlChkNormal    = new CheckBox { Text = "normal",    Checked = true,  AutoSize = true };
            _dlChkOverload  = new CheckBox { Text = "overload",  Checked = false, AutoSize = true };
            _dlChkLooseness = new CheckBox { Text = "looseness", Checked = true,  AutoSize = true };
            _dlChkOverspeed = new CheckBox { Text = "overspeed", Checked = false, AutoSize = true };
            clsFlow.Controls.AddRange(new Control[] { _dlChkNormal, _dlChkOverload, _dlChkLooseness, _dlChkOverspeed });
            tl.Controls.Add(clsFlow, 1, row++);

            grp.Controls.Add(tl);
            return grp;
        }

        /// <summary>
        /// 데이터 폴더를 스캔하여 클래스 목록·신호 타입·채널·축 번호·출력 경로를 자동 설정합니다.
        /// 기대 구조: {root}/{클래스명}/{Accel|Torque}/[{device}/]*.csv
        /// </summary>
        private void ScanDataFolder()
        {
            string root = _dlDataDir?.Text?.Trim() ?? "";
            if (!System.IO.Directory.Exists(root))
            {
                MessageBox.Show("폴더가 존재하지 않습니다:\n" + root, "스캔 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            // ── 1. 클래스 디렉터리 탐색 (첫 번째 레벨 서브폴더 중 Accel/Torque를 포함한 것) ──
            var classDirs = System.IO.Directory.GetDirectories(root)
                .Where(d =>
                    System.IO.Directory.GetDirectories(d, "Accel", System.IO.SearchOption.TopDirectoryOnly).Any() ||
                    System.IO.Directory.GetDirectories(d, "Torque", System.IO.SearchOption.TopDirectoryOnly).Any())
                .Select(d => System.IO.Path.GetFileName(d))
                .OrderBy(n => n)
                .ToList();

            if (classDirs.Count == 0)
            {
                MessageBox.Show(
                    "클래스 폴더를 찾지 못했습니다.\n\n" +
                    "기대 구조:\n  {루트}/{클래스명}/Accel/*.csv\n  {루트}/{클래스명}/Torque/*.csv",
                    "스캔 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            // ── 2. 신호 타입 감지 ────────────────────────────────────────────────────
            bool hasAccel  = System.IO.Directory.GetDirectories(root, "Accel",  System.IO.SearchOption.AllDirectories).Any();
            bool hasTorque = System.IO.Directory.GetDirectories(root, "Torque", System.IO.SearchOption.AllDirectories).Any();
            bool useAccel  = hasAccel; // Accel 우선

            // ── 3. 대표 CSV 헤더 읽기 ───────────────────────────────────────────────
            string sampleCsv = System.IO.Directory
                .EnumerateFiles(root, "*.csv", System.IO.SearchOption.AllDirectories)
                .FirstOrDefault(f =>
                {
                    string seg = System.IO.Path.GetDirectoryName(f) ?? "";
                    return useAccel
                        ? seg.IndexOf(System.IO.Path.DirectorySeparatorChar + "Accel", StringComparison.OrdinalIgnoreCase) >= 0
                        : seg.IndexOf(System.IO.Path.DirectorySeparatorChar + "Torque", StringComparison.OrdinalIgnoreCase) >= 0;
                });

            string[] headers = new string[0];
            if (sampleCsv != null)
            {
                try
                {
                    string headerLine = System.IO.File.ReadLines(sampleCsv).FirstOrDefault() ?? "";
                    headers = headerLine.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None)
                                        .Select(h => h.Trim()).ToArray();
                }
                catch { }
            }

            // ── 4. 축 번호 추출 (폴더명 "Axis0" → 0) ────────────────────────────────
            var axisMatch = System.Text.RegularExpressions.Regex.Match(
                System.IO.Path.GetFileName(root), @"[Aa]xis(\d+)");
            int axisNum = axisMatch.Success ? int.Parse(axisMatch.Groups[1].Value) : 0;

            // ── 5. UI 적용 ────────────────────────────────────────────────────────────
            // 클래스 리스트
            // 미리 정의된 클래스는 체크박스로 반영
            var knownMap = new System.Collections.Generic.Dictionary<string, CheckBox>(StringComparer.OrdinalIgnoreCase)
            {
                { "normal",    _dlChkNormal    },
                { "overload",  _dlChkOverload  },
                { "looseness", _dlChkLooseness },
                { "overspeed", _dlChkOverspeed },
            };
            foreach (var kv in knownMap) kv.Value.Checked = false;

            foreach (var c in classDirs)
            {
                if (knownMap.TryGetValue(c, out CheckBox chk))
                    chk.Checked = true;
            }

            // 신호 타입 + 채널
            if (useAccel)
            {
                _dlRdoAccel.Checked = true;
                if (_dlChX != null) _dlChX.Checked = headers.Any(h => string.Equals(h, "x", StringComparison.OrdinalIgnoreCase));
                if (_dlChY != null) _dlChY.Checked = headers.Any(h => string.Equals(h, "y", StringComparison.OrdinalIgnoreCase));
                if (_dlChZ != null) _dlChZ.Checked = headers.Any(h => string.Equals(h, "z", StringComparison.OrdinalIgnoreCase));
            }
            else
            {
                _dlRdoTorque.Checked = true;
                if (_dlChTrq != null)
                {
                    // 토크 CSV 헤더: time_s, Ax{n}_Trq(%) — "Trq(%)" 접미사 매칭
                    var trqSuffixes = new[] { "Trq(%)" };
                    for (int i = 0; i < _dlChTrq.Length && i < trqSuffixes.Length; i++)
                    {
                        string sfx = trqSuffixes[i];
                        _dlChTrq[i].Checked = headers.Any(h =>
                            System.Text.RegularExpressions.Regex.IsMatch(
                                h, @"^Ax\d+_" + System.Text.RegularExpressions.Regex.Escape(sfx) + @"$",
                                System.Text.RegularExpressions.RegexOptions.IgnoreCase));
                    }
                }
            }

            // Label 컬럼
            bool hasLabelCol = headers.Any(h => string.Equals(h, "Label", StringComparison.OrdinalIgnoreCase));
            if (_dlLabelColumn != null)
                _dlLabelColumn.Text = hasLabelCol ? "Label" : "";

            // 출력 경로 자동 생성
            if (_dlOutputPath != null)
            {
                string modelsDir = System.IO.Path.Combine(
                    System.IO.Path.GetDirectoryName(_dlOutputPath.Text)
                    ?? @"C:\Data\PHM_Logs\models");
                string sigTag   = useAccel ? "accel" : "torque";
                bool   scanIsAe = _dlRdoAe?.Checked == true;
                string scanPfx  = scanIsAe ? "ae_cnn1d" : "cnn1d";
                _dlOutputPath.Text = System.IO.Path.Combine(modelsDir, $"{scanPfx}_axis{axisNum}_{sigTag}.onnx");
            }

            string sigTypeText = useAccel ? "가속도계(Accel)" : "토크(Torque)";
            string bothText    = (hasAccel && hasTorque) ? " (Accel + Torque 모두 존재, Accel 우선)" : "";
            MessageBox.Show(
                $"스캔 완료{bothText}\n\n" +
                $"  신호 타입  : {sigTypeText}\n" +
                $"  클래스 수  : {classDirs.Count}개\n" +
                $"  클래스    : {string.Join(", ", classDirs)}\n" +
                $"  헤더 채널  : {string.Join(", ", headers)}\n" +
                $"  축 번호   : {axisNum}",
                "데이터셋 스캔 결과", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private GroupBox BuildDlRightPanel()
        {
            var grp = new GroupBox { Text = "모델 설정 / 출력", Dock = DockStyle.Fill, Padding = new Padding(8) };
            var tl  = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 11 };
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 120));
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            for (int i = 0; i < 11; i++) tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));

            int row = 0;

            // 모델 유형: 분류(CLS) / AE(이상탐지)
            tl.Controls.Add(Lbl("모델 유형:"), 0, row);
            var modeFlow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight };
            _dlRdoCls = new RadioButton { Text = "분류(CLS)", Checked = true, AutoSize = true };
            _dlRdoAe  = new RadioButton { Text = "AE(이상탐지)", AutoSize = true, Margin = new Padding(8, 0, 0, 0) };
            _dlRdoCls.CheckedChanged += (s, e) => { if (_dlRdoCls.Checked) { UpdateDlModeUi(); UpdateAflTrainModeItems(); UpdateTriggerButtonText(); } };
            _dlRdoAe.CheckedChanged  += (s, e) => { if (_dlRdoAe.Checked)  { UpdateDlModeUi(); UpdateAflTrainModeItems(); UpdateTriggerButtonText(); } };
            modeFlow.Controls.AddRange(new Control[] { _dlRdoCls, _dlRdoAe });
            tl.Controls.Add(modeFlow, 1, row++);

            // 윈도우 크기
            tl.Controls.Add(Lbl("윈도우(샘플):"), 0, row);
            _dlWindowSize = Nud(64, 65536, 256); tl.Controls.Add(_dlWindowSize, 1, row++);

            // 스트라이드
            tl.Controls.Add(Lbl("스트라이드:"), 0, row);
            _dlStride = Nud(1, 65536, 128); tl.Controls.Add(_dlStride, 1, row++);

            // 에포크
            tl.Controls.Add(Lbl("에포크:"), 0, row);
            _dlEpochs = Nud(1, 10000, 30); tl.Controls.Add(_dlEpochs, 1, row++);

            // 배치 크기
            tl.Controls.Add(Lbl("배치 크기:"), 0, row);
            _dlBatch = Nud(1, 1024, 32); tl.Controls.Add(_dlBatch, 1, row++);

            // 학습률
            tl.Controls.Add(Lbl("학습률:"), 0, row);
            _dlLr = new TextBox { Dock = DockStyle.Fill, Text = "0.001" };
            tl.Controls.Add(_dlLr, 1, row++);

            // 검증 비율
            tl.Controls.Add(Lbl("검증 비율(%):"), 0, row);
            _dlValSplit = Nud(5, 50, 20); tl.Controls.Add(_dlValSplit, 1, row++);

            // AE 임계값 퍼센타일 (AE 모드 전용)
            var lblThrPct = Lbl("임계값(%):");
            tl.Controls.Add(lblThrPct, 0, row);
            _dlThresholdPct = new NumericUpDown
            {
                Dock = DockStyle.Fill, Minimum = 90m, Maximum = 100m,
                DecimalPlaces = 1, Increment = 0.5m, Value = 99.5m
            };
            var tipThrPct = new ToolTip();
            tipThrPct.SetToolTip(_dlThresholdPct, "AE 정상 MAE의 N% → threshold 로 사용 (높을수록 민감도↓)");
            tl.Controls.Add(_dlThresholdPct, 1, row++);
            // 베이스 필터 수 (AE 모드 전용)
            var lblBaseFilters = Lbl("베이스필터:");
            tl.Controls.Add(lblBaseFilters, 0, row);
            _dlBaseFilters = new NumericUpDown
            {
                Dock = DockStyle.Fill, Minimum = 4m, Maximum = 128m,
                DecimalPlaces = 0, Increment = 4m, Value = 32m
            };
            var tipBF = new ToolTip();
            tipBF.SetToolTip(_dlBaseFilters,
                "낮출수록 이상 패턴 재구성 실패 확률 ↑ (이상탐지 민감도 ↑)\n" +
                "권장: 8(민감) / 16 / 32(기본) / 64(둔감)");
            tl.Controls.Add(_dlBaseFilters, 1, row++);

            // AE/CLS 모드 전환 시 가시성 연동 (임계값 + 베이스 필터 함께)
            Action syncThrPct = () =>
            {
                bool ae = _dlRdoAe?.Checked ?? true;
                lblThrPct.Enabled        = ae;
                _dlThresholdPct.Enabled  = ae;
                lblThrPct.ForeColor      = ae ? SystemColors.ControlText : SystemColors.GrayText;
                lblBaseFilters.Enabled   = ae;
                _dlBaseFilters.Enabled   = ae;
                lblBaseFilters.ForeColor = ae ? SystemColors.ControlText : SystemColors.GrayText;
            };
            if (_dlRdoAe  != null) _dlRdoAe.CheckedChanged  += (s, e) => syncThrPct();
            if (_dlRdoCls != null) _dlRdoCls.CheckedChanged += (s, e) => syncThrPct();
            syncThrPct();

            // 출력 경로
            tl.Controls.Add(Lbl("출력 모델:"), 0, row);
            _dlOutputPath = new TextBox { Dock = DockStyle.Fill, Text = @"C:\Data\PHM_Logs\models\cnn1d_fd.onnx" };
            var outRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            _dlOutputPath.Width = 180;
            var btnOutPath = new Button { Text = "…", Width = 28, Height = 22 };
            btnOutPath.Click += (s, e) => {
                using (var sfd = new SaveFileDialog { Filter = "ONNX|*.onnx", FileName = System.IO.Path.GetFileName(_dlOutputPath.Text) })
                    if (sfd.ShowDialog() == DialogResult.OK) _dlOutputPath.Text = sfd.FileName;
            };
            outRow.Controls.Add(_dlOutputPath);
            outRow.Controls.Add(btnOutPath);
            tl.Controls.Add(outRow, 1, row++);

            // Python 경로 (로컬 학습 시에만 사용 — .venv 가 있으면 이 값보다 우선함)
            tl.Controls.Add(Lbl("Python:"), 0, row);
            _dlPythonPath = new TextBox { Dock = DockStyle.Fill, Text = "python" };
            var tipPy = new ToolTip();
            tipPy.SetToolTip(_dlPythonPath, "로컬 학습(▶ 학습 시작/⚡ 전체 축 일괄)에 사용할 Python 경로.\nscripts/.venv 가 있으면 이 값보다 우선 사용됩니다.");
            tl.Controls.Add(_dlPythonPath, 1, row++);

            grp.Controls.Add(tl);
            return grp;
        }

        // ════════════════════════════════════════════════════════════════════
        //  설정 영속화 — 자동 저장/복원 & 수동 프리셋
        // ════════════════════════════════════════════════════════════════════

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            SaveDlSettings(DlSettingsFile);  // 항상 마지막 상태 자동 저장
            StopDlTraining();
            base.OnFormClosed(e);
        }

        /// <summary>현재 설정을 JSON 파일로 저장합니다.</summary>
        private void SaveDlSettings(string path)
        {
            try
            {
                System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path));
                bool isTorque  = _dlRdoTorque?.Checked == true;
                bool isAe      = _dlRdoAe?.Checked     == true;
                var trqChecked = _dlChTrq?.Select(c => c.Checked).ToArray() ?? new bool[0];

                var obj = new
                {
                    dataDir      = _dlDataDir?.Text    ?? "",
                    sensorType   = isTorque ? "torque" : "accel",
                    chX          = _dlChX?.Checked     ?? true,
                    chY          = _dlChY?.Checked     ?? true,
                    chZ          = _dlChZ?.Checked     ?? true,
                    chTrq        = trqChecked,
                    labelColumn  = _dlLabelColumn?.Text ?? "Label",
                    clsNormal    = _dlChkNormal?.Checked    ?? true,
                    clsOverload  = _dlChkOverload?.Checked  ?? false,
                    clsLooseness = _dlChkLooseness?.Checked ?? true,
                    clsOverspeed = _dlChkOverspeed?.Checked ?? false,
                    modelType    = isAe ? "AE" : "CLS",
                    windowSize   = (int)NudCurrent(_dlWindowSize, 256m),
                    stride       = (int)NudCurrent(_dlStride,     128m),
                    epochs       = (int)NudCurrent(_dlEpochs,     30m),
                    batch        = (int)NudCurrent(_dlBatch,      32m),
                    lr              = _dlLr?.Text           ?? "0.001",
                    valSplit        = (int)NudCurrent(_dlValSplit,      20m),
                    thresholdPct    = (double)NudCurrent(_dlThresholdPct, 99.5m),
                    baseFilters     = (int)NudCurrent(_dlBaseFilters,    32m),
                    outputPath      = _dlOutputPath?.Text   ?? "",
                    pythonPath      = _dlPythonPath?.Text   ?? "python",
                };
                var json = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true });
                System.IO.File.WriteAllText(path, json, System.Text.Encoding.UTF8);
            }
            catch { }
        }

        /// <summary>JSON 파일에서 설정을 복원합니다.</summary>
        private void LoadDlSettings(string path)
        {
            try
            {
                if (!System.IO.File.Exists(path)) return;
                var txt  = System.IO.File.ReadAllText(path, System.Text.Encoding.UTF8);
                var root = JsonDocument.Parse(txt).RootElement;

                string Str(string key, string def)
                { return root.TryGetProperty(key, out var v) ? v.GetString() ?? def : def; }
                bool Bool(string key, bool def)
                { return root.TryGetProperty(key, out var v) ? v.GetBoolean() : def; }
                int Int(string key, int def)
                { return root.TryGetProperty(key, out var v) && v.TryGetInt32(out int i) ? i : def; }

                if (_dlDataDir    != null) _dlDataDir.Text    = Str("dataDir",     _dlDataDir.Text);
                if (_dlLabelColumn!= null) _dlLabelColumn.Text= Str("labelColumn", "Label");
                if (_dlLr         != null) _dlLr.Text         = Str("lr",          "0.001");
                if (_dlOutputPath != null) _dlOutputPath.Text = Str("outputPath",  _dlOutputPath.Text);
                if (_dlPythonPath != null) _dlPythonPath.Text = Str("pythonPath",  "python");

                bool isTorque = string.Equals(Str("sensorType", "accel"), "torque", StringComparison.OrdinalIgnoreCase);
                if (_dlRdoAccel  != null) _dlRdoAccel.Checked  = !isTorque;
                if (_dlRdoTorque != null) _dlRdoTorque.Checked = isTorque;

                if (_dlChX != null) _dlChX.Checked = Bool("chX", true);
                if (_dlChY != null) _dlChY.Checked = Bool("chY", true);
                if (_dlChZ != null) _dlChZ.Checked = Bool("chZ", true);

                if (_dlChTrq != null && root.TryGetProperty("chTrq", out var trqArr))
                {
                    var arr = trqArr.EnumerateArray().Select(v => v.GetBoolean()).ToArray();
                    for (int i = 0; i < _dlChTrq.Length && i < arr.Length; i++)
                        _dlChTrq[i].Checked = arr[i];
                }

                if (_dlChkNormal   != null) _dlChkNormal.Checked    = Bool("clsNormal",    true);
                if (_dlChkOverload != null) _dlChkOverload.Checked  = Bool("clsOverload",  false);
                if (_dlChkLooseness!= null) _dlChkLooseness.Checked = Bool("clsLooseness", true);
                if (_dlChkOverspeed!= null) _dlChkOverspeed.Checked = Bool("clsOverspeed", false);

                bool isAe = string.Equals(Str("modelType", "CLS"), "AE", StringComparison.OrdinalIgnoreCase);
                if (_dlRdoCls != null) _dlRdoCls.Checked = !isAe;
                if (_dlRdoAe  != null) _dlRdoAe.Checked  = isAe;

                void SetNud(NumericUpDown nud, int val)
                { if (nud != null) nud.Value = Math.Max(nud.Minimum, Math.Min(nud.Maximum, val)); }
                void SetNudD(NumericUpDown nud, double val)
                { if (nud != null) nud.Value = (decimal)Math.Max((double)nud.Minimum, Math.Min((double)nud.Maximum, val)); }
                SetNud(_dlWindowSize, Int("windowSize", 256));
                SetNud(_dlStride,     Int("stride",     128));
                SetNud(_dlEpochs,     Int("epochs",     30));
                SetNud(_dlBatch,      Int("batch",      32));
                SetNud(_dlValSplit,   Int("valSplit",   20));
                double tpct = 99.5;
                if (root.TryGetProperty("thresholdPct", out var tprop)) tprop.TryGetDouble(out tpct);
                SetNudD(_dlThresholdPct, tpct);
                int bfilt = 32;
                if (root.TryGetProperty("baseFilters", out var bfProp)) bfProp.TryGetInt32(out bfilt);
                SetNud(_dlBaseFilters, bfilt);
            }
            catch { }
        }

        /// <summary>dl_presets 폴더의 프리셋 목록으로 콤보박스를 채웁니다.</summary>
        private void RefreshPresetCombo()
        {
            if (_dlPresetCombo == null) return;
            try
            {
                _dlPresetCombo.Items.Clear();
                if (!System.IO.Directory.Exists(DlPresetsDir)) return;
                foreach (var f in System.IO.Directory.GetFiles(DlPresetsDir, "*.json").OrderBy(x => x))
                    _dlPresetCombo.Items.Add(System.IO.Path.GetFileNameWithoutExtension(f));
                if (_dlPresetCombo.Items.Count > 0) _dlPresetCombo.SelectedIndex = 0;
            }
            catch { }
        }

        /// <summary>현재 설정을 이름 붙여 프리셋으로 저장합니다.</summary>
        private void ShowSavePresetDialog()
        {
            // 이름 입력 미니 다이얼로그
            var dlg = new Form
            {
                Text = "프리셋 저장", Width = 320, Height = 120,
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition = FormStartPosition.CenterParent,
                MaximizeBox = false, MinimizeBox = false,
                Font = new Font("Segoe UI", 9f), BackColor = Color.White
            };
            var lbl = new Label { Text = "프리셋 이름:", Left = 12, Top = 14, AutoSize = true };
            var txt = new TextBox { Left = 12, Top = 32, Width = 276, Text = "preset_" + DateTime.Now.ToString("MMdd_HHmm") };
            var btnOk = new Button
            {
                Text = "저장", DialogResult = DialogResult.OK,
                Left = 128, Top = 58, Width = 76, Height = 26,
                BackColor = Color.FromArgb(0, 120, 212), ForeColor = Color.White, FlatStyle = FlatStyle.Flat
            };
            btnOk.FlatAppearance.BorderSize = 0;
            var btnCancel = new Button
            {
                Text = "취소", DialogResult = DialogResult.Cancel,
                Left = 212, Top = 58, Width = 76, Height = 26, FlatStyle = FlatStyle.Flat
            };
            dlg.Controls.AddRange(new Control[] { lbl, txt, btnOk, btnCancel });
            dlg.AcceptButton = btnOk; dlg.CancelButton = btnCancel;

            if (dlg.ShowDialog(this) != DialogResult.OK) return;
            string name = txt.Text.Trim();
            if (string.IsNullOrEmpty(name)) return;
            // 파일명에 사용 불가한 문자 제거
            foreach (char c in System.IO.Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');

            try
            {
                System.IO.Directory.CreateDirectory(DlPresetsDir);
                string presetPath = System.IO.Path.Combine(DlPresetsDir, name + ".json");
                SaveDlSettings(presetPath);
                RefreshPresetCombo();
                // 방금 저장한 항목 선택
                int idx = _dlPresetCombo.Items.IndexOf(name);
                if (idx >= 0) _dlPresetCombo.SelectedIndex = idx;
                AppendDlLog($"[프리셋] 저장 완료: {name}.json");
            }
            catch (Exception ex) { MessageBox.Show("저장 실패: " + ex.Message); }
        }

        /// <summary>모델 유형(CLS/AE) 전환 시 관련 UI를 동기화합니다.</summary>
        private void UpdateDlModeUi()
        {
            bool isAe = _dlRdoAe?.Checked == true;

            // 클래스 레이블 전환
            if (_dlClassListLbl != null)
                _dlClassListLbl.Text = isAe ? "정상 클래스:" : "결함 클래스:";

            // 출력 경로 파일명 접두사 ae_ 추가/제거
            if (_dlOutputPath != null)
            {
                string dir = System.IO.Path.GetDirectoryName(_dlOutputPath.Text) ?? "";
                string fn  = System.IO.Path.GetFileNameWithoutExtension(_dlOutputPath.Text);
                if (isAe && !fn.StartsWith("ae_", StringComparison.OrdinalIgnoreCase))
                    _dlOutputPath.Text = System.IO.Path.Combine(dir, "ae_" + fn + ".onnx");
                else if (!isAe && fn.StartsWith("ae_", StringComparison.OrdinalIgnoreCase))
                    _dlOutputPath.Text = System.IO.Path.Combine(dir, fn.Substring(3) + ".onnx");
            }
        }

        private static Label Lbl(string text) =>
            new Label { Text = text, AutoSize = false, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft };

        /// <summary>AutoSize 컬럼용 짧은 레이블 (TableLayoutPanel의 AutoSize 컬럼과 Dock=Fill을 같이 쓰면
        /// 컬럼 너비가 0으로 붕괴하므로, 이런 자리에는 AutoSize 레이블을 따로 둔다.)</summary>
        private static Label Lbl2(string text) =>
            new Label { Text = text, AutoSize = true, TextAlign = ContentAlignment.MiddleLeft, Margin = new Padding(0, 7, 4, 0) };

        private static NumericUpDown Nud(int min, int max, int val) =>
            new NumericUpDown { Minimum = min, Maximum = max, Value = val, Dock = DockStyle.Fill };

        // ── 로컬 학습 실행 ────────────────────────────────────────────────────

        /// <summary>
        /// NumericUpDown 의 현재 사용자 입력 값을 가져옵니다.
        /// .Value 는 Enter/Tab 등으로 포커스를 잃을 때만 commit 되므로,
        /// 사용자가 텍스트만 변경한 채 트리거하면 옛 값이 반환되는 문제가 있습니다.
        /// 이 헬퍼는 .Text 를 우선 파싱하고, 실패 시 .Value 로 폴백합니다.
        /// </summary>
        private static decimal NudCurrent(NumericUpDown nud, decimal fallback)
        {
            if (nud == null) return fallback;
            string txt = (nud.Text ?? "").Trim();
            if (!string.IsNullOrEmpty(txt) &&
                decimal.TryParse(txt,
                    System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.CurrentCulture,
                    out decimal v))
            {
                if (v < nud.Minimum) v = nud.Minimum;
                if (v > nud.Maximum) v = nud.Maximum;
                return v;
            }
            return nud.Value;
        }

        /// <summary>현재 UI 설정을 기반으로 학습 params를 빌드합니다. 실패 시 null 반환.</summary>
        private Dictionary<string, object> BuildDlParams(string dataDir, string outputPath)
        {
            var channels = new List<string>();
            bool isTorque = _dlRdoTorque?.Checked == true;
            if (isTorque) { if (_dlChTrq != null) foreach (var cb in _dlChTrq) if (cb.Checked) channels.Add(cb.Text); }
            else { if (_dlChX.Checked) channels.Add("x"); if (_dlChY.Checked) channels.Add("y"); if (_dlChZ.Checked) channels.Add("z"); }
            if (channels.Count == 0) return null;

            bool isAe = _dlRdoAe?.Checked == true;
            // 체크박스에서 선택된 클래스 수집 (표시 순서 유지)
            var classNames = new System.Collections.Generic.List<string>();
            var clsTogglePairs = new (CheckBox Chk, string Name)[]
            {
                (_dlChkNormal,    "normal"),
                (_dlChkOverload,  "overload"),
                (_dlChkLooseness, "looseness"),
                (_dlChkOverspeed, "overspeed"),
            };
            foreach (var pair in clsTogglePairs)
                if (pair.Chk != null && pair.Chk.Checked) classNames.Add(pair.Name);
            // CLS: 클래스 2개 이상, AE: 정상 클래스 1개 이상
            if (!isAe && classNames.Count < 2) return null;
            if (isAe  && classNames.Count < 1) return null;

            if (!double.TryParse(_dlLr.Text.Trim(), System.Globalization.NumberStyles.Float,
                CultureInfo.InvariantCulture, out double lr) || lr <= 0) return null;

            var p = new Dictionary<string, object>
            {
                ["data_dir"]            = dataDir,
                ["output"]              = outputPath,
                ["channels"]            = channels.ToArray(),
                ["sensor_type"]         = isTorque ? "torque" : "accel",
                ["label_column"]        = _dlLabelColumn?.Text?.Trim() ?? "Label",
                ["class_names"]         = classNames.ToArray(),
                ["window_size"]         = (int)NudCurrent(_dlWindowSize, 256m),
                ["stride"]              = (int)NudCurrent(_dlStride,     128m),
                ["epochs"]              = (int)NudCurrent(_dlEpochs,     30m),
                ["batch_size"]          = (int)NudCurrent(_dlBatch,      32m),
                ["lr"]                  = lr,
                ["val_split"]           = (double)NudCurrent(_dlValSplit, 20m) / 100.0,
                ["seed"]                = 42,
                ["mlflow_tracking_uri"] = Services.ServerSettings.Current.MlflowUrl ?? "",
                ["mlflow_experiment"]   = "PHM-DL",
                ["session"]             = isAe ? "AE" : "CLS",
            };
            // AE: 어떤 폴더를 "정상"으로 볼지 명시 + threshold percentile 상향
            if (isAe)
            {
                p["normal_classes"]          = classNames.ToArray();
                p["ae_threshold_percentile"] = (double)NudCurrent(_dlThresholdPct, 99.5m);
                p["ae_base_filters"]         = (int)NudCurrent(_dlBaseFilters,    32m);
            }
            return p;
        }

        /// <summary>데이터 폴더에서 Axis* 하위 폴더를 스캔해 일괄 학습합니다.</summary>
        private async void StartBatchTrainingAsync()
        {
            string currentDir = _dlDataDir?.Text?.Trim() ?? "";
            if (!System.IO.Directory.Exists(currentDir))
            { MessageBox.Show("데이터 폴더가 없습니다.", "오류", MessageBoxButtons.OK, MessageBoxIcon.Error); return; }

            // Axis 폴더 탐색: 현재 폴더 우선 → 부모 폴더 차선 → 단일 폴더 fallback
            // 우선순위 ①: currentDir 안에 Axis* 하위폴더 (e.g. train\20260414_Axis0)
            var axisDirs = System.IO.Directory.GetDirectories(currentDir)
                .Where(d => System.Text.RegularExpressions.Regex.IsMatch(
                    System.IO.Path.GetFileName(d), @"[Aa]xis\d+"))
                .OrderBy(d => d)
                .ToList();

            if (axisDirs.Count == 0)
            {
                // 우선순위 ②: 부모 폴더 안에 Axis* 폴더 (currentDir 자체가 Axis 폴더인 경우)
                string parentDir = System.IO.Path.GetDirectoryName(currentDir) ?? currentDir;
                if (parentDir != currentDir && System.IO.Directory.Exists(parentDir))
                {
                    axisDirs = System.IO.Directory.GetDirectories(parentDir)
                        .Where(d => System.Text.RegularExpressions.Regex.IsMatch(
                            System.IO.Path.GetFileName(d), @"[Aa]xis\d+"))
                        .OrderBy(d => d)
                        .ToList();
                }
            }

            if (axisDirs.Count == 0)
                axisDirs = new List<string> { currentDir }; // 우선순위 ③: 단일 폴더

            var dlg = MessageBox.Show(
                $"다음 {axisDirs.Count}개 축을 순차 학습합니다:\n\n" +
                string.Join("\n", axisDirs.Select(d => "  • " + System.IO.Path.GetFileName(d))) +
                "\n\n계속할까요?",
                "일괄 학습 확인", MessageBoxButtons.OKCancel, MessageBoxIcon.Question);
            if (dlg != DialogResult.OK) return;

            // Python + 스크립트 확인 (공통)
            string scriptsDir = System.IO.Path.Combine(
                System.IO.Path.GetDirectoryName(Application.ExecutablePath) ?? ".", "scripts");
            string venvPython = System.IO.Path.Combine(scriptsDir, ".venv", "Scripts", "python.exe");
            string python = System.IO.File.Exists(venvPython) ? venvPython : FindPythonExe(_dlPythonPath?.Text?.Trim() ?? "");
            if (python == null) { MessageBox.Show("Python을 찾을 수 없습니다.", "오류", MessageBoxButtons.OK, MessageBoxIcon.Error); return; }
            string scriptPath = System.IO.Path.Combine(scriptsDir, "train_dl_model.py");
            if (!System.IO.File.Exists(scriptPath)) { MessageBox.Show("train_dl_model.py 없음.", "오류", MessageBoxButtons.OK, MessageBoxIcon.Error); return; }

            string modelsDir = System.IO.Path.GetDirectoryName(_dlOutputPath?.Text ?? "") ?? @"C:\Data\PHM_Logs\models";
            bool isTorque = _dlRdoTorque?.Checked == true;
            string sigTag = isTorque ? "torque" : "accel";

            _dlBtnTrain.Enabled = false; _dlBtnBatch.Enabled = false; _dlBtnStop.Enabled = true;
            _dlLog.Clear();

            int total = axisDirs.Count, done = 0;
            foreach (var axisDir in axisDirs)
            {
                if (_dlProc != null) break; // 중지 체크

                string axisName = System.IO.Path.GetFileName(axisDir);
                var axisMatch = System.Text.RegularExpressions.Regex.Match(axisName, @"[Aa]xis(\d+)");
                int axisNum = axisMatch.Success ? int.Parse(axisMatch.Groups[1].Value) : done;
                bool   batchIsAe = _dlRdoAe?.Checked == true;
                string batchPfx  = batchIsAe ? "ae_cnn1d" : "cnn1d";
                string outputPath = System.IO.Path.Combine(modelsDir, $"{batchPfx}_axis{axisNum}_{sigTag}.onnx");
                try { System.IO.Directory.CreateDirectory(modelsDir); } catch { }

                var paramsObj = BuildDlParams(axisDir, outputPath);
                if (paramsObj == null)
                {
                    AppendDlLog($"[SKIP] {axisName} — params 빌드 실패 (채널/클래스 미설정)", Color.Yellow);
                    done++; continue;
                }

                AppendDlLog($"\n━━━ [{done + 1}/{total}] {axisName} ━━━", Color.Cyan);
                _dlStatus.Text = $"일괄 [{done + 1}/{total}] {axisName}";

                string paramsPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"phm_dl_batch_{axisNum}.json");
                System.IO.File.WriteAllText(paramsPath,
                    JsonSerializer.Serialize(paramsObj, new JsonSerializerOptions { WriteIndented = true }),
                    new System.Text.UTF8Encoding(false));

                int totalEpochs = (int)NudCurrent(_dlEpochs, 30m);
                bool success = await System.Threading.Tasks.Task.Run(() => RunTrainingProcess(python, scriptPath, paramsPath, totalEpochs));

                done++;
                _dlProgress.Value = (int)(done * 100.0 / total);
                if (!success) AppendDlLog($"[FAIL] {axisName}", Color.Red);
                else          AppendDlLog($"[OK]   모델 저장: {outputPath}", Color.LightGreen);
            }

            _dlBtnTrain.Enabled = true; _dlBtnBatch.Enabled = true; _dlBtnStop.Enabled = false;
            _dlStatus.Text = $"일괄 완료 ({done}/{total})";
            MessageBox.Show($"일괄 학습 완료: {done}/{total}개 축\n모델 폴더: {modelsDir}",
                "완료", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        /// <summary>학습 프로세스를 실행하고 완료 여부를 반환합니다. 동기 블로킹 — Task.Run에서 호출.</summary>
        /// <param name="onProgress">에포크 진행률(0-100)을 받는 콜백 (선택적)</param>
        private bool RunTrainingProcess(string python, string scriptPath, string paramsPath, int totalEpochs, Action<int> onProgress = null)
        {
            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo(python, $"\"{scriptPath}\" --params \"{paramsPath}\"")
                {
                    UseShellExecute = false, RedirectStandardOutput = true,
                    RedirectStandardError = true, CreateNoWindow = true,
                    StandardOutputEncoding = System.Text.Encoding.UTF8,
                    StandardErrorEncoding  = System.Text.Encoding.UTF8,
                };
                // Python I/O 인코딩 강제 — MLflow 등 라이브러리가 이모지/한글을 stderr로 출력할 때
                // 시스템 기본 인코딩(cp949)과 충돌하는 문제 방지
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                psi.EnvironmentVariables["PYTHONUTF8"]       = "1";
                _dlProc = System.Diagnostics.Process.Start(psi);
                _dlProc.ErrorDataReceived += (s2, ea) =>
                {
                    if (ea.Data != null) BeginInvoke(new Action(() => AppendDlLog("[ERR] " + ea.Data, Color.Orange)));
                };
                _dlProc.BeginErrorReadLine();

                string line;
                while ((line = _dlProc.StandardOutput.ReadLine()) != null)
                {
                    string captured = line;
                    BeginInvoke(new Action(() =>
                    {
                        AppendDlLog(captured);
                        try
                        {
                            using (var doc = JsonDocument.Parse(captured))
                            {
                                if (doc.RootElement.TryGetProperty("epoch", out var ep))
                                {
                                    int pct = Math.Min(100, (int)(ep.GetInt32() * 100.0 / totalEpochs));
                                    onProgress?.Invoke(pct);
                                    if (doc.RootElement.TryGetProperty("val_acc", out var va))
                                        _dlStatus.Text = $"에포크 {ep.GetInt32()}/{totalEpochs}  val_acc={va.GetDouble():F3}";
                                    else if (doc.RootElement.TryGetProperty("val_mse", out var vm))
                                        _dlStatus.Text = $"에포크 {ep.GetInt32()}/{totalEpochs}  val_mse={vm.GetDouble():F5}";
                                }
                                if (doc.RootElement.TryGetProperty("accuracy", out var acc))
                                    _dlStatus.Text = $"완료  최종={acc.GetDouble():F3}";
                                else if (doc.RootElement.TryGetProperty("threshold", out var thr))
                                    _dlStatus.Text = $"완료  임계값={thr.GetDouble():F4}";
                            }
                        }
                        catch { }
                    }));
                }
                _dlProc.WaitForExit();
                int exitCode = _dlProc.ExitCode;
                _dlProc = null;
                return exitCode == 0;
            }
            catch (Exception ex)
            {
                BeginInvoke(new Action(() => AppendDlLog("프로세스 오류: " + ex.Message, Color.Red)));
                _dlProc = null;
                return false;
            }
        }

        private async void StartDlTrainingAsync()
        {
            // ── 유효성 검사 ──────────────────────────────────────────────────
            string dataDir = _dlDataDir?.Text?.Trim() ?? "";
            if (!System.IO.Directory.Exists(dataDir))
            { MessageBox.Show("데이터 폴더가 존재하지 않습니다:\n" + dataDir, "오류", MessageBoxButtons.OK, MessageBoxIcon.Error); return; }

            string outputPath = _dlOutputPath?.Text?.Trim() ?? "";
            if (string.IsNullOrEmpty(outputPath))
            { MessageBox.Show("출력 모델 경로를 입력하세요.", "오류", MessageBoxButtons.OK, MessageBoxIcon.Error); return; }

            // ── params 빌드 ─────────────────────────────────────────────────
            var paramsObj = BuildDlParams(dataDir, outputPath);
            if (paramsObj == null)
            {
                bool trainIsAe = _dlRdoAe?.Checked == true;
                MessageBox.Show(
                    trainIsAe ? "입력 채널(1개 이상), 정상 클래스(1개 이상), 학습률을 확인하세요."
                              : "입력 채널(1개 이상), 결함 클래스(2개 이상), 학습률을 확인하세요.",
                    "설정 오류", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // ── Python 및 스크립트 확인 ──────────────────────────────────────
            string scriptsDir = System.IO.Path.Combine(
                System.IO.Path.GetDirectoryName(Application.ExecutablePath) ?? ".", "scripts");
            string venvPython = System.IO.Path.Combine(scriptsDir, ".venv", "Scripts", "python.exe");
            string python = System.IO.File.Exists(venvPython)
                ? venvPython
                : FindPythonExe(_dlPythonPath?.Text?.Trim() ?? "");
            if (python == null)
            {
                MessageBox.Show(
                    "Python을 찾을 수 없습니다.\n\n" +
                    "• [🐍 가상환경] 버튼을 클릭하여 .venv 생성\n" +
                    "• 또는 Python 경로를 직접 지정하세요.",
                    "Python 없음", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            AppendDlLog($"[Python] {python}");

            string scriptPath = System.IO.Path.Combine(scriptsDir, "train_dl_model.py");
            if (!System.IO.File.Exists(scriptPath))
            { MessageBox.Show("train_dl_model.py 를 찾을 수 없습니다:\n" + scriptPath, "스크립트 없음", MessageBoxButtons.OK, MessageBoxIcon.Error); return; }

            // ── 출력 폴더 생성 + params JSON 저장 ─────────────────────────────
            try { System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(outputPath)); } catch { }
            int totalEpochs = (int)NudCurrent(_dlEpochs, 30m);
            string paramsPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "phm_dl_params.json");
            System.IO.File.WriteAllText(paramsPath,
                JsonSerializer.Serialize(paramsObj, new JsonSerializerOptions { WriteIndented = true }),
                new System.Text.UTF8Encoding(false));

            // ── UI 상태 전환 ─────────────────────────────────────────────────
            _dlLog.Clear();
            _dlProgress.Value   = 0;
            _dlBtnTrain.Enabled = false;
            _dlBtnBatch.Enabled = false;
            _dlBtnStop.Enabled  = true;
            _dlStatus.Text = "학습 중...";
            AppendDlLog($"[시작] python \"{scriptPath}\"");
            AppendDlLog($"[params] {paramsPath}");
            AppendDlLog("");

            // ── 비동기 프로세스 실행 ─────────────────────────────────────────
            bool success = await System.Threading.Tasks.Task.Run(() =>
                RunTrainingProcess(python, scriptPath, paramsPath, totalEpochs,
                    pct => BeginInvoke(new Action(() => _dlProgress.Value = pct))));

            _dlProgress.Value   = success ? 100 : _dlProgress.Value;
            _dlBtnTrain.Enabled = true;
            _dlBtnBatch.Enabled = true;
            _dlBtnStop.Enabled  = false;

            if (success)
            {
                _dlStatus.Text = "완료 ✓";
                AppendDlLog($"\n모델 저장: {outputPath}", Color.Cyan);
                bool doneIsAe = _dlRdoAe?.Checked == true;
                MessageBox.Show(
                    doneIsAe
                        ? $"AE 모델 학습 완료!\n{outputPath}\n\n대시보드에서 ONNX AE 모델로 로드하세요."
                        : $"DL 모델 학습 완료!\n{outputPath}\n\n대시보드에서 ONNX 분류 모델로 로드하세요.",
                    "학습 완료", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                _dlStatus.Text = "오류";
                AppendDlLog("학습 실패", Color.Red);
            }
        }

        private void RunSetupVenv()
        {
            // ── 선택 다이얼로그 ────────────────────────────────────────
            string choice;
            using (var dlg = new Form
            {
                Text            = "환경 설치",
                Width           = 360,
                Height          = 160,
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition   = FormStartPosition.CenterParent,
                MaximizeBox     = false,
                MinimizeBox     = false,
            })
            {
                var lbl = new Label
                {
                    Text   = "설치 항목을 선택하세요:",
                    Left   = 16, Top = 16, Width = 320, Height = 20,
                };
                var btnLocal = new Button
                {
                    Text = "🐍  로컬 가상환경 설치",
                    Left = 16, Top = 48, Width = 150, Height = 36,
                };
                var btnServer = new Button
                {
                    Text = "🚀  서버 GPU Docker 빌드",
                    Left = 178, Top = 48, Width = 155, Height = 36,
                };
                var btnCancel = new Button
                {
                    Text         = "취소",
                    Left         = 254, Top = 92, Width = 79, Height = 26,
                    DialogResult = DialogResult.Cancel,
                };
                dlg.Controls.AddRange(new Control[] { lbl, btnLocal, btnServer, btnCancel });
                dlg.CancelButton = btnCancel;
                btnLocal.Click  += (s, e) => { dlg.Tag = "local";  dlg.Close(); };
                btnServer.Click += (s, e) => { dlg.Tag = "server"; dlg.Close(); };

                dlg.ShowDialog(this);
                choice = dlg.Tag as string;
            }
            if (choice == null) return;

            // ── 공통 helpers ───────────────────────────────────────────
            string scriptsDir = System.IO.Path.Combine(
                System.IO.Path.GetDirectoryName(Application.ExecutablePath) ?? ".", "scripts");

            void RunBat(string fileName, string logMsg)
            {
                string batPath = System.IO.Path.Combine(scriptsDir, fileName);
                if (!System.IO.File.Exists(batPath))
                {
                    MessageBox.Show($"{fileName} 를 찾을 수 없습니다:\n{batPath}",
                        "파일 없음", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                try
                {
                    System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
                    {
                        FileName         = batPath,
                        WorkingDirectory = scriptsDir,
                        UseShellExecute  = true,
                    });
                    AppendDlLog(logMsg, Color.Cyan);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"{fileName} 실행 실패:\n{ex.Message}",
                        "오류", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }

            // ── 선택 분기 ──────────────────────────────────────────────
            if (choice == "local")
                RunBat("setup_venv.bat",
                    "[가상환경] setup_venv.bat 실행 중 — 완료 후 학습 시작 가능합니다.");
            else
                RunBat("setup_docker_gpu.bat",
                    "[서버 GPU] setup_docker_gpu.bat 실행 중 — SSH로 서버에 접속합니다.");
        }

        private void StopDlTraining()
        {
            try { _dlProc?.Kill(); } catch { }
            _dlBtnTrain.Enabled = true;
            _dlBtnStop.Enabled  = false;
            _dlStatus.Text = "중단됨";
            AppendDlLog("사용자에 의해 중단됨", Color.Yellow);
        }

        private void AppendDlLog(string text, Color? color = null)
        {
            if (_dlLog == null) return;
            _dlLog.SelectionStart  = _dlLog.TextLength;
            _dlLog.SelectionLength = 0;
            _dlLog.SelectionColor  = color ?? Color.LightGreen;
            _dlLog.AppendText(DateTime.Now.ToString("HH:mm:ss") + "  " + text + "\n");
            _dlLog.ScrollToCaret();
        }

        /// <summary>사용 가능한 Python 실행 파일을 탐색합니다. 없으면 null 반환.</summary>
        private static string FindPythonExe(string userPath)
        {
            var candidates = new List<string>();

            // 1) 사용자 지정 경로
            if (!string.IsNullOrWhiteSpace(userPath)) candidates.Add(userPath);

            // 2) PATH에 등록된 명령어
            candidates.AddRange(new[] { "py", "python", "python3" });

            // 3) Windows 레지스트리에서 설치 경로 탐색
            foreach (var exePath in FindPythonFromRegistry())
                candidates.Add(exePath);

            // 4) 일반 설치 폴더 탐색 (PATH 미등록 케이스 대응)
            var searchRoots = new[]
            {
                System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python"),
                System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Programs", "Python"),
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                @"C:\",
            };
            foreach (var root in searchRoots)
            {
                if (!System.IO.Directory.Exists(root)) continue;
                try
                {
                    foreach (var dir in System.IO.Directory.GetDirectories(root, "Python3*")
                                            .OrderByDescending(d => d))
                    {
                        var exe = System.IO.Path.Combine(dir, "python.exe");
                        if (System.IO.File.Exists(exe)) candidates.Add(exe);
                    }
                }
                catch { }
            }

            // 후보 중 실제 실행 가능한 첫 번째 반환
            foreach (var cand in candidates)
            {
                if (string.IsNullOrWhiteSpace(cand)) continue;
                try
                {
                    var psi = new System.Diagnostics.ProcessStartInfo(cand, "--version")
                    {
                        UseShellExecute = false,
                        CreateNoWindow = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true
                    };
                    using (var p = System.Diagnostics.Process.Start(psi))
                    {
                        if (p.WaitForExit(3000) && p.ExitCode == 0)
                            return cand;
                    }
                }
                catch { }
            }
            return null;
        }

        /// <summary>레지스트리 HKCU/HKLM에서 Python InstallPath를 읽어 python.exe 경로 목록 반환.</summary>
        private static IEnumerable<string> FindPythonFromRegistry()
        {
            var result = new List<string>();
            var hives = new[]
            {
                Microsoft.Win32.Registry.CurrentUser,
                Microsoft.Win32.Registry.LocalMachine
            };
            foreach (var hive in hives)
            {
                try
                {
                    using (var key = hive.OpenSubKey(@"SOFTWARE\Python\PythonCore"))
                    {
                        if (key == null) continue;
                        foreach (var ver in key.GetSubKeyNames().OrderByDescending(v => v))
                        {
                            try
                            {
                                using (var instKey = key.OpenSubKey(ver + @"\InstallPath"))
                                {
                                    if (instKey == null) continue;
                                    var exePath = instKey.GetValue("ExecutablePath") as string;
                                    if (string.IsNullOrEmpty(exePath))
                                    {
                                        var folder = instKey.GetValue("") as string;
                                        if (!string.IsNullOrEmpty(folder))
                                            exePath = System.IO.Path.Combine(folder.TrimEnd('\\', '/'), "python.exe");
                                    }
                                    if (!string.IsNullOrEmpty(exePath) && System.IO.File.Exists(exePath))
                                        result.Add(exePath);
                                }
                            }
                            catch { }
                        }
                    }
                }
                catch { }
            }
            return result;
        }

        // ════════════════════════════════════════════════════════════════════
        //  Airflow 연동 패널
        // ════════════════════════════════════════════════════════════════════

        /// <summary>서버 설정 저장 후 Airflow 패널 텍스트박스를 최신값으로 갱신합니다.</summary>
        private void SyncAirflowPanel(Services.ServerSettings s)
        {
            if (_aflUrl   != null) _aflUrl.Text   = s.AirflowUrl   ?? "";
            if (_aflDagId != null) _aflDagId.Text = s.AirflowDagId ?? "phm_retrain";
            // User/Password 는 ServerSettings.Current 에서 직접 읽으므로 별도 TextBox 불필요

            // 서버(URL/계정)가 바뀌었을 수 있으므로 자동 재학습 On/Off 상태를 다시 조회
            _ = RefreshAutoRetrainStatusAsync();
        }

        /// <summary>
        /// Airflow 트리거 패널. 이전에는 URL/DAG/프로파일/축/체크박스 9개가 한 줄에
        /// 몰려 있었는데, "연결 → 모델 선택 → 저장 위치 → 실행" 4단으로 나눠 한눈에 읽히게 함.
        /// </summary>
        private GroupBox BuildAirflowPanel()
        {
            var grp = new GroupBox
            {
                Text = "Airflow 학습 실행",
                Dock = DockStyle.Fill,
                Padding = new Padding(8, 4, 8, 4),
                ForeColor = Color.FromArgb(0, 140, 220),
            };

            var stack = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 6 };
            stack.RowStyles.Add(new RowStyle(SizeType.Absolute, 22));  // row0: 서버 연결
            stack.RowStyles.Add(new RowStyle(SizeType.Absolute, 46));  // row1: 모델 선택 (박스로 구분)
            stack.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));  // row2: 축 / 프로파일 / 라벨
            stack.RowStyles.Add(new RowStyle(SizeType.Percent, 100));  // row3: 실행 버튼 + 상태
            stack.RowStyles.Add(new RowStyle(SizeType.Absolute, 26));  // row4: 자동 재학습 On/Off
            stack.RowStyles.Add(new RowStyle(SizeType.Absolute, 26));  // row5: 자동 재학습 주기(스케줄) 지정

            // ── row0: 서버 연결 (URL / DAG — 자주 안 바뀌므로 작고 옅게) ─────────
            var connRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            connRow.Controls.Add(new Label { Text = "서버:", AutoSize = true, Margin = new Padding(0, 4, 4, 0), ForeColor = Color.Gray, Font = new Font(Font.FontFamily, 8f) });
            _aflUrl = new TextBox { Width = 190, Margin = new Padding(0, 1, 12, 0), Font = new Font(Font.FontFamily, 8f),
                Text = Services.ServerSettings.Current.AirflowUrl ?? "http://localhost:8080" };
            connRow.Controls.Add(_aflUrl);
            connRow.Controls.Add(new Label { Text = "DAG:", AutoSize = true, Margin = new Padding(0, 4, 4, 0), ForeColor = Color.Gray, Font = new Font(Font.FontFamily, 8f) });
            _aflDagId = new TextBox { Width = 110, Font = new Font(Font.FontFamily, 8f),
                Text = Services.ServerSettings.Current.AirflowDagId ?? "phm_retrain" };
            connRow.Controls.Add(_aflDagId);

            // ── row1: 모델 선택 체크박스 ────────────────────────────────────────
            Action onModelChk = () => { if (!_aflUpdatingAll) UpdateTriggerButtonText(); };

            _aflChkAll = AflMakeChk("▣ 전체", true);
            _aflChkAll.Font = new Font(_aflChkAll.Font, FontStyle.Bold);
            _aflChkAll.CheckedChanged += (s, e) =>
            {
                if (_aflUpdatingAll) return;
                _aflUpdatingAll = true;
                bool chk = _aflChkAll.Checked;
                foreach (Control c in _aflModelFlow.Controls)
                    if (c is CheckBox cb && cb != _aflChkAll && cb.Visible) cb.Checked = chk;
                _aflUpdatingAll = false;
                UpdateTriggerButtonText();
            };

            // AE 체크박스
            _aflChkAeAccelG   = AflMakeChk("Accel 전역",   true); _aflChkAeAccelG.CheckedChanged   += (s, e) => onModelChk();
            _aflChkAeTorqueG  = AflMakeChk("Torque 전역",  true); _aflChkAeTorqueG.CheckedChanged  += (s, e) => onModelChk();
            _aflChkAeTorqueAx = AflMakeChk("Torque 축별",  true); _aflChkAeTorqueAx.CheckedChanged += (s, e) => onModelChk();
            _aflChkAeCombG    = AflMakeChk("Comb 전역",    true); _aflChkAeCombG.CheckedChanged    += (s, e) => onModelChk();
            _aflChkAeCombAx   = AflMakeChk("Comb 축별",    true); _aflChkAeCombAx.CheckedChanged   += (s, e) => onModelChk();

            // Accel IsolationForest — AE-CNN(ae_accel.onnx)의 선택 가능한 대안.
            // 파일명이 같아서 별도 프로파일(<프로파일>_isoforest)에 저장되며, default
            // AE-CNN 모델을 덮어쓰지 않습니다. 대시보드 프로파일 콤보박스로 전환해서 씁니다.
            _aflChkIsoAccel = AflMakeChk("Accel IsoForest", true);
            _aflChkIsoAccel.CheckedChanged += (s, e) => onModelChk();
            var tipIso = new ToolTip();
            tipIso.SetToolTip(_aflChkIsoAccel,
                "RobustScaler+IsolationForest 기반 가속도 이상탐지 (crest/kurtosis/주파수 피크 특징).\n" +
                "AE-CNN과 파일명이 같아 별도 프로파일 '<프로파일>_isoforest'에 저장됩니다.\n" +
                "대시보드 프로파일 콤보박스에서 전환해서 비교하세요.");

            // CLS 체크박스
            _aflChkClsAccel  = AflMakeChk("Accel",    true); _aflChkClsAccel.CheckedChanged  += (s, e) => onModelChk();
            _aflChkClsTorque = AflMakeChk("Torque",   true); _aflChkClsTorque.CheckedChanged += (s, e) => onModelChk();
            _aflChkClsComb   = AflMakeChk("Combined", true); _aflChkClsComb.CheckedChanged   += (s, e) => onModelChk();

            _aflModelFlow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill, AutoSize = false,
                FlowDirection = FlowDirection.LeftToRight, WrapContents = true,
                Padding = new Padding(6, 4, 6, 4), Margin = new Padding(0),
            };
            _aflModelFlow.Controls.AddRange(new Control[]
            {
                _aflChkAll,
                _aflChkAeAccelG, _aflChkAeTorqueG, _aflChkAeTorqueAx,
                _aflChkAeCombG,  _aflChkAeCombAx,
                _aflChkIsoAccel,
                _aflChkClsAccel, _aflChkClsTorque, _aflChkClsComb,
            });
            // 체크박스 영역을 옅은 배경 + 테두리로 감싸 나머지 여백과 시각적으로 구분
            var modelBox = new Panel
            {
                Dock = DockStyle.Fill, BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(240, 243, 248),
            };
            modelBox.Controls.Add(_aflModelFlow);

            // ── row2: 축 선택 / 저장 프로파일 / 라벨 (한 줄로 폭을 고르게 사용) ──
            var midRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 6, RowCount = 1 };
            midRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            midRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 70));
            midRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            midRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 45));
            midRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            midRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 55));

            // "전체" 선택 시 DAG 가 axis_count 자동 감지 → 모든 축 학습
            // "Ax{N}" 선택 시 conf["axes"]=[N] 으로 해당 축만 학습
            _aflAxisCombo = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList, Dock = DockStyle.Left,
                Width = 66, Margin = new Padding(0, 3, 0, 0),
            };
            _aflAxisCombo.Items.AddRange(new object[] { "전체", "Ax0", "Ax1", "Ax2" });
            _aflAxisCombo.SelectedIndex = 0;
            _aflAxisCombo.SelectedIndexChanged += (s, e) => UpdateTriggerButtonText();
            var tipAxis = new ToolTip();
            tipAxis.SetToolTip(_aflAxisCombo,
                "축별 모델(Torque 축별 / Comb 축별 / CLS Accel-Torque-Combined) 학습 시 대상 축을 선택합니다.\n" +
                "전체: 모든 축(0~N-1) 학습 (기본)\n" +
                "Ax{N}: 해당 축 하나만 학습");

            _aflProfile = new TextBox { Dock = DockStyle.Fill, Margin = new Padding(0, 4, 14, 0), Text = "default" };
            _aflProfile.Enter += (s, e) => { if (_aflProfile.Text == "default") _aflProfile.SelectAll(); };
            var tipProfile = new ToolTip();
            tipProfile.SetToolTip(_aflProfile, "모델 저장 디렉토리 이름\n예: default, accel_only, v2026-05-11");

            _aflProfileLabel = new TextBox { Dock = DockStyle.Fill, Margin = new Padding(0, 4, 0, 0) };
            var tipProfileLbl = new ToolTip();
            tipProfileLbl.SetToolTip(_aflProfileLabel, "프로파일 표시 이름 (선택)\n예: 5월 재학습");

            midRow.Controls.Add(Lbl2("축:"),        0, 0);
            midRow.Controls.Add(_aflAxisCombo,       1, 0);
            midRow.Controls.Add(Lbl2("프로파일:"),  2, 0);
            midRow.Controls.Add(_aflProfile,         3, 0);
            midRow.Controls.Add(Lbl2("라벨:"),      4, 0);
            midRow.Controls.Add(_aflProfileLabel,    5, 0);

            // ── row3: 실행 버튼 + 상태 조회 + 상태 문구 (남는 폭을 상태 문구가 채움) ──
            var runRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 1 };
            runRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 150));
            runRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 100));
            runRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));

            _aflBtnTrigger = new Button
            {
                Text = "▶ 지금 트리거", Dock = DockStyle.Fill, Height = 26,
                BackColor = Color.FromArgb(0, 120, 60), ForeColor = Color.White, FlatStyle = FlatStyle.Flat,
                Margin = new Padding(0, 4, 6, 0),
            };
            _aflBtnTrigger.Click += async (s, e) => await TriggerAirflowAsync();

            _aflBtnStatus = new Button { Text = "🔄 상태 조회", Dock = DockStyle.Fill, Height = 26, Margin = new Padding(0, 4, 10, 0) };
            _aflBtnStatus.Click += async (s, e) => await RefreshAirflowStatusAsync();

            _aflStatusLbl = new Label
            {
                Text = "—", Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                ForeColor = Color.Gray, Font = new Font(Font.FontFamily, 8.5f),
            };

            runRow.Controls.Add(_aflBtnTrigger, 0, 0);
            runRow.Controls.Add(_aflBtnStatus,  1, 0);
            runRow.Controls.Add(_aflStatusLbl,  2, 0);

            // ── row4: 자동 재학습 On/Off (phm_auto_retrain DAG 일시정지 전환) ──────
            var autoRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            _aflChkAutoRetrain = new CheckBox
            {
                Text = "자동 재학습 (AE+IsoForest, 아래 주기로 실행)",
                AutoSize = true,
                Margin = new Padding(0, 5, 10, 0),
            };
            _aflChkAutoRetrain.CheckedChanged += async (s, e) =>
            {
                if (_aflSyncingAutoRetrain) return;
                await ToggleAutoRetrainAsync(_aflChkAutoRetrain.Checked);
            };
            var tipAuto = new ToolTip();
            tipAuto.SetToolTip(_aflChkAutoRetrain,
                "Airflow의 phm_auto_retrain DAG를 켜고 끕니다 — 아래 '주기'에서 지정한 스케줄로 실행됩니다.\n" +
                "켜면 CLS(레이블 필요)는 제외하고 AE-CNN + IsolationForest 이상탐지 모델만 자동 재학습합니다.\n" +
                "서버가(이 앱이) 꺼져 있어도 Airflow 컨테이너가 살아있으면 그대로 실행됩니다.\n" +
                "대상 모델을 바꾸려면 서버의 phm_auto_retrain_dag.py 환경변수(PHM_AUTO_RETRAIN_MODES)를 수정하세요.");

            _aflAutoStatusLbl = new Label
            {
                Text = "확인 중…", AutoSize = true,
                ForeColor = Color.Gray, Font = new Font(Font.FontFamily, 8.5f),
                Margin = new Padding(0, 8, 0, 0),
            };

            autoRow.Controls.Add(_aflChkAutoRetrain);
            autoRow.Controls.Add(_aflAutoStatusLbl);

            // ── row5: 자동 재학습 주기(스케줄) 지정 ────────────────────────────
            var scheduleRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            scheduleRow.Controls.Add(Lbl2("주기:"));

            _aflSchedulePreset = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList, Width = 150,
                Margin = new Padding(0, 3, 6, 0),
            };
            foreach (var preset in AutoRetrainSchedulePresets)
                _aflSchedulePreset.Items.Add(preset.Label);
            _aflSchedulePreset.SelectedIndex = 0;
            _aflSchedulePreset.SelectedIndexChanged += (s, e) =>
            {
                if (_aflSyncingSchedulePreset) return;
                int idx = _aflSchedulePreset.SelectedIndex;
                if (idx < 0 || idx >= AutoRetrainSchedulePresets.Length) return;
                string cron = AutoRetrainSchedulePresets[idx].Cron;
                if (cron != null) _aflScheduleCron.Text = cron;   // "사용자 지정"이면 null → 텍스트박스 유지
            };

            _aflScheduleCron = new TextBox { Width = 110, Margin = new Padding(0, 4, 6, 0), Text = "0 2 * * *" };
            _aflScheduleCron.TextChanged += (s, e) => SyncSchedulePresetFromCron();
            var tipCron = new ToolTip();
            tipCron.SetToolTip(_aflScheduleCron,
                "cron 표현식(분 시 일 월 요일) 또는 \"@daily\" 같은 Airflow 매크로.\n예: \"0 2 * * *\" = 매일 새벽 2시");

            _aflBtnScheduleApply = new Button { Text = "적용", Width = 50, Height = 22, Margin = new Padding(0, 2, 0, 0) };
            _aflBtnScheduleApply.Click += async (s, e) => await ApplyAutoRetrainScheduleAsync();

            scheduleRow.Controls.Add(_aflSchedulePreset);
            scheduleRow.Controls.Add(_aflScheduleCron);
            scheduleRow.Controls.Add(_aflBtnScheduleApply);

            stack.Controls.Add(connRow,     0, 0);
            stack.Controls.Add(modelBox,    0, 1);
            stack.Controls.Add(midRow,      0, 2);
            stack.Controls.Add(runRow,      0, 3);
            stack.Controls.Add(autoRow,     0, 4);
            stack.Controls.Add(scheduleRow, 0, 5);

            // 초기 세션(CLS)에 맞는 항목 채우기
            UpdateAflTrainModeItems();

            grp.Controls.Add(stack);
            return grp;
        }

        /// <summary>체크박스 생성 헬퍼.</summary>
        private static CheckBox AflMakeChk(string text, bool isChecked) => new CheckBox
        {
            Text = text, Checked = isChecked, AutoSize = true,
            Margin = new Padding(0, 4, 10, 0),
        };

        /// <summary>AE/CLS 세션에 따라 모델 선택 체크박스를 표시/숨깁니다.</summary>
        private void UpdateAflTrainModeItems()
        {
            if (_aflModelFlow == null) return;
            if (this.InvokeRequired) { this.BeginInvoke(new Action(UpdateAflTrainModeItems)); return; }

            bool isAe = (_dlRdoAe?.Checked == true);

            // AE 체크박스
            _aflChkAeAccelG.Visible   = isAe;
            _aflChkAeTorqueG.Visible  = isAe;
            _aflChkAeTorqueAx.Visible = isAe;
            _aflChkAeCombG.Visible    = isAe;
            _aflChkAeCombAx.Visible   = isAe;
            _aflChkIsoAccel.Visible   = isAe;

            // CLS 체크박스
            _aflChkClsAccel.Visible  = !isAe;
            _aflChkClsTorque.Visible = !isAe;
            _aflChkClsComb.Visible   = !isAe;

            // 세션 전환 시 전체 선택 상태 리셋
            _aflUpdatingAll = true;
            _aflChkAll.Checked = true;
            foreach (Control c in _aflModelFlow.Controls)
                if (c is CheckBox cb && cb != _aflChkAll) cb.Checked = true;
            _aflUpdatingAll = false;

            UpdateTriggerButtonText();
        }

        /// <summary>세션/센서 선택 조합을 버튼 툴팁으로 표시합니다.</summary>
        private void UpdateTriggerButtonText()
        {
            if (_aflBtnTrigger == null) return;
            bool isAe = (_dlRdoAe?.Checked == true);

            var tasks = new System.Collections.Generic.List<string>();
            if (isAe)
            {
                if (_aflChkAeAccelG?.Checked   == true) tasks.Add("Accel 전역");
                if (_aflChkAeTorqueG?.Checked  == true) tasks.Add("Torque 전역");
                if (_aflChkAeTorqueAx?.Checked == true) tasks.Add("Torque 축별");
                if (_aflChkAeCombG?.Checked    == true) tasks.Add("Comb 전역");
                if (_aflChkAeCombAx?.Checked   == true) tasks.Add("Comb 축별");
                if (_aflChkIsoAccel?.Checked   == true) tasks.Add("Accel IsoForest");
            }
            else
            {
                if (_aflChkClsAccel?.Checked  == true) tasks.Add("Accel");
                if (_aflChkClsTorque?.Checked == true) tasks.Add("Torque");
                if (_aflChkClsComb?.Checked   == true) tasks.Add("Combined");
            }

            string label   = isAe ? "AE" : "CLS";
            string preview = tasks.Count > 0
                ? string.Join(" + ", tasks)
                : "⚠ 없음 (1개 이상 선택 필요)";
            string axisSel = _aflAxisCombo?.SelectedItem?.ToString() ?? "전체";
            string axisSfx = (axisSel == "전체") ? "" : $" / 축={axisSel}";
            string tipText = $"[{label}{axisSfx}] 실행 태스크: {preview}";

            _aflBtnTrigger.Text = $"▶ 트리거 [{label}{axisSfx}]";
            if (_aflBtnTrigger.Tag is ToolTip tt)
                tt.SetToolTip(_aflBtnTrigger, tipText);
            else
            {
                var newTip = new ToolTip();
                newTip.SetToolTip(_aflBtnTrigger, tipText);
                _aflBtnTrigger.Tag = newTip;
            }
        }

        private async System.Threading.Tasks.Task TriggerAirflowAsync()
        {
            // NumericUpDown 등의 사용자 입력값이 .Value 에 commit 되지 않은 채로
            // 트리거 버튼을 누르면 이전 .Value 가 사용되는 WinForms quirk 방지.
            // ValidateChildren() 은 모든 자식 컨트롤의 Validating 이벤트를 발생시켜
            // NumericUpDown 등의 Text → Value commit 을 강제한다.
            // (예: "임계값 퍼센타일" 90 입력 → Enter 안 누르고 트리거 시 99.5/99.9 가 전송되던 버그)
            try { this.ValidateChildren(); } catch { /* 무시 */ }

            string dataDir    = _dlDataDir?.Text?.Trim() ?? "";
            string outputPath = _dlOutputPath?.Text?.Trim() ?? "";
            string dagId      = _aflDagId?.Text?.Trim() ?? "phm_retrain";
            string airflowUrl = _aflUrl?.Text?.Trim()
                                ?? Services.ServerSettings.Current.AirflowUrl;

            if (!System.IO.Directory.Exists(dataDir))
            {
                MessageBox.Show("데이터 폴더가 없습니다:\n" + dataDir, "오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            var paramsObj = BuildDlParams(dataDir, outputPath);
            if (paramsObj == null)
            {
                MessageBox.Show("학습 설정이 올바르지 않습니다. 채널/클래스/학습률을 확인하세요.",
                    "설정 오류", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            // ── train_modes 결정: 체크박스 상태 읽기 ─────────────────────────────
            bool isAeSession = (_dlRdoAe?.Checked == true);

            var modeList = new System.Collections.Generic.List<string>();
            if (isAeSession)
            {
                if (_aflChkAeAccelG?.Checked   == true) modeList.Add("ae_accel");
                if (_aflChkAeTorqueG?.Checked  == true) modeList.Add("ae_torque_global");
                if (_aflChkAeTorqueAx?.Checked == true) modeList.Add("ae_torque");
                if (_aflChkAeCombG?.Checked    == true) modeList.Add("ae_combined_global");
                if (_aflChkAeCombAx?.Checked   == true) modeList.Add("ae_combined");
                if (_aflChkIsoAccel?.Checked   == true) modeList.Add("isoforest_accel");
            }
            else
            {
                if (_aflChkClsAccel?.Checked  == true) modeList.Add("accel");
                if (_aflChkClsTorque?.Checked == true) modeList.Add("torque");
                if (_aflChkClsComb?.Checked   == true) modeList.Add("combined");
            }
            string[] trainModes = modeList.ToArray();

            if (trainModes.Length == 0)
            {
                MessageBox.Show("학습할 모델을 1개 이상 선택하세요.",
                                "모델 미선택", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            paramsObj["train_modes"] = trainModes;
            // Airflow DAG 태스크는 session을 내부 고정값 사용 → conf의 session 제거
            paramsObj.Remove("session");

            // ── 축 선택: "Ax{N}" 단일 선택 시 conf["axes"]=[N] 전송 ────────────
            // "전체" 선택이면 conf 에 axes 미전송 → DAG 가 axis_count 자동 감지
            string axisSel = _aflAxisCombo?.SelectedItem?.ToString();
            if (!string.IsNullOrEmpty(axisSel) && axisSel.StartsWith("Ax") &&
                int.TryParse(axisSel.Substring(2), out int axIdx))
            {
                paramsObj["axes"] = new int[] { axIdx };
            }

            // ── 프로파일 설정 ────────────────────────────────────────────────
            string profileName = _aflProfile?.Text?.Trim();
            if (string.IsNullOrWhiteSpace(profileName)) profileName = "default";
            paramsObj["profile"] = profileName;

            string profileLabel = _aflProfileLabel?.Text?.Trim();
            if (!string.IsNullOrWhiteSpace(profileLabel))
                paramsObj["profile_label"] = profileLabel;

            // ── IsolationForest 전용 프로파일 ────────────────────────────────
            // ae_accel.onnx 파일명이 AE-CNN과 겹치므로 "<프로파일>_isoforest"로
            // 분리해 저장한다 (DAG의 _get_isoforest_profile_dir 기본값과 별개로,
            // 여기서 명시해 사용자가 고른 프로파일 이름과 연관되도록 함).
            if (modeList.Contains("isoforest_accel"))
            {
                paramsObj["isoforest_profile"] = profileName + "_isoforest";
                if (!string.IsNullOrWhiteSpace(profileLabel))
                    paramsObj["isoforest_profile_label"] = profileLabel + " (IsoForest)";

                // "모델 설정 / 출력"의 윈도우(샘플)/스트라이드 입력을 IsoForest 전용 키로도
                // 전달한다 — 안 넘기면 DAG 자체 기본값(train.py 기준 500/250)이 쓰이므로,
                // 사용자가 이 화면에서 값을 바꿔도 IsoForest 학습에는 반영되지 않던 문제를 고침.
                paramsObj["isoforest_window_size"] = (int)NudCurrent(_dlWindowSize, 500m);
                paramsObj["isoforest_stride"]      = (int)NudCurrent(_dlStride,     250m);
            }

            _aflBtnTrigger.Enabled = false;
            _aflStatusLbl.ForeColor = Color.DodgerBlue;
            _aflStatusLbl.Text = "트리거 중…";

            var s2 = Services.ServerSettings.Current;
            using (var client = new Services.Core.AirflowClient(airflowUrl, s2.AirflowUser, s2.AirflowPassword))
            {
                var trigResult = await client.TriggerDagAsync(dagId, paramsObj);
                if (trigResult.Ok)
                {
                    string prof = paramsObj.TryGetValue("profile", out var pv) ? pv?.ToString() : "default";
                    _aflLastRunId           = trigResult.RunId;
                    _aflStatusLbl.Text      = $"queued — {trigResult.RunId}  [프로파일: {prof}]";
                    _aflStatusLbl.ForeColor = Color.LightGreen;
                    AppendDlLog($"[Airflow] DAG 트리거 성공: {trigResult.RunId}  프로파일={prof}", Color.LightGreen);
                }
                else
                {
                    _aflStatusLbl.Text      = "트리거 실패: " + trigResult.Error;
                    _aflStatusLbl.ForeColor = Color.OrangeRed;
                    AppendDlLog($"[Airflow] 트리거 실패: {trigResult.Error}", Color.OrangeRed);
                }
            }

            _aflBtnTrigger.Enabled = true;
        }

        /// <summary>마지막으로 트리거된 DAG 실행 상태를 조회합니다.</summary>
        private async System.Threading.Tasks.Task RefreshAirflowStatusAsync()
        {
            string dagId      = _aflDagId?.Text?.Trim() ?? "phm_retrain";
            string airflowUrl = _aflUrl?.Text?.Trim()
                                ?? Services.ServerSettings.Current.AirflowUrl;

            _aflBtnStatus.Enabled   = false;
            _aflStatusLbl.ForeColor = Color.DodgerBlue;
            _aflStatusLbl.Text      = "조회 중…";

            var s2 = Services.ServerSettings.Current;
            using (var client = new Services.Core.AirflowClient(airflowUrl, s2.AirflowUser, s2.AirflowPassword))
            {
                if (!string.IsNullOrEmpty(_aflLastRunId))
                {
                    var stRes = await client.GetDagRunStatusAsync(dagId, _aflLastRunId);
                    if (stRes.State != null)
                    {
                        _aflStatusLbl.Text      = $"{stRes.State} — {_aflLastRunId}";
                        _aflStatusLbl.ForeColor = StateColor(stRes.State);
                    }
                    else
                    {
                        _aflStatusLbl.Text      = "오류: " + stRes.Error;
                        _aflStatusLbl.ForeColor = Color.OrangeRed;
                    }
                }
                else
                {
                    // run_id 없으면 최신 실행 조회
                    var latRes = await client.GetLatestDagRunAsync(dagId);
                    if (latRes.State != null)
                    {
                        _aflLastRunId           = latRes.RunId;
                        _aflStatusLbl.Text      = string.IsNullOrEmpty(latRes.RunId)
                            ? "실행 이력 없음"
                            : $"{latRes.State} — {latRes.RunId}";
                        _aflStatusLbl.ForeColor = StateColor(latRes.State ?? "none");
                    }
                    else
                    {
                        _aflStatusLbl.Text      = "오류: " + latRes.Error;
                        _aflStatusLbl.ForeColor = Color.OrangeRed;
                    }
                }
            }

            _aflBtnStatus.Enabled = true;

            // 같은 버튼으로 자동 재학습 On/Off 상태도 함께 갱신
            await RefreshAutoRetrainStatusAsync();
        }

        /// <summary>
        /// phm_auto_retrain DAG의 현재 일시정지(is_paused) 상태를 조회해 체크박스/상태 문구를
        /// 동기화합니다. DAG가 아직 서버에 배포되지 않았으면 그 사실을 안내합니다.
        /// </summary>
        private async System.Threading.Tasks.Task RefreshAutoRetrainStatusAsync()
        {
            if (_aflChkAutoRetrain == null || _aflAutoStatusLbl == null) return;

            string airflowUrl = _aflUrl?.Text?.Trim() ?? Services.ServerSettings.Current.AirflowUrl;
            var s2 = Services.ServerSettings.Current;

            _aflAutoStatusLbl.ForeColor = Color.Gray;
            _aflAutoStatusLbl.Text      = "확인 중…";

            using (var client = new Services.Core.AirflowClient(airflowUrl, s2.AirflowUser, s2.AirflowPassword))
            {
                var info = await client.GetDagInfoAsync(AutoRetrainDagId);
                if (IsDisposed) return;

                if (info.IsPaused.HasValue)
                {
                    _aflSyncingAutoRetrain = true;
                    _aflChkAutoRetrain.Checked = !info.IsPaused.Value;
                    _aflSyncingAutoRetrain = false;

                    string sched = string.IsNullOrEmpty(info.Schedule) ? "" : $" ({info.Schedule})";
                    _aflAutoStatusLbl.Text      = info.IsPaused.Value ? "꺼짐" : $"켜짐{sched}";
                    _aflAutoStatusLbl.ForeColor = info.IsPaused.Value ? Color.Gray : Color.LightGreen;

                    // 서버에 실제 반영된(마지막 DAG 재파싱 시점 기준) 주기를 텍스트박스에 표시
                    if (!string.IsNullOrEmpty(info.Schedule) && _aflScheduleCron != null)
                        _aflScheduleCron.Text = info.Schedule;
                }
                else
                {
                    _aflAutoStatusLbl.Text      = "DAG 없음 — 서버에 phm_auto_retrain_dag.py 배포 필요";
                    _aflAutoStatusLbl.ForeColor = Color.OrangeRed;
                }
            }
        }

        /// <summary>cron 텍스트박스가 알려진 프리셋과 일치하면 콤보를 그 프리셋으로, 아니면 "사용자 지정"으로 맞춥니다.</summary>
        private void SyncSchedulePresetFromCron()
        {
            if (_aflSchedulePreset == null || _aflScheduleCron == null) return;
            string cron = _aflScheduleCron.Text.Trim();

            int matchIdx = -1;
            for (int i = 0; i < AutoRetrainSchedulePresets.Length; i++)
                if (AutoRetrainSchedulePresets[i].Cron == cron) { matchIdx = i; break; }

            int target = matchIdx >= 0 ? matchIdx : AutoRetrainSchedulePresets.Length - 1; // 마지막 = "사용자 지정"
            if (_aflSchedulePreset.SelectedIndex == target) return;

            _aflSyncingSchedulePreset = true;
            _aflSchedulePreset.SelectedIndex = target;
            _aflSyncingSchedulePreset = false;
        }

        /// <summary>표준 5필드 cron("분 시 일 월 요일") 또는 "@daily" 류 Airflow 매크로인지 검사합니다.</summary>
        private static bool IsValidCronOrMacro(string cron)
        {
            if (string.IsNullOrWhiteSpace(cron)) return false;
            if (cron.StartsWith("@")) return true;
            var fields = cron.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return fields.Length == 5;
        }

        /// <summary>
        /// 자동 재학습 주기(cron)를 Airflow Variable(phm_auto_retrain_schedule)로 저장합니다.
        /// Airflow가 DAG 파일을 다음에 재파싱할 때(보통 수십 초~수 분 내) 반영됩니다.
        /// </summary>
        private async System.Threading.Tasks.Task ApplyAutoRetrainScheduleAsync()
        {
            string cron = _aflScheduleCron?.Text?.Trim();
            if (!IsValidCronOrMacro(cron))
            {
                MessageBox.Show(
                    "주기(cron 표현식)가 올바르지 않습니다.\n" +
                    "예: \"0 2 * * *\" (분 시 일 월 요일, 매일 새벽 2시) 또는 \"@daily\"",
                    "형식 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            string airflowUrl = _aflUrl?.Text?.Trim() ?? Services.ServerSettings.Current.AirflowUrl;
            var s2 = Services.ServerSettings.Current;

            _aflBtnScheduleApply.Enabled = false;
            _aflAutoStatusLbl.ForeColor  = Color.DodgerBlue;
            _aflAutoStatusLbl.Text       = "주기 저장 중…";

            using (var client = new Services.Core.AirflowClient(airflowUrl, s2.AirflowUser, s2.AirflowPassword))
            {
                var result = await client.SetVariableAsync(AutoRetrainScheduleVar, cron);
                if (result.Ok)
                {
                    _aflAutoStatusLbl.Text      = $"주기 저장됨: {cron} (다음 DAG 재스캔 시 반영, 최대 수 분 소요)";
                    _aflAutoStatusLbl.ForeColor = Color.LightGreen;
                    AppendDlLog($"[Airflow] 자동 재학습 주기 변경: {cron} ({AutoRetrainScheduleVar})", Color.LightGreen);
                }
                else
                {
                    _aflAutoStatusLbl.Text      = "주기 저장 실패: " + result.Error;
                    _aflAutoStatusLbl.ForeColor = Color.OrangeRed;
                    AppendDlLog($"[Airflow] 자동 재학습 주기 저장 실패: {result.Error}", Color.OrangeRed);
                }
            }

            _aflBtnScheduleApply.Enabled = true;
        }

        /// <summary>
        /// 자동 재학습 체크박스 On/Off를 phm_auto_retrain DAG의 일시정지 상태로 반영합니다.
        /// On = 일시정지 해제(is_paused=false), Off = 일시정지(is_paused=true).
        /// </summary>
        private async System.Threading.Tasks.Task ToggleAutoRetrainAsync(bool enable)
        {
            string airflowUrl = _aflUrl?.Text?.Trim() ?? Services.ServerSettings.Current.AirflowUrl;
            var s2 = Services.ServerSettings.Current;

            _aflChkAutoRetrain.Enabled  = false;
            _aflAutoStatusLbl.ForeColor = Color.DodgerBlue;
            _aflAutoStatusLbl.Text      = enable ? "켜는 중…" : "끄는 중…";

            using (var client = new Services.Core.AirflowClient(airflowUrl, s2.AirflowUser, s2.AirflowPassword))
            {
                var result = await client.SetDagPausedAsync(AutoRetrainDagId, !enable);
                if (result.Ok)
                {
                    _aflAutoStatusLbl.Text      = enable ? "켜짐" : "꺼짐";
                    _aflAutoStatusLbl.ForeColor = enable ? Color.LightGreen : Color.Gray;
                    AppendDlLog($"[Airflow] 자동 재학습 {(enable ? "활성화" : "비활성화")} ({AutoRetrainDagId})",
                        enable ? Color.LightGreen : Color.Gray);
                }
                else
                {
                    _aflAutoStatusLbl.Text      = "오류: " + result.Error;
                    _aflAutoStatusLbl.ForeColor = Color.OrangeRed;
                    AppendDlLog($"[Airflow] 자동 재학습 전환 실패: {result.Error}", Color.OrangeRed);

                    // 실패했으므로 체크박스를 실제 서버 상태(직전 상태)로 되돌림
                    _aflSyncingAutoRetrain = true;
                    _aflChkAutoRetrain.Checked = !enable;
                    _aflSyncingAutoRetrain = false;
                }
            }

            _aflChkAutoRetrain.Enabled = true;
        }

        private static Color StateColor(string state)
        {
            switch (state?.ToLowerInvariant())
            {
                case "success":  return Color.LightGreen;
                case "running":  return Color.DodgerBlue;
                case "queued":   return Color.Cyan;
                case "failed":
                case "upstream_failed": return Color.OrangeRed;
                default:         return Color.Gray;
            }
        }
    }
}
