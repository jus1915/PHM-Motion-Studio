using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services.Core;
using PHM_Project_DockPanel.Services; // SignalFeatures

namespace PHM_Project_DockPanel.UI.DataAnalysis
{
    /// <summary>
    /// 데이터 수집 → 특징 추출 → 모델 학습/저장 → 실시간 추론
    /// 절차 안내형 올인원 마법사 DockContent
    /// </summary>
    public class PHMPipelineWizard : DockContent
    {
        // ====== 내부 모델 포맷 (저장용) ======
        private sealed class PersistedKnnModel
        {
            public string ModelType { get; set; } = "KNN";
            public int K { get; set; }
            public bool Standardize { get; set; }
            public string[] Features { get; set; }      // Feature keys in order
            public double Threshold { get; set; }       // 저장 임계값
            public double[] Mean { get; set; }          // null if !Standardize
            public double[] Std { get; set; }           // null if !Standardize
            public double[][] Train { get; set; }       // NxD training vectors (RAW)
            public string YColumn { get; set; }         // 특징 추출에 사용한 Y 컬럼명
        }

        private struct FeatureDef { public string Key; public string Title; }
        private static readonly FeatureDef[] FeatureList = new[]
        {
            new FeatureDef{Key="AbsMax",    Title="AbsMax"},
            new FeatureDef{Key="AbsMean",   Title="AbsMean"},
            new FeatureDef{Key="P2P",       Title="P2P"},
            new FeatureDef{Key="RMS",       Title="RMS"},
            new FeatureDef{Key="Skewness",  Title="Skewness"},
            new FeatureDef{Key="Kurtosis",  Title="Kurtosis"},
            new FeatureDef{Key="Crest",     Title="Crest"},
            new FeatureDef{Key="Shape",     Title="Shape"},
            new FeatureDef{Key="Impulse",   Title="Impulse"},
            new FeatureDef{Key="Peak1Freq", Title="1st Freq"},
            new FeatureDef{Key="Peak1Amp",  Title="1st Amp"},
            new FeatureDef{Key="Peak2Freq", Title="2nd Freq"},
            new FeatureDef{Key="Peak2Amp",  Title="2nd Amp"},
            new FeatureDef{Key="Peak3Freq", Title="3rd Freq"},
            new FeatureDef{Key="Peak3Amp",  Title="3rd Amp"},
            new FeatureDef{Key="Peak4Freq", Title="4th Freq"},
            new FeatureDef{Key="Peak4Amp",  Title="4th Amp"},
        };

        // ====== 상태 ======
        private string _csvRoot;
        private string[] _yHeaders = Array.Empty<string>();
        private string _selectedY;

        // 특징 테이블
        private readonly BindingList<SignalFeatures.FeatureRow> _featureRows = new BindingList<SignalFeatures.FeatureRow>();

        // 학습 상태
        private string[] _selectedKeys = Array.Empty<string>();
        private int _k = 5;
        private bool _useStd = true;
        private double[][] _trainVectors;
        private double[] _mu, _sd;
        private double _threshold = 0; // 사용자가 지정 안 하면 LOO 자동 계산

        // ====== UI ======
        private TabControl tabs;

        // Step1
        private TextBox txtCsvRoot;
        private Button btnPickCsv, btnScanCsv;
        private ComboBox cmbY;
        private Label lblCsvSummary;

        // Step2: 특징 추출
        private CheckedListBox clbTime, clbFreq;
        private Button btnExtract;
        private Label lblFeatureSummary;
        private DataGridView gridFeature;

        // Step3: 학습/저장
        private NumericUpDown numK;
        private CheckBox chkStd;
        private Button btnDefaultSelect, btnDefaultClear, btnTrain, btnAutoThr, btnSaveModel;
        private NumericUpDown numManualThr;
        private Label lblTrainInfo;

        // Step4: 실시간 추론
        private Button btnOpenInference;
        private Label lblInferHelp;

        public PHMPipelineWizard()
        {
            Text = "PHM 파이프라인 마법사";
            BuildUI();
        }

        private void BuildUI()
        {
            tabs = new TabControl { Dock = DockStyle.Fill, Alignment = TabAlignment.Top };

            // ---------- STEP 1: 데이터 ----------
            var p1 = new TabPage("1) 데이터 선택");
            {
                var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4, Padding = new Padding(8) };
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

                var row1 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                row1.Controls.Add(new Label { Text = "CSV 루트 폴더:", AutoSize = true, Margin = new Padding(0, 10, 6, 0) });
                txtCsvRoot = new TextBox { Width = 520 };
                btnPickCsv = new Button { Text = "찾기...", Width = 80 };
                btnPickCsv.Click += (s, e) => PickCsvRoot();
                btnScanCsv = new Button { Text = "스캔", Width = 80 };
                btnScanCsv.Click += async (s, e) => await ScanCsvHeaders();
                row1.Controls.AddRange(new Control[] { txtCsvRoot, btnPickCsv, btnScanCsv });

                var row2 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                row2.Controls.Add(new Label { Text = "Y 컬럼:", AutoSize = true, Margin = new Padding(0, 10, 6, 0) });
                cmbY = new ComboBox { DropDownStyle = ComboBoxStyle.DropDownList, Width = 200 };
                cmbY.SelectedIndexChanged += (s, e) => { _selectedY = cmbY.SelectedItem as string; };
                row2.Controls.Add(cmbY);

                lblCsvSummary = new Label { Dock = DockStyle.Fill, AutoSize = false, TextAlign = ContentAlignment.TopLeft };
                var help = new Label
                {
                    Dock = DockStyle.Fill,
                    AutoSize = false,
                    Text = "① CSV 루트 폴더를 지정하면 하위 폴더까지 모두 스캔합니다.\r\n② 첫 CSV의 헤더로부터 Y 컬럼 후보를 불러옵니다.\r\n③ 선택한 Y 컬럼으로 특징을 추출합니다.",
                };

                root.Controls.Add(row1, 0, 0);
                root.Controls.Add(row2, 0, 1);
                root.Controls.Add(lblCsvSummary, 0, 2);
                root.Controls.Add(help, 0, 3);
                p1.Controls.Add(root);
            }

            // ---------- STEP 2: 특징 추출 ----------
            var p2 = new TabPage("2) 특징 추출");
            {
                var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2, Padding = new Padding(8) };
                root.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 320));
                root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 160));
                root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

                // 왼패널 - 피커
                var left = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 5 };
                left.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
                left.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
                left.RowStyles.Add(new RowStyle(SizeType.Absolute, 24));
                left.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
                left.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));

                left.Controls.Add(new Label { Text = "시간 영역", AutoSize = true }, 0, 0);
                clbTime = new CheckedListBox { Dock = DockStyle.Fill, CheckOnClick = true };
                left.Controls.Add(clbTime, 0, 1);
                left.Controls.Add(new Label { Text = "주파수 영역", AutoSize = true }, 0, 2);
                clbFreq = new CheckedListBox { Dock = DockStyle.Fill, CheckOnClick = true };
                left.Controls.Add(clbFreq, 0, 3);

                btnExtract = new Button { Text = "특징 계산/갱신", Dock = DockStyle.Fill, Height = 36 };
                btnExtract.Click += async (s, e) => await ExtractFeaturesAsync();
                left.Controls.Add(btnExtract, 0, 4);

                // 오른패널 - 그리드
                gridFeature = new DataGridView
                {
                    Dock = DockStyle.Fill,
                    ReadOnly = true,
                    AllowUserToAddRows = false,
                    AllowUserToDeleteRows = false,
                    AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                    DataSource = _featureRows
                };

                lblFeatureSummary = new Label { Dock = DockStyle.Fill, AutoSize = false, TextAlign = ContentAlignment.MiddleLeft };

                root.Controls.Add(left, 0, 0);
                root.SetRowSpan(left, 2);
                root.Controls.Add(lblFeatureSummary, 1, 0);
                root.Controls.Add(gridFeature, 1, 1);
                p2.Controls.Add(root);

                PopulateFeaturePickersDefault();
            }

            // ---------- STEP 3: 학습/저장 ----------
            var p3 = new TabPage("3) 학습 ▸ 임계값 ▸ 저장");
            {
                var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4, Padding = new Padding(8) };
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

                var r1 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                r1.Controls.Add(new Label { Text = "k:", AutoSize = true, Margin = new Padding(0, 10, 6, 0) });
                numK = new NumericUpDown { Minimum = 1, Maximum = 50, Value = 5, Width = 60 };
                r1.Controls.Add(numK);
                chkStd = new CheckBox { Text = "표준화", Checked = true, AutoSize = true, Margin = new Padding(12, 8, 0, 0) };
                r1.Controls.Add(chkStd);
                btnDefaultSelect = new Button { Text = "기본특징 선택", Width = 110 };
                btnDefaultSelect.Click += (s, e) => DefaultSelectFeatures(true);
                btnDefaultClear = new Button { Text = "전체 해제", Width = 90 };
                btnDefaultClear.Click += (s, e) => DefaultSelectFeatures(false);
                r1.Controls.Add(btnDefaultSelect);
                r1.Controls.Add(btnDefaultClear);
                btnTrain = new Button { Text = "학습", Width = 80, Margin = new Padding(12, 0, 0, 0) };
                btnTrain.Click += (s, e) => TrainModel();
                r1.Controls.Add(btnTrain);

                var r2 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                btnAutoThr = new Button { Text = "임계값 자동(LOO 99%)", Width = 160 };
                btnAutoThr.Click += (s, e) => AutoThresholdByLOO();
                r2.Controls.Add(btnAutoThr);
                r2.Controls.Add(new Label { Text = "수동 임계값:", AutoSize = true, Margin = new Padding(12, 10, 6, 0) });
                numManualThr = new NumericUpDown { DecimalPlaces = 3, Maximum = 1_000_000, Width = 100, Value = 0 };
                numManualThr.ValueChanged += (s, e) => { _threshold = (double)numManualThr.Value; };
                r2.Controls.Add(numManualThr);

                var r3 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                btnSaveModel = new Button { Text = "모델 저장", Width = 100 };
                btnSaveModel.Click += (s, e) => SaveModel();
                r3.Controls.Add(btnSaveModel);
                lblTrainInfo = new Label { AutoSize = true, Margin = new Padding(16, 8, 0, 0) };
                r3.Controls.Add(lblTrainInfo);

                var help = new Label
                {
                    Dock = DockStyle.Fill,
                    AutoSize = false,
                    Text = "① 특징을 선택하고 학습을 누르세요.\r\n② 검증을 하지 않아도 자동 임계값(LOO 99%)을 계산해 저장할 수 있습니다.\r\n③ 저장된 JSON에는 YColumn/Threshold/Train/Mean/Std/Features/k가 포함됩니다.",
                };

                root.Controls.Add(r1, 0, 0);
                root.Controls.Add(r2, 0, 1);
                root.Controls.Add(r3, 0, 2);
                root.Controls.Add(help, 0, 3);
                p3.Controls.Add(root);
            }

            // ---------- STEP 4: 실시간 추론 ----------
            var p4 = new TabPage("4) 실시간 추론");
            {
                var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3, Padding = new Padding(8) };
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
                root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

                var r1 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                btnOpenInference = new Button { Text = "실시간 추론 창 열기", Width = 160 };
                btnOpenInference.Click += (s, e) => OpenInferenceForm();
                r1.Controls.Add(btnOpenInference);

                var r2 = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
                r2.Controls.Add(new Label { Text = "TIP:", AutoSize = true, Margin = new Padding(0, 10, 6, 0) });
                r2.Controls.Add(new Label
                {
                    Text = "모델 폴더 일괄 로드(axis0.json, axis1.json, ...) → CSV 폴더 선택(하위 포함) → 시작",
                    AutoSize = true,
                    ForeColor = Color.DimGray
                });

                lblInferHelp = new Label
                {
                    Dock = DockStyle.Fill,
                    AutoSize = false,
                    Text = "※ InferenceForm은 각 모델 JSON의 YColumn/Threshold를 사용하며, 움직임이 감지된 축만 해당 축 모델로 추론합니다.",
                };

                root.Controls.Add(r1, 0, 0);
                root.Controls.Add(r2, 0, 1);
                root.Controls.Add(lblInferHelp, 0, 2);
                p4.Controls.Add(root);
            }

            tabs.TabPages.AddRange(new[] { p1, p2, p3, p4 });
            Controls.Add(tabs);

            ClientSize = new Size(1120, 760);
        }

        // ===================== STEP 1: 데이터 =====================

        private void PickCsvRoot()
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Title = "CSV 루트 폴더 선택";
                ofd.ValidateNames = false;     // 파일명이 유효하지 않아도 OK
                ofd.CheckFileExists = false;   // 실제 파일 선택 강제 X → 폴더 선택용
                ofd.CheckPathExists = true;
                ofd.FileName = "폴더 선택";     // 폴더 픽커 트릭

                // 초기 디렉터리: 기존 경로가 유효하면 그곳, 아니면 C:\PHM_Logs
                string start = (!string.IsNullOrEmpty(_csvRoot) && Directory.Exists(_csvRoot))
                    ? _csvRoot
                    : @"C:\Data\PHM_Logs\Signals";

                try { Directory.CreateDirectory(start); } catch { /* ignore */ }
                ofd.InitialDirectory = start;

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    // 사용자가 '열기'를 누른 시점의 폴더 경로
                    string picked = Path.GetDirectoryName(ofd.FileName);
                    if (!string.IsNullOrEmpty(picked))
                    {
                        _csvRoot = picked;
                        txtCsvRoot.Text = _csvRoot;
                    }
                }
            }
        }

        private async Task ScanCsvHeaders()
        {
            try
            {
                if (string.IsNullOrEmpty(_csvRoot) || !Directory.Exists(_csvRoot))
                {
                    MessageBox.Show("CSV 루트 폴더를 먼저 지정하세요.");
                    return;
                }

                var files = Directory.EnumerateFiles(_csvRoot, "*.csv", SearchOption.AllDirectories).ToList();
                if (files.Count == 0)
                {
                    MessageBox.Show("CSV 파일이 없습니다.");
                    return;
                }

                // 모든 CSV의 헤더에서 Y 후보(시간축 제외) 합집합을 수집
                var allHeaders = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                await Task.Run(() =>
                {
                    foreach (var f in files)
                    {
                        string[] headers = null;
                        try { headers = SignalFeatures.GetCsvHeaders(f); } catch { /* skip */ }
                        if (headers == null || headers.Length == 0) continue;

                        foreach (var raw in headers)
                        {
                            if (string.IsNullOrWhiteSpace(raw)) continue;
                            var h = raw.Trim();
                            if (IsTimeColumn(h)) continue;              // 시간축 컬럼 제외
                            if (!allHeaders.ContainsKey(h)) allHeaders[h] = h; // 대소문자 무시 중복 제거
                        }
                    }
                });

                _yHeaders = allHeaders.Values
                                      .OrderBy(x => x, StringComparer.OrdinalIgnoreCase)
                                      .ToArray();

                cmbY.Items.Clear();
                foreach (var h in _yHeaders) cmbY.Items.Add(h);

                if (_yHeaders.Length > 0)
                {
                    cmbY.SelectedIndex = 0;
                    _selectedY = _yHeaders[0];
                }
                else
                {
                    _selectedY = null;
                    MessageBox.Show("Y 컬럼 후보를 찾지 못했습니다. (시간축 컬럼만 있거나 헤더를 읽을 수 없습니다)");
                }

                lblCsvSummary.Text = $"파일 {files.Count}개 스캔 완료 (하위 폴더 포함) · Y 후보 {_yHeaders.Length}개";
                if (_yHeaders.Length > 0) tabs.SelectedIndex = 1; // 다음 단계로 이동
            }
            catch (Exception ex)
            {
                MessageBox.Show("스캔 중 오류: " + ex.Message);
            }
        }

        private static bool IsTimeColumn(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return false;
            var n = name.Trim();
            return n.Equals("time_s", StringComparison.OrdinalIgnoreCase) ||
                   n.Equals("cycle", StringComparison.OrdinalIgnoreCase);
        }

        // ===================== STEP 2: 특징 추출 =====================

        private void PopulateFeaturePickersDefault()
        {
            clbTime.Items.Clear();
            clbFreq.Items.Clear();

            foreach (var f in FeatureList)
            {
                bool isFreq =
                    f.Key.StartsWith("Peak", StringComparison.OrdinalIgnoreCase) ||
                    f.Key.IndexOf("Freq", StringComparison.OrdinalIgnoreCase) >= 0 ||
                    f.Key.IndexOf("Amp", StringComparison.OrdinalIgnoreCase) >= 0;

                // 기본 선택 규칙
                bool defTime =
                    (f.Key == "RMS") || (f.Key == "AbsMean") || (f.Key == "AbsMax") || (f.Key == "P2P") ||
                    (f.Key == "Skewness") || (f.Key == "Kurtosis") ||
                    (f.Key == "Crest") || (f.Key == "Shape") || (f.Key == "Impulse");

                bool defFreq =
                    (f.Key == "Peak1Freq") || (f.Key == "Peak1Amp") ||
                    (f.Key == "Peak2Freq") || (f.Key == "Peak2Amp") ||
                    (f.Key == "Peak3Freq") || (f.Key == "Peak3Amp") ||
                    (f.Key == "Peak4Freq") || (f.Key == "Peak4Amp");

                if (isFreq) clbFreq.Items.Add(f.Title, defFreq);
                else clbTime.Items.Add(f.Title, defTime);
            }
        }

        private string[] GetSelectedKeys()
        {
            var keys = new List<string>();
            for (int i = 0; i < clbTime.Items.Count; i++)
                if (clbTime.GetItemChecked(i)) keys.Add(TitleToKey(clbTime.Items[i].ToString()));
            for (int i = 0; i < clbFreq.Items.Count; i++)
                if (clbFreq.GetItemChecked(i)) keys.Add(TitleToKey(clbFreq.Items[i].ToString()));
            return keys.ToArray();
        }

        private static string TitleToKey(string title)
        {
            for (int i = 0; i < FeatureList.Length; i++)
                if (FeatureList[i].Title == title) return FeatureList[i].Key;
            return title;
        }
        private static string KeyToTitle(string key)
        {
            for (int i = 0; i < FeatureList.Length; i++)
                if (FeatureList[i].Key == key) return FeatureList[i].Title;
            return key;
        }

        private async Task ExtractFeaturesAsync()
        {
            if (string.IsNullOrEmpty(_csvRoot) || !Directory.Exists(_csvRoot))
            {
                MessageBox.Show("CSV 루트 폴더를 먼저 지정하세요.");
                return;
            }
            if (string.IsNullOrEmpty(_selectedY))
            {
                MessageBox.Show("Y 컬럼을 선택하세요.");
                return;
            }

            var files = Directory.EnumerateFiles(_csvRoot, "*.csv", SearchOption.AllDirectories).ToList();
            if (files.Count == 0)
            {
                MessageBox.Show("CSV 파일이 없습니다.");
                return;
            }

            var selectedKeys = GetSelectedKeys();
            if (selectedKeys.Length == 0)
            {
                MessageBox.Show("특징을 하나 이상 선택하세요.");
                return;
            }

            btnExtract.Enabled = false;
            UseWaitCursor = true;
            lblFeatureSummary.Text = $"특징 계산 중... (0/{files.Count})";

            try
            {
                // 1) 백그라운드에서 결과만 수집 (UI/BindingList 절대 접근 금지)
                var results = await Task.Run(() =>
                {
                    var list = new List<SignalFeatures.FeatureRow>(capacity: Math.Min(files.Count, 4096));
                    int processed = 0;

                    foreach (var path in files)
                    {
                        try
                        {
                            if (!SignalFeatures.TryParseCsvColumn(path, _selectedY, out var ys) || ys.Count < 4)
                                continue;

                            var sr = AppState.GetForColumn(_selectedY);           // Y 컬럼명으로 가속도/토크 구분
                            var fr = SignalFeatures.ExtractFeatures(ys, sr, maxSamples: 16384);
                            if (fr == null) continue;

                            fr.FileName = MakeRelativePath(_csvRoot, path);
                            list.Add(fr);
                        }
                        catch
                        {
                            // 파일 하나 실패는 무시
                        }
                        finally
                        {
                            processed++;
                            // 진행률은 너무 자주 갱신하지 말고 가끔만 (예: 50개마다)
                            if (processed % 50 == 0)
                            {
                                int p = processed;
                                this.BeginInvoke(new Action(() =>
                                {
                                    lblFeatureSummary.Text = $"특징 계산 중... ({p}/{files.Count})";
                                }));
                            }
                        }
                    }
                    return list;
                });

                // 2) UI 스레드에서 한 번에 바인딩 (프리징 방지)
                _featureRows.RaiseListChangedEvents = false;
                _featureRows.Clear();
                foreach (var row in results) _featureRows.Add(row);
                _featureRows.RaiseListChangedEvents = true;
                _featureRows.ResetBindings();

                lblFeatureSummary.Text = $"특징 샘플: {_featureRows.Count}개 (하위 폴더 포함)";
                tabs.SelectedIndex = 2; // Step3로 이동
            }
            finally
            {
                UseWaitCursor = false;
                btnExtract.Enabled = true;
            }
        }

        private static string MakeRelativePath(string root, string full)
        {
            try
            {
                var r = new Uri(AppendDirSep(root));
                var f = new Uri(full);
                return Uri.UnescapeDataString(r.MakeRelativeUri(f).ToString()).Replace('/', Path.DirectorySeparatorChar);
            }
            catch { return Path.GetFileName(full); }

            string AppendDirSep(string p) => p.EndsWith(Path.DirectorySeparatorChar.ToString()) ? p : p + Path.DirectorySeparatorChar;
        }

        // ===================== STEP 3: 학습/저장 =====================

        private void DefaultSelectFeatures(bool select)
        {
            for (int i = 0; i < clbTime.Items.Count; i++) clbTime.SetItemChecked(i, select);
            for (int i = 0; i < clbFreq.Items.Count; i++) clbFreq.SetItemChecked(i, select);
        }

        private void TrainModel()
        {
            _selectedKeys = GetSelectedKeys();
            if (_selectedKeys.Length == 0)
            {
                MessageBox.Show("특징을 하나 이상 선택하세요.");
                return;
            }
            if (_featureRows.Count == 0)
            {
                MessageBox.Show("특징 테이블이 비었습니다. 먼저 '특징 계산/갱신'을 실행하세요.");
                return;
            }

            // 모든 샘플을 정상(0)으로 간주하여 훈련(비지도 이상탐지 시나리오)
            var normals = _featureRows.ToList(); // Label UX 단순화를 위해 모두 정상 처리

            // 학습 벡터 구성 (선택 키 → row 속성 접근)
            var list = new List<double[]>();
            foreach (var r in normals)
            {
                var x = new double[_selectedKeys.Length];
                bool bad = false;
                for (int j = 0; j < _selectedKeys.Length; j++)
                {
                    double? v = GetValueByKey(r, _selectedKeys[j]);
                    if (!v.HasValue || double.IsNaN(v.Value) || double.IsInfinity(v.Value)) { bad = true; break; }
                    x[j] = v.Value;
                }
                if (!bad) list.Add(x);
            }

            if (list.Count < 3)
            {
                MessageBox.Show("유효한 정상 샘플이 부족합니다(최소 3개 이상 권장).");
                return;
            }

            _trainVectors = list.ToArray();
            _k = (int)numK.Value;
            _useStd = chkStd.Checked;

            _mu = null; _sd = null;
            if (_useStd)
            {
                int d = _selectedKeys.Length;
                _mu = new double[d];
                _sd = new double[d];

                for (int j = 0; j < d; j++)
                {
                    double mu = 0;
                    for (int i = 0; i < _trainVectors.Length; i++) mu += _trainVectors[i][j];
                    mu /= Math.Max(1, _trainVectors.Length);

                    double var = 0;
                    for (int i = 0; i < _trainVectors.Length; i++)
                    {
                        double t = _trainVectors[i][j] - mu;
                        var += t * t;
                    }
                    var /= Math.Max(1, _trainVectors.Length - 1);
                    double sd = Math.Sqrt(var);
                    _mu[j] = mu;
                    _sd[j] = (sd <= 1e-12) ? 1.0 : sd;
                }
            }

            lblTrainInfo.Text = $"학습 완료 — 정상:{_trainVectors.Length} / feat:{_selectedKeys.Length} / k={_k} / 표준화={_useStd}";
            // 임계값 초기화
            if (_threshold <= 0) numManualThr.Value = 0;
        }

        private double? GetValueByKey(SignalFeatures.FeatureRow r, string key)
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

        private void AutoThresholdByLOO()
        {
            if (_trainVectors == null || _trainVectors.Length < 3)
            {
                MessageBox.Show("학습부터 진행하세요. (정상 샘플 최소 3개 이상)");
                return;
            }

            var loo = ComputeTrainScoresLOO();
            if (loo.Length >= 2)
            {
                _threshold = Percentile(loo, 0.99);
                if (_threshold < 0) _threshold = 0;
                try { numManualThr.Value = (decimal)_threshold; } catch { /* ignore */ }
                MessageBox.Show($"자동 임계값(LOO 99%) = {_threshold:F3}");
            }
            else
            {
                MessageBox.Show("LOO 임계값 계산 불가(샘플 부족).");
            }
        }

        private double[] ComputeTrainScoresLOO()
        {
            var arr = new double[_trainVectors.Length];
            for (int i = 0; i < _trainVectors.Length; i++)
            {
                if (_trainVectors.Length <= 1) { arr[i] = 0; continue; }
                var tmp = new double[_trainVectors.Length - 1][];
                int k = 0;
                for (int j = 0; j < _trainVectors.Length; j++)
                    if (j != i) tmp[k++] = _trainVectors[j];

                int kEff = Math.Min(_k, tmp.Length);
                if (kEff <= 0) { arr[i] = 0; continue; }

                arr[i] = SignalFeatures.ScoreKnn(_trainVectors[i], tmp, kEff, _useStd, _mu, _sd);
            }
            return arr;
        }

        private static double Percentile(double[] arr, double p)
        {
            if (arr == null || arr.Length == 0) return 0;
            if (p <= 0) return arr.Min();
            if (p >= 1) return arr.Max();
            var copy = (double[])arr.Clone();
            Array.Sort(copy);
            double pos = (copy.Length - 1) * p;
            int i = (int)Math.Floor(pos);
            double f = pos - i;
            if (i + 1 < copy.Length) return copy[i] * (1.0 - f) + copy[i + 1] * f;
            return copy[i];
        }

        private void SaveModel()
        {
            if (_trainVectors == null || _trainVectors.Length == 0)
            {
                MessageBox.Show("학습을 먼저 수행하세요.");
                return;
            }
            if (_selectedKeys == null || _selectedKeys.Length == 0)
            {
                MessageBox.Show("특징 선택이 비었습니다.");
                return;
            }
            if (string.IsNullOrEmpty(_selectedY))
            {
                MessageBox.Show("Y 컬럼이 비어 있습니다. 1단계에서 선택하세요.");
                return;
            }

            double thr = _threshold > 0 ? _threshold : (double)numManualThr.Value;
            if (thr <= 0)
            {
                // 자동 계산 fallback
                var loo = ComputeTrainScoresLOO();
                if (loo.Length >= 2) thr = Percentile(loo, 0.99);
            }
            if (thr <= 0)
            {
                MessageBox.Show("임계값을 결정하지 못했습니다. 자동 계산 또는 수동 입력을 사용하세요.");
                return;
            }

            var payload = new PersistedKnnModel
            {
                K = _k,
                Standardize = _useStd,
                Features = _selectedKeys,
                Threshold = thr,
                Mean = _mu,
                Std = _sd,
                Train = _trainVectors,
                YColumn = _selectedY
            };

            using (var sfd = new SaveFileDialog
            {
                Filter = "PHM Model (*.json)|*.json",
                FileName = "axis0.json"
            })
            {
                if (sfd.ShowDialog() != DialogResult.OK) return;

                var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(sfd.FileName, json);
                MessageBox.Show("모델 저장 완료:\r\n" + sfd.FileName);
                tabs.SelectedIndex = 3;
            }
        }

        // ===================== STEP 4: 실시간 추론 =====================

        private void OpenInferenceForm()
        {
            // 기존 InferenceForm을 사용(축별 모델, YColumn, 임계값, 움직임 판정, 하위 폴더 감시 포함)
            var existing = this.DockPanel?.Contents.OfType<InferenceForm>().FirstOrDefault();
            InferenceForm f = existing ?? new InferenceForm();

            if (this.DockPanel == null)
            {
                if (!f.Visible) f.Show(this);
                f.Activate();
            }
            else
            {
                if (!f.Visible) f.Show(this.DockPanel, DockState.Document);
                f.DockHandler.Activate();
            }

            MessageBox.Show(
                "1) [모델 폴더 일괄]로 axis0.json, axis1.json ... 로드\n" +
                "2) [CSV 폴더] 선택 (하위 폴더 포함)\n" +
                "3) [시작] 클릭 → 움직인 축만 해당 축 모델로 추론\n\n" +
                "※ 모델 JSON에는 YColumn/Threshold가 저장되어 있으며, 임계선은 축별로 표시됩니다.",
                "실시간 추론 안내", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
