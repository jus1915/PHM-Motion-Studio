using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services.Core; // SignalFeatures

namespace PHM_Project_DockPanel.UI.DataAnalysis
{
    public class InferenceForm : DockContent
    {
        // ===== 모델 포맷 (kNN 전용, 확장 가능) =====
        private class PersistedKnnModel
        {
            public string ModelType { get; set; }       // "KNN"
            public int K { get; set; }
            public bool Standardize { get; set; }
            public string[] Features { get; set; }      // Feature keys in order
            public double Threshold { get; set; }       // 축별 임계값
            public double[] Mean { get; set; }          // null if !Standardize
            public double[] Std { get; set; }           // null if !Standardize
            public double[][] Train { get; set; }       // NxD training vectors (RAW)
            public string YColumn { get; set; }         // 특징 추출에 사용한 Y 컬럼명
        }

        // 축 모델 엔트리
        private class AxisModel
        {
            public int AxisId { get; set; }
            public string ModelPath { get; set; }
            public PersistedKnnModel Model { get; set; }
        }

        private struct AxisCol { public int AxisId; public int ColIndex; }

        // ===== 설정 상수 =====
        private const bool WatchSubdirectories = true;
        private const int ChartKeepPoints = 300;

        // 움직임 판정용 컬럼 패턴 (환경에 맞게 보완 가능)
        private static readonly Regex[] AxisPosRegexes =
        {
            new Regex(@"^CMDPOS(?<id>\d+)$", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex(@"^POS(?<id>\d+)$",    RegexOptions.IgnoreCase | RegexOptions.Compiled),
        };

        // ====== UI ======
        private Button btnLoadModelSingle, btnLoadModelFolder, btnSelectFolder, btnStart, btnStop;
        private NumericUpDown numThresholdDefault, numMotionEps;
        private Label lblFolder, lblStatus;
        private Chart chartScore;
        private DataGridView gridLive, gridAxisModels;

        // ====== 상태 ======
        private string _watchFolder;
        private FileSystemWatcher _watcher;

        private readonly Dictionary<int, AxisModel> _axisModels = new Dictionary<int, AxisModel>();

        private readonly HashSet<string> _processing = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, CancellationTokenSource> _debouncers = new Dictionary<string, CancellationTokenSource>(StringComparer.OrdinalIgnoreCase);
        private readonly object _sync = new object();

        private readonly HashSet<string> _processedOnce = new HashSet<string>(StringComparer.OrdinalIgnoreCase);


        public InferenceForm()
        {
            Text = "실시간 추론 (멀티축 / 모델 YColumn 사용 / 축별 임계)";
            BuildUI();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            try { StopWatch(); } catch { }
            base.OnFormClosing(e);
        }

        // ========================= UI 구성 =========================
        private void BuildUI()
        {
            var root = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4 };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 56));
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 140));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 50));

            // 상단 바
            var top = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false, Padding = new Padding(6, 8, 6, 6) };

            btnLoadModelSingle = new Button { Text = "축별 모델 추가", Width = 120 };
            btnLoadModelSingle.Click += (s, e) => LoadAxisModelSingle();
            top.Controls.Add(btnLoadModelSingle);

            btnLoadModelFolder = new Button { Text = "모델 폴더 일괄", Width = 120, Margin = new Padding(6, 0, 0, 0) };
            btnLoadModelFolder.Click += (s, e) => LoadAxisModelsFromFolder();
            top.Controls.Add(btnLoadModelFolder);

            top.Controls.Add(new Label { Text = "기본 임계값:", AutoSize = true, Margin = new Padding(12, 8, 4, 0) });
            numThresholdDefault = new NumericUpDown { DecimalPlaces = 3, Maximum = 1_000_000, Width = 100, Value = 1 };
            numThresholdDefault.ValueChanged += (s, e) => DrawThresholdLines();
            top.Controls.Add(numThresholdDefault);

            top.Controls.Add(new Label { Text = "움직임 판정(Δpos):", AutoSize = true, Margin = new Padding(12, 8, 4, 0) });
            numMotionEps = new NumericUpDown { DecimalPlaces = 3, Maximum = 1000, Width = 100, Value = 0.010M };
            numMotionEps.ValueChanged += (s, e) => { /* UI만, 판정은 파일 읽을 때 적용 */ };
            top.Controls.Add(numMotionEps);

            btnSelectFolder = new Button { Text = "CSV 폴더", Width = 90, Margin = new Padding(12, 0, 0, 0) };
            btnSelectFolder.Click += (s, e) => SelectFolder();
            top.Controls.Add(btnSelectFolder);

            btnStart = new Button { Text = "시작", Width = 80, Margin = new Padding(6, 0, 0, 0) };
            btnStop = new Button { Text = "중지", Width = 80, Enabled = false, Margin = new Padding(6, 0, 0, 0) };
            btnStart.Click += (s, e) => StartWatch();
            btnStop.Click += (s, e) => StopWatch();
            top.Controls.Add(btnStart);
            top.Controls.Add(btnStop);

            lblFolder = new Label { AutoSize = true, Margin = new Padding(12, 8, 0, 0) };
            lblStatus = new Label { AutoSize = true, Margin = new Padding(12, 8, 0, 0) };
            top.Controls.Add(lblFolder);
            top.Controls.Add(lblStatus);

            // 축-모델 매핑 그리드
            gridAxisModels = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = false,  // Threshold 편집 허용
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };
            var dtAxis = new DataTable();
            dtAxis.Columns.Add("Axis", typeof(int));
            dtAxis.Columns.Add("ModelPath", typeof(string));
            dtAxis.Columns.Add("YColumn", typeof(string));
            dtAxis.Columns.Add("Threshold", typeof(double));   // ★ 축별 임계값 편집 컬럼
            dtAxis.Columns.Add("K/Std/Feat/Train", typeof(string));
            gridAxisModels.DataSource = dtAxis;
            gridAxisModels.DataBindingComplete += OnAxisGridBound;
            gridAxisModels.CellEndEdit += GridAxisModels_CellEndEdit;

            // 점수 차트 (축별 시리즈 동적 생성)
            chartScore = new Chart { Dock = DockStyle.Fill };
            var ca = new ChartArea("Score");
            ca.AxisX.Title = "Time";
            ca.AxisX.LabelStyle.Format = "HH:mm:ss";
            ca.AxisX.IntervalAutoMode = IntervalAutoMode.VariableCount;
            ca.AxisY.Title = "Anomaly Score";
            chartScore.ChartAreas.Add(ca);
            chartScore.Legends.Clear();
            chartScore.Legends.Add(new Legend { Docking = Docking.Top, Alignment = StringAlignment.Near });

            // 최근 결과 테이블
            gridLive = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill
            };
            var dt = new DataTable();
            dt.Columns.Add("Time", typeof(string));
            dt.Columns.Add("Axis", typeof(int));
            dt.Columns.Add("File", typeof(string));
            dt.Columns.Add("Score", typeof(double));
            dt.Columns.Add("Decision", typeof(string));
            gridLive.DataSource = dt;
            foreach (DataGridViewColumn col in gridLive.Columns)
            {
                if (col.ValueType == typeof(double))
                {
                    col.DefaultCellStyle.Format = "F3";
                    col.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
                }
            }

            root.Controls.Add(top, 0, 0);
            root.Controls.Add(gridAxisModels, 0, 1);
            root.Controls.Add(chartScore, 0, 2);
            root.Controls.Add(gridLive, 0, 3);
            Controls.Add(root);

            ClientSize = new Size(1200, 800);
        }

        // ========= Axis Grid 편집 가능 컬럼 설정 =========
        private void OnAxisGridBound(object sender, DataGridViewBindingCompleteEventArgs e)
        {
            foreach (DataGridViewColumn col in gridAxisModels.Columns)
            {
                col.ReadOnly = col.Name != "Threshold";
                if (col.Name == "Threshold")
                {
                    col.DefaultCellStyle.Format = "F3";
                    col.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
                }
            }
        }

        private void GridAxisModels_CellEndEdit(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex < 0) return;
            var grid = gridAxisModels;
            var dt = grid.DataSource as DataTable;
            if (dt == null) return;
            if (grid.Columns[e.ColumnIndex].Name != "Threshold") return;

            try
            {
                int axis = Convert.ToInt32(dt.Rows[e.RowIndex]["Axis"]);
                double thr = Convert.ToDouble(dt.Rows[e.RowIndex]["Threshold"], CultureInfo.InvariantCulture);
                AxisModel am;
                if (_axisModels.TryGetValue(axis, out am) && am != null && am.Model != null)
                {
                    am.Model.Threshold = thr;   // 모델 임계값 갱신
                    DrawThresholdLines();       // 차트 반영
                }
            }
            catch { /* ignore */ }
        }

        // ========================= 모델 로드 =========================

        private void LoadAxisModelSingle()
        {
            using (var dlgAxis = new InputBox("축 번호 입력", "모델을 연결할 축 번호(0,1,2,...)를 입력하세요:"))
            {
                if (dlgAxis.ShowDialog() != DialogResult.OK) return;
                int axis;
                if (!int.TryParse(dlgAxis.InputText, out axis) || axis < 0)
                {
                    MessageBox.Show("유효한 축 번호가 아닙니다.");
                    return;
                }

                using (var ofd = new OpenFileDialog { Filter = "PHM Model (*.json)|*.json|All files (*.*)|*.*" })
                {
                    if (ofd.ShowDialog() != DialogResult.OK) return;

                    PersistedKnnModel model;
                    string err;
                    if (!TryLoadModelFromPath(ofd.FileName, out model, out err))
                    {
                        MessageBox.Show("모델 로드 실패: " + err);
                        return;
                    }

                    _axisModels[axis] = new AxisModel { AxisId = axis, ModelPath = ofd.FileName, Model = model };
                    RefreshAxisGrid();
                    DrawThresholdLines();
                }
            }
        }

        private void LoadAxisModelsFromFolder()
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Title = "모델 폴더 선택";
                ofd.ValidateNames = false;
                ofd.CheckFileExists = false;
                ofd.CheckPathExists = true;
                ofd.FileName = "폴더 선택";              // 폴더 픽커 트릭

                // 초기 디렉터리: C:\PHM_Logs (없으면 생성)
                string start = @"E:\Data\PHM_Logs";
                try { Directory.CreateDirectory(start); } catch { /* ignore */ }
                ofd.InitialDirectory = start;

                if (ofd.ShowDialog() != DialogResult.OK) return;

                // 사용자가 '열기'를 누른 시점의 폴더
                string folder = Path.GetDirectoryName(ofd.FileName);
                if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder)) return;

                int loaded = 0, failed = 0;

                // 하위 폴더까지 포함하려면 SearchOption.AllDirectories 로 변경하세요.
                foreach (var file in Directory.EnumerateFiles(folder, "*.json", SearchOption.TopDirectoryOnly))
                {
                    var name = Path.GetFileNameWithoutExtension(file);
                    var m = Regex.Match(name, @"axis(?<id>\d+)", RegexOptions.IgnoreCase);
                    if (!m.Success) continue;

                    if (!int.TryParse(m.Groups["id"].Value, out int axis)) continue;

                    if (!TryLoadModelFromPath(file, out PersistedKnnModel model, out string err))
                    {
                        failed++;
                        continue;
                    }

                    _axisModels[axis] = new AxisModel { AxisId = axis, ModelPath = file, Model = model };
                    loaded++;
                }

                RefreshAxisGrid();
                DrawThresholdLines();
                MessageBox.Show($"일괄 로드 완료: {loaded}개 성공, {failed}개 실패\n경로: {folder}");
            }
        }

        private bool TryLoadModelFromPath(string path, out PersistedKnnModel model, out string error)
        {
            model = null; error = null;
            try
            {
                string json = File.ReadAllText(path);
                var m = JsonSerializer.Deserialize<PersistedKnnModel>(json);

                if (m == null || !string.Equals(m.ModelType, "KNN_AD", StringComparison.OrdinalIgnoreCase) ||
                    m.Features == null || m.Train == null)
                {
                    error = "지원하지 않는 모델 형식(KNN JSON만).";
                    return false;
                }
                if (m.Standardize && (m.Mean == null || m.Std == null))
                {
                    error = "표준화 파라미터(Mean/Std)가 누락되었습니다.";
                    return false;
                }
                if (string.IsNullOrEmpty(m.YColumn))
                {
                    error = "모델 JSON에 YColumn이 없습니다. (AIForm에서 저장한 모델인지 확인)";
                    return false;
                }

                model = m;
                return true;
            }
            catch (Exception ex)
            {
                error = ex.Message;
                return false;
            }
        }

        private void RefreshAxisGrid()
        {
            var dt = (DataTable)gridAxisModels.DataSource;
            dt.Rows.Clear();
            foreach (var kv in _axisModels.OrderBy(k => k.Key))
            {
                var m = kv.Value.Model;
                string meta = "k=" + m.K + ", Std=" + (m.Standardize ? "Y" : "N") +
                              ", Feat=" + (m.Features != null ? m.Features.Length : 0) +
                              ", Train=" + (m.Train != null ? m.Train.Length : 0);

                dt.Rows.Add(kv.Key, kv.Value.ModelPath, m.YColumn, m.Threshold, meta);
            }
            gridAxisModels.Refresh();
        }

        // ========================= CSV 폴더 & Watcher =========================

        private void SelectFolder()
        {
            using (var fbd = new FolderBrowserDialog())
            {
                if (!string.IsNullOrEmpty(_watchFolder) && Directory.Exists(_watchFolder))
                    fbd.SelectedPath = _watchFolder;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    _watchFolder = fbd.SelectedPath;
                    lblFolder.Text = "폴더: " + _watchFolder;

                    lock (_sync)
                    {
                        _processing.Clear();
                        foreach (var kv in _debouncers) kv.Value.Cancel();
                        _debouncers.Clear();

                        // ★ 새 폴더로 바꿨으니 '이미 처리됨' 집합 초기화
                        _processedOnce.Clear();
                    }
                }
            }
        }

        private void StartWatch()
        {
            if (_axisModels.Count == 0)
            {
                MessageBox.Show("먼저 축별 모델을 추가하세요.");
                return;
            }
            if (string.IsNullOrEmpty(_watchFolder) || !Directory.Exists(_watchFolder))
            {
                MessageBox.Show("CSV 폴더를 먼저 선택하세요.");
                return;
            }
            foreach (var am in _axisModels.Values)
            {
                if (am.Model == null || string.IsNullOrEmpty(am.Model.YColumn))
                {
                    MessageBox.Show("축 " + am.AxisId + " 모델에 YColumn 정보가 없습니다.");
                    return;
                }
            }

            StopWatch(); // 기존 watcher 정리

            _watcher = new FileSystemWatcher(_watchFolder, "*.csv")
            {
                IncludeSubdirectories = WatchSubdirectories,
                NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.CreationTime
            };
            _watcher.Created += OnFileCreatedOrChanged;
            _watcher.Changed += OnFileCreatedOrChanged;
            _watcher.Renamed += OnFileRenamed;
            _watcher.EnableRaisingEvents = true;

            btnStart.Enabled = false;
            btnStop.Enabled = true;
            lblStatus.Text = "상태: 수집 중..." + (WatchSubdirectories ? " (하위 폴더 포함)" : "");
        }

        private void StopWatch()
        {
            if (_watcher != null)
            {
                try
                {
                    _watcher.EnableRaisingEvents = false;
                    _watcher.Created -= OnFileCreatedOrChanged;
                    _watcher.Changed -= OnFileCreatedOrChanged;
                    _watcher.Renamed -= OnFileRenamed;
                    _watcher.Dispose();
                }
                catch { }
                finally { _watcher = null; }
            }

            lock (_sync)
            {
                foreach (var kv in _debouncers) kv.Value.Cancel();
                _debouncers.Clear();
                _processing.Clear();
            }

            btnStart.Enabled = true;
            btnStop.Enabled = false;
            lblStatus.Text = "상태: 중지";
        }

        private void OnFileRenamed(object sender, RenamedEventArgs e)
        {
            OnFileCreatedOrChanged(sender, new FileSystemEventArgs(WatcherChangeTypes.Changed, Path.GetDirectoryName(e.FullPath), Path.GetFileName(e.FullPath)));
        }

        private void OnFileCreatedOrChanged(object sender, FileSystemEventArgs e)
        {
            try
            {
                CancellationTokenSource cts;
                lock (_sync)
                {
                    // ★ 이미 처리된 파일은 디바운스 스케줄 자체를 건지지 않음
                    if (_processedOnce.Contains(e.FullPath)) return;

                    if (_debouncers.TryGetValue(e.FullPath, out var old))
                    {
                        try { old.Cancel(); } catch { }
                    }
                    cts = new CancellationTokenSource();
                    _debouncers[e.FullPath] = cts;
                }

                // 300ms 디바운스 후 처리
                Task.Delay(300, cts.Token).ContinueWith(t =>
                {
                    if (t.IsCanceled) return;
                    ProcessCsvSafe(e.FullPath);
                    lock (_sync) { _debouncers.Remove(e.FullPath); }
                }, TaskScheduler.Default);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex);
            }
        }

        // ========================= 파일 처리 =========================

        private void ProcessCsvSafe(string path)
        {
            // ★ 혹시 이미 스케줄된 태스크가 남아 있어도, 진입 시 한 번 더 차단
            lock (_sync) { if (_processedOnce.Contains(path)) return; }

            if (!TryMarkProcessing(path)) return;

            try
            {
                WaitUntilStable(path, timeoutMs: 5000, sampleMs: 200, stableSamples: 3);

                string[] headers = Retry(() => SignalFeatures.GetCsvHeaders(path), 5, 100);
                if (headers == null || headers.Length == 0) return;

                var headerSet = new HashSet<string>(
                    headers.Select(h => h?.Trim()).Where(h => !string.IsNullOrEmpty(h)),
                    StringComparer.OrdinalIgnoreCase);

                bool anyAxisProcessed = false; // ★ 실제로 한 축이라도 스코어링하면 '처리 완료'로 본다.

                foreach (var kv in _axisModels.OrderBy(k => k.Key))
                {
                    int axis = kv.Key;
                    var axisModel = kv.Value;
                    if (axisModel?.Model == null) continue;

                    var m = axisModel.Model;
                    string yColumn = m.YColumn;

                    if (string.IsNullOrWhiteSpace(yColumn) || !headerSet.Contains(yColumn))
                        continue;

                    double[] sample = Retry(() => BuildFeatureVectorFromCsv(path, yColumn, m.Features), 5, 100);
                    if (sample == null) continue;

                    double score = SignalFeatures.ScoreKnn(sample, m.Train, m.K, m.Standardize, m.Mean, m.Std);
                    double thr = (m.Threshold > 0) ? m.Threshold : (double)numThresholdDefault.Value;
                    bool isAnom = score >= thr;

                    anyAxisProcessed = true; // ★ 최소 1축 처리됨

                    this.BeginInvoke(new Action(() =>
                    {
                        AppendScoreToChart(axis, score, DateTime.Now);
                        AddRowToGrid(axis, Path.GetFileName(path), score, isAnom);
                        lblStatus.Text = "상태: 처리완료 " + DateTime.Now.ToString("HH:mm:ss") +
                                         " (axis " + axis + ", " + Path.GetFileName(path) + ")";
                    }));
                }

                // ★ 처리 성공으로 간주되는 경우에만 '이미 처리됨'으로 마킹
                if (anyAxisProcessed)
                {
                    lock (_sync) { _processedOnce.Add(path); }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex);
            }
            finally
            {
                UnmarkProcessing(path);
            }
        }

        private static bool WaitUntilStable(string path, int timeoutMs = 5000, int sampleMs = 200, int stableSamples = 3)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            long? lastLen = null;
            int stableCount = 0;

            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                long len;
                DateTime lastWrite;
                try
                {
                    var fi = new FileInfo(path);
                    fi.Refresh();
                    len = fi.Length;
                    lastWrite = fi.LastWriteTimeUtc;
                }
                catch
                {
                    System.Threading.Thread.Sleep(sampleMs);
                    continue;
                }

                if (lastLen.HasValue && len == lastLen.Value)
                {
                    stableCount++;
                    if (stableCount >= stableSamples) return true; // 충분히 안정
                }
                else
                {
                    stableCount = 0;
                    lastLen = len;
                }

                System.Threading.Thread.Sleep(sampleMs);
            }
            return false; // 시간 초과: 그래도 읽고 싶으면 true로 바꾸거나 호출부에서 재시도
        }

        /// <summary>
        /// 헤더/CSV 내용을 기준으로 “움직인 축” 탐지.
        /// 정책: CMDPOS{n} 또는 POS{n} 열의 (max-min) > motionEps 면 움직임으로 간주.
        /// </summary>
        private List<int> DetermineMovedAxes(string filePath, string[] headers, double motionEps)
        {
            var axisCols = new List<AxisCol>();
            for (int i = 0; i < headers.Length; i++)
            {
                for (int r = 0; r < AxisPosRegexes.Length; r++)
                {
                    var m = AxisPosRegexes[r].Match(headers[i]);
                    int axisIdParsed;
                    if (m.Success && int.TryParse(m.Groups["id"].Value, out axisIdParsed))
                    {
                        axisCols.Add(new AxisCol { AxisId = axisIdParsed, ColIndex = i });
                        break;
                    }
                }
            }
            if (axisCols.Count == 0) return new List<int>();

            var moved = new HashSet<int>();
            try
            {
                using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs))
                {
                    sr.ReadLine(); // header

                    var mins = new Dictionary<int, double>();
                    var maxs = new Dictionary<int, double>();
                    for (int i = 0; i < axisCols.Count; i++)
                    {
                        if (!mins.ContainsKey(axisCols[i].AxisId)) mins[axisCols[i].AxisId] = double.PositiveInfinity;
                        if (!maxs.ContainsKey(axisCols[i].AxisId)) maxs[axisCols[i].AxisId] = double.NegativeInfinity;
                    }

                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        string[] cells = SplitCsvLine(line);
                        for (int i = 0; i < axisCols.Count; i++)
                        {
                            int AxisId = axisCols[i].AxisId;
                            int ColIndex = axisCols[i].ColIndex;

                            if (ColIndex >= cells.Length) continue;

                            double v;
                            if (double.TryParse(cells[ColIndex], NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                            {
                                if (v < mins[AxisId]) mins[AxisId] = v;
                                if (v > maxs[AxisId]) maxs[AxisId] = v;
                            }
                        }
                    }

                    foreach (int axisId in mins.Keys.ToList())
                    {
                        if (double.IsInfinity(mins[axisId]) || double.IsInfinity(maxs[axisId])) continue;
                        if (maxs[axisId] - mins[axisId] > motionEps)
                            moved.Add(axisId);
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex);
            }

            return moved.OrderBy(a => a).ToList();
        }

        // ========================= Chart (축별) =========================

        private Series EnsureScoreSeries(int axis)
        {
            string name = "Score_axis" + axis;
            var s = chartScore.Series.FindByName(name);
            if (s == null)
            {
                s = new Series(name)
                {
                    ChartType = SeriesChartType.FastLine,
                    BorderWidth = 2,
                    IsVisibleInLegend = true,
                    LegendText = "Score a" + axis,
                    XValueType = ChartValueType.DateTime
                };
                chartScore.Series.Add(s);
            }
            return s;
        }

        private Series EnsureThrSeries(int axis)
        {
            string name = "Thr_axis" + axis;
            var s = chartScore.Series.FindByName(name);
            if (s == null)
            {
                s = new Series(name)
                {
                    ChartType = SeriesChartType.FastLine,
                    BorderDashStyle = ChartDashStyle.Dash,
                    IsVisibleInLegend = true,
                    LegendText = "Thr a" + axis,
                    XValueType = ChartValueType.DateTime
                };
                chartScore.Series.Add(s);
            }
            return s;
        }

        private void AppendScoreToChart(int axis, double score, DateTime when)
        {
            var s = EnsureScoreSeries(axis);
            s.Points.AddXY(when.ToOADate(), score);

            while (s.Points.Count > ChartKeepPoints) s.Points.RemoveAt(0);

            var area = chartScore.ChartAreas["Score"];
            area.AxisX.Minimum = double.NaN; area.AxisX.Maximum = double.NaN;
            area.AxisY.Minimum = double.NaN; area.AxisY.Maximum = double.NaN;
            area.RecalculateAxesScale();

            DrawThresholdLines();
        }

        private void DrawThresholdLines()
        {
            // X 구간: 모든 축 score 시리즈 구간의 합집합
            double? xmin = null, xmax = null;
            foreach (var s in chartScore.Series.Cast<Series>().Where(ss => ss.Name.StartsWith("Score_axis")))
            {
                if (s.Points.Count == 0) continue;
                double sxmin = s.Points[0].XValue;
                double sxmax = s.Points[s.Points.Count - 1].XValue;
                xmin = xmin.HasValue ? Math.Min(xmin.Value, sxmin) : sxmin;
                xmax = xmax.HasValue ? Math.Max(xmax.Value, sxmax) : sxmax;
            }

            if (!xmin.HasValue || !xmax.HasValue)
            {
                // 점수 시리즈가 아직 없으면 모델만큼 임계선 시리즈 초기화(짧은 라인)
                double now = DateTime.Now.ToOADate();
                foreach (var kv in _axisModels)
                {
                    int axis = kv.Key;
                    var model = kv.Value.Model;
                    double thr = (model != null && model.Threshold > 0)
                        ? model.Threshold
                        : (double)numThresholdDefault.Value;

                    var sThrInit = EnsureThrSeries(axis);
                    sThrInit.Points.Clear();
                    sThrInit.Points.AddXY(now, thr);
                    sThrInit.Points.AddXY(now + TimeSpan.FromSeconds(1).TotalDays, thr);
                }
                return;
            }

            foreach (var kv in _axisModels)
            {
                int axis = kv.Key;
                var model = kv.Value.Model;
                double thr = (model != null && model.Threshold > 0)
                    ? model.Threshold
                    : (double)numThresholdDefault.Value;

                var sThr = EnsureThrSeries(axis);
                sThr.Points.Clear();
                sThr.Points.AddXY(xmin.Value, thr);
                sThr.Points.AddXY(xmax.Value, thr);
            }

            // 사용하지 않는 Thr_* 제거
            var validThrNames = new HashSet<string>(_axisModels.Keys.Select(a => "Thr_axis" + a));
            var remove = chartScore.Series.Cast<Series>()
                .Where(s => s.Name.StartsWith("Thr_axis") && !validThrNames.Contains(s.Name))
                .ToList();
            foreach (var s in remove) chartScore.Series.Remove(s);
        }

        // ========================= 표/유틸 =========================
        private void AddRowToGrid(int axis, string file, double score, bool anom)
        {
            var t = (DataTable)gridLive.DataSource;
            t.Rows.Add(DateTime.Now.ToString("HH:mm:ss"), axis, file, score, anom ? "Anomaly" : "Normal");

            if (gridLive.Rows.Count > 0)
            {
                try { gridLive.FirstDisplayedScrollingRowIndex = gridLive.Rows.Count - 1; }
                catch { }
            }
        }

        private bool TryMarkProcessing(string path)
        {
            lock (_sync) { return _processing.Add(path); }
        }

        private void UnmarkProcessing(string path)
        {
            lock (_sync) { _processing.Remove(path); }
        }

        private T Retry<T>(Func<T> action, int maxAttempts = 5, int initialDelayMs = 100)
        {
            int delay = initialDelayMs;
            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                try { return action(); }
                catch (Exception ex) when ((ex is IOException || ex is UnauthorizedAccessException) && attempt < maxAttempts)
                {
                    Thread.Sleep(delay);
                    delay = Math.Min(delay * 2, 2000);
                }
            }
            return action();
        }

        private static string[] SplitCsvLine(string line) => line.Split(',');

        // ===== 특징 벡터 만들기: 모델에 저장된 YColumn/Features 사용 =====
        private double[] BuildFeatureVectorFromCsv(string filePath, string yColumn, string[] featureKeysInOrder)
        {
            if (featureKeysInOrder == null || featureKeysInOrder.Length == 0) return null;

            return SignalFeatures.BuildFeatureVectorFromCsv(
                filePath: filePath,
                yColumn: yColumn,
                featureKeysInOrder: featureKeysInOrder);
        }

        // ===== 간단 입력 박스 =====
        private class InputBox : Form
        {
            public string InputText { get { return _tb.Text; } }
            private TextBox _tb;
            public InputBox(string title, string message)
            {
                Text = title;
                Width = 360; Height = 160;
                StartPosition = FormStartPosition.CenterParent;
                var lbl = new Label { Text = message, Dock = DockStyle.Top, Height = 32, TextAlign = ContentAlignment.MiddleLeft, Padding = new Padding(10, 8, 10, 4) };
                _tb = new TextBox { Dock = DockStyle.Top, Margin = new Padding(10) };
                var flow = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 40, FlowDirection = FlowDirection.RightToLeft };
                var ok = new Button { Text = "확인", DialogResult = DialogResult.OK, Width = 80 };
                var cancel = new Button { Text = "취소", DialogResult = DialogResult.Cancel, Width = 80 };
                flow.Controls.Add(ok); flow.Controls.Add(cancel);
                Controls.Add(flow); Controls.Add(_tb); Controls.Add(lbl);
                AcceptButton = ok; CancelButton = cancel;
            }
        }
    }
}
