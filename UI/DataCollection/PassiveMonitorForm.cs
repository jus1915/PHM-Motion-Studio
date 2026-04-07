using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using NationalInstruments.DAQmx;
using PHM_Project_DockPanel.Services;
using PHM_Project_DockPanel.Services.DAQ;
using WeifenLuo.WinFormsUI.Docking;

using DaqTask = NationalInstruments.DAQmx.Task;

namespace PHM_Project_DockPanel.Windows
{
    public class PassiveMonitorForm : DockContent
    {
        // ── DAQ ──────────────────────────────────────────────────────────────
        private DaqTask _aiTask;
        private AnalogMultiChannelReader _reader;
        private AsyncCallback _cb;
        private volatile bool _monitoring;
        private int _blockSize;

        /// <summary>DAQ 콜백 블록마다 발행됩니다. (module, block[3,samples], timestampUtc)</summary>
        public Action<string, double[,], DateTime> BlockPublished;

        // ── 트리거 엔진 ──────────────────────────────────────────────────────
        private EventTrigger _trigger;
        private DaqSensorConfig _cfg;

        // ── 스레드 안전 UI 상태 (DAQ 스레드에서 쓰고, UI 타이머에서 읽음) ───
        private volatile float _uiRms;
        private volatile float _uiPeak;
        private volatile float _uiStaLta;
        private volatile float _uiNoiseFloor;
        private volatile float _uiOnsetThr;
        private volatile int   _uiStateCode; // 0=Idle, 1=Active, 2=PostBuffer
        private bool _suspendedForRun; // PHM 모션 실행 중 자동 일시 정지 여부

        // ── UI ───────────────────────────────────────────────────────────────
        private Button         _btnStart;
        private Button         _btnStop;
        private Label          _lblState;
        private Label          _lblRms, _lblNf, _lblOnset, _lblStaLta, _lblPeak;
        private ProgressBar    _rmsBar;
        private DataGridView   _grid;
        private System.Windows.Forms.Timer _uiTimer;

        // ── 설정 패널 ─────────────────────────────────────────────────────────
        private TableLayoutPanel  _root;
        private Panel             _settingsPanel;
        private CheckBox          _chkAdaptive, _chkRequireStaLta;
        private NumericUpDown     _nudOnsetMult, _nudOnsetFixed, _nudPeak, _nudStaLta;
        private Label             _lblOnsetMult, _lblOnsetFixed;

        private const string BaseLogDir = @"C:\Data\PHM_Logs\Signals";

        public PassiveMonitorForm()
        {
            Text = "Passive Monitor";
            MinimumSize = new Size(500, 320);
            _cfg = DaqSensorConfig.LoadOrDefault(MainForm.DaqSensorConfigPath);
            BuildUI();

            _uiTimer = new System.Windows.Forms.Timer { Interval = 150 };
            _uiTimer.Tick += (s, e) => RefreshMetrics();
            _uiTimer.Start();

            // PHM 모션과 NI 모듈 리소스 공유 조정
            AppEvents.PassiveMonitorSuspendRequested += OnSuspendRequested;
            AppEvents.PassiveMonitorResumeRequested  += OnResumeRequested;
            FormClosed += (s, e) =>
            {
                AppEvents.PassiveMonitorSuspendRequested -= OnSuspendRequested;
                AppEvents.PassiveMonitorResumeRequested  -= OnResumeRequested;
                SafeStopDaq();
            };
        }

        // =====================================================================
        // UI 구성
        // =====================================================================
        private void BuildUI()
        {
            _root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4,
                Padding = new Padding(4)
            };
            _root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            _root.RowStyles.Add(new RowStyle(SizeType.AutoSize));     // 0: 툴바
            _root.RowStyles.Add(new RowStyle(SizeType.Absolute, 0));  // 1: 설정 (기본 숨김)
            _root.RowStyles.Add(new RowStyle(SizeType.AutoSize));     // 2: 지표
            _root.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); // 3: 이벤트 그리드
            Controls.Add(_root);

            _root.Controls.Add(BuildToolbar(),  0, 0);
            _root.Controls.Add(BuildSettings(), 0, 1);
            _root.Controls.Add(BuildMetrics(),  0, 2);
            _root.Controls.Add(BuildGrid(),     0, 3);
        }

        private Control BuildToolbar()
        {
            var p = new FlowLayoutPanel
            {
                AutoSize = true, Dock = DockStyle.Top, WrapContents = false,
                Padding = new Padding(0, 4, 0, 6)
            };

            _btnStart = new Button { Text = "▶ 모니터링 시작", Width = 130, Height = 28 };
            _btnStop  = new Button { Text = "■ 정지",          Width = 80,  Height = 28, Enabled = false };

            var btnClear = new Button { Text = "Clear", Width = 60, Height = 28 };
            btnClear.Click += (s, e) => _grid.Rows.Clear();

            var btnSettings = new Button { Text = "⚙ 설정", Width = 70, Height = 28 };
            btnSettings.Click += (s, e) => ToggleSettings();

            _lblState = new Label
            {
                Text = "● Idle", AutoSize = true, ForeColor = Color.Gray,
                Font = new Font("Segoe UI", 10f, FontStyle.Bold),
                Margin = new Padding(12, 5, 0, 0)
            };

            _btnStart.Click += (s, e) => StartMonitoring();
            _btnStop.Click  += (s, e) => StopMonitoring();

            p.Controls.Add(_btnStart);
            p.Controls.Add(_btnStop);
            p.Controls.Add(btnClear);
            p.Controls.Add(btnSettings);
            p.Controls.Add(_lblState);
            return p;
        }

        private Control BuildSettings()
        {
            var cfg = new TriggerConfig(); // 기본값 참조용

            _settingsPanel = new Panel { Dock = DockStyle.Top, AutoSize = true, Visible = false };
            var flow = new FlowLayoutPanel
            {
                AutoSize = true, WrapContents = true, Dock = DockStyle.Top,
                Padding = new Padding(2, 2, 2, 4)
            };

            // ── 적응형 임계값 ───────────────────────────────────────────────
            _chkAdaptive = new CheckBox
            {
                Text = "적응형 임계값", Checked = cfg.UseAdaptiveThreshold,
                AutoSize = true, Margin = new Padding(4, 6, 8, 0)
            };
            flow.Controls.Add(_chkAdaptive);

            // 적응형 ON: Onset 배수
            _lblOnsetMult = new Label { Text = "Onset 배수:", AutoSize = true, Margin = new Padding(4, 8, 2, 0) };
            _nudOnsetMult = new NumericUpDown
            {
                Minimum = 1.0m, Maximum = 10.0m, Increment = 0.5m, DecimalPlaces = 1,
                Value = (decimal)cfg.OnsetNoiseMultiplier, Width = 60
            };
            flow.Controls.Add(_lblOnsetMult);
            flow.Controls.Add(_nudOnsetMult);

            // 적응형 OFF: 고정 Onset(g)
            _lblOnsetFixed = new Label { Text = "Onset(g):", AutoSize = true, Margin = new Padding(12, 8, 2, 0) };
            _nudOnsetFixed = new NumericUpDown
            {
                Minimum = 0.001m, Maximum = 1.0m, Increment = 0.005m, DecimalPlaces = 3,
                Value = (decimal)cfg.OnsetThreshold, Width = 70
            };
            flow.Controls.Add(_lblOnsetFixed);
            flow.Controls.Add(_nudOnsetFixed);

            // ── Peak 임계값 ─────────────────────────────────────────────────
            flow.Controls.Add(new Label { Text = "Peak(g):", AutoSize = true, Margin = new Padding(12, 8, 2, 0) });
            _nudPeak = new NumericUpDown
            {
                Minimum = 0.001m, Maximum = 2.0m, Increment = 0.005m, DecimalPlaces = 3,
                Value = (decimal)cfg.PeakThreshold, Width = 70
            };
            flow.Controls.Add(_nudPeak);

            // ── STA/LTA ─────────────────────────────────────────────────────
            _chkRequireStaLta = new CheckBox
            {
                Text = "STA/LTA 필요", Checked = cfg.RequireStaLtaForOnset,
                AutoSize = true, Margin = new Padding(12, 6, 4, 0)
            };
            flow.Controls.Add(_chkRequireStaLta);

            _nudStaLta = new NumericUpDown
            {
                Minimum = 1.0m, Maximum = 20.0m, Increment = 0.5m, DecimalPlaces = 1,
                Value = (decimal)cfg.StaLtaRatio, Width = 60
            };
            flow.Controls.Add(_nudStaLta);

            _settingsPanel.Controls.Add(flow);

            // 적응형 토글 시 관련 컨트롤 표시/숨김
            void RefreshAdaptiveVisibility()
            {
                bool adaptive = _chkAdaptive.Checked;
                _lblOnsetMult.Visible  = adaptive;
                _nudOnsetMult.Visible  = adaptive;
                _lblOnsetFixed.Visible = !adaptive;
                _nudOnsetFixed.Visible = !adaptive;
            }
            RefreshAdaptiveVisibility();
            _chkAdaptive.CheckedChanged += (s, e) => { RefreshAdaptiveVisibility(); ApplyConfigToTrigger(); };

            // 값 변경 시 즉시 적용
            _nudOnsetMult.ValueChanged  += (s, e) => ApplyConfigToTrigger();
            _nudOnsetFixed.ValueChanged += (s, e) => ApplyConfigToTrigger();
            _nudPeak.ValueChanged       += (s, e) => ApplyConfigToTrigger();
            _nudStaLta.ValueChanged     += (s, e) => ApplyConfigToTrigger();
            _chkRequireStaLta.CheckedChanged += (s, e) => ApplyConfigToTrigger();

            return _settingsPanel;
        }

        private void ToggleSettings()
        {
            bool show = !_settingsPanel.Visible;
            _settingsPanel.Visible = show;
            _root.RowStyles[1] = show
                ? new RowStyle(SizeType.AutoSize)
                : new RowStyle(SizeType.Absolute, 0);
        }

        private TriggerConfig BuildTriggerConfig() => new TriggerConfig
        {
            UseAdaptiveThreshold  = _chkAdaptive?.Checked ?? false,
            OnsetNoiseMultiplier  = (double)(_nudOnsetMult?.Value  ?? 2.0m),
            OffsetNoiseMultiplier = (double)(_nudOnsetMult?.Value  ?? 2.0m) * 0.75,
            OnsetThreshold        = (double)(_nudOnsetFixed?.Value ?? 0.005m),
            OffsetThreshold       = (double)(_nudOnsetFixed?.Value ?? 0.005m) * 0.5,
            PeakThreshold         = (double)(_nudPeak?.Value       ?? 0.025m),
            RequirePeakForOnset   = true,
            RequireStaLtaForOnset = _chkRequireStaLta?.Checked ?? false,
            StaLtaRatio           = (double)(_nudStaLta?.Value     ?? 2.0m),
        };

        private void ApplyConfigToTrigger()
        {
            if (_trigger == null) return;
            _trigger.Config = BuildTriggerConfig();
        }

        private Control BuildMetrics()
        {
            var panel = new Panel { Dock = DockStyle.Top, Height = 60, Padding = new Padding(2) };

            var tbl = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 5, RowCount = 2
            };
            for (int i = 0; i < 5; i++)
                tbl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 20));
            tbl.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
            tbl.RowStyles.Add(new RowStyle(SizeType.Percent, 50));

            // Row 0: RMS 바 (5열 span)
            _rmsBar = new ProgressBar { Dock = DockStyle.Fill, Minimum = 0, Maximum = 100 };
            tbl.Controls.Add(_rmsBar, 0, 0);
            tbl.SetColumnSpan(_rmsBar, 5);

            // Row 1: 수치 레이블
            _lblRms    = MetricLabel("RMS: —");
            _lblNf     = MetricLabel("NF: —");
            _lblOnset  = MetricLabel("Onset: —");
            _lblStaLta = MetricLabel("STA/LTA: —");
            _lblPeak   = MetricLabel("Peak: —");

            tbl.Controls.Add(_lblRms,    0, 1);
            tbl.Controls.Add(_lblNf,     1, 1);
            tbl.Controls.Add(_lblOnset,  2, 1);
            tbl.Controls.Add(_lblStaLta, 3, 1);
            tbl.Controls.Add(_lblPeak,   4, 1);

            panel.Controls.Add(tbl);
            return panel;
        }

        private static Label MetricLabel(string text) => new Label
        {
            Text = text, AutoSize = false, Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            Font = new Font("Segoe UI", 8.5f)
        };

        private DataGridView BuildGrid()
        {
            _grid = new DataGridView
            {
                Dock = DockStyle.Fill, ReadOnly = true,
                AllowUserToAddRows = false, RowHeadersVisible = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                BackgroundColor = SystemColors.Window
            };

            void Col(string name, string header, int weight)
                => _grid.Columns.Add(new DataGridViewTextBoxColumn
                   { Name = name, HeaderText = header, FillWeight = weight });

            Col("Time",     "시작 시각",   18);
            Col("Duration", "지속(s)",     10);
            Col("Type",     "유형",        12);
            Col("Frames",   "프레임",      10);
            Col("PeakX",    "Peak X(g)",  12);
            Col("PeakY",    "Peak Y(g)",  12);
            Col("PeakZ",    "Peak Z(g)",  12);
            Col("Path",     "CSV",        40);

            // 더블클릭 → Log Graph에서 열기
            _grid.CellDoubleClick += (s, e) =>
            {
                if (e.RowIndex < 0) return;
                var path = _grid.Rows[e.RowIndex].Cells["Path"].Value?.ToString();
                if (!string.IsNullOrEmpty(path) && File.Exists(path))
                    AppEvents.RaiseShowLogGraph(AppEvents.LogDataKind.Accel, path);
            };

            return _grid;
        }

        // =====================================================================
        // 모니터링 시작 / 정지
        // =====================================================================
        private void StartMonitoring()
        {
            if (_monitoring) return;

            _cfg = DaqSensorConfig.LoadOrDefault(MainForm.DaqSensorConfigPath);

            try
            {
                _trigger = new EventTrigger(BuildTriggerConfig());
                _trigger.EventDetected += OnEventDetected;

                _aiTask  = new DaqTask("cDAQ_Passive");
                _blockSize = _cfg.ReadBlock > 0 ? _cfg.ReadBlock : 100;

                CreateChannel($"{_cfg.Module}/ai0", _cfg.SensX);
                CreateChannel($"{_cfg.Module}/ai1", _cfg.SensY);
                CreateChannel($"{_cfg.Module}/ai2", _cfg.SensZ);

                double rate = _cfg.SampleRate > 0 ? _cfg.SampleRate : 1000;
                _aiTask.Timing.ConfigureSampleClock(
                    "", rate,
                    SampleClockActiveEdge.Rising,
                    SampleQuantityMode.ContinuousSamples,
                    (int)rate * 10);
                _aiTask.Stream.Timeout = 10000;

                _reader = new AnalogMultiChannelReader(_aiTask.Stream)
                    { SynchronizeCallbacks = false };
                _cb = new AsyncCallback(DaqCallback);
                _aiTask.Start();
                _monitoring = true;
                _reader.BeginReadMultiSample(_blockSize, _cb, null);

                _btnStart.Enabled = false;
                _btnStop.Enabled  = true;
                AppEvents.RaiseLog($"[Passive Monitor] 시작 — {_cfg.Module}, {rate} Hz, 블록 {_blockSize}");
            }
            catch (Exception ex)
            {
                SafeStopDaq();
                AppEvents.RaiseLog($"[Passive Monitor 오류] {ex.Message}");
                MessageBox.Show(ex.Message, "모니터링 시작 실패",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void StopMonitoring()
        {
            SafeStopDaq();
            _btnStart.Enabled = true;
            _btnStop.Enabled  = false;
            _uiStateCode      = 0;
            AppEvents.RaiseLog("[Passive Monitor] 정지");
        }

        private void SafeStopDaq()
        {
            _monitoring = false;
            _cb     = null;
            _reader = null;
            try { _aiTask?.Control(TaskAction.Abort); } catch { }
            try { _aiTask?.Dispose();                 } catch { }
            _aiTask = null;
        }

        // PHM 모션 실행 전 호출 → DAQ 태스크 해제 (배경 스레드에서 호출될 수 있음)
        private void OnSuspendRequested()
        {
            if (!_monitoring) return;
            _suspendedForRun = true;
            SafeStopDaq();
            BeginInvoke(new Action(() =>
            {
                _btnStart.Enabled = true;
                _btnStop.Enabled  = false;
                _uiStateCode      = 0;
                AppEvents.RaiseLog("[Passive Monitor] 모션 시작 — DAQ 일시 해제");
            }));
        }

        // PHM 모션 종료 후 호출 → 모니터링 자동 재시작
        private void OnResumeRequested()
        {
            if (!_suspendedForRun) return;
            _suspendedForRun = false;
            BeginInvoke(new Action(() =>
            {
                StartMonitoring();
                AppEvents.RaiseLog("[Passive Monitor] 모션 완료 — 모니터링 재시작");
            }));
        }

        private void CreateChannel(string phys, double sens_mVpg)
        {
            _aiTask.AIChannels.CreateAccelerometerChannel(
                phys, "",
                AITerminalConfiguration.Pseudodifferential,
                -_cfg.GRange, _cfg.GRange,
                sens_mVpg,
                AIAccelerometerSensitivityUnits.MillivoltsPerG,
                AIExcitationSource.Internal,
                _cfg.IepeCurrentAmps,
                AIAccelerationUnits.G);
        }

        // =====================================================================
        // DAQ 콜백 (백그라운드 스레드)
        // =====================================================================
        private void DaqCallback(IAsyncResult ar)
        {
            try
            {
                if (!_monitoring || _reader == null) return;

                double[,] block = _reader.EndReadMultiSample(ar);
                BlockPublished?.Invoke(_cfg.Module, block, DateTime.UtcNow);
                int ch = Math.Min(block.GetLength(0), 3);
                int n  = block.GetLength(1);

                // 채널별 RMS
                var rms = new double[ch];
                for (int c = 0; c < ch; c++)
                {
                    double sum = 0;
                    for (int i = 0; i < n; i++) { double v = block[c, i]; sum += v * v; }
                    rms[c] = Math.Sqrt(sum / n);
                }

                // 전체 피크
                double peak = 0;
                for (int c = 0; c < ch; c++)
                    for (int i = 0; i < n; i++) { double v = Math.Abs(block[c, i]); if (v > peak) peak = v; }

                // EventTrigger 공급
                _trigger.Feed(
                    _cfg.Module, ch,
                    _cfg.SampleRate > 0 ? (int)_cfg.SampleRate : 1000,
                    rms, block, DateTime.UtcNow);

                // UI 상태 기록
                _uiRms        = (float)rms.Max();
                _uiPeak       = (float)peak;
                _uiStaLta     = (float)_trigger.CurrentStaLta;
                _uiNoiseFloor = (float)_trigger.CurrentNoiseFloor;
                _uiOnsetThr   = (float)_trigger.EffectiveOnsetThreshold;
                _uiStateCode  = _trigger.CurrentStateName == "Active"     ? 1
                              : _trigger.CurrentStateName == "PostBuffer" ? 2 : 0;

                if (_monitoring && _reader != null && _cb != null)
                    _reader.BeginReadMultiSample(_blockSize, _cb, null);
            }
            catch (DaqException) { }
            catch (ObjectDisposedException) { }
        }

        // =====================================================================
        // 이벤트 감지 콜백 (백그라운드 스레드에서 호출됨)
        // =====================================================================
        private void OnEventDetected(VibrationEvent ev)
        {
            string csvPath = null;
            try   { csvPath = SaveEventToCsv(ev); }
            catch (Exception ex)
            { AppEvents.RaiseLog($"[Passive Monitor] CSV 저장 오류: {ex.Message}"); }

            double[] peak = ev.ComputePeak();
            double peakMax = peak.Length > 0 ? peak.Max() : 0;

            BeginInvoke(new Action(() =>
            {
                _grid.Rows.Add(
                    ev.StartUtc.ToLocalTime().ToString("HH:mm:ss.fff"),
                    ev.DurationSeconds.ToString("F3"),
                    ev.EventType,
                    ev.Frames.Count,
                    peak.Length > 0 ? peak[0].ToString("F3") : "-",
                    peak.Length > 1 ? peak[1].ToString("F3") : "-",
                    peak.Length > 2 ? peak[2].ToString("F3") : "-",
                    csvPath ?? "");

                if (_grid.Rows.Count > 0)
                    _grid.FirstDisplayedScrollingRowIndex = _grid.Rows.Count - 1;

                AppEvents.RaiseLog(
                    $"[Passive Monitor] 이벤트 감지 — {ev.EventType}, " +
                    $"{ev.DurationSeconds:F2} s, Peak={peakMax:F3} g");

                if (!string.IsNullOrEmpty(csvPath))
                {
                    AppState.LastAccelCsvs = new List<string> { csvPath };
                    AppEvents.RaiseShowLogGraph(AppEvents.LogDataKind.Accel, csvPath);
                }
            }));
        }

        // =====================================================================
        // CSV 저장
        // =====================================================================
        private string SaveEventToCsv(VibrationEvent ev)
        {
            string date = ev.StartUtc.ToLocalTime().ToString("yyyyMMdd");
            var dir = Path.Combine(BaseLogDir, $"{date}_Passive", "Accel", ev.Device);
            Directory.CreateDirectory(dir);

            string ts   = ev.StartUtc.ToLocalTime().ToString("yyyyMMdd_HHmmss");
            string path = Path.Combine(dir, $"{ts}_Passive_{ev.EventType}_Accel.csv");

            using (var sw = new StreamWriter(path, false, new UTF8Encoding(false)))
            {
                sw.WriteLine("time_s,x,y,z");
                long idx  = 0;
                double sr = ev.SampleRate > 0 ? ev.SampleRate : 1000;
                foreach (var frame in ev.Frames)
                {
                    int n  = frame.GetLength(1);
                    int ch = frame.GetLength(0);
                    for (int i = 0; i < n; i++)
                    {
                        double t = idx / sr;
                        double x = ch > 0 ? frame[0, i] : 0;
                        double y = ch > 1 ? frame[1, i] : 0;
                        double z = ch > 2 ? frame[2, i] : 0;
                        sw.WriteLine(FormattableString.Invariant($"{t:F6},{x:G6},{y:G6},{z:G6}"));
                        idx++;
                    }
                }
            }
            return path;
        }

        // =====================================================================
        // UI 타이머 (150 ms 주기)
        // =====================================================================
        private void RefreshMetrics()
        {
            if (!_monitoring)
            {
                _lblState.Text      = "● Idle";
                _lblState.ForeColor = Color.Gray;
                return;
            }

            float rms    = _uiRms;
            float peak   = _uiPeak;
            float staLta = _uiStaLta;
            float nf     = _uiNoiseFloor;
            float onset  = _uiOnsetThr;
            int   code   = _uiStateCode;

            switch (code)
            {
                case 1:  _lblState.Text = "● Active";     _lblState.ForeColor = Color.OrangeRed;  break;
                case 2:  _lblState.Text = "● PostBuffer"; _lblState.ForeColor = Color.DodgerBlue; break;
                default: _lblState.Text = "● Idle";       _lblState.ForeColor = Color.SeaGreen;   break;
            }

            // RMS 바: onset 임계값의 0~200% → 0~100%
            float pct = onset > 0 ? (rms / onset) * 50f : 0f;
            _rmsBar.Value = Math.Max(0, Math.Min(100, (int)pct));

            _lblRms.Text    = $"RMS: {rms:F4} g";
            _lblNf.Text     = $"NF: {nf:F4} g";
            _lblOnset.Text  = $"Onset: {onset:F4} g";
            _lblStaLta.Text = $"STA/LTA: {staLta:F1}";
            _lblPeak.Text   = $"Peak: {peak:F4} g";
        }

        // =====================================================================
        // Dispose
        // =====================================================================
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _uiTimer?.Stop();
                _uiTimer?.Dispose();
                SafeStopDaq();
            }
            base.Dispose(disposing);
        }
    }
}
