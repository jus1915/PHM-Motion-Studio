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

        // ── UI ───────────────────────────────────────────────────────────────
        private Button         _btnStart;
        private Button         _btnStop;
        private Label          _lblState;
        private Label          _lblRms, _lblNf, _lblOnset, _lblStaLta, _lblPeak;
        private ProgressBar    _rmsBar;
        private DataGridView   _grid;
        private System.Windows.Forms.Timer _uiTimer;

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
        }

        // =====================================================================
        // UI 구성
        // =====================================================================
        private void BuildUI()
        {
            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3,
                Padding = new Padding(4)
            };
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));    // 0: 툴바
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));    // 1: 지표
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100)); // 2: 이벤트 그리드
            Controls.Add(root);

            root.Controls.Add(BuildToolbar(), 0, 0);
            root.Controls.Add(BuildMetrics(), 0, 1);
            root.Controls.Add(BuildGrid(),    0, 2);
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
            p.Controls.Add(_lblState);
            return p;
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
                _trigger = new EventTrigger(new TriggerConfig());
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
                        sw.WriteLine($"{t:F6},{x:G6},{y:G6},{z:G6}");
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
