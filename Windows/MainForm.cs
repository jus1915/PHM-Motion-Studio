using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows.Forms;
using System.Xml;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Windows;
using PHM_Project_DockPanel.Services;
using PHM_Project_DockPanel.Services.WMX;
using PHM_Project_DockPanel.Services.DAQ;
using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.UI.DataAnalysis;
using PHM_Project_DockPanel.UI.Dashboard;
using PHM_Project_DockPanel.DebugTools;
using static PHM_Project_DockPanel.Windows.AxisInfoForm;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel
{
    public partial class MainForm : Form
    {
        // ── 상수 ────────────────────────────────────────────────────────────
        private const string LayoutFile = "layout.xml";
        private const string AxisConfigFile = "axis_config.json";
        private const string HttpServerUrl = "http://10.100.17.221:8000/api/ingest";

        // ── 핵심 서비스 ──────────────────────────────────────────────────────
        private ControllerManager _controller;
        private DaqSensorConfig _daqCfg;
        private AxisConfig[] _axisConfigs;
        private WmxTorqueLogger _torqueLogger;
        private DaqAccelCsvLogger _daq;
        private AccelInfluxPublisher _influxPublisher;
        private PHM_Motion _motion;

        // ── DockPanel ────────────────────────────────────────────────────────
        private DockPanel _dockPanel;
        private VS2015BlueTheme _theme;
        private readonly Dictionary<string, ToolStripMenuItem> _dockMenuMap
            = new Dictionary<string, ToolStripMenuItem>();

        // ── 캐시된 DockContent 인스턴스 ─────────────────────────────────────
        private AxisInfoForm _axisInfo;
        private TeachingForm _teaching;
        private SimulatorForm _simulator;
        private LogGraphForm _logGraph;
        private LogWriterForm _logWriter;

        // ────────────────────────────────────────────────────────────────────
        // 생성자
        // ────────────────────────────────────────────────────────────────────
        public MainForm()
        {
            // 1. 제어기 선택 → 실패(취소)하면 앱 종료
            if (!TrySelectController())
            {
                Load += (s, e) => Close();
                return;
            }

            // 2. 각 서브시스템 초기화
            InitAxisConfigs();
            InitMotion();
            _daqCfg = DaqSensorConfig.LoadOrDefault(DaqSensorConfigPath);
            InitDaq();
            InitInfluxPublisher();

            // 3. WinForms 초기화
            InitializeComponent();
            InitDockPanel();

            // 4. 저장된 축 설정 불러오기
            LoadAxisConfigs();

            // 5. 앱 이벤트 구독
            SubscribeAppEvents();

            // 6. LogWriter는 AppState에서 공유
            _logWriter = new LogWriterForm();
            AppState.LogWriter = _logWriter;
        }

        // ────────────────────────────────────────────────────────────────────
        // 초기화 헬퍼
        // ────────────────────────────────────────────────────────────────────

        /// <summary>시작 시 제어기 선택 다이얼로그를 표시하고 ControllerManager를 구성합니다.</summary>
        private bool TrySelectController()
        {
            using (var dlg = new ControllerSelectDialog())
            {
                if (dlg.ShowDialog() != DialogResult.OK)
                    return false;

                _controller = new ControllerManager();
                _controller.SetController(dlg.SelectedController);
                return true;
            }
        }

        private void InitAxisConfigs()
        {
            _axisConfigs = new AxisConfig[5];
            for (int i = 0; i < _axisConfigs.Length; i++)
                _axisConfigs[i] = new AxisConfig();
        }

        private void InitMotion()
        {
            // WMX3 제어기일 때만 Log(WMX3 전용 클래스)를 생성합니다.
            // Ajin 또는 시뮬레이션에서 new Log()를 호출하면
            // LogApi_CLRLib.dll 로드를 시도해 FileNotFoundException이 발생합니다.
            if (_controller.IsWmx3)
            {
                _torqueLogger = new WmxTorqueLogger(
                    new Log(), channelIndex: 0, logAction: msg => AppEvents.RaiseLog(msg));
            }
            else
            {
                AppEvents.RaiseLog($"[모션] WmxTorqueLogger 건너뜀 ({(_controller.IsSimulationMode ? "시뮬레이션" : "Ajin")} 모드)");
            }

            // _axisInfo는 나중에 생성되므로 람다에서 null-safe하게 접근
            _motion = new PHM_Motion(
                _controller,
                _axisConfigs,
                _torqueLogger,           // WMX3가 아니면 null → PHM_Motion 내부에서 null-safe 처리됨
                isAccelEnabled: () => _axisInfo?.AccelCheckBox?.Checked ?? false,
                isTorqueEnabled: () => _axisInfo?.TorqueCheckBox?.Checked ?? false);

            // Ajin 전용: 폴링 방식 모션 데이터 로거 주입
            if (_controller.IsAjin)
            {
                var ajin = _controller.AsAjin;
                var ajinLogger = new AjinCsvLogger(
                    getPos:    ax => ajin.GetActPos(ax),
                    getTorque: ax => ajin.GetTorque(ax),
                    log:       msg => AppEvents.RaiseLog(msg),
                    getCmdPos: ax => ajin.GetCmdPos(ax));
                _motion.SetAjinLogger(ajinLogger);
                AppEvents.RaiseLog("[Ajin] AjinCsvLogger 주입 완료");
            }

            // 시뮬레이션: 가상 토크 데이터 폴링 로거 주입
            var sim = _controller.AsSimulation;
            if (sim != null)
            {
                var simLogger = new AjinCsvLogger(
                    getPos:     ax => _controller.GetStatus().AxesStatus[ax].ActualPos,
                    getTorque:  ax => sim.GetTorque(ax),
                    log:        msg => AppEvents.RaiseLog(msg),
                    getVel:     ax => sim.GetVelocity(ax),
                    fileSuffix: "Simulator");
                _motion.SetAjinLogger(simLogger);
                AppEvents.RaiseLog("[Sim] 가상 토크 로거 주입 완료");
            }
        }

        private void InitDaq()
        {
            _daq = new DaqAccelCsvLogger(AppEvents.RaiseLog);
            ApplyDaqConfig(_daqCfg);
            _motion.SetAccelLogger(_daq);
        }

        internal static string DaqSensorConfigPath =>
            Path.Combine(ResolveCfgDir(), "daq_sensor_config.json");

        /// <summary>DaqSensorConfig 를 DaqAccelCsvLogger 에 적용합니다.</summary>
        private void ApplyDaqConfig(DaqSensorConfig cfg)
        {
            _daqCfg = cfg;

            _daq.Modules    = new[] { cfg.Module };
            _daq.AiRange    = cfg.Channel;
            _daq.SampleRate = cfg.SampleRate;
            _daq.ReadBlock  = cfg.ReadBlock;
            _daq.MinG       = -cfg.GRange;
            _daq.MaxG       =  cfg.GRange;
            _daq.SetModuleSensitivity(cfg.Module, cfg.SensX, cfg.SensY, cfg.SensZ);
            _daq.SetModuleOffset(cfg.Module, 0.0, 0.0, 0.0);
            _daq.IepeCurrentAmps = cfg.IepeCurrentAmps;

            if (_influxPublisher != null)
                _influxPublisher.SampleRate = cfg.SampleRate;

            AppEvents.RaiseLog($"[DAQ 설정 적용] {cfg.Module}/{cfg.Channel}  " +
                               $"Sens X={cfg.SensX} Y={cfg.SensY} Z={cfg.SensZ} mV/g  " +
                               $"Rate={cfg.SampleRate} Hz  ±{cfg.GRange} g");
        }

        private void InitInfluxPublisher()
        {
            // 우선순위 순으로 탐색: 로그 폴더 → DAQ_Test infra 폴더 → 실행 파일 폴더
            var candidatePaths = new[]
            {
                Path.Combine(ResolveCfgDir(), "influx_config.json"),
                @"D:\Dev\hvs\WorkingSource\DAQ_Test\infra\influx_config.json",
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "influx_config.json"),
            };

            string foundPath = null;
            foreach (var p in candidatePaths)
                if (File.Exists(p)) { foundPath = p; break; }

            // 어디에도 없으면 기본 위치에 샘플 파일 생성
            string savePath = foundPath ?? candidatePaths[0];
            var cfg = InfluxConfig.LoadOrDefault(savePath);

            if (foundPath == null)
            {
                cfg.Save(savePath);
                AppEvents.RaiseLog($"[InfluxDB] 설정 파일 생성: {savePath}  (기본값 — URL/Token 수정 후 재시작)");
            }
            else
            {
                AppEvents.RaiseLog($"[InfluxDB] 설정 파일 로드: {foundPath}");
            }

            _influxPublisher = new AccelInfluxPublisher(cfg, AppEvents.RaiseLog);

            // DaqAccelCsvLogger 블록 → InfluxDB (모션 런 중)
            _daq.BlockReceived += _influxPublisher.Feed;

            _motion.SetAccelInfluxPublisher(_influxPublisher);
        }

        /// <summary>E:\Data\PHM_Logs → C:\Data\PHM_Logs → C:\PHM_Logs 순으로 존재하는 루트 사용.</summary>
        private static string ResolveCfgDir()
        {
            foreach (var root in new[] { @"E:\Data\PHM_Logs", @"C:\Data\PHM_Logs", @"C:\PHM_Logs" })
                if (Directory.Exists(root)) return Path.Combine(root, "Tests");
            return Path.Combine(@"C:\Data\PHM_Logs", "Tests"); // 폴더 없으면 기본 생성 경로
        }

        // ────────────────────────────────────────────────────────────────────
        // 앱 이벤트 구독
        // ────────────────────────────────────────────────────────────────────
        private void SubscribeAppEvents()
        {
            // 실시간 가속도 스트리밍 토글
            AppEvents.AccelRealtimeToggled += OnAccelRealtimeToggled;
            // InfluxDB 저장 레이블
            AppEvents.InfluxLabelChanged += label =>
            {
                if (_influxPublisher != null)
                    _influxPublisher.Label = label;
            };

            // Simulator 창 닫기 요청
            AppEvents.RequestCloseSimulator += () =>
            {
                if (_dockMenuMap.TryGetValue(typeof(SimulatorForm).Name, out var menu))
                    menu.Checked = false;
            };

            // LogGraph에 파일 로드 요청 (단순)
            AppEvents.ShowLogGraphRequested += filePath =>
            {
                this.BeginInvoke(new Action(() =>
                {
                    var logGraph = EnsureLogGraphOpen();
                    logGraph.LoadCsv(filePath);
                }));
            };

            // LogGraph에 파일 로드 요청 (종류 포함)
            AppEvents.ShowLogGraphRequestedEx += (kind, filePath) =>
            {
                this.BeginInvoke(new Action(() =>
                {
                    var logGraph = EnsureLogGraphOpen();
                    var tgt = kind == AppEvents.LogDataKind.Accel
                        ? LogGraphForm.LogKind.Accel
                        : LogGraphForm.LogKind.Torque;
                    logGraph.LoadCsv(filePath, tgt);
                }));
            };
        }

        private void OnAccelRealtimeToggled(bool enabled)
        {
            if (enabled)
            {
                _influxPublisher?.Enable();
                AppEvents.RaiseLog("[InfluxDB] 실시간 게시 활성화");
            }
            else
            {
                _influxPublisher?.Disable();
                AppEvents.RaiseLog("[InfluxDB] 실시간 게시 비활성화");
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // DockPanel & 메뉴
        // ────────────────────────────────────────────────────────────────────
        private void InitDockPanel()
        {
            _theme = new VS2015BlueTheme();
            _dockPanel = new DockPanel
            {
                Dock = DockStyle.Fill,
                Theme = _theme
            };
            Controls.Add(_dockPanel);

            InitMenu();

            Load += MainForm_Load;
            FormClosing += MainForm_FormClosing;
        }

        private void InitMenu()
        {
            var menuStrip = new MenuStrip();

            // ── 데이터 수집 ──────────────────────────────────────────────────
            var menuDataCollection = new ToolStripMenuItem("데이터 수집");

            var monitorMenu = new ToolStripMenuItem("장비 상태 모니터링");
            monitorMenu.DropDownItems.Add(
                CreateDockMenuItem("Axis Info", DockState.DockBottom,
                    typeof(AxisInfoForm), () => new AxisInfoForm(_motion, _axisConfigs)));
            monitorMenu.DropDownItems.Add(
                CreateDockMenuItem<SimulatorForm>("Simulator", DockState.DockBottom));
            monitorMenu.DropDownItems.Add(
                CreateDockMenuItem("Teaching", DockState.DockBottom,
                    typeof(TeachingForm), () => new TeachingForm(_motion)));
            monitorMenu.DropDownItems.Add(
                CreateDockMenuItem("Passive Monitor", DockState.Document,
                    typeof(PassiveMonitorForm), () =>
                    {
                        var pmf = new PassiveMonitorForm();
                        if (_influxPublisher != null)
                            pmf.BlockPublished += _influxPublisher.Feed;
                        return pmf;
                    }));

            var logMenu = new ToolStripMenuItem("로그 관리");
            logMenu.DropDownItems.Add(CreateDockMenuItem<LogWriterForm>("Log Writer", DockState.DockBottom));
            logMenu.DropDownItems.Add(CreateDockMenuItem<LogGraphForm>("Log Graph", DockState.Document));

            var configMenu = new ToolStripMenuItem("환경 설정");
            configMenu.DropDownItems.Add(new ToolStripMenuItem("축 설정 관리", null, (s, e) => { }));
            configMenu.DropDownItems.Add(new ToolStripMenuItem("연결 설정",    null, (s, e) => { }));
            configMenu.DropDownItems.Add(
                CreateDockMenuItem("DAQ 센서 설정", DockState.DockRight,
                    typeof(DaqSettingsForm),
                    () => new DaqSettingsForm(DaqSensorConfigPath, ApplyDaqConfig)));

            menuDataCollection.DropDownItems.Add(monitorMenu);
            menuDataCollection.DropDownItems.Add(logMenu);
            menuDataCollection.DropDownItems.Add(configMenu);

            // ── 데이터 분석 ──────────────────────────────────────────────────
            var menuDataAnalysis = new ToolStripMenuItem("데이터 분석");
            menuDataAnalysis.DropDownItems.Add(
                CreateDockMenuItem<PHMPipelineWizard>("PHM 파이프라인(마법사)", DockState.Document));
            menuDataAnalysis.DropDownItems.Add(
                CreateDockMenuItem<SignalExplorerForm>("신호 탐색기", DockState.Document));
            menuDataAnalysis.DropDownItems.Add(
                CreateDockMenuItem<AnomalyDetectionForm>("이상 탐지", DockState.Document));
            menuDataAnalysis.DropDownItems.Add(
                CreateDockMenuItem<AIForm>("AI", DockState.Document));
            menuDataAnalysis.DropDownItems.Add(
                CreateDockMenuItem<DashboardForm>("실시간 추론", DockState.Document));

            menuStrip.Items.Add(menuDataCollection);
            menuStrip.Items.Add(menuDataAnalysis);
            menuStrip.Items.Add(new ToolStripMenuItem("프로그램 종료", null, (s, e) => Close()));

            MainMenuStrip = menuStrip;
            Controls.Add(menuStrip);
        }

        // ────────────────────────────────────────────────────────────────────
        // 폼 로드 / 닫기
        // ────────────────────────────────────────────────────────────────────
        private void MainForm_Load(object sender, EventArgs e)
        {
            try
            {
                if (File.Exists(LayoutFile))
                    RestoreLayout();
                else
                    OpenDefaultLayout();

                // 메뉴 체크 상태를 현재 열린 창과 동기화
                foreach (var kv in _dockMenuMap)
                    kv.Value.Checked = _dockPanel.Contents.Any(c => c.GetType().Name == kv.Key);
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[로드 실패] 레이아웃 로드 중 오류: {ex.Message}");
            }
        }

        private void RestoreLayout()
        {
            var doc = new XmlDocument();
            doc.Load(LayoutFile);

            var layoutNode = doc.SelectSingleNode("Layout");
            if (layoutNode != null)
                RestoreWindowBounds(layoutNode);

            var dockNode = doc.SelectSingleNode("Layout/DockPanel");
            if (dockNode != null)
                RestoreDockPanel(dockNode);
        }

        private void RestoreWindowBounds(XmlNode node)
        {
            int x = int.Parse(node.Attributes["X"].Value);
            int y = int.Parse(node.Attributes["Y"].Value);
            int w = int.Parse(node.Attributes["Width"].Value);
            int h = int.Parse(node.Attributes["Height"].Value);
            var state = (FormWindowState)Enum.Parse(typeof(FormWindowState), node.Attributes["State"].Value);

            StartPosition = FormStartPosition.Manual;
            Location = new Point(x, y);
            Size = new Size(Math.Max(200, w), Math.Max(150, h));
            WindowState = state;

            // 화면 밖에 위치한 경우 안전 위치로 리셋
            if (!IsOnAnyScreen(new Rectangle(Location, Size)))
            {
                WindowState = FormWindowState.Normal;
                StartPosition = FormStartPosition.CenterScreen;
                Size = new Size(1200, 800);
                CenterToScreen();
            }
        }

        private void RestoreDockPanel(XmlNode dockNode)
        {
            string tempFile = Path.GetTempFileName();
            try
            {
                File.WriteAllText(tempFile, dockNode.InnerXml);
                _dockPanel.LoadFromXml(tempFile, DeserializeDockContent);
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[복원 실패] DockPanel 로드 중 오류: {ex.Message}");
                OpenDefaultLayout();
            }
            finally
            {
                File.Delete(tempFile);
            }
        }

        private void OpenDefaultLayout()
        {
            // AxisInfoForm — _axisInfo 필드를 채워서 나중에 재활용
            _axisInfo = new AxisInfoForm(_motion, _axisConfigs);
            _axisInfo.Show(_dockPanel, DockState.DockBottom);

            // LogWriter — 이미 생성된 _logWriter를 재사용 (중복 생성 방지)
            _logWriter.Show(_dockPanel, DockState.Document);
        }

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            SaveLayout();
            SaveAxisConfigs();
            DisposeServices();
        }

        // ────────────────────────────────────────────────────────────────────
        // 레이아웃 저장
        // ────────────────────────────────────────────────────────────────────
        private void SaveLayout()
        {
            try
            {
                // DockPanel XML 추출 (임시 파일 경유)
                string tempFile = Path.GetTempFileName();
                _dockPanel.SaveAsXml(tempFile);
                string dockXml = File.ReadAllText(tempFile);
                File.Delete(tempFile);

                // XML 선언 제거
                if (dockXml.StartsWith("<?xml"))
                {
                    int idx = dockXml.IndexOf("?>");
                    if (idx != -1) dockXml = dockXml.Substring(idx + 2).Trim();
                }

                var doc = new XmlDocument();
                var root = doc.CreateElement("Layout");

                // 최소화 상태는 Normal로 저장 (복원 시 최소화 방지)
                var saveState = WindowState == FormWindowState.Minimized
                    ? FormWindowState.Normal
                    : WindowState;

                root.SetAttribute("X", Location.X.ToString());
                root.SetAttribute("Y", Location.Y.ToString());
                root.SetAttribute("Width", Size.Width.ToString());
                root.SetAttribute("Height", Size.Height.ToString());
                root.SetAttribute("State", saveState.ToString());
                doc.AppendChild(root);

                var dockNode = doc.CreateElement("DockPanel");
                dockNode.InnerXml = dockXml;
                root.AppendChild(dockNode);

                doc.Save(LayoutFile);
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[저장 실패] 레이아웃 저장 중 오류: {ex.Message}");
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // 축 설정 저장 / 불러오기
        // ────────────────────────────────────────────────────────────────────
        private void SaveAxisConfigs()
        {
            try
            {
                if (_axisConfigs == null || _axisConfigs.Length == 0) return;

                var json = JsonSerializer.Serialize(_axisConfigs,
                    new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(AxisConfigFile, json);
                AppEvents.RaiseLog("[저장] AxisConfig 저장 완료");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[저장 실패] AxisConfig 저장 중 오류: {ex.Message}");
            }
        }

        private void LoadAxisConfigs()
        {
            try
            {
                if (!File.Exists(AxisConfigFile)) return;

                var json = File.ReadAllText(AxisConfigFile);
                var configs = JsonSerializer.Deserialize<AxisConfig[]>(json);
                if (configs == null) return;

                _axisConfigs = configs;
                _motion.SetAxisConfigs(_axisConfigs);
                AppEvents.RaiseLog("[불러오기] AxisConfig 로드 완료");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[불러오기 실패] AxisConfig 로드 중 오류: {ex.Message}");
            }
        }

        // ────────────────────────────────────────────────────────────────────
        // 리소스 정리
        // ────────────────────────────────────────────────────────────────────
        private void DisposeServices()
        {
            try { _influxPublisher?.Disable(); } catch { }
            try { _influxPublisher?.Dispose(); } catch { }
            try { _daq?.Dispose(); } catch { }
            try { _controller?.Dispose(); } catch { }
        }

        // ────────────────────────────────────────────────────────────────────
        // DockContent 역직렬화
        // ────────────────────────────────────────────────────────────────────
        private IDockContent DeserializeDockContent(string persistString)
        {
            if (persistString == typeof(AxisInfoForm).ToString())
                return _axisInfo ?? (_axisInfo = new AxisInfoForm(_motion, _axisConfigs));

            if (persistString == typeof(TeachingForm).ToString())
                return _teaching ?? (_teaching = new TeachingForm(_motion));

            if (persistString == typeof(SimulatorForm).ToString())
                return _simulator ?? (_simulator = new SimulatorForm());

            if (persistString == typeof(LogWriterForm).ToString())
                return _logWriter ?? (_logWriter = new LogWriterForm());

            if (persistString == typeof(LogGraphForm).ToString())
                return _logGraph ?? (_logGraph = new LogGraphForm());

            if (persistString == typeof(PassiveMonitorForm).ToString())
            {
                var pmf = new PassiveMonitorForm();
                if (_influxPublisher != null)
                    pmf.BlockPublished += _influxPublisher.Feed;
                return pmf;
            }

            // 데이터 분석 폼 — 상태 없으므로 매번 새로 생성
            if (persistString == typeof(PHMPipelineWizard).ToString()) return new PHMPipelineWizard();
            if (persistString == typeof(SignalExplorerForm).ToString() ||
                persistString == "PHM_Project_DockPanel.UI.DataAnalysis.PreprocessingForm") return new SignalExplorerForm();
            if (persistString == typeof(AnomalyDetectionForm).ToString()) return new AnomalyDetectionForm();
            if (persistString == typeof(AIForm).ToString()) return new AIForm();
            if (persistString == typeof(DashboardForm).ToString()) return new DashboardForm();

            return null;
        }

        // ────────────────────────────────────────────────────────────────────
        // DockMenuItem 팩토리
        // ────────────────────────────────────────────────────────────────────

        /// <summary>기본 생성자가 있는 폼용 단축 오버로드.</summary>
        private ToolStripMenuItem CreateDockMenuItem<T>(string title, DockState defaultState)
            where T : DockContent, new()
            => CreateDockMenuItem(title, defaultState, typeof(T), () => new T());

        /// <summary>생성자 인자가 필요한 폼용 팩토리 기반 오버로드.</summary>
        private ToolStripMenuItem CreateDockMenuItem(
            string title, DockState defaultState, Type formType, Func<DockContent> factory)
        {
            var menuItem = new ToolStripMenuItem(title) { CheckOnClick = true };

            menuItem.CheckedChanged += (s, e) =>
            {
                if (menuItem.Checked)
                {
                    var existing = _dockPanel.Contents.FirstOrDefault(c => c.GetType() == formType);
                    if (existing == null)
                    {
                        var form = factory();
                        form.FormClosed += (fs, fe) => menuItem.Checked = false;
                        form.Show(_dockPanel, defaultState);
                    }
                    else
                    {
                        ((DockContent)existing).Show();
                    }
                }
                else
                {
                    (_dockPanel.Contents.FirstOrDefault(c => c.GetType() == formType) as DockContent)?.Close();
                }
            };

            _dockMenuMap[formType.Name] = menuItem;
            menuItem.Checked = _dockPanel.Contents.Any(c => c.GetType() == formType);

            return menuItem;
        }

        // ────────────────────────────────────────────────────────────────────
        // 유틸리티
        // ────────────────────────────────────────────────────────────────────

        /// <summary>DockPanel에서 현재 열려 있는 특정 타입의 창을 반환합니다.</summary>
        private T FindOpenForm<T>() where T : DockContent
            => _dockPanel.Contents.OfType<T>().FirstOrDefault(f => !f.IsDisposed);

        private LogGraphForm EnsureLogGraphOpen()
        {
            var existing = FindOpenForm<LogGraphForm>();
            if (existing != null) { existing.Activate(); return existing; }
            _logGraph = _logGraph ?? new LogGraphForm();
            _logGraph.Show(_dockPanel, DockState.Document);
            return _logGraph;
        }

        /// <summary>윈도우 사각형이 어느 화면과도 50×50 이상 겹치지 않으면 false.</summary>
        private static bool IsOnAnyScreen(Rectangle rect)
        {
            foreach (var screen in Screen.AllScreens)
            {
                var inter = Rectangle.Intersect(screen.WorkingArea, rect);
                if (inter.Width >= 50 && inter.Height >= 50)
                    return true;
            }
            return false;
        }
    }
}