using System;
using System.Drawing;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services;
using PHM_Project_DockPanel.Services.DAQ;

namespace PHM_Project_DockPanel.Windows
{
    /// <summary>
    /// DAQ 가속도 센서 파라미터 설정 패널.
    /// 설정을 JSON 파일로 저장하고, 저장 시 즉시 적용합니다.
    /// </summary>
    public class DaqSettingsForm : DockContent
    {
        private readonly string _configPath;
        private readonly Action<DaqSensorConfig> _onApply;

        // ── 하드웨어 ──────────────────────────────────────────────────────
        private TextBox _txtModule, _txtChannel;

        // ── 민감도 ────────────────────────────────────────────────────────
        private NumericUpDown _numSensX, _numSensY, _numSensZ;

        // ── 오프셋 ────────────────────────────────────────────────────────
        private NumericUpDown _numOffX, _numOffY, _numOffZ;

        // ── 수집 설정 ─────────────────────────────────────────────────────
        private NumericUpDown _numRate, _numBlock, _numGRange;

        private Label _lblVoltRange;   // 계산된 전압 범위 표시

        public DaqSettingsForm(string configPath, Action<DaqSensorConfig> onApply)
        {
            _configPath = configPath;
            _onApply    = onApply;

            Text    = "DAQ 센서 설정";
            TabText = "DAQ 센서 설정";

            Build();
            LoadValues(DaqSensorConfig.LoadOrDefault(_configPath));
        }

        // ─────────────────────────────────────────────────────────────────
        // UI 구성
        // ─────────────────────────────────────────────────────────────────
        private void Build()
        {
            this.AutoScroll = true;
            this.BackColor  = Color.WhiteSmoke;
            this.Padding    = new Padding(12);

            var root = new FlowLayoutPanel
            {
                Dock          = DockStyle.Fill,
                FlowDirection = FlowDirection.TopDown,
                WrapContents  = false,
                AutoSize      = true,
                AutoSizeMode  = AutoSizeMode.GrowAndShrink,
                Padding       = new Padding(0)
            };

            root.Controls.Add(MakeSection("하드웨어"));
            root.Controls.Add(MakeRow("모듈",         out _txtModule,  isText: true,  def: "cDAQ2Mod2"));
            root.Controls.Add(MakeRow("채널",          out _txtChannel, isText: true,  def: "ai0:2"));

            root.Controls.Add(MakeSection("민감도 (mV/g)"));
            root.Controls.Add(MakeNumRow("X",          out _numSensX, 0, 10000, 1, 1026.0));
            root.Controls.Add(MakeNumRow("Y",          out _numSensY, 0, 10000, 1, 991.0));
            root.Controls.Add(MakeNumRow("Z",          out _numSensZ, 0, 10000, 1, 985.0));

            root.Controls.Add(MakeSection("오프셋 (g)"));
            root.Controls.Add(MakeNumRow("X",          out _numOffX, -100, 100, 3, 0.0));
            root.Controls.Add(MakeNumRow("Y",          out _numOffY, -100, 100, 3, 0.0));
            root.Controls.Add(MakeNumRow("Z",          out _numOffZ, -100, 100, 3, 0.0));

            root.Controls.Add(MakeSection("수집 설정"));
            root.Controls.Add(MakeNumRow("샘플레이트 (Hz)", out _numRate,   1, 100000, 0, 1000.0));
            root.Controls.Add(MakeNumRow("프레임 크기 (samples)", out _numBlock,  1, 100000, 0, 100.0));
            root.Controls.Add(MakeNumRow("측정 범위 ± (g)", out _numGRange, 0.1m, 1000, 2, 5.0));

            // 전압 범위 표시 (읽기 전용)
            _lblVoltRange = new Label
            {
                AutoSize  = true,
                ForeColor = Color.SteelBlue,
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Italic),
                Margin    = new Padding(8, 0, 0, 8)
            };
            root.Controls.Add(_lblVoltRange);

            // 저장 & 적용 버튼
            var btnSave = new Button
            {
                Text      = "저장 & 적용",
                Width     = 340,
                Height    = 32,
                Margin    = new Padding(0, 8, 0, 0),
                BackColor = Color.SteelBlue,
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat
            };
            btnSave.FlatAppearance.BorderSize = 0;
            btnSave.Click += BtnSave_Click;
            root.Controls.Add(btnSave);

            Controls.Add(root);

            // 민감도 / g범위 변경 시 전압 범위 실시간 갱신
            _numSensX.ValueChanged += (s, e) => RefreshVoltageLabel();
            _numSensY.ValueChanged += (s, e) => RefreshVoltageLabel();
            _numSensZ.ValueChanged += (s, e) => RefreshVoltageLabel();
            _numGRange.ValueChanged += (s, e) => RefreshVoltageLabel();

            // 샘플레이트 변경 시 프레임 크기 자동 추천 (SampleRate / 10)
            _numRate.ValueChanged += (s, e) =>
            {
                int suggested = Math.Max(1, (int)_numRate.Value / 10);
                _numBlock.Value = suggested;
            };
        }

        // ─────────────────────────────────────────────────────────────────
        // 저장 & 적용
        // ─────────────────────────────────────────────────────────────────
        private void BtnSave_Click(object sender, EventArgs e)
        {
            var cfg = ReadValues();

            try
            {
                cfg.Save(_configPath);
                _onApply?.Invoke(cfg);
                RefreshVoltageLabel();
                AppEvents.RaiseLog($"[DAQ 설정] 저장 완료 → {_configPath}");
                MessageBox.Show("설정이 저장되고 적용되었습니다.", "완료",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show("저장 중 오류: " + ex.Message, "오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // 값 읽기 / 쓰기
        // ─────────────────────────────────────────────────────────────────
        private void LoadValues(DaqSensorConfig cfg)
        {
            _txtModule.Text  = cfg.Module;
            _txtChannel.Text = cfg.Channel;
            _numSensX.Value  = (decimal)Clamp(cfg.SensX,  0, 10000);
            _numSensY.Value  = (decimal)Clamp(cfg.SensY,  0, 10000);
            _numSensZ.Value  = (decimal)Clamp(cfg.SensZ,  0, 10000);
            _numOffX.Value   = (decimal)Clamp(cfg.OffsetX, -100, 100);
            _numOffY.Value   = (decimal)Clamp(cfg.OffsetY, -100, 100);
            _numOffZ.Value   = (decimal)Clamp(cfg.OffsetZ, -100, 100);
            _numRate.Value   = (decimal)Clamp(cfg.SampleRate, 1, 100000);
            _numBlock.Value  = Math.Max(1, Math.Min(100000, cfg.ReadBlock));
            _numGRange.Value = (decimal)Clamp(cfg.GRange, 0.1, 1000);
            RefreshVoltageLabel();
        }

        private DaqSensorConfig ReadValues() => new DaqSensorConfig
        {
            Module     = _txtModule.Text.Trim(),
            Channel    = _txtChannel.Text.Trim(),
            SensX      = (double)_numSensX.Value,
            SensY      = (double)_numSensY.Value,
            SensZ      = (double)_numSensZ.Value,
            OffsetX    = (double)_numOffX.Value,
            OffsetY    = (double)_numOffY.Value,
            OffsetZ    = (double)_numOffZ.Value,
            SampleRate = (double)_numRate.Value,
            ReadBlock  = (int)_numBlock.Value,
            GRange     = (double)_numGRange.Value
        };

        private void RefreshVoltageLabel()
        {
            double maxSens  = Math.Max((double)_numSensX.Value,
                              Math.Max((double)_numSensY.Value, (double)_numSensZ.Value));
            double voltRange = (double)_numGRange.Value * maxSens / 1000.0;
            _lblVoltRange.Text = $"→ 전압 범위 ±{voltRange:F3} V  (NI 9230 max ±30 V)";
        }

        // ─────────────────────────────────────────────────────────────────
        // UI 헬퍼
        // ─────────────────────────────────────────────────────────────────
        private const int ROW_W = 340;

        private Label MakeSection(string title)
        {
            return new Label
            {
                Text      = title,
                AutoSize  = false,
                Width     = ROW_W,
                Height    = 22,
                Font      = new Font("Segoe UI", 9f, FontStyle.Bold),
                ForeColor = Color.DimGray,
                Margin    = new Padding(0, 10, 0, 2),
                BackColor = Color.Gainsboro,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding   = new Padding(4, 0, 0, 0)
            };
        }

        private Panel MakeRow(string label, out TextBox txt, bool isText, string def)
        {
            var p   = MakeRowPanel();
            var lbl = MakeLabel(label);
            txt = new TextBox
            {
                Text   = def,
                Width  = 180,
                Anchor = AnchorStyles.Left
            };
            p.Controls.Add(lbl);
            p.Controls.Add(txt);
            return p;
        }

        private Panel MakeNumRow(string label, out NumericUpDown num,
            decimal min, decimal max, int decimals, double def)
        {
            var p   = MakeRowPanel();
            var lbl = MakeLabel(label);
            num = new NumericUpDown
            {
                Minimum       = min,
                Maximum       = max,
                DecimalPlaces = decimals,
                Value         = (decimal)Clamp(def, (double)min, (double)max),
                Width         = 120,
                TextAlign     = HorizontalAlignment.Right
            };
            p.Controls.Add(lbl);
            p.Controls.Add(num);
            return p;
        }

        private static Panel MakeRowPanel() => new Panel
        {
            Width     = ROW_W,
            Height    = 28,
            Margin    = new Padding(0, 2, 0, 2)
        };

        private static Label MakeLabel(string text) => new Label
        {
            Text      = text,
            Width     = 155,
            Height    = 24,
            TextAlign = ContentAlignment.MiddleLeft,
            Location  = new Point(0, 2)
        };

        private static double Clamp(double v, double lo, double hi)
            => v < lo ? lo : v > hi ? hi : v;
    }
}
