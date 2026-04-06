using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services;
using PHM_Project_DockPanel.Services.DAQ;
using NationalInstruments.DAQmx;

namespace PHM_Project_DockPanel.Windows
{
    /// <summary>
    /// DAQ 가속도 센서 파라미터 설정 패널.
    /// 데이터 수집 > 환경 설정 > DAQ 센서 설정 메뉴로 엽니다.
    /// </summary>
    public class DaqSettingsForm : DockContent
    {
        private readonly string _configPath;
        private readonly Action<DaqSensorConfig> _onApply;

        // ── 하드웨어 ──────────────────────────────────────────────────────
        private TextBox    _txtModule;
        private ComboBox   _cmbChannel;

        // ── 민감도 (mV/g) ─────────────────────────────────────────────────
        private NumericUpDown _numSensX, _numSensY, _numSensZ;

        // ── 수집 설정 ─────────────────────────────────────────────────────
        private NumericUpDown _numRate, _numBlock, _numGRange, _numIepe;

        // ── 상태 표시 ─────────────────────────────────────────────────────
        private Label _lblVoltRange;

        // ─────────────────────────────────────────────────────────────────
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
            AutoScroll  = true;
            BackColor   = Color.WhiteSmoke;
            MinimumSize = new Size(420, 500);

            var scroll = new Panel { Dock = DockStyle.Fill, AutoScroll = true };

            var root = new FlowLayoutPanel
            {
                FlowDirection = FlowDirection.TopDown,
                WrapContents  = false,
                AutoSize      = true,
                AutoSizeMode  = AutoSizeMode.GrowAndShrink,
                Padding       = new Padding(12),
                BackColor     = Color.WhiteSmoke
            };

            // ── 하드웨어 섹션 ────────────────────────────────────────────
            root.Controls.Add(MakeSectionLabel("하드웨어"));

            var tblHw = MakeTable(2);
            AddLabelRow(tblHw, 0, "모듈");
            var pnlModule = new Panel { Dock = DockStyle.Fill, Height = 28 };
            _txtModule = new TextBox
            {
                Anchor   = AnchorStyles.Left | AnchorStyles.Right | AnchorStyles.Top,
                Location = new Point(0, 2),
                Width    = 160,
                Height   = 22
            };
            var btnSearch = new Button
            {
                Text     = "검색...",
                Location = new Point(166, 1),
                Width    = 64,
                Height   = 24,
                FlatStyle = FlatStyle.Flat
            };
            btnSearch.FlatAppearance.BorderColor = Color.SteelBlue;
            btnSearch.Click += BtnSearch_Click;
            pnlModule.Controls.Add(_txtModule);
            pnlModule.Controls.Add(btnSearch);
            tblHw.Controls.Add(pnlModule, 1, 0);

            AddLabelRow(tblHw, 1, "채널");
            _cmbChannel = new ComboBox
            {
                Dock          = DockStyle.Fill,
                DropDownStyle = ComboBoxStyle.DropDown
            };
            _cmbChannel.Items.AddRange(new object[] {
                "ai0", "ai0:1", "ai0:2", "ai0:3",
                "ai1:3", "ai2:4"
            });
            tblHw.Controls.Add(_cmbChannel, 1, 1);
            root.Controls.Add(tblHw);

            // ── 민감도 섹션 ──────────────────────────────────────────────
            root.Controls.Add(MakeSectionLabel("민감도 (mV/g)"));
            var tblSens = MakeTable(3);
            AddLabelRow(tblSens, 0, "X (채널 ai0)");
            AddLabelRow(tblSens, 1, "Y (채널 ai1)");
            AddLabelRow(tblSens, 2, "Z (채널 ai2)");
            _numSensX = MakeNumeric(1, 9999, 1, 1026); tblSens.Controls.Add(_numSensX, 1, 0);
            _numSensY = MakeNumeric(1, 9999, 1, 991);  tblSens.Controls.Add(_numSensY, 1, 1);
            _numSensZ = MakeNumeric(1, 9999, 1, 985);  tblSens.Controls.Add(_numSensZ, 1, 2);
            root.Controls.Add(tblSens);

            // ── 수집 설정 섹션 ───────────────────────────────────────────
            root.Controls.Add(MakeSectionLabel("수집 설정"));
            var tblAcq = MakeTable(4);
            AddLabelRow(tblAcq, 0, "샘플레이트 (Hz)");
            AddLabelRow(tblAcq, 1, "프레임 크기 (samples)");
            AddLabelRow(tblAcq, 2, "측정 범위 ± (g)");
            AddLabelRow(tblAcq, 3, "IEPE 전류 (A)");

            _numRate  = MakeNumeric(1, 1000000, 0, 1000);  tblAcq.Controls.Add(_numRate,  1, 0);
            _numBlock = MakeNumeric(1, 1000000, 0, 100);   tblAcq.Controls.Add(_numBlock, 1, 1);
            _numGRange = MakeNumeric(0.01m, 9999, 2, 5);   tblAcq.Controls.Add(_numGRange, 1, 2);
            _numIepe  = MakeNumeric(0.001m, 0.02m, 4, 0.004m); tblAcq.Controls.Add(_numIepe, 1, 3);

            root.Controls.Add(tblAcq);

            // 전압 범위 표시
            _lblVoltRange = new Label
            {
                AutoSize  = true,
                ForeColor = Color.SteelBlue,
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Italic),
                Margin    = new Padding(4, 2, 0, 6)
            };
            root.Controls.Add(_lblVoltRange);

            // 저장 & 적용 버튼
            var btnSave = new Button
            {
                Text      = "저장 & 적용",
                Width     = 370,
                Height    = 36,
                Margin    = new Padding(0, 8, 0, 0),
                BackColor = Color.SteelBlue,
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font      = new Font("Segoe UI", 10f, FontStyle.Bold)
            };
            btnSave.FlatAppearance.BorderSize = 0;
            btnSave.Click += BtnSave_Click;
            root.Controls.Add(btnSave);

            scroll.Controls.Add(root);
            Controls.Add(scroll);

            // 이벤트: 민감도·범위 변경 시 전압 범위 실시간 갱신
            _numSensX.ValueChanged  += (s, e) => RefreshVoltLabel();
            _numSensY.ValueChanged  += (s, e) => RefreshVoltLabel();
            _numSensZ.ValueChanged  += (s, e) => RefreshVoltLabel();
            _numGRange.ValueChanged += (s, e) => RefreshVoltLabel();

            // 샘플레이트 변경 시 프레임 크기 자동 추천 (rate / 10)
            _numRate.ValueChanged += (s, e) =>
                _numBlock.Value = Math.Max(1, (int)_numRate.Value / 10);
        }

        // ─────────────────────────────────────────────────────────────────
        // 모듈 검색
        // ─────────────────────────────────────────────────────────────────
        private void BtnSearch_Click(object sender, EventArgs e)
        {
            try
            {
                string[] all = DaqSystem.Local.Devices;

                // AI 채널이 있는 모듈만 필터 (cDAQxModY 형태 우선)
                var modules = new List<string>();
                foreach (var d in all)
                {
                    if (d.IndexOf("Mod", StringComparison.OrdinalIgnoreCase) >= 0)
                        modules.Add(d);
                }
                if (modules.Count == 0)
                    modules.AddRange(all);   // 없으면 전체 표시

                if (modules.Count == 0)
                {
                    MessageBox.Show("연결된 NI-DAQ 장치가 없습니다.", "검색",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                ShowModuleSelectDialog(modules);
            }
            catch (Exception ex)
            {
                MessageBox.Show("장치 검색 실패 (NI-DAQmx 설치 확인):\n" + ex.Message,
                    "오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        private void ShowModuleSelectDialog(List<string> modules)
        {
            using (var dlg = new Form
            {
                Text            = "DAQ 모듈 선택",
                Size            = new Size(320, 380),
                StartPosition   = FormStartPosition.CenterParent,
                FormBorderStyle = FormBorderStyle.FixedDialog,
                MaximizeBox     = false, MinimizeBox = false
            })
            {
                var lb = new ListBox
                {
                    Dock     = DockStyle.Fill,
                    Font     = new Font("Consolas", 10f),
                    IntegralHeight = false
                };
                lb.Items.AddRange(modules.ToArray());
                // 현재 선택된 모듈 미리 선택
                var cur = _txtModule.Text.Trim();
                int idx = lb.Items.IndexOf(cur);
                if (idx >= 0) lb.SelectedIndex = idx;
                else if (lb.Items.Count > 0) lb.SelectedIndex = 0;

                var pnlBtn = new Panel { Dock = DockStyle.Bottom, Height = 44, Padding = new Padding(8) };
                var btnOk = new Button
                {
                    Text = "선택", DialogResult = DialogResult.OK,
                    Width = 90, Height = 28,
                    Location = new Point(8, 8)
                };
                var btnCancel = new Button
                {
                    Text = "취소", DialogResult = DialogResult.Cancel,
                    Width = 90, Height = 28,
                    Location = new Point(106, 8)
                };
                pnlBtn.Controls.Add(btnOk);
                pnlBtn.Controls.Add(btnCancel);
                dlg.AcceptButton = btnOk;
                dlg.CancelButton = btnCancel;

                lb.DoubleClick += (s, e) => { dlg.DialogResult = DialogResult.OK; };
                dlg.Controls.Add(lb);
                dlg.Controls.Add(pnlBtn);

                if (dlg.ShowDialog(this) == DialogResult.OK && lb.SelectedItem != null)
                {
                    _txtModule.Text = lb.SelectedItem.ToString();
                    RefreshChannelCombo(lb.SelectedItem.ToString());
                }
            }
        }

        private void RefreshChannelCombo(string moduleName)
        {
            var current = _cmbChannel.Text;
            _cmbChannel.Items.Clear();

            try
            {
                // NI-DAQmx에서 AI 채널 목록 가져오기
                var dev = DaqSystem.Local.LoadDevice(moduleName);
                var aiChans = dev.AIPhysicalChannels;

                if (aiChans != null && aiChans.Length > 0)
                {
                    int count = aiChans.Length;
                    // 채널 범위 옵션 생성 (ai0, ai0:1, ai0:2, ...)
                    for (int i = 0; i < count; i++)
                        _cmbChannel.Items.Add($"ai0:{i}");
                    // 단일 채널도 추가
                    _cmbChannel.Items.Insert(0, "ai0");
                }
            }
            catch { }

            // 기본 옵션 보장
            if (_cmbChannel.Items.Count == 0)
                _cmbChannel.Items.AddRange(new object[] { "ai0", "ai0:1", "ai0:2", "ai0:3" });

            // 이전 선택값 복원 또는 3채널(X/Y/Z) 기본 선택
            if (_cmbChannel.Items.Contains(current))
                _cmbChannel.Text = current;
            else if (_cmbChannel.Items.Contains("ai0:2"))
                _cmbChannel.Text = "ai0:2";
            else
                _cmbChannel.SelectedIndex = 0;
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
                RefreshVoltLabel();
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
            _txtModule.Text   = cfg.Module;
            _cmbChannel.Text  = cfg.Channel;

            _numSensX.Value   = ClampD(cfg.SensX,  1, 9999);
            _numSensY.Value   = ClampD(cfg.SensY,  1, 9999);
            _numSensZ.Value   = ClampD(cfg.SensZ,  1, 9999);

            _numRate.Value    = ClampD(cfg.SampleRate,      1, 1000000);
            _numBlock.Value   = (decimal)Math.Max(1, Math.Min(1000000, cfg.ReadBlock));
            _numGRange.Value  = ClampD(cfg.GRange,          0.01m, 9999);
            _numIepe.Value    = ClampD(cfg.IepeCurrentAmps, 0.001m, 0.02m);

            RefreshVoltLabel();
        }

        private DaqSensorConfig ReadValues() => new DaqSensorConfig
        {
            Module         = _txtModule.Text.Trim(),
            Channel        = _cmbChannel.Text.Trim(),
            SensX          = (double)_numSensX.Value,
            SensY          = (double)_numSensY.Value,
            SensZ          = (double)_numSensZ.Value,
            SampleRate     = (double)_numRate.Value,
            ReadBlock      = (int)_numBlock.Value,
            GRange         = (double)_numGRange.Value,
            IepeCurrentAmps= (double)_numIepe.Value
        };

        private void RefreshVoltLabel()
        {
            double maxSens   = Math.Max((double)_numSensX.Value,
                               Math.Max((double)_numSensY.Value, (double)_numSensZ.Value));
            double voltRange = (double)_numGRange.Value * maxSens / 1000.0;
            _lblVoltRange.Text = $"→ 전압 범위 ±{voltRange:F3} V  (NI 9230 max ±30 V)";
        }

        // ─────────────────────────────────────────────────────────────────
        // UI 헬퍼
        // ─────────────────────────────────────────────────────────────────
        private const int LABEL_W = 160;
        private const int ROW_H   = 30;
        private const int TABLE_W = 370;

        private static Label MakeSectionLabel(string title) => new Label
        {
            Text      = "  " + title,
            AutoSize  = false,
            Width     = TABLE_W,
            Height    = 24,
            Font      = new Font("Segoe UI", 9f, FontStyle.Bold),
            ForeColor = Color.White,
            BackColor = Color.SteelBlue,
            TextAlign = ContentAlignment.MiddleLeft,
            Margin    = new Padding(0, 10, 0, 4)
        };

        /// <summary>rows 행 × 2 열(Label | Control) TableLayoutPanel 생성.</summary>
        private static TableLayoutPanel MakeTable(int rows)
        {
            var tbl = new TableLayoutPanel
            {
                ColumnCount = 2,
                RowCount    = rows,
                AutoSize    = true,
                Width       = TABLE_W,
                Margin      = new Padding(0, 0, 0, 4),
                BackColor   = Color.White,
                Padding     = new Padding(4)
            };
            tbl.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, LABEL_W));
            tbl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,  100f));
            for (int r = 0; r < rows; r++)
                tbl.RowStyles.Add(new RowStyle(SizeType.Absolute, ROW_H));
            return tbl;
        }

        private static void AddLabelRow(TableLayoutPanel tbl, int row, string text)
        {
            tbl.Controls.Add(new Label
            {
                Text      = text,
                Dock      = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                Font      = new Font("Segoe UI", 9f),
                ForeColor = Color.DimGray
            }, 0, row);
        }

        private static NumericUpDown MakeNumeric(decimal min, decimal max, int decimals, decimal def)
        {
            return new NumericUpDown
            {
                Minimum       = min,
                Maximum       = max,
                DecimalPlaces = decimals,
                Value         = Math.Max(min, Math.Min(max, def)),
                Dock          = DockStyle.Fill,
                TextAlign     = HorizontalAlignment.Right,
                Font          = new Font("Segoe UI", 9.5f)
            };
        }

        private static decimal ClampD(double v, decimal lo, decimal hi)
        {
            var d = (decimal)v;
            return d < lo ? lo : d > hi ? hi : d;
        }
    }
}
