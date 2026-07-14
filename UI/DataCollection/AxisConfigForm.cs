using System;
using System.Drawing;
using System.IO;
using System.Text.Json;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using PHM_Project_DockPanel.Services;

namespace PHM_Project_DockPanel.Windows
{
    /// <summary>
    /// 축별 모션 파라미터(Pitch, 엔코더 펄스/rev, 이동 명령 보정 배율 등) 편집 패널.
    /// 데이터 수집 > 환경 설정 > 축 설정 관리 메뉴로 엽니다.
    /// </summary>
    public class AxisConfigForm : DockContent
    {
        private readonly string _configPath;
        private readonly Action<AxisConfig[]> _onApply;
        private readonly AxisConfig[] _configs;

        private DataGridView _grid;
        private Label _lblHelp;

        public AxisConfigForm(string configPath, AxisConfig[] configs, Action<AxisConfig[]> onApply)
        {
            _configPath = configPath;
            _configs    = configs ?? new AxisConfig[0];
            _onApply    = onApply;

            Text    = "축 설정 관리";
            TabText = "축 설정 관리";

            Build();
            LoadValues();
        }

        // ─────────────────────────────────────────────────────────────────
        // UI 구성
        // ─────────────────────────────────────────────────────────────────
        private void Build()
        {
            BackColor   = Color.WhiteSmoke;
            MinimumSize = new Size(760, 360);

            var root = new TableLayoutPanel
            {
                Dock       = DockStyle.Fill,
                ColumnCount = 1,
                RowCount    = 3,
                Padding     = new Padding(12),
            };
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var lblTitle = new Label
            {
                Text     = "축별 모션 파라미터",
                AutoSize = true,
                Font     = new Font("Segoe UI", 11f, FontStyle.Bold),
                Margin   = new Padding(0, 0, 0, 4),
            };

            _grid = new DataGridView
            {
                Dock                     = DockStyle.Fill,
                AutoGenerateColumns      = false,
                AllowUserToAddRows       = false,
                AllowUserToDeleteRows    = false,
                RowHeadersWidth          = 70,
                RowHeadersVisible        = true,
                ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize,
                AutoSizeColumnsMode      = DataGridViewAutoSizeColumnsMode.Fill,
                BackgroundColor          = Color.White,
                BorderStyle              = BorderStyle.FixedSingle,
            };

            _grid.Columns.Add(MakeCol(nameof(AxisConfig.PositionMax),     "Max Stroke (mm)"));
            _grid.Columns.Add(MakeCol(nameof(AxisConfig.PitchMmPerRev),   "Pitch (mm/rev)"));
            _grid.Columns.Add(MakeCol(nameof(AxisConfig.PulsePerRev),     "엔코더 펄스/rev\n(0=기본값 8388608)"));
            _grid.Columns.Add(MakeCol(nameof(AxisConfig.MoveCommandScale),"이동 명령 보정 배율\n(1=보정없음)"));
            _grid.Columns.Add(MakeCol(nameof(AxisConfig.MaxVel),          "최대 속도"));
            _grid.Columns.Add(MakeCol(nameof(AxisConfig.Acc),             "가속도"));
            _grid.Columns.Add(MakeCol(nameof(AxisConfig.Dec),             "감속도"));

            _grid.RowPrePaint += (s, e) =>
            {
                _grid.Rows[e.RowIndex].HeaderCell.Value = $"Axis {e.RowIndex}";
            };

            var btnRow = new FlowLayoutPanel
            {
                Dock          = DockStyle.Fill,
                FlowDirection = FlowDirection.LeftToRight,
                AutoSize      = true,
                Margin        = new Padding(0, 8, 0, 0),
            };
            var btnSave = new Button
            {
                Text      = "저장 & 적용",
                Width     = 140,
                Height    = 32,
                BackColor = Color.SteelBlue,
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Font      = new Font("Segoe UI", 9.5f, FontStyle.Bold),
            };
            btnSave.FlatAppearance.BorderSize = 0;
            btnSave.Click += BtnSave_Click;
            var btnReload = new Button { Text = "다시 불러오기", Width = 110, Height = 32 };
            btnReload.Click += (s, e) => LoadValues();
            btnRow.Controls.Add(btnSave);
            btnRow.Controls.Add(btnReload);

            _lblHelp = new Label
            {
                AutoSize  = true,
                ForeColor = Color.DimGray,
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Italic),
                Margin    = new Padding(0, 8, 0, 0),
                Text      =
                    "엔코더 펄스/rev: 축마다 다른 서보 드라이브(예: 이노반스 vs 파나소닉)를 쓰면 실제 값이 다를 수 있습니다.\n" +
                    "이동 명령 보정 배율: 드라이브 쪽 원인으로 명령 거리와 실제 이동 거리가 일정 비율로 어긋날 때 임시로 보정하는 값입니다. " +
                    "위치 표시(피드백)에는 영향을 주지 않고 이동 명령에만 적용됩니다.",
            };

            root.Controls.Add(lblTitle, 0, 0);
            root.Controls.Add(_grid,    0, 1);

            var bottom = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2, AutoSize = true };
            bottom.Controls.Add(btnRow,    0, 0);
            bottom.Controls.Add(_lblHelp,  0, 1);
            root.Controls.Add(bottom, 0, 2);

            Controls.Add(root);
        }

        private static DataGridViewTextBoxColumn MakeCol(string propName, string header) =>
            new DataGridViewTextBoxColumn
            {
                DataPropertyName = propName,
                HeaderText       = header,
                Name             = propName,
                DefaultCellStyle = new DataGridViewCellStyle { Format = "0.######" },
            };

        // ─────────────────────────────────────────────────────────────────
        // 데이터 바인딩
        // ─────────────────────────────────────────────────────────────────
        private void LoadValues()
        {
            var bs = new BindingSource { DataSource = _configs };
            _grid.DataSource = bs;
        }

        private void BtnSave_Click(object sender, EventArgs e)
        {
            try
            {
                _grid.EndEdit();
                _grid.Refresh();

                var json = JsonSerializer.Serialize(_configs, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_configPath, json);

                _onApply?.Invoke(_configs);

                AppEvents.RaiseLog("[축 설정] 저장 & 적용 완료");
                MessageBox.Show("축 설정을 저장하고 적용했습니다.", "완료",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[축 설정] 저장 실패: {ex.Message}");
                MessageBox.Show("저장 중 오류가 발생했습니다:\n" + ex.Message, "오류",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
