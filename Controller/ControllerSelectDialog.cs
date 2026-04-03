using System;
using System.Drawing;
using System.Windows.Forms;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// 앱 시작 시 표시되는 제어기 선택 다이얼로그.
    /// 선택 결과는 SelectedController 프로퍼티로 꺼내 씁니다.
    /// </summary>
    public class ControllerSelectDialog : Form
    {
        public IMotionController SelectedController { get; private set; }

        private NumericUpDown _numAxisCount;

        public ControllerSelectDialog()
        {
            Text = "모션 제어기 선택";
            Size = new Size(340, 270);
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;

            var lbl = new Label
            {
                Text = "사용할 모션 제어기를 선택하세요.",
                Left = 20, Top = 20, Width = 290, Height = 24,
                Font = new Font("Segoe UI", 10f)
            };

            int btnW = 280, btnH = 38, btnX = 20;

            var btnWMX3 = new Button
            {
                Text = "WMX3  (SoftServo)",
                Left = btnX, Top = 60, Width = btnW, Height = btnH,
                Font = new Font("Segoe UI", 10f)
            };
            btnWMX3.Click += (s, e) =>
            {
                SelectedController = new WMX3Controller();
                DialogResult = DialogResult.OK;
            };

            var btnAjin = new Button
            {
                Text = "Ajin  (AMP 제어기)",
                Left = btnX, Top = 106, Width = btnW, Height = btnH,
                Font = new Font("Segoe UI", 10f)
            };
            btnAjin.Click += (s, e) =>
            {
                SelectedController = new AjinController();
                DialogResult = DialogResult.OK;
            };

            var btnSim = new Button
            {
                Text = "시뮬레이션  (하드웨어 없음)",
                Left = btnX, Top = 152, Width = btnW, Height = btnH,
                Font = new Font("Segoe UI", 10f),
                BackColor = Color.FromArgb(230, 240, 255)
            };
            btnSim.Click += (s, e) =>
            {
                SelectedController = new SimulationController((int)_numAxisCount.Value);
                DialogResult = DialogResult.OK;
            };

            // ── 시뮬레이션 옵션: 축 수 ────────────────────────────────
            var lblAxisCount = new Label
            {
                Text = "시뮬레이션 축 수:",
                Left = btnX, Top = 204, Width = 130, Height = 22,
                Font = new Font("Segoe UI", 9f),
                ForeColor = Color.DimGray,
                TextAlign = ContentAlignment.MiddleLeft
            };

            _numAxisCount = new NumericUpDown
            {
                Left = btnX + 135, Top = 204, Width = 60, Height = 22,
                Minimum = 1, Maximum = 16, Value = 5,
                Font = new Font("Segoe UI", 9f)
            };

            Controls.AddRange(new Control[] { lbl, btnWMX3, btnAjin, btnSim, lblAxisCount, _numAxisCount });
        }
    }
}
