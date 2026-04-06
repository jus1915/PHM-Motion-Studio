using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.Services;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using static PHM_Project_DockPanel.Windows.AxisInfoForm;

namespace PHM_Project_DockPanel.Windows
{
    public class SimulatorForm : DockContent
    {
        private float[] _positions;
        private float[] _maxPositions;

        public SimulatorForm()
        {
            InitializeComponent();

            AppEvents.SimulatorInitializeRequested += InitializeSimulator;
            AppEvents.SimulatorInitializeWithMaxPositionsRequested += InitializeSimulator;
            AppEvents.SimulatorMaxPositionUpdateRequested += SetMaxPosition;
            AppEvents.SimulatorPositionUpdateRequested += UpdatePositions;
            AppEvents.RequestClearSimulator += ClearSimulator;

        }

        private void InitializeComponent()
        {
            Text = "Simulator";
            DoubleBuffered = true;
            BackColor = Color.White;
        }

        public void SetMaxPosition(int axisIndex, float maxPos)
        {
            if (_maxPositions == null || axisIndex < 0 || axisIndex >= _maxPositions.Length) return;
            _maxPositions[axisIndex] = maxPos;
            Invalidate();
        }

        public void SetPosition(int axisIndex, float pos)
        {
            if (_positions == null || axisIndex < 0 || axisIndex >= _positions.Length) return;
            _positions[axisIndex] = Math.Max(0, Math.Min(_maxPositions[axisIndex], pos));
            Invalidate();
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                AppEvents.SimulatorInitializeRequested -= InitializeSimulator;
                AppEvents.SimulatorInitializeWithMaxPositionsRequested -= InitializeSimulator;
                AppEvents.SimulatorMaxPositionUpdateRequested -= SetMaxPosition;
                AppEvents.SimulatorPositionUpdateRequested -= UpdatePositions;
                AppEvents.RequestClearSimulator -= ClearSimulator;
            }
            base.Dispose(disposing);
        }

        private void ClearSimulator()
        {
            _positions = null;
            _maxPositions = null;
            Invalidate();
        }

        private void InitializeSimulator(int axisCount, float defaultMaxPos)
        {
            if (axisCount <= 0) { _positions = null; _maxPositions = null; Invalidate(); return; }

            _positions = new float[axisCount];
            _maxPositions = Enumerable.Repeat(defaultMaxPos, axisCount).ToArray();
            Invalidate();
        }

        private void InitializeSimulator(float[] maxPositions)
        {
            if (maxPositions == null || maxPositions.Length == 0) { _positions = null; _maxPositions = null; Invalidate(); return; }

            _positions = new float[maxPositions.Length];
            _maxPositions = maxPositions.ToArray();
            Invalidate();
        }

        private void UpdatePositions(float[] positionsMm)
        {
            if (_positions == null || positionsMm.Length != _positions.Length) return;
            for (int i = 0; i < _positions.Length; i++)
                _positions[i] = Math.Max(0, Math.Min(_maxPositions[i], positionsMm[i]));
            Invalidate();
        }


        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.White);

            if (_positions == null || _maxPositions == null || _positions.Length == 0)
                return;

            // 현재 시뮬레이터에 있는 모든 축 중 최대 길이 찾기
            float maxRobotLength = _maxPositions.Max();
            if (maxRobotLength <= 0) maxRobotLength = 1f; // 0 방지

            int startX = 20;              // 왼쪽 시작 위치 (왼쪽 정렬)
            int rectSize = 30;            // 블록 크기
            int railSpacing = 60;         // 축 간격
            int maxRailDrawLength = Width - 40; // 최대 레일 표시 길이

            for (int i = 0; i < _positions.Length; i++)
            {
                // 해당 축의 최대 이동거리 비율
                float lengthRatio = _maxPositions[i] / maxRobotLength;

                // 실제 표시할 레일 길이
                int railLength = (int)(maxRailDrawLength * lengthRatio);

                int railY = 40 + i * railSpacing;
                g.DrawLine(Pens.Gray, startX, railY, startX + railLength, railY);

                // 현재 위치 비율
                float posRatio = _maxPositions[i] > 0 ? _positions[i] / _maxPositions[i] : 0;
                int rectX = (int)(startX + posRatio * railLength);
                int rectY = railY - rectSize / 2;

                // 블록 그리기
                g.FillRectangle(Brushes.SteelBlue, rectX - rectSize / 2, rectY, rectSize, rectSize);
                g.DrawRectangle(Pens.Black, rectX - rectSize / 2, rectY, rectSize, rectSize);

                // 텍스트
                g.DrawString(
                    $"Axis {i}: {_positions[i]:F1} / {_maxPositions[i]:F1} mm",
                    Font,
                    Brushes.Black,
                    10,
                    railY - 30
                );
            }
        }
    }
}
