using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.Windows;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace PHM_Project_DockPanel.Services
{
    public class AxisMonitor
    {
        private readonly ControllerManager _controller;
        private SimulatorForm _simulatorWindow;
        private readonly Label[,] _labels;
        private readonly int _axisCount;
        private readonly AxisConfig[] _configs;
        private bool _running;

        public event Action<float[]> PositionUpdated;
        public event Action<int, bool, bool, string> StatusUpdated;

        public AxisMonitor(ControllerManager controller, Label[,] labels, int axisCount, AxisConfig[] configs)
        {
            _controller = controller;
            _labels = labels;
            _axisCount = axisCount;
            _configs = configs;
        }

        public void Start()
        {
            _running = true;
            Task.Run(async () =>
            {
                while (_running)
                {
                    var status = _controller.GetStatus();
                    if (status != null)
                    {
                        UpdateUI(status);
                    }
                    await Task.Delay(200);
                }
            });
        }

        public void Stop() => _running = false;

        private void UpdateUI(dynamic status)
        {
            if (_configs == null || _axisCount <= 0)
                return;

            // Ajin은 AxmMotSetMoveUnitPerPulse로 이미 mm 단위 반환 → 변환 불필요
            // WMX3는 encoder pulse 단위 → UnitConverter.EncoderToMm 필요
            bool posAlreadyMm = _controller.PosIsAlreadyMm;

            float[] positionsMm = new float[_axisCount];

            for (int i = 0; i < _axisCount; i++)
            {
                double rawPos = status.AxesStatus[i].ActualPos;
                double posMm = posAlreadyMm
                    ? rawPos
                    : UnitConverter.EncoderToMm(rawPos, _configs[i]?.PitchMmPerRev ?? 30.0);

                bool servoOn = status.AxesStatus[i].ServoOn;
                bool alarm = status.AxesStatus[i].AmpAlarm;
                string opStr = status.AxesStatus[i].OpState.ToString();

                positionsMm[i] = (float)posMm;
                StatusUpdated?.Invoke(i, servoOn, !alarm, opStr);
            }

            PositionUpdated?.Invoke(positionsMm);
        }
    }
}