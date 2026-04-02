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

            float[] positionsMm = new float[_axisCount];

            for (int i = 0; i < _axisCount; i++)
            {
                double posMm = UnitConverter.EncoderToMm(status.AxesStatus[i].ActualPos, _configs[i].PitchMmPerRev);
                bool servoOn = status.AxesStatus[i].ServoOn;
                bool alarm = status.AxesStatus[i].AmpAlarm; // true=알람 발생
                string opStr = status.AxesStatus[i].OpState.ToString();

                positionsMm[i] = (float)posMm;

                // UI는 폼 쪽에서: alarmOk = !alarm
                StatusUpdated?.Invoke(i, servoOn, !alarm, opStr);
            }

            PositionUpdated?.Invoke(positionsMm);
        }
    }
}
