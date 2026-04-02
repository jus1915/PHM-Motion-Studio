using PHM_Project_DockPanel.Services;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// SoftServo WMX3 하드웨어 제어기.
    /// 하드웨어가 없거나 초기화 실패 시 IsSimulationMode = true 로 전환됩니다.
    /// </summary>
    public class WMX3Controller : IMotionController
    {
        private readonly WMX3 _wmx;

        public WMX3Controller()
        {
            _wmx = new WMX3();
        }

        public bool IsConnected => _wmx.IsConnected;
        public bool IsSimulationMode => _wmx.IsSimulationMode;

        public void SetAxisConfigs(AxisConfig[] configs) => _wmx.SetAxisConfigs(configs);

        public void Connect()
        {
            _wmx.Initialize();
            _wmx.StartCommunication();

            if (_wmx.IsSimulationMode)
                AppEvents.RaiseLog("[WMX3] 하드웨어 없음 → 시뮬레이션 모드");
            else
                AppEvents.RaiseLog("[WMX3] 연결 완료");
        }

        public void Disconnect() => _wmx.Dispose();
        public void Dispose() => _wmx.Dispose();

        public CoreMotionStatus GetStatus() =>
            _wmx.IsSimulationMode ? new CoreMotionStatus() : _wmx.GetStatus();

        public void SetServo(int axis, bool state) =>
            _wmx.SetServoState(axis, state);

        public void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec) =>
            _wmx.MoveAbsoluteMm(axes, target, vel, acc, dec);

        public void MoveRel(int[] axes, double[] delta, double[] vel, double[] acc, double[] dec) =>
            _wmx.MoveRelativeMm(axes, delta, vel, acc, dec);
    }
}