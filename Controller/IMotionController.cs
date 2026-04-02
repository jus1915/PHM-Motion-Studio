using PHM_Project_DockPanel.Services;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// 모든 모션 제어기가 구현해야 하는 공통 인터페이스.
    /// WMX3, Ajin, 시뮬레이션 모두 이 계약을 따릅니다.
    /// </summary>
    public interface IMotionController
    {
        bool IsConnected { get; }
        bool IsSimulationMode { get; }

        void Connect();
        void Disconnect();
        void Dispose();

        CoreMotionStatus GetStatus();
        void SetServo(int axis, bool state);
        void SetAxisConfigs(AxisConfig[] configs);

        void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec);
        void MoveRel(int[] axes, double[] delta, double[] vel, double[] acc, double[] dec);
    }
}