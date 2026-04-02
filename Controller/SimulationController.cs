using PHM_Project_DockPanel.Services;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// 하드웨어 없이 실행할 때 사용하는 시뮬레이션 제어기.
    /// 모든 동작은 로그만 남기고 실제 하드웨어 호출은 없습니다.
    /// </summary>
    public class SimulationController : IMotionController
    {
        public bool IsConnected => true;   // UI상 "연결됨"으로 표시
        public bool IsSimulationMode => true;

        public void SetAxisConfigs(AxisConfig[] configs) { }

        public void Connect() => AppEvents.RaiseLog("[Sim] 시뮬레이션 모드로 연결됨");
        public void Disconnect() => AppEvents.RaiseLog("[Sim] 연결 해제");
        public void Dispose() { }

        public CoreMotionStatus GetStatus() => new CoreMotionStatus();

        public void SetServo(int axis, bool state) =>
            AppEvents.RaiseLog($"[Sim] SetServo axis={axis} {(state ? "ON" : "OFF")}");

        public void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec) =>
            AppEvents.RaiseLog($"[Sim] MoveAbs axes=[{string.Join(",", axes)}]");

        public void MoveRel(int[] axes, double[] delta, double[] vel, double[] acc, double[] dec) =>
            AppEvents.RaiseLog($"[Sim] MoveRel axes=[{string.Join(",", axes)}]");
    }
}