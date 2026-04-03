using PHM_Project_DockPanel.Services;
using System;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// IMotionController 를 주입받아 동작하는 제어기 관리자.
    /// 어떤 제어기를 쓸지는 외부(ControllerSelectDialog)에서 결정합니다.
    /// </summary>
    public class ControllerManager
    {
        private IMotionController _controller;
        private AxisConfig[] _axisConfigs;

        public bool IsConnected => _controller?.IsConnected ?? false;
        public bool IsSimulationMode => _controller?.IsSimulationMode ?? true;

        /// <summary>WMX3 제어기일 때만 true — Log(WMX3 전용 클래스) 생성 여부 판단에 사용</summary>
        public bool IsWmx3 => _controller is WMX3Controller;

        /// <summary>Ajin 제어기일 때만 true — 축 수 감지 등 Ajin 전용 분기에 사용</summary>
        public bool IsAjin => _controller is AjinController;

        /// <summary>Ajin 제어기로 캐스팅하여 반환. Ajin이 아니면 null.</summary>
        public AjinController AsAjin => _controller as AjinController;

        /// <summary>시뮬레이션 제어기로 캐스팅하여 반환. 시뮬레이션이 아니면 null.</summary>
        public SimulationController AsSimulation => _controller as SimulationController;

        public bool PosIsAlreadyMm
        {
            get
            {
                // Ajin/Simulation: 이미 mm
                if (IsAjin) return true;
                if (_controller is SimulationController) return true;

                // WMX3: pulse → 변환 필요
                if (IsWmx3) return false;

                // 기본값
                return false;
            }
        }
        /// <summary>
        /// 선택된 제어기를 주입합니다. Connect() 호출 전에 반드시 설정하세요.
        /// </summary>
        public void SetController(IMotionController controller)
        {
            _controller = controller ?? throw new ArgumentNullException(nameof(controller));
        }

        public void SetAxisConfigs(AxisConfig[] configs)
        {
            _axisConfigs = configs;
            _controller?.SetAxisConfigs(configs);
        }

        public void Connect()
        {
            if (_controller == null)
                throw new InvalidOperationException("제어기가 선택되지 않았습니다. SetController()를 먼저 호출하세요.");

            if (_axisConfigs != null)
                _controller.SetAxisConfigs(_axisConfigs);

            _controller.Connect();
        }

        public void Disconnect()
        {
            try { _controller?.Disconnect(); } catch { }
        }

        public CoreMotionStatus GetStatus()
        {
            if (_controller == null)
                return new CoreMotionStatus();
            return _controller.GetStatus();
        }

        public void SetServo(int axis, bool state) =>
            _controller?.SetServo(axis, state);

        public void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec)
        {
            if (axes == null || axes.Length == 0) throw new ArgumentException("axes empty");
            if (target?.Length != axes.Length || vel?.Length != axes.Length ||
                acc?.Length != axes.Length || dec?.Length != axes.Length)
                throw new ArgumentException("array length mismatch");

            _controller?.MoveAbs(axes, target, vel, acc, dec);
        }

        public void MoveRel(int[] axes, double[] delta, double[] vel, double[] acc, double[] dec)
        {
            if (axes == null || axes.Length == 0) throw new ArgumentException("axes empty");
            if (delta?.Length != axes.Length || vel?.Length != axes.Length ||
                acc?.Length != axes.Length || dec?.Length != axes.Length)
                throw new ArgumentException("array length mismatch");

            _controller?.MoveRel(axes, delta, vel, acc, dec);
        }

        public void Dispose()
        {
            try { _controller?.Dispose(); } catch { }
        }
    }
}