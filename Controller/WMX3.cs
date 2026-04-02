using PHM_Project_DockPanel.DebugTools;
using PHM_Project_DockPanel.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms.DataVisualization.Charting;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    public class WMX3
    {
        private WMX3Api _wmxlib;
        private CoreMotion _wmxlibCm;
        private const double EncoderResolution = 8388608; // 엔코더 펄스 수

        private AxisConfig[] _axisConfigs;

        // ✅ 하드웨어 없이 실행 중인지 여부 (true = 시뮬레이션 모드)
        public bool IsSimulationMode { get; private set; } = false;

        public void SetAxisConfigs(AxisConfig[] configs)
        {
            _axisConfigs = configs;
        }

        public void Initialize()
        {
            try
            {
                _wmxlib = new WMX3Api();
                _wmxlibCm = new CoreMotion(_wmxlib);

                int err = _wmxlib.CreateDevice(@"C:\Program Files\SoftServo\WMX3", DeviceType.DeviceTypeNormal);
                if (err == ErrorCode.None)
                {
                    AppEvents.RaiseLog("[WMX3] CreateDevice 성공");
                    IsSimulationMode = false;
                }
                else
                {
                    AppEvents.RaiseLog($"[WMX3 경고] CreateDevice 실패 (ErrorCode: {err}) → 시뮬레이션 모드로 전환");
                    IsSimulationMode = true;
                    _wmxlib = null;
                    _wmxlibCm = null;
                }
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[WMX3 경고] 초기화 중 예외 발생: {ex.Message} → 시뮬레이션 모드로 전환");
                IsSimulationMode = true;
                _wmxlib = null;
                _wmxlibCm = null;
            }
        }

        public void StartCommunication()
        {
            if (IsSimulationMode)
            {
                AppEvents.RaiseLog("[WMX3] 시뮬레이션 모드: StartCommunication 건너뜀");
                return;
            }

            try
            {
                int err = _wmxlib.StartCommunication(5000);
                if (err == ErrorCode.None)
                {
                    AppEvents.RaiseLog("[WMX3] StartCommunication 성공");
                }
                else
                {
                    AppEvents.RaiseLog($"[WMX3 경고] StartCommunication 실패 (ErrorCode: {err}) → 시뮬레이션 모드로 전환");
                    IsSimulationMode = true;
                }
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[WMX3 경고] StartCommunication 중 예외: {ex.Message} → 시뮬레이션 모드로 전환");
                IsSimulationMode = true;
            }
        }

        public void StopCommunication()
        {
            if (IsSimulationMode || _wmxlib == null) return;

            try
            {
                int err = _wmxlib.StopCommunication();
                if (err == ErrorCode.None)
                    AppEvents.RaiseLog("[WMX3] StopCommunication 성공");
                else
                    AppEvents.RaiseLog($"[WMX3 에러] StopCommunication 실패 (ErrorCode: {err})");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[WMX3] StopCommunication 예외 무시: {ex.Message}");
            }
        }

        public void Dispose()
        {
            StopCommunication();
            CloseDevice();
            try { _wmxlibCm?.Dispose(); } catch { }
            try { _wmxlib?.Dispose(); } catch { }
        }

        public void CloseDevice()
        {
            if (IsSimulationMode || _wmxlib == null) return;

            try { _wmxlib.CloseDevice(); }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[WMX3] CloseDevice 예외 무시: {ex.Message}");
            }
        }

        public bool IsConnected
        {
            get
            {
                if (IsSimulationMode || _wmxlibCm == null)
                    return false;

                try
                {
                    var status = GetStatus();
                    return status.EngineState == EngineState.Communicating;
                }
                catch
                {
                    return false;
                }
            }
        }

        public CoreMotionStatus GetStatus()
        {
            if (IsSimulationMode || _wmxlibCm == null)
                throw new InvalidOperationException("WMX3가 초기화되지 않았습니다. (시뮬레이션 모드)");

            CoreMotionStatus status = new CoreMotionStatus();
            _wmxlibCm.GetStatus(ref status);
            return status;
        }

        public void SetServoState(int axis, bool on)
        {
            if (IsSimulationMode || _wmxlibCm == null)
            {
                AppEvents.RaiseLog($"[서보] 시뮬레이션 모드: Axis {axis} Servo {(on ? "ON" : "OFF")} (실제 동작 없음)");
                return;
            }

            int err = _wmxlibCm.AxisControl.SetServoOn(axis, on ? 1 : 0);
            if (err == ErrorCode.None)
                AppEvents.RaiseLog($"[서보] Axis {axis} Servo {(on ? "ON" : "OFF")}");
            else
                AppEvents.RaiseLog($"[에러] Axis {axis} {(on ? "Servo ON" : "Servo OFF")} 실패 (ErrorCode: {err})");
        }

        public void MoveAbsoluteMm(int[] axes, double[] targetMm, double[] maxVelMm, double[] accMm, double[] decMm)
        {
            if (axes == null || axes.Length == 0) throw new ArgumentException("axes empty");
            if (targetMm?.Length != axes.Length || maxVelMm?.Length != axes.Length
                || accMm?.Length != axes.Length || decMm?.Length != axes.Length)
                throw new ArgumentException("array length mismatch");

            if (IsSimulationMode || _wmxlibCm == null)
            {
                AppEvents.RaiseLog($"[모션] 시뮬레이션 모드: MoveAbsoluteMm 호출됨 (실제 동작 없음)");
                return;
            }

            var cmds = new Motion.PosCommand[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                var pos = new Motion.PosCommand();
                pos.Axis = ax;
                pos.Profile.Type = ProfileType.Trapezoidal;
                pos.Target = ConvertMmToEncoder(ax, targetMm[i]);
                pos.Profile.Velocity = ConvertMmToEncoder(ax, maxVelMm[i]);
                pos.Profile.Acc = ConvertMmToEncoder(ax, accMm[i]);
                pos.Profile.Dec = ConvertMmToEncoder(ax, decMm[i]);
                cmds[i] = pos;
            }

            int err = _wmxlibCm.Motion.StartPos((uint)cmds.Length, cmds);
            if (err != 0) throw new InvalidOperationException($"WMX StartPos(multi) failed: {err}");
        }

        public void MoveRelativeMm(int[] axes, double[] deltaMm, double[] maxVelMm, double[] accMm, double[] decMm)
        {
            if (axes == null || axes.Length == 0) throw new ArgumentException("axes empty");
            if (deltaMm?.Length != axes.Length || maxVelMm?.Length != axes.Length
                || accMm?.Length != axes.Length || decMm?.Length != axes.Length)
                throw new ArgumentException("array length mismatch");

            if (IsSimulationMode || _wmxlibCm == null)
            {
                AppEvents.RaiseLog($"[모션] 시뮬레이션 모드: MoveRelativeMm 호출됨 (실제 동작 없음)");
                return;
            }

            var cmds = new Motion.PosCommand[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                var pos = new Motion.PosCommand();
                pos.Axis = ax;
                pos.Profile.Type = ProfileType.Trapezoidal;
                pos.Target = ConvertMmToEncoder(ax, deltaMm[i]);
                pos.Profile.Velocity = ConvertMmToEncoder(ax, maxVelMm[i]);
                pos.Profile.Acc = ConvertMmToEncoder(ax, accMm[i]);
                pos.Profile.Dec = ConvertMmToEncoder(ax, decMm[i]);
                cmds[i] = pos;
            }

            int err = _wmxlibCm.Motion.StartMov((uint)cmds.Length, cmds);
            if (err != 0) throw new InvalidOperationException($"WMX StartMov(multi) failed: {err}");
        }

        public double ConvertEncoderToMm(int axis, double encoderPos)
        {
            double pitch = GetPitch(axis);
            return encoderPos / EncoderResolution * pitch;
        }

        private double ConvertMmToEncoder(int axis, double mmPos)
        {
            double pitch = GetPitch(axis);
            return mmPos / pitch * EncoderResolution;
        }

        private double GetPitch(int axis)
        {
            if (_axisConfigs != null &&
                axis >= 0 &&
                axis < _axisConfigs.Length &&
                _axisConfigs[axis] != null)
            {
                return _axisConfigs[axis].PitchMmPerRev;
            }

            return 30.0; // 기본값 (30mm/rev)
        }
    }
}