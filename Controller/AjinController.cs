using PHM_Project_DockPanel.Services;
using System;
using System.Runtime.InteropServices;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// Ajin AMP 모션 제어기 (AXL.dll DllImport 방식).
    /// AXL.dll을 실행 파일과 같은 폴더(bin\x64\Debug 등)에 배치하세요.
    /// </summary>
    public class AjinController : IMotionController, IDisposable
    {
        // ────────────────────────────────────────────────────────────
        // AXL.dll P/Invoke 선언
        // ────────────────────────────────────────────────────────────
        private const string DLL = "AXL.dll";
        private const uint AXT_RT_SUCCESS = 0;
        private const uint SERVO_ON = 1;
        private const uint SERVO_OFF = 0;
        private const uint ABS_MODE = 0;   // 절대 좌표
        private const uint REL_MODE = 1;   // 상대 좌표

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxlOpen(int nIRQ);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxlClose();
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmInfoGetAxisCount(ref int lpAxisCount);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmSignalServoOn(int lAxisNo, uint dwOnOff);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmSignalIsServoOn(int lAxisNo, ref uint dwpOnOff);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMotSetAbsRelMode(int lAxisNo, uint dwAbsRelMode);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMotSetProfileMode(int lAxisNo, uint dwProfileMode);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMotSetMaxVel(int lAxisNo, double dMaxVel);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMotSetMaxAccel(int lAxisNo, double dMaxAccel);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMotSetMaxDecel(int lAxisNo, double dMaxDecel);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMotSetMoveUnitPerPulse(int lAxisNo, double dUnit, int lPulse);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMoveStartPos(int lAxisNo, double dPos, double dVel, double dAccel, double dDecel);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMoveMultiPos(int lArraySize, int[] lpAxisNo, double[] dpPos, double dVel, double dAccel, double dDecel);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMoveSStop(int lAxisNo);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmMoveEStop(int lAxisNo);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusReadInMotion(int lAxisNo, ref uint dwpInMotion);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusGetActPos(int lAxisNo, ref double dpActPos);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusSetActPos(int lAxisNo, double dActPos);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusSetCmdPos(int lAxisNo, double dCmdPos);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusReadTorque(int lAxisNo, ref double dpTorque);

        // ────────────────────────────────────────────────────────────
        // 필드
        // ────────────────────────────────────────────────────────────
        private AxisConfig[] _axisConfigs;
        private bool _connected = false;
        private bool _disposed = false;

        // WMX3와 동일 기준: pitch=20mm, encoder=8388608 pulse/rev
        private const double DefaultUnit = 20.0;
        private const int DefaultPulse = 8388608;

        // ────────────────────────────────────────────────────────────
        // IMotionController 구현
        // ────────────────────────────────────────────────────────────
        public bool IsConnected => _connected;
        public bool IsSimulationMode => false;

        public void SetAxisConfigs(AxisConfig[] configs)
        {
            _axisConfigs = configs;
            if (_connected && _axisConfigs != null)
                ApplyAxisParams();
        }

        /// <summary>
        /// AXL 라이브러리를 초기화하고 모든 축의 기본 파라미터를 설정합니다.
        /// IRQ: -1(자동) 또는 7(권장)
        /// </summary>
        public void Connect(int irq = -1)
        {
            try
            {
                uint ret = AxlOpen(irq);
                if (ret != AXT_RT_SUCCESS)
                    throw new InvalidOperationException($"[Ajin] AxlOpen 실패 (0x{ret:X})");

                _connected = true;
                AppEvents.RaiseLog("[Ajin] 연결 완료");

                if (_axisConfigs != null)
                    ApplyAxisParams();
            }
            catch (DllNotFoundException)
            {
                AppEvents.RaiseLog("[Ajin] AXL.dll을 찾을 수 없습니다. 실행 파일 폴더에 배치하세요.");
                throw;
            }
            catch (SEHException ex)
            {
                AppEvents.RaiseLog($"[Ajin] 하드웨어 초기화 중 네이티브 예외: {ex.Message}");
                throw;
            }
        }

        // IMotionController.Connect() — 파라미터 없는 버전
        void IMotionController.Connect() => Connect();

        public void Disconnect()
        {
            if (!_connected) return;
            try
            {
                int count = GetAxisCount();
                for (int i = 0; i < count; i++)
                {
                    try { AxmSignalServoOn(i, SERVO_OFF); } catch { }
                }
                AxlClose();
                AppEvents.RaiseLog("[Ajin] 연결 해제 완료");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[Ajin] Disconnect 오류: {ex.Message}");
            }
            finally
            {
                _connected = false;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Disconnect();
            _disposed = true;
        }

        /// <summary>
        /// Ajin 상태를 WMX3 호환 CoreMotionStatus로 변환합니다.
        /// </summary>
        public CoreMotionStatus GetStatus()
        {
            var status = new CoreMotionStatus();
            if (!_connected || _axisConfigs == null) return status;

            try
            {
                int count = Math.Min(_axisConfigs.Length, status.AxesStatus?.Length ?? 0);
                for (int i = 0; i < count; i++)
                {
                    uint inMotion = 0; AxmStatusReadInMotion(i, ref inMotion);
                    double actPos = 0; AxmStatusGetActPos(i, ref actPos);
                    uint servoOn = 0; AxmSignalIsServoOn(i, ref servoOn);

                    status.AxesStatus[i].OpState = inMotion != 0 ? OperationState.Moving : OperationState.Idle;
                    status.AxesStatus[i].ActualPos = actPos;
                    status.AxesStatus[i].ServoOn = servoOn != 0;
                }
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[Ajin] GetStatus 오류: {ex.Message}");
            }

            return status;
        }

        public void SetServo(int axis, bool state)
        {
            CheckConnected();
            uint ret = AxmSignalServoOn(axis, state ? SERVO_ON : SERVO_OFF);
            if (ret == AXT_RT_SUCCESS)
                AppEvents.RaiseLog($"[Ajin] Axis {axis} Servo {(state ? "ON" : "OFF")}");
            else
                AppEvents.RaiseLog($"[Ajin] Axis {axis} ServoOn 실패 (0x{ret:X})");
        }

        /// <summary>절대 좌표 다축 동시 이동.</summary>
        public void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec)
        {
            ValidateArgs(axes, target, vel, acc, dec);
            CheckConnected();

            foreach (int ax in axes)
                AxmMotSetAbsRelMode(ax, ABS_MODE);

            if (axes.Length == 1)
            {
                // 단축 이동
                uint ret = AxmMoveStartPos(axes[0], target[0], vel[0], acc[0], dec[0]);
                if (ret != AXT_RT_SUCCESS)
                    throw new InvalidOperationException($"[Ajin] MoveAbs axis={axes[0]} 실패 (0x{ret:X})");
            }
            else
            {
                // 다축 동시 이동 — AxmMoveMultiPos는 vel/acc/dec를 단일값으로 받음
                uint ret = AxmMoveMultiPos(axes.Length, axes, target, vel[0], acc[0], dec[0]);
                if (ret != AXT_RT_SUCCESS)
                    throw new InvalidOperationException($"[Ajin] MoveMultiPos 실패 (0x{ret:X})");
            }
        }

        /// <summary>
        /// 상대 좌표 이동 — 현재 실제 위치(ActPos)에 delta를 더해 절대 이동으로 변환합니다.
        /// </summary>
        public void MoveRel(int[] axes, double[] delta, double[] vel, double[] acc, double[] dec)
        {
            ValidateArgs(axes, delta, vel, acc, dec);
            CheckConnected();

            double[] absTarget = new double[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                double cur = 0;
                AxmStatusGetActPos(axes[i], ref cur);
                absTarget[i] = cur + delta[i];
            }

            MoveAbs(axes, absTarget, vel, acc, dec);
        }

        // ────────────────────────────────────────────────────────────
        // 추가 공개 유틸
        // ────────────────────────────────────────────────────────────
        public void Stop(int axisNo) { if (_connected) AxmMoveSStop(axisNo); }
        public void EStop(int axisNo) { if (_connected) AxmMoveEStop(axisNo); }

        public double GetActPos(int axisNo)
        {
            CheckConnected();
            double pos = 0;
            AxmStatusGetActPos(axisNo, ref pos);
            return pos;
        }

        public double GetTorque(int axisNo)
        {
            CheckConnected();
            double tor = 0;
            AxmStatusReadTorque(axisNo, ref tor);
            return tor;
        }

        public void SetZeroPos(int axisNo)
        {
            CheckConnected();
            AxmStatusSetActPos(axisNo, 0);
            AxmStatusSetCmdPos(axisNo, 0);
        }

        public int GetAxisCount()
        {
            int count = 0;
            if (_connected) AxmInfoGetAxisCount(ref count);
            return count;
        }

        // ────────────────────────────────────────────────────────────
        // 내부 헬퍼
        // ────────────────────────────────────────────────────────────

        /// <summary>연결 후 _axisConfigs 기준으로 각 축 파라미터를 일괄 적용합니다.</summary>
        private void ApplyAxisParams()
        {
            for (int i = 0; i < _axisConfigs.Length; i++)
            {
                var cfg = _axisConfigs[i];
                if (cfg == null) continue;

                // UnitPerPulse: WMX3와 동일 기준 사용
                AxmMotSetMoveUnitPerPulse(i, DefaultUnit, DefaultPulse);

                // 절대 모드 + 대칭 사다리꼴 프로파일
                AxmMotSetAbsRelMode(i, ABS_MODE);
                AxmMotSetProfileMode(i, 0);

                if (cfg.MaxVel > 0)
                {
                    AxmMotSetMaxVel(i, cfg.MaxVel);
                    AxmMotSetMaxAccel(i, cfg.Acc);
                    AxmMotSetMaxDecel(i, cfg.Dec);
                }

                AppEvents.RaiseLog($"[Ajin] Axis {i} 파라미터 적용 (vel={cfg.MaxVel}, acc={cfg.Acc})");
            }
        }

        private void CheckConnected()
        {
            if (!_connected)
                throw new InvalidOperationException("[Ajin] 연결되지 않았습니다. Connect()를 먼저 호출하세요.");
        }

        private static void ValidateArgs(int[] axes, double[] vals, double[] vel, double[] acc, double[] dec)
        {
            if (axes == null || axes.Length == 0) throw new ArgumentException("axes empty");
            if (vals?.Length != axes.Length || vel?.Length != axes.Length ||
                acc?.Length != axes.Length || dec?.Length != axes.Length)
                throw new ArgumentException("배열 길이 불일치");
        }
    }
}