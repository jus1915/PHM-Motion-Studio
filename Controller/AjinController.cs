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
        private const uint ABS_MODE = 0;
        private const uint REL_MODE = 1;

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxlOpen(int nIRQ);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxlClose();
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmInfoGetAxisCount(ref int lpAxisCount);

        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmSignalServoOn(int lAxisNo, uint dwOnOff);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmSignalIsServoOn(int lAxisNo, ref uint dwpOnOff);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmSignalServoAlarmReset(int lAxisNo, uint dwOnOff);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmSignalIsAlarm(int lAxisNo, ref uint dwpStatus);

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
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusGetCmdPos(int lAxisNo, ref double dpCmdPos);
        [DllImport(DLL, CallingConvention = CallingConvention.Cdecl)] private static extern uint AxmStatusReadActVel(int lAxisNo, ref double dpActVel);
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

        // OperationState.Pos = 절대 이동 중 상태 (WMX3 enum 기준)
        // PHM_Motion.WaitForMotionsEnd()가 OpState != Idle 로 모션 중 판단
        private const OperationState StateMoving = OperationState.Pos;

        // ────────────────────────────────────────────────────────────
        // IMotionController 구현
        // ────────────────────────────────────────────────────────────
        public bool IsConnected => _connected;
        public bool IsSimulationMode => false;
        // Ajin: AxmMotSetMoveUnitPerPulse(unit=20mm, pulse=8388608) 설정으로
        // AxmStatusGetActPos가 이미 mm 단위 반환
        public bool PosIsAlreadyMm => true;

        public void SetAxisConfigs(AxisConfig[] configs)
        {
            _axisConfigs = configs;
            if (_connected && _axisConfigs != null)
                ApplyAxisParams();
        }

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

        void IMotionController.Connect() => Connect();

        public void Disconnect()
        {
            if (!_connected) return;
            try
            {
                int count = GetAxisCount();
                for (int i = 0; i < count; i++)
                    try { AxmSignalServoOn(i, SERVO_OFF); } catch { }

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

                    // OperationState.Moving이 WMX3 enum에 없을 수 있으므로
                    // inMotion 플래그로 Idle/Moving 구분
                    status.AxesStatus[i].OpState = inMotion != 0 ? StateMoving : OperationState.Idle;
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

        public void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec)
        {
            ValidateArgs(axes, target, vel, acc, dec);
            CheckConnected();

            foreach (int ax in axes)
                AxmMotSetAbsRelMode(ax, ABS_MODE);

            if (axes.Length == 1)
            {
                uint ret = AxmMoveStartPos(axes[0], target[0], vel[0], acc[0], dec[0]);
                if (ret != AXT_RT_SUCCESS)
                    throw new InvalidOperationException($"[Ajin] MoveAbs axis={axes[0]} 실패 (0x{ret:X})");
            }
            else
            {
                uint ret = AxmMoveMultiPos(axes.Length, axes, target, vel[0], acc[0], dec[0]);
                if (ret != AXT_RT_SUCCESS)
                    throw new InvalidOperationException($"[Ajin] MoveMultiPos 실패 (0x{ret:X})");
            }
        }

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

        public double GetCmdPos(int axisNo)
        {
            CheckConnected();
            double pos = 0;
            AxmStatusGetCmdPos(axisNo, ref pos);
            return pos;
        }

        public double GetActVel(int axisNo)
        {
            CheckConnected();
            double vel = 0;
            AxmStatusReadActVel(axisNo, ref vel);
            return vel;
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

        /// <summary>
        /// 서보 드라이브 알람을 클리어합니다.
        /// AlarmReset 신호를 100ms ON → OFF 하고, 서보를 재활성화합니다.
        /// </summary>
        public void ClearAlarm(int axisNo)
        {
            CheckConnected();
            try
            {
                // 1) AlarmReset 신호 ON
                AxmSignalServoAlarmReset(axisNo, 1);
                System.Threading.Thread.Sleep(100);
                // 2) AlarmReset 신호 OFF
                AxmSignalServoAlarmReset(axisNo, 0);
                System.Threading.Thread.Sleep(50);

                AppEvents.RaiseLog($"[Ajin] Axis {axisNo} 알람 클리어 완료");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[Ajin] Axis {axisNo} 알람 클리어 실패: {ex.Message}");
                throw;
            }
        }

        /// <summary>알람 상태 조회. true = 알람 발생 중.</summary>
        public bool IsAlarm(int axisNo)
        {
            if (!_connected) return false;
            uint status = 0;
            AxmSignalIsAlarm(axisNo, ref status);
            return status != 0;
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
        private void ApplyAxisParams()
        {
            int actualCount = GetAxisCount();
            int applyCount = actualCount > 0
                ? Math.Min(actualCount, _axisConfigs.Length)
                : _axisConfigs.Length;

            for (int i = 0; i < applyCount; i++)
            {
                var cfg = _axisConfigs[i];
                if (cfg == null) continue;

                // ✅ 핵심: Pitch 사용
                double unit = cfg.PitchMmPerRev > 0 ? cfg.PitchMmPerRev : 10.0;

                AxmMotSetMoveUnitPerPulse(i, unit, DefaultPulse);

                AxmMotSetAbsRelMode(i, ABS_MODE);
                AxmMotSetProfileMode(i, 0);

                if (cfg.MaxVel > 0)
                {
                    AxmMotSetMaxVel(i, cfg.MaxVel);
                    AxmMotSetMaxAccel(i, cfg.Acc);
                    AxmMotSetMaxDecel(i, cfg.Dec);
                }

                AppEvents.RaiseLog($"[Ajin] Axis {i} unit={unit} 적용");
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