using PHM_Project_DockPanel.Services;
using System;
using System.Threading;
using System.Threading.Tasks;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Controller
{
    /// <summary>
    /// 하드웨어 없이 실행할 때 사용하는 시뮬레이션 제어기.
    /// 트라페조이드 속도 프로파일로 실제 모션을 모사하며,
    /// 서보 상태·위치·OpState를 내부에서 추적합니다.
    /// </summary>
    public class SimulationController : IMotionController
    {
        private int _axisCount;
        private double[] _positions;
        private double[] _velocities;   // mm/s, 토크 계산용
        private double[] _torques;      // %, 폴링 로거용
        private bool[] _servoOn;
        private OperationState[] _opStates;
        private CancellationTokenSource[] _motionCts;
        private readonly object _stateLock = new object();
        private readonly Random _rng = new Random();

        public bool IsConnected { get; private set; }
        public bool IsSimulationMode => true;
        public bool PosIsAlreadyMm => true;

        /// <param name="axisCount">시뮬레이션할 축 수. SetAxisConfigs 호출 시 자동으로 재설정됩니다.</param>
        public SimulationController(int axisCount = 5)
        {
            InitArrays(Math.Max(1, axisCount));
        }

        // ── IMotionController 구현 ─────────────────────────────────────

        /// <summary>사용자가 선택한 축 수는 변경하지 않음. 설정값만 저장합니다.</summary>
        public void SetAxisConfigs(AxisConfig[] configs) { }

        public int GetAxisCount() => _axisCount;

        public void Connect()
        {
            IsConnected = true;
            AppEvents.RaiseLog($"[Sim] 시뮬레이션 모드로 연결됨 (축 수: {_axisCount})");
        }

        public void Disconnect()
        {
            CancelAllMotions();
            IsConnected = false;
            AppEvents.RaiseLog("[Sim] 연결 해제");
        }

        public void Dispose() => CancelAllMotions();

        public CoreMotionStatus GetStatus()
        {
            var status = new CoreMotionStatus();
            lock (_stateLock)
            {
                for (int i = 0; i < _axisCount; i++)
                {
                    status.AxesStatus[i].ActualPos   = _positions[i];
                    status.AxesStatus[i].ServoOn      = _servoOn[i];
                    status.AxesStatus[i].OpState      = _opStates[i];
                    status.AxesStatus[i].AmpAlarm     = false;
                    status.AxesStatus[i].ServoOffline = false;
                }

                // CoreMotionStatus는 N개 이상의 AxisStatus를 기본 할당함.
                // 우리 축 수를 넘는 항목은 ServoOffline=true로 마킹해야
                // AxisInfoForm의 Count(a => !a.ServoOffline) 로직이 정확한 축 수를 반환함.
                int totalSlots = 0;
                foreach (var _ in status.AxesStatus) totalSlots++;
                for (int i = _axisCount; i < totalSlots; i++)
                    status.AxesStatus[i].ServoOffline = true;
            }
            return status;
        }

        public void SetServo(int axis, bool state)
        {
            if (axis < 0 || axis >= _axisCount) return;
            lock (_stateLock) { _servoOn[axis] = state; }
            AppEvents.RaiseLog($"[Sim] Servo axis={axis} {(state ? "ON" : "OFF")}");
        }

        public void MoveAbs(int[] axes, double[] target, double[] vel, double[] acc, double[] dec)
        {
            if (axes == null) return;
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                if (ax < 0 || ax >= _axisCount) continue;
                double start;
                lock (_stateLock) { start = _positions[ax]; }
                StartMotion(ax, start, target[i], vel[i], acc[i], dec[i]);
            }
            AppEvents.RaiseLog($"[Sim] MoveAbs axes=[{string.Join(",", axes)}] targets=[{string.Join(",", target)}]");
        }

        public void MoveRel(int[] axes, double[] delta, double[] vel, double[] acc, double[] dec)
        {
            if (axes == null) return;
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                if (ax < 0 || ax >= _axisCount) continue;
                double start;
                lock (_stateLock) { start = _positions[ax]; }
                StartMotion(ax, start, start + delta[i], vel[i], acc[i], dec[i]);
            }
            AppEvents.RaiseLog($"[Sim] MoveRel axes=[{string.Join(",", axes)}] deltas=[{string.Join(",", delta)}]");
        }

        // ── 내부 구현 ──────────────────────────────────────────────────

        /// <summary>현재 축의 시뮬레이션 토크(%)를 반환합니다. AjinCsvLogger 주입에 사용.</summary>
        public double GetTorque(int axis)
        {
            if (axis < 0 || axis >= _axisCount) return 0.0;
            lock (_stateLock) { return _torques[axis]; }
        }

        private void InitArrays(int count)
        {
            _axisCount  = count;
            _positions  = new double[count];
            _velocities = new double[count];
            _torques    = new double[count];
            _servoOn    = new bool[count];
            _opStates   = new OperationState[count]; // 기본값: Idle (= 0)
            _motionCts  = new CancellationTokenSource[count];
        }

        private void StartMotion(int axis, double start, double target,
                                  double vel, double acc, double dec)
        {
            _motionCts[axis]?.Cancel();
            _motionCts[axis] = new CancellationTokenSource();
            var cts = _motionCts[axis];

            lock (_stateLock) { _opStates[axis] = OperationState.Pos; }

            _ = Task.Run(async () =>
            {
                try
                {
                    await SimulateMotion(axis, start, target, vel, acc, dec, cts.Token);
                }
                catch (OperationCanceledException) { }
                finally
                {
                    lock (_stateLock) { _opStates[axis] = OperationState.Idle; }
                }
            }, cts.Token);
        }

        private async Task SimulateMotion(int axis, double start, double target,
            double vel, double acc, double dec, CancellationToken ct)
        {
            double distance = Math.Abs(target - start);
            if (distance < 0.001)
            {
                lock (_stateLock) { _positions[axis] = target; }
                return;
            }

            double sign      = target > start ? 1.0 : -1.0;
            double totalTime = CalcTotalTime(distance, vel, acc, dec);
            if (totalTime <= 0)
            {
                lock (_stateLock) { _positions[axis] = target; }
                return;
            }

            const int IntervalMs = 20;
            const double IntervalSec = IntervalMs * 0.001;
            double elapsed = 0;
            double prevTraveled = 0;

            while (elapsed < totalTime)
            {
                ct.ThrowIfCancellationRequested();
                await Task.Delay(IntervalMs, ct);
                elapsed += IntervalSec;

                double t = Math.Min(elapsed, totalTime);
                double traveled = CalcPosition(t, distance, vel, acc, dec);
                double curVel = sign * (traveled - prevTraveled) / IntervalSec; // mm/s
                double torque = CalcTorque(t, distance, vel, acc, dec, curVel);
                prevTraveled = traveled;

                lock (_stateLock)
                {
                    _positions[axis]  = start + sign * traveled;
                    _velocities[axis] = curVel;
                    _torques[axis]    = torque;
                }
            }

            lock (_stateLock)
            {
                _positions[axis]  = target;
                _velocities[axis] = 0;
                _torques[axis]    = 0;
            }
        }

        /// <summary>
        /// 트라페조이드/삼각 속도 프로파일에서 시간 t 에서의 이동 거리(0~distance)를 반환합니다.
        /// </summary>
        private static double CalcPosition(double t, double distance,
                                            double vel, double acc, double dec)
        {
            double da = 0.5 * vel * vel / acc;   // 가속 구간 거리
            double dd = 0.5 * vel * vel / dec;   // 감속 구간 거리

            if (da + dd >= distance)
            {
                // 삼각형 프로파일 (크루즈 없음)
                double vpeak = Math.Sqrt(2.0 * distance / (1.0 / acc + 1.0 / dec));
                double ta    = vpeak / acc;
                double dapeak = 0.5 * vpeak * vpeak / acc;

                if (t <= ta)
                    return 0.5 * acc * t * t;

                double t2  = t - ta;
                double pos = dapeak + vpeak * t2 - 0.5 * dec * t2 * t2;
                return Math.Max(0, Math.Min(distance, pos));
            }
            else
            {
                // 트라페조이드 프로파일
                double ta = vel / acc;
                double dc = distance - da - dd;
                double tc = dc / vel;

                if (t <= ta)
                    return 0.5 * acc * t * t;

                if (t <= ta + tc)
                    return da + vel * (t - ta);

                double t3  = t - ta - tc;
                double pos = da + dc + vel * t3 - 0.5 * dec * t3 * t3;
                return Math.Max(0, Math.Min(distance, pos));
            }
        }

        private static double CalcTotalTime(double distance, double vel, double acc, double dec)
        {
            if (distance <= 0 || vel <= 0 || acc <= 0 || dec <= 0) return 0;
            double da = 0.5 * vel * vel / acc;
            double dd = 0.5 * vel * vel / dec;
            if (distance >= da + dd)
                return vel / acc + (distance - da - dd) / vel + vel / dec;
            double vpeak = Math.Sqrt(2.0 * distance / (1.0 / acc + 1.0 / dec));
            return vpeak / acc + vpeak / dec;
        }

        /// <summary>
        /// 트라페조이드 프로파일 기반 가상 토크(%) 계산.
        /// 가속 구간: 관성+마찰, 크루즈: 마찰만, 감속: 제동+마찰
        /// </summary>
        private double CalcTorque(double t, double distance,
                                   double vel, double acc, double dec, double curVelMmS)
        {
            double da = 0.5 * vel * vel / acc;
            double dd = 0.5 * vel * vel / dec;
            double ta, tc;

            if (da + dd >= distance)
            {
                double vpeak = Math.Sqrt(2.0 * distance / (1.0 / acc + 1.0 / dec));
                ta = vpeak / acc;
                tc = 0;
            }
            else
            {
                ta = vel / acc;
                tc = (distance - da - dd) / vel;
            }

            const double InertiaFactor  = 0.40;   // 관성 성분 최대 40%
            const double FrictionFactor = 0.08;   // 마찰 성분 8%
            const double NoiseFactor    = 0.03;   // 노이즈 ±3%
            double noise = (_rng.NextDouble() * 2 - 1) * NoiseFactor;

            double torque;
            if (t <= ta)
                torque = InertiaFactor * (t / ta) + FrictionFactor + noise;       // 가속
            else if (t <= ta + tc)
                torque = FrictionFactor + noise;                                    // 크루즈
            else
                torque = -InertiaFactor * ((t - ta - tc) / (vel / dec)) + FrictionFactor + noise; // 감속

            return Math.Max(-1.0, Math.Min(1.0, torque)) * 100.0;  // −100% ~ +100%
        }

        private void CancelAllMotions()
        {
            if (_motionCts == null) return;
            foreach (var cts in _motionCts)
            {
                try { cts?.Cancel(); } catch { }
            }
        }
    }
}
