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
        private bool[] _servoOn;
        private OperationState[] _opStates;
        private CancellationTokenSource[] _motionCts;
        private readonly object _stateLock = new object();

        public bool IsConnected { get; private set; }
        public bool IsSimulationMode => true;
        public bool PosIsAlreadyMm => true;

        /// <param name="axisCount">시뮬레이션할 축 수. SetAxisConfigs 호출 시 자동으로 재설정됩니다.</param>
        public SimulationController(int axisCount = 5)
        {
            InitArrays(Math.Max(1, axisCount));
        }

        // ── IMotionController 구현 ─────────────────────────────────────

        public void SetAxisConfigs(AxisConfig[] configs)
        {
            if (configs == null || configs.Length == 0) return;
            if (configs.Length == _axisCount) return;
            CancelAllMotions();
            InitArrays(configs.Length);
        }

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

        private void InitArrays(int count)
        {
            _axisCount = count;
            _positions = new double[count];
            _servoOn   = new bool[count];
            _opStates  = new OperationState[count]; // 기본값: Idle (= 0)
            _motionCts = new CancellationTokenSource[count];
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
            double elapsed = 0;

            while (elapsed < totalTime)
            {
                ct.ThrowIfCancellationRequested();
                await Task.Delay(IntervalMs, ct);
                elapsed += IntervalMs * 0.001;

                double traveled = CalcPosition(Math.Min(elapsed, totalTime), distance, vel, acc, dec);
                lock (_stateLock)
                {
                    _positions[axis] = start + sign * traveled;
                }
            }

            lock (_stateLock) { _positions[axis] = target; }
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
