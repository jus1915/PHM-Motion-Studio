using PHM_Project_DockPanel.Services;
using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.WMX
{
    /// <summary>
    /// Ajin AMP 전용 CSV 로거.
    /// SDK 로그 API가 없으므로 폴링 방식으로 기록합니다.
    /// 속도는 AxmStatusReadActVel 대신 위치 차분으로 계산합니다 (단위 보장).
    /// 기록 항목: Timestamp(ms), Pos(mm), Vel(mm/s), Torque(%)
    /// </summary>
    public class AjinCsvLogger : IDisposable
    {
        // ── 폴링 주기 ──────────────────────────────────────────────
        public int IntervalMs { get; set; } = 10;   // 기본 10ms (100 Hz)

        // ── 외부 주입 ──────────────────────────────────────────────
        private readonly Func<int, double> _getPos;     // axis → actual pos(mm)
        private readonly Func<int, double> _getVel;     // axis → actual vel(mm/s), null이면 위치 차분으로 계산
        private readonly Func<int, double> _getTorque;  // axis → torque(%)
        private readonly Func<int, double> _getCmdPos;  // axis → command pos(mm), null이면 기록 안 함
        private readonly Action<string> _log;

        // ── 상태 ──────────────────────────────────────────────────
        private CancellationTokenSource _cts;
        private Task _task;
        private bool _disposed;
        private int[] _axes;
        private string _filePath;

        public bool IsLogging => _task != null && !_task.IsCompleted;
        public string OutputPath => _filePath;

        public AjinCsvLogger(
            Func<int, double> getPos,
            Func<int, double> getTorque,
            Action<string> log = null,
            Func<int, double> getVel = null,
            Func<int, double> getCmdPos = null)
        {
            _getPos    = getPos    ?? throw new ArgumentNullException(nameof(getPos));
            _getTorque = getTorque ?? throw new ArgumentNullException(nameof(getTorque));
            _getVel    = getVel;
            _getCmdPos = getCmdPos; // null이면 CmdPos 컬럼 생략
            _log       = log ?? (_ => { });
        }

        public bool Start(int[] axes, string dir, string baseName)
        {
            if (IsLogging) { _log("[AjinLog] 이미 로깅 중입니다."); return false; }

            try
            {
                Directory.CreateDirectory(dir);
                _axes = axes;
                _filePath = Path.Combine(dir, baseName + "_AjinMotion.csv");
                _cts = new CancellationTokenSource();
                _task = Task.Run(() => PollLoop(_cts.Token), _cts.Token);
                _log($"[AjinLog] 시작 → {_filePath}");
                return true;
            }
            catch (Exception ex)
            {
                _log($"[AjinLog] 시작 실패: {ex.Message}");
                return false;
            }
        }

        public void Stop()
        {
            if (!IsLogging) return;
            try { _cts?.Cancel(); _task?.Wait(2000); _log("[AjinLog] 중지 완료"); }
            catch { }
            finally { _cts?.Dispose(); _cts = null; _task = null; }
        }

        // ── 폴링 루프 ─────────────────────────────────────────────
        private void PollLoop(CancellationToken token)
        {
            // 헤더
            bool hasCmdPos = _getCmdPos != null;
            var header = new StringBuilder("Timestamp_ms");
            foreach (int ax in _axes)
            {
                header.Append($",Ax{ax}_Pos(mm),Ax{ax}_Vel(mm/s),Ax{ax}_Trq(%)");
                if (hasCmdPos) header.Append($",Ax{ax}_CmdPos(mm),Ax{ax}_CmdVel(mm/s)");
            }

            // 이전 위치 (차분 속도 계산용)
            double[] prevPos    = new double[_axes.Length];
            double[] prevCmdPos = new double[_axes.Length];
            long prevTime = 0;
            bool first = true;

            var sw = System.Diagnostics.Stopwatch.StartNew();

            try
            {
                using (var writer = new StreamWriter(_filePath, false, Encoding.UTF8))
                {
                    writer.WriteLine(header.ToString());

                    while (!token.IsCancellationRequested)
                    {
                        long t = sw.ElapsedMilliseconds;
                        double dtSec = first ? 0.0 : (t - prevTime) / 1000.0;

                        var line = new StringBuilder();
                        line.Append(t.ToString(CultureInfo.InvariantCulture));

                        for (int i = 0; i < _axes.Length; i++)
                        {
                            int ax = _axes[i];
                            double pos = SafeGet(_getPos, ax);
                            double trq = SafeGet(_getTorque, ax);

                            // getVel 콜백이 있으면 직접 사용, 없으면 위치 차분으로 계산
                            double vel = (_getVel != null)
                                ? SafeGet(_getVel, ax)
                                : ((first || dtSec <= 0) ? 0.0 : (pos - prevPos[i]) / dtSec);

                            line.Append($",{pos:F4},{vel:F4},{trq:F4}");
                            if (hasCmdPos)
                            {
                                double cmdPos = SafeGet(_getCmdPos, ax);
                                double cmdVel = (first || dtSec <= 0) ? 0.0 : (cmdPos - prevCmdPos[i]) / dtSec;
                                line.Append($",{cmdPos:F4},{cmdVel:F4}");
                                prevCmdPos[i] = cmdPos;
                            }
                            prevPos[i] = pos;
                        }

                        writer.WriteLine(line.ToString());
                        prevTime = t;
                        first = false;

                        // 정밀 슬립
                        long elapsed = sw.ElapsedMilliseconds - t;
                        int wait = IntervalMs - (int)elapsed;
                        if (wait > 0) Thread.Sleep(wait);
                    }

                    writer.Flush();
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { _log($"[AjinLog] 폴링 오류: {ex.Message}"); }
        }

        private static double SafeGet(Func<int, double> fn, int ax)
        {
            try { return fn(ax); }
            catch { return double.NaN; }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Stop();
            _disposed = true;
        }
    }
}