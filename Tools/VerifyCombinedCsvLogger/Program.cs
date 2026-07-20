using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using PHM_Project_DockPanel.Services.DAQ;

namespace VerifyCombinedCsvLogger
{
    /// <summary>
    /// CombinedCsvLogger를 합성(synthetic) 데이터로 구동해 CSV 스키마/동기화/MotionID/
    /// SequenceID 요구사항을 자동 점검하는 검증 도구. 실제 하드웨어/컨트롤러 없이
    /// PHM_Project_DockPanel.exe의 CombinedCsvLogger를 그대로 참조해 실행합니다.
    /// 배포 대상이 아니며, 개발자가 수동으로 실행해 결과를 확인하는 용도입니다.
    /// </summary>
    internal static class Program
    {
        private const double SampleRate = 1000.0;
        private const int BlockSize = 100; // 실제 DAQ 블록(_readBlock)과 유사한 주기(≈100ms)

        private class AxisWindow
        {
            public double StartSec, EndSec, StartPos, TargetPos;
            public int MotionId;
        }

        // 합성 시나리오: Axis0은 두 번 이동(첫 번째는 단독, 두 번째는 Axis1과 동시),
        // Axis1은 Axis0의 두 번째 이동과 겹치는 구간에서 한 번 이동 → "동시 이동 시 MotionID 독립성" 검증용.
        private static readonly Dictionary<int, List<AxisWindow>> Windows = new Dictionary<int, List<AxisWindow>>
        {
            [0] = new List<AxisWindow>
            {
                new AxisWindow { StartSec = 0.3, EndSec = 0.8, StartPos = 0,  TargetPos = 50, MotionId = 1 },
                new AxisWindow { StartSec = 1.0, EndSec = 1.6, StartPos = 50, TargetPos = 10, MotionId = 2 },
            },
            [1] = new List<AxisWindow>
            {
                new AxisWindow { StartSec = 1.0, EndSec = 1.6, StartPos = 0,  TargetPos = 30, MotionId = 1 },
            },
        };

        private static DateTime _t0;
        private static int _sequenceId;
        private static readonly object StateLock = new object();

        private static double NowSec() => (DateTime.UtcNow - _t0).TotalSeconds;

        private static AxisWindow CurrentWindow(int ax, double t)
        {
            if (!Windows.TryGetValue(ax, out var list)) return null;
            foreach (var w in list)
                if (t >= w.StartSec && t < w.EndSec) return w;
            return null;
        }

        private static AxisWindow LastEndedWindow(int ax, double t)
        {
            if (!Windows.TryGetValue(ax, out var list)) return null;
            AxisWindow last = null;
            foreach (var w in list) if (w.EndSec <= t) last = w;
            return last;
        }

        private static AxisWindow LastStartedWindow(int ax, double t)
        {
            if (!Windows.TryGetValue(ax, out var list)) return null;
            AxisWindow last = null;
            foreach (var w in list) if (w.StartSec <= t) last = w;
            return last;
        }

        private static string GetOp(int ax) => CurrentWindow(ax, NowSec()) != null ? "Pos" : "Idle";

        private static int GetMotionId(int ax) => CurrentWindow(ax, NowSec())?.MotionId ?? 0;

        private static double GetActPos(int ax)
        {
            double t = NowSec();
            var w = CurrentWindow(ax, t);
            if (w != null)
            {
                double frac = Math.Max(0, Math.Min(1, (t - w.StartSec) / (w.EndSec - w.StartSec)));
                return w.StartPos + (w.TargetPos - w.StartPos) * frac;
            }
            var last = LastEndedWindow(ax, t);
            return last?.TargetPos ?? 0.0;
        }

        private static double GetCmdPos(int ax)
        {
            double t = NowSec();
            var last = LastStartedWindow(ax, t);
            return last?.TargetPos ?? 0.0;
        }

        private static double GetActVel(int ax)
        {
            var w = CurrentWindow(ax, NowSec());
            return w == null ? 0.0 : (w.TargetPos - w.StartPos) / (w.EndSec - w.StartSec);
        }

        private static double GetTorque(int ax)
        {
            double t = NowSec();
            var w = CurrentWindow(ax, t);
            return w != null
                ? 15.0 + 5.0 * Math.Sin(t * 10 + ax)
                : 0.5 * Math.Sin(t * 3 + ax);
        }

        private static int GetSequenceId() { lock (StateLock) return _sequenceId; }

        private static int Main()
        {
            string outDir = Path.Combine(Path.GetTempPath(), "CombinedCsvLoggerVerify");
            Directory.CreateDirectory(outDir);

            bool ok = true;

            // ── 시나리오 1: 정상 경로 (모든 콜백 제공) ────────────────────────────
            string path1 = RunScenario(outDir, "Verify_full", includeAllCallbacks: true);
            Console.WriteLine();
            Console.WriteLine("========== 시나리오 1: 전체 컬럼 검증 (" + path1 + ") ==========");
            ok &= Verify.RunFullChecks(path1);

            // ── 시나리오 2: ActPos/CmdPos/ActVel 콜백 미주입 → NaN 결측 정책 확인 ──
            string path2 = RunScenario(outDir, "Verify_missing", includeAllCallbacks: false);
            Console.WriteLine();
            Console.WriteLine("========== 시나리오 2: 결측값 NaN 정책 검증 (" + path2 + ") ==========");
            ok &= Verify.RunMissingValueCheck(path2);

            Console.WriteLine();
            Console.WriteLine(ok ? "===== 전체 검증 통과 =====" : "===== 일부 검증 실패 =====");
            return ok ? 0 : 1;
        }

        private static string RunScenario(string outDir, string baseName, bool includeAllCallbacks)
        {
            _t0 = DateTime.UtcNow;
            lock (StateLock) _sequenceId = 0;

            var axes = new[] { 0, 1 };
            var logger = new CombinedCsvLogger(
                getTorque:     GetTorque,
                getAxisOp:     GetOp,
                axes:          axes,
                log:           msg => Console.WriteLine("[LOG] " + msg),
                getActPos:     includeAllCallbacks ? (Func<int, double>)GetActPos : null,
                getCmdPos:     includeAllCallbacks ? (Func<int, double>)GetCmdPos : null,
                getActVel:     includeAllCallbacks ? (Func<int, double>)GetActVel : null,
                getMotionId:   GetMotionId,
                getSequenceId: GetSequenceId);

            if (!logger.Start(outDir, baseName))
                throw new InvalidOperationException("logger.Start() 실패: " + baseName);

            Console.WriteLine("시작: " + logger.OutputPath);

            // t=0.3s에 시퀀스 시작(합성) — TeachingForm의 RaiseSequenceStarted에 대응
            bool seqRaised = false;

            var rng = new Random(42);
            const double simEnd = 2.0;
            while (NowSec() < simEnd)
            {
                if (!seqRaised && NowSec() >= 0.3)
                {
                    lock (StateLock) _sequenceId = 1;
                    seqRaised = true;
                }

                double t = NowSec();
                var block = new double[3, BlockSize];
                for (int i = 0; i < BlockSize; i++)
                {
                    block[0, i] = 0.05 * Math.Sin((t + i / SampleRate) * 50) + rng.NextDouble() * 0.01;
                    block[1, i] = 0.05 * Math.Cos((t + i / SampleRate) * 50) + rng.NextDouble() * 0.01;
                    block[2, i] = 1.0 + rng.NextDouble() * 0.01;
                }
                logger.ProcessAccelBlock(0, block, BlockSize, SampleRate);
                Thread.Sleep(100);
            }

            logger.Stop();
            Console.WriteLine("종료: " + logger.OutputPath);
            return logger.OutputPath;
        }
    }
}
