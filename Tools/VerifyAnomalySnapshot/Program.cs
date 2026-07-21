using System;
using System.IO;
using PHM_Project_DockPanel.Services.Core;

namespace VerifyAnomalySnapshot
{
    /// <summary>
    /// AnomalySnapshotWriter를 합성 데이터로 구동해 PNG가 실제로 만들어지는지,
    /// 백그라운드 스레드(Task.Run 시뮬레이션)에서 호출해도 문제없는지 확인합니다.
    /// </summary>
    internal static class Program
    {
        private static int Main()
        {
            string outDir = Path.Combine(Path.GetTempPath(), "AnomalySnapshotVerify");
            Directory.CreateDirectory(outDir);
            bool ok = true;

            // ── 시나리오 1: 토크 축별 이상 — CLS 결함 분류 포함 ─────────────────
            ok &= RunOnBackgroundThread(() =>
            {
                int windowSize = 500;
                var channelNames = new[] { "Ax1_Trq(%)" };
                var window = new float[windowSize * 1];
                var rng = new Random(7);
                for (int t = 0; t < windowSize; t++)
                    window[t] = (float)(10.0 + 40.0 * Math.Sin(t * 0.2) + rng.NextDouble() * 5.0);

                AnomalySnapshotWriter.Save(
                    outDir, window, 1, channelNames,
                    "torque", 1, "AE-CNN1D",
                    score: 1.85f, threshold: 1.0f,
                    clsClassName: "overload", clsConfidence: 0.87f,
                    log: msg => Console.WriteLine("[LOG] " + msg));
            });

            // ── 시나리오 2: 가속도 전역 이상 — CLS 없음(비활성) ─────────────────
            ok &= RunOnBackgroundThread(() =>
            {
                int windowSize = 512;
                var channelNames = new[] { "x", "y", "z" };
                var window = new float[windowSize * 3];
                var rng = new Random(11);
                for (int t = 0; t < windowSize; t++)
                {
                    window[t * 3 + 0] = (float)(0.05 * Math.Sin(t * 0.5) + rng.NextDouble() * 0.02);
                    window[t * 3 + 1] = (float)(0.03 * Math.Cos(t * 0.3) + rng.NextDouble() * 0.02);
                    window[t * 3 + 2] = (float)(1.0 + rng.NextDouble() * 0.05); // 중력 성분
                }

                AnomalySnapshotWriter.Save(
                    outDir, window, 3, channelNames,
                    "accel", null, "IF-STAT",
                    score: 0.012f, threshold: 0.0102f,
                    clsClassName: null, clsConfidence: null,
                    log: msg => Console.WriteLine("[LOG] " + msg));
            });

            // ── 시나리오 3: 잘못된 입력(윈도우 없음) — 예외 없이 조용히 스킵되는지 ──
            ok &= RunOnBackgroundThread(() =>
            {
                AnomalySnapshotWriter.Save(
                    outDir, null, 0, null,
                    "torque", 0, "AE-CNN1D",
                    score: 2.0f, threshold: 1.0f,
                    clsClassName: null, clsConfidence: null,
                    log: msg => Console.WriteLine("[LOG] " + msg));
            });

            Console.WriteLine();
            var files = Directory.GetFiles(outDir, "*.png");
            Console.WriteLine($"생성된 PNG 파일 수: {files.Length} (기대: 2 — 시나리오 3은 데이터 없어 저장 안 함)");
            foreach (var f in files)
            {
                var info = new FileInfo(f);
                bool validPng = IsValidPng(f);
                Console.WriteLine($"  {Path.GetFileName(f)}  {info.Length:N0} bytes  PNG시그니처={(validPng ? "OK" : "FAIL")}");
                ok &= info.Length > 1000 && validPng;
            }

            Console.WriteLine();
            Console.WriteLine(ok && files.Length == 2 ? "===== 전체 검증 통과 =====" : "===== 일부 검증 실패 =====");
            return (ok && files.Length == 2) ? 0 : 1;
        }

        private static bool IsValidPng(string path)
        {
            byte[] sig = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
            using (var fs = File.OpenRead(path))
            {
                var buf = new byte[8];
                if (fs.Read(buf, 0, 8) != 8) return false;
                for (int i = 0; i < 8; i++)
                    if (buf[i] != sig[i]) return false;
            }
            return true;
        }

        /// <summary>ContinuousInferenceService의 Task.Run 백그라운드 루프와 동일한 조건(비-UI 스레드)에서 실행합니다.</summary>
        private static bool RunOnBackgroundThread(Action action)
        {
            bool ok = true;
            var t = new System.Threading.Thread(() =>
            {
                try { action(); }
                catch (Exception ex)
                {
                    Console.WriteLine("[FAIL] 백그라운드 스레드 예외: " + ex);
                    ok = false;
                }
            });
            t.IsBackground = true;
            t.Start();
            t.Join();
            return ok;
        }
    }
}
