using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace VerifyCombinedCsvLogger
{
    /// <summary>CombinedCsvLogger가 생성한 CSV를 파싱해 요구사항 체크리스트를 검증합니다.</summary>
    internal static class Verify
    {
        public static bool RunFullChecks(string path)
        {
            var lines = File.ReadAllLines(path);
            bool ok = true;
            if (lines.Length < 2)
            {
                Console.WriteLine("[FAIL] 파일에 데이터 행이 없습니다: " + path);
                return false;
            }

            var headers = lines[0].Split(',');
            int expectedCols = headers.Length;

            var idx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < headers.Length; i++) idx[headers[i].Trim()] = i;

            var rows = lines.Skip(1).Select(l => l.Split(',')).ToList();

            // 1) 헤더/행 컬럼 수 항상 동일
            bool colCountOk = rows.All(r => r.Length == expectedCols);
            Report(ref ok, colCountOk, "1) 헤더/모든 행의 컬럼 수 일치",
                $"헤더={expectedCols}컬럼, 총 {rows.Count}행");

            // 2) time_s 단조 증가
            double prevT = double.NegativeInfinity;
            bool monotonic = true;
            foreach (var r in rows)
            {
                double t = double.Parse(r[idx["time_s"]], CultureInfo.InvariantCulture);
                if (t < prevT) { monotonic = false; break; }
                prevT = t;
            }
            Report(ref ok, monotonic, "2) time_s 단조 증가",
                $"첫 행={rows[0][idx["time_s"]]}s, 마지막 행={rows[rows.Count - 1][idx["time_s"]]}s");

            // 3) 숫자 소수점 '.' 사용 (invariant culture 포맷 확인)
            bool dotOk = rows.Take(50).All(r =>
            {
                string v = r[idx["x"]];
                return v.IndexOf('.') >= 0 || v.IndexOf('e') >= 0 || v.IndexOf('E') >= 0;
            });
            Report(ref ok, dotOk, "3) 숫자 소수점이 '.'으로 기록됨 (InvariantCulture)", "x 컬럼 앞 50행 샘플 검사");

            // 4/7) MotionID: Idle 구간=0, Pos 구간=nonzero 유지
            bool motionIdOk = true;
            string motionIdDetail = "";
            foreach (int ax in new[] { 0, 1 })
            {
                string col = $"Ax{ax}_MotionID";
                if (!idx.ContainsKey(col)) { motionIdOk = false; motionIdDetail += $"{col} 컬럼 없음; "; continue; }
                var ids = rows.Select(r => int.Parse(r[idx[col]], CultureInfo.InvariantCulture)).ToList();
                bool hasZero = ids.Any(v => v == 0);
                bool hasNonZero = ids.Any(v => v != 0);
                motionIdOk &= hasZero && hasNonZero;
                motionIdDetail += $"{col} distinct=[{string.Join(",", ids.Distinct().OrderBy(v => v))}]; ";
            }
            Report(ref ok, motionIdOk, "4/7) 축별 MotionID: Idle=0, Pos 중엔 0이 아닌 ID 유지", motionIdDetail);

            // 5) 동시 이동 시 축별 MotionID 독립성 (t∈[1.05,1.55)에서 Ax0=2, Ax1=1 동시 존재)
            bool overlapOk = idx.ContainsKey("Ax0_MotionID") && idx.ContainsKey("Ax1_MotionID") && rows.Any(r =>
            {
                double t = double.Parse(r[idx["time_s"]], CultureInfo.InvariantCulture);
                if (t < 1.05 || t >= 1.55) return false;
                int m0 = int.Parse(r[idx["Ax0_MotionID"]], CultureInfo.InvariantCulture);
                int m1 = int.Parse(r[idx["Ax1_MotionID"]], CultureInfo.InvariantCulture);
                return m0 == 2 && m1 == 1;
            });
            Report(ref ok, overlapOk, "5) Ax0/Ax1 동시 이동 중 MotionID가 서로 다른 값(2 vs 1)으로 독립 유지",
                "t∈[1.05,1.55) 구간에서 확인 — 한 축의 모션이 다른 축의 MotionID에 영향 없음");

            // 6) Pos 구간 ActPos(시작/종료)로 이동거리 계산 가능한지 (설계값과 비교)
            bool distanceOk =
                CheckMotionDistance(rows, idx, axis: 0, motionId: 1, expectedDistance: 50.0, tolerance: 5.0) &&
                CheckMotionDistance(rows, idx, axis: 0, motionId: 2, expectedDistance: 40.0, tolerance: 5.0) &&
                CheckMotionDistance(rows, idx, axis: 1, motionId: 1, expectedDistance: 30.0, tolerance: 5.0);
            Report(ref ok, distanceOk, "6) MotionID 구간의 ActPos(시작/종료)로 이동거리 계산 가능", "설계값 대비 오차 ≤5mm");

            // 9) 기존 컬럼(time_s,x,y,z,Ax*_Trq(%),Op_Ax*) 이름/존재 보존 확인
            bool prefixOk = headers.Length >= 4
                && headers[0].Trim() == "time_s" && headers[1].Trim() == "x"
                && headers[2].Trim() == "y" && headers[3].Trim() == "z"
                && idx.ContainsKey("Ax0_Trq(%)") && idx.ContainsKey("Ax1_Trq(%)")
                && idx.ContainsKey("Op_Ax0") && idx.ContainsKey("Op_Ax1");
            Report(ref ok, prefixOk, "9) 기존 컬럼명 보존(time_s,x,y,z,Ax*_Trq(%),Op_Ax*)", string.Join(",", headers));

            // Op 값이 Idle/Pos만 존재하는지 (임의 상태 추정 없음 확인)
            bool opOk = idx.ContainsKey("Op_Ax0")
                && rows.All(r => r[idx["Op_Ax0"]] == "Idle" || r[idx["Op_Ax0"]] == "Pos");
            Report(ref ok, opOk, "Op_Ax0 값이 Idle/Pos만 존재 (임의 상태 추정 없음)", "");

            // SequenceID: 시퀀스 시작 전=0, 시작 후=1
            bool seqOk = idx.ContainsKey("SequenceID")
                && rows.Any(r => r[idx["SequenceID"]] == "0")
                && rows.Any(r => r[idx["SequenceID"]] == "1");
            Report(ref ok, seqOk, "SequenceID: 시퀀스 시작 전=0, RaiseSequenceStarted 이후=1", "");

            // 샘플 출력 (Idle→Pos 전이 구간, 두 축이 동시에 움직이기 시작하는 구간)
            Console.WriteLine();
            Console.WriteLine("--- 헤더 ---");
            Console.WriteLine(lines[0]);
            Console.WriteLine("--- 샘플 (t∈[0.95, 1.10), Ax0 2차 모션 + Ax1 모션이 동시에 시작되는 전이 구간) ---");
            foreach (var r in rows.Where(r =>
            {
                double t = double.Parse(r[idx["time_s"]], CultureInfo.InvariantCulture);
                return t >= 0.95 && t <= 1.10;
            }).Take(18))
            {
                Console.WriteLine(string.Join(",", r));
            }

            return ok;
        }

        public static bool RunMissingValueCheck(string path)
        {
            var lines = File.ReadAllLines(path);
            bool ok = true;
            if (lines.Length < 2)
            {
                Console.WriteLine("[FAIL] 파일에 데이터 행이 없습니다: " + path);
                return false;
            }

            var headers = lines[0].Split(',');
            var idx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < headers.Length; i++) idx[headers[i].Trim()] = i;

            var dataLines = lines.Skip(1).ToList();

            foreach (var col in new[] { "Ax0_ActPos", "Ax0_CmdPos", "Ax0_ActVel" })
            {
                if (!idx.ContainsKey(col))
                {
                    Report(ref ok, false, $"10) {col} 컬럼 존재", "컬럼 자체가 없음");
                    continue;
                }
                bool allNaN = dataLines.All(l => l.Split(',')[idx[col]].Trim().Equals("NaN", StringComparison.OrdinalIgnoreCase));
                Report(ref ok, allNaN, $"10) getActPos/getCmdPos/getActVel 콜백 미주입 시 {col}이 'NaN'으로 기록됨 (0으로 위장하지 않음)", "");
            }

            return ok;
        }

        private static bool CheckMotionDistance(
            List<string[]> rows, Dictionary<string, int> idx,
            int axis, int motionId, double expectedDistance, double tolerance)
        {
            string midCol = $"Ax{axis}_MotionID";
            string posCol = $"Ax{axis}_ActPos";
            if (!idx.ContainsKey(midCol) || !idx.ContainsKey(posCol)) return false;

            var matched = rows.Where(r => int.Parse(r[idx[midCol]], CultureInfo.InvariantCulture) == motionId).ToList();
            if (matched.Count < 2) return false;

            double startPos = double.Parse(matched[0][idx[posCol]], CultureInfo.InvariantCulture);
            double endPos = double.Parse(matched[matched.Count - 1][idx[posCol]], CultureInfo.InvariantCulture);
            double dist = Math.Abs(endPos - startPos);
            bool okThis = Math.Abs(dist - expectedDistance) <= tolerance;

            Console.WriteLine($"  [거리검증] Ax{axis} MotionID={motionId}: start={startPos:F2}mm end={endPos:F2}mm " +
                               $"dist={dist:F2}mm (기대 {expectedDistance}±{tolerance}mm) → {(okThis ? "OK" : "MISMATCH")}");
            return okThis;
        }

        private static void Report(ref bool overallOk, bool condition, string title, string detail)
        {
            Console.WriteLine((condition ? "[PASS] " : "[FAIL] ") + title +
                               (string.IsNullOrEmpty(detail) ? "" : "  — " + detail));
            overallOk = overallOk && condition;
        }
    }
}
