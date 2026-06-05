using System;

namespace PHM_Project_DockPanel.Services.Core
{
    // =========================================================================
    //  WindowState — 윈도우 상태 열거형
    // =========================================================================
    public enum WindowState
    {
        /// <summary>정지 구간 — AE 추론 스킵, rule 기반 감시만 수행</summary>
        Idle,
        /// <summary>구동 구간 — AE 추론 실행</summary>
        Motion,
        /// <summary>경계 구간 — 추론 제외 또는 별도 로그</summary>
        Ambiguous,
    }

    // =========================================================================
    //  WindowFeatures — 윈도우별 상태 분류 특징
    // =========================================================================
    public sealed class WindowFeatures
    {
        /// <summary>가속도 크기(sqrt(x²+y²+z²)) RMS</summary>
        public double AccMagRms          { get; set; }
        /// <summary>가속도 크기 Peak</summary>
        public double AccMagPeak         { get; set; }
        /// <summary>토크 detrended RMS (축별 평균 제거 후) — 축 평균</summary>
        public double TrqDetrendedRms    { get; set; }
        /// <summary>토크 표준편차 — 축 평균</summary>
        public double TrqStdMean         { get; set; }
        /// <summary>토크 peak-to-peak — 축 최댓값</summary>
        public double TrqPeakToPeakMax   { get; set; }
        /// <summary>acc_mag < zero_threshold 비율 (0=항상 활성, 1=항상 정지)</summary>
        public double ZeroRatio          { get; set; }
        /// <summary>motion_score (0~1 클리핑)</summary>
        public double MotionScore        { get; set; }
    }

    // =========================================================================
    //  WindowStateThresholds — 상태 분류 임계값 (서버 /model_info 또는 기본값)
    // =========================================================================
    public sealed class WindowStateThresholds
    {
        // ── 분류 경계 ─────────────────────────────────────────────────────────
        /// <summary>motion_score >= 이 값 → Motion</summary>
        public double MotionScoreThreshold  { get; set; } = 0.25;
        /// <summary>motion_score <= 이 값 → Idle</summary>
        public double IdleScoreThreshold    { get; set; } = 0.05;
        /// <summary>acc_mag 이 값 미만이면 zero 샘플로 간주</summary>
        public double AccZeroThreshold      { get; set; } = 0.01;

        // ── motion_score 정규화 기준값 (학습 데이터 통계 기반) ────────────────
        /// <summary>acc_mag_rms 정규화 기준 (g 단위 기본값)</summary>
        public double RefAccMagRms          { get; set; } = 0.5;
        /// <summary>trq_detrended_rms 정규화 기준 (토크% 단위 기본값)</summary>
        public double RefTrqDetrendedRms    { get; set; } = 5.0;
        /// <summary>trq_peak_to_peak_max 정규화 기준 (토크% 단위 기본값)</summary>
        public double RefTrqPeakToPeakMax   { get; set; } = 10.0;

        // ── motion_score 가중치 (합=1.0) ──────────────────────────────────────
        /// <summary>가속도 RMS 가중치</summary>
        public double WeightAccRms          { get; set; } = 0.2;
        /// <summary>토크 detrended RMS 가중치 (torque dynamic 중심)</summary>
        public double WeightTrqRms          { get; set; } = 0.5;
        /// <summary>토크 peak-to-peak 가중치</summary>
        public double WeightTrqP2p          { get; set; } = 0.3;
    }

    // =========================================================================
    //  WindowStateClassifier — 정적 유틸
    // =========================================================================
    public static class WindowStateClassifier
    {
        /// <summary>
        /// 윈도우 float 배열에서 상태 분류 특징을 계산합니다.
        /// </summary>
        /// <param name="window">
        ///   float[windowSize × nChannels], row-major (time × channel).
        ///   window[t * nChannels + ch]
        /// </param>
        /// <param name="windowSize">시간축 샘플 수</param>
        /// <param name="nChannels">전체 채널 수</param>
        /// <param name="accelChIdx">가속도 채널 인덱스 (x=0, y=1, z=2 순서)</param>
        /// <param name="torqueChIdx">토크 채널 인덱스 (Ax0, Ax1, Ax2 ...)</param>
        /// <param name="thr">임계값/정규화 파라미터</param>
        public static WindowFeatures ComputeFeatures(
            float[]               window,
            int                   windowSize,
            int                   nChannels,
            int[]                 accelChIdx,
            int[]                 torqueChIdx,
            WindowStateThresholds thr)
        {
            var f = new WindowFeatures();

            // ── 가속도 크기 특징 ───────────────────────────────────────────────
            bool hasAccel = accelChIdx != null && accelChIdx.Length >= 3;
            if (hasAccel)
            {
                double sumSq = 0, peak = 0, zeroCnt = 0;
                for (int t = 0; t < windowSize; t++)
                {
                    int b  = t * nChannels;
                    double x = window[b + accelChIdx[0]];
                    double y = window[b + accelChIdx[1]];
                    double z = window[b + accelChIdx[2]];
                    double mag = Math.Sqrt(x * x + y * y + z * z);
                    sumSq += mag * mag;
                    if (mag > peak) peak = mag;
                    if (mag < thr.AccZeroThreshold) zeroCnt++;
                }
                f.AccMagRms  = Math.Sqrt(sumSq / windowSize);
                f.AccMagPeak = peak;
                f.ZeroRatio  = zeroCnt / windowSize;
            }
            else
            {
                f.AccMagRms = 0; f.AccMagPeak = 0; f.ZeroRatio = 1.0;
            }

            // ── 토크 특징 ──────────────────────────────────────────────────────
            bool hasTrq = torqueChIdx != null && torqueChIdx.Length > 0;
            if (hasTrq)
            {
                double sumDetRms = 0, sumStd = 0, maxP2P = 0;
                foreach (int ci in torqueChIdx)
                {
                    // 축별 평균 계산 (detrending 기준)
                    double mean = 0;
                    for (int t = 0; t < windowSize; t++)
                        mean += window[t * nChannels + ci];
                    mean /= windowSize;

                    double sumDetSq = 0;
                    double chMin = double.MaxValue, chMax = double.MinValue;
                    for (int t = 0; t < windowSize; t++)
                    {
                        double v = window[t * nChannels + ci];
                        double d = v - mean;  // detrended
                        sumDetSq += d * d;
                        if (v < chMin) chMin = v;
                        if (v > chMax) chMax = v;
                    }
                    double detRms = Math.Sqrt(sumDetSq / windowSize);
                    double p2p    = chMax - chMin;

                    sumDetRms += detRms;
                    sumStd    += detRms;   // detrended RMS = std (mean 제거 후)
                    if (p2p > maxP2P) maxP2P = p2p;
                }
                int n = torqueChIdx.Length;
                f.TrqDetrendedRms  = sumDetRms / n;
                f.TrqStdMean       = sumStd    / n;
                f.TrqPeakToPeakMax = maxP2P;
            }

            // ── motion_score 계산 (정규화 후 가중합, 0~1 클리핑) ──────────────
            double refAcc = Math.Max(thr.RefAccMagRms,       1e-9);
            double refTrq = Math.Max(thr.RefTrqDetrendedRms, 1e-9);
            double refP2P = Math.Max(thr.RefTrqPeakToPeakMax,1e-9);

            double sAcc = Math.Min(f.AccMagRms       / refAcc, 1.0);
            double sTrq = Math.Min(f.TrqDetrendedRms / refTrq, 1.0);
            double sP2P = Math.Min(f.TrqPeakToPeakMax/ refP2P, 1.0);

            if (hasAccel && hasTrq)
            {
                f.MotionScore = thr.WeightAccRms * sAcc
                              + thr.WeightTrqRms * sTrq
                              + thr.WeightTrqP2p * sP2P;
            }
            else if (hasAccel)
            {
                // 가속도만 있는 경우 (sensorType="accel")
                f.MotionScore = sAcc;
            }
            else if (hasTrq)
            {
                // 토크만 있는 경우 (sensorType="torque")
                double wSum = thr.WeightTrqRms + thr.WeightTrqP2p;
                f.MotionScore = wSum > 0
                    ? (thr.WeightTrqRms * sTrq + thr.WeightTrqP2p * sP2P) / wSum
                    : sTrq;
            }
            // else: 채널 없음 → score=0

            return f;
        }

        /// <summary>
        /// motion_score 기반 윈도우 상태를 분류합니다.
        /// </summary>
        public static WindowState Classify(WindowFeatures f, WindowStateThresholds thr)
        {
            if (f.MotionScore >= thr.MotionScoreThreshold) return WindowState.Motion;
            if (f.MotionScore <= thr.IdleScoreThreshold)   return WindowState.Idle;
            return WindowState.Ambiguous;
        }

        /// <summary>
        /// sensor_type 문자열에서 채널 역할(accel / torque 인덱스)을 추론합니다.
        ///   "accel"    → accelIdx=[0,1,2],     torqueIdx=[]
        ///   "torque"   → accelIdx=[],           torqueIdx=[0..nCh-1]
        ///   "combined" → accelIdx=[0,1,2],      torqueIdx=[3..nCh-1]
        /// GetSignalColumnIndices 가 컬럼을 이 순서로 채우기 때문에 재파싱 없이 인덱스 직접 생성.
        /// </summary>
        public static void GetChannelRoles(
            string sensorType,
            int    nChannels,
            out int[] accelIdx,
            out int[] torqueIdx)
        {
            if (sensorType == "accel")
            {
                accelIdx  = nChannels >= 3 ? new[] { 0, 1, 2 } : new int[0];
                torqueIdx = new int[0];
            }
            else if (sensorType == "torque")
            {
                accelIdx  = new int[0];
                torqueIdx = new int[nChannels];
                for (int i = 0; i < nChannels; i++) torqueIdx[i] = i;
            }
            else  // "combined"
            {
                accelIdx = nChannels >= 3 ? new[] { 0, 1, 2 } : new int[0];
                int trqCount = Math.Max(0, nChannels - 3);
                torqueIdx = new int[trqCount];
                for (int i = 0; i < trqCount; i++) torqueIdx[i] = 3 + i;
            }
        }
    }
}
