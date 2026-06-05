using System;
using System.Collections.Generic;

namespace PHM_Project_DockPanel.Services.Core
{
    // =========================================================================
    //  WindowState
    // =========================================================================
    public enum WindowState { Idle, Motion, Ambiguous }

    // =========================================================================
    //  WindowFeatures — 윈도우별 상태 분류 특징
    // =========================================================================
    public sealed class WindowFeatures
    {
        public double AccMagRms          { get; set; }
        public double AccMagPeak         { get; set; }
        public double TrqDetrendedRms    { get; set; }
        public double TrqStdMean         { get; set; }
        public double TrqPeakToPeakMax   { get; set; }
        public double ZeroRatio          { get; set; }
        public double MotionScore        { get; set; }
    }

    // =========================================================================
    //  FeatureScaler — robust_scale_01 per feature
    //  (value - q05) / (q95 - q05), clipped [0, 1]
    // =========================================================================
    public sealed class FeatureScaler
    {
        public double Q05 { get; set; } = 0.0;
        public double Q95 { get; set; } = 1.0;

        public double Normalize(double value)
        {
            double range = Q95 - Q05;
            if (range < 1e-9) return 0.0;
            double v = (value - Q05) / range;
            return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
        }
    }

    // =========================================================================
    //  WindowStateThresholds — 학습 데이터 기반 분류 파라미터
    //  서버 /model_info → window_state 블록에서 로드
    // =========================================================================
    public sealed class WindowStateThresholds
    {
        // ── per-feature robust scaler (q05 / q95) ─────────────────────────
        public FeatureScaler AccMagRms          { get; set; } = new FeatureScaler { Q05 = 0.0, Q95 = 0.05  };
        public FeatureScaler AccMagPeak         { get; set; } = new FeatureScaler { Q05 = 0.0, Q95 = 0.08  };
        public FeatureScaler TrqDetrendedRms    { get; set; } = new FeatureScaler { Q05 = 0.0, Q95 = 5.0   };
        public FeatureScaler TrqStdMean         { get; set; } = new FeatureScaler { Q05 = 0.0, Q95 = 5.0   };
        public FeatureScaler TrqPeakToPeakMax   { get; set; } = new FeatureScaler { Q05 = 0.0, Q95 = 10.0  };

        // ── motion_score 가중치 ────────────────────────────────────────────
        public double WeightAccMagRms        { get; set; } = 0.15;
        public double WeightAccMagPeak       { get; set; } = 0.10;
        public double WeightTrqDetrendedRms  { get; set; } = 0.35;
        public double WeightTrqStdMean       { get; set; } = 0.20;
        public double WeightTrqPeakToPeakMax { get; set; } = 0.20;

        // ── 분류 임계값 ────────────────────────────────────────────────────
        public double MotionScoreThreshold   { get; set; } = 0.45;
        public double IdleScoreThreshold     { get; set; } = 0.08;

        // ── zero_ratio 임계값 ─────────────────────────────────────────────
        public double AccZeroThreshold       { get; set; } = 0.03;
        public double TrqDynZeroThreshold    { get; set; } = 0.15;
        public double ZeroRatioIdleThreshold { get; set; } = 0.70;
    }

    // =========================================================================
    //  WindowStateClassifier — 정적 유틸
    // =========================================================================
    public static class WindowStateClassifier
    {
        /// <summary>
        /// 윈도우에서 상태 분류 특징을 계산합니다.
        ///
        /// run_motion_window_pipeline.py extract_window_features() 와 동일한 로직:
        ///   - robust_scale_01(q05~q95) 정규화
        ///   - zero_ratio: acc_mag &lt; acc_zero_thr AND trq_dyn_mag &lt; trq_zero_thr 동시 조건
        ///   - motion_score: 5개 feature 가중합
        ///
        /// window layout: float[windowSize * nChannels], row-major (time × channel)
        ///   window[t * nChannels + ch]
        /// </summary>
        public static WindowFeatures ComputeFeatures(
            float[]               window,
            int                   windowSize,
            int                   nChannels,
            int[]                 accelChIdx,
            int[]                 torqueChIdx,
            WindowStateThresholds thr)
        {
            var f = new WindowFeatures();

            bool hasAccel  = accelChIdx  != null && accelChIdx.Length  >= 3;
            bool hasTorque = torqueChIdx != null && torqueChIdx.Length  > 0;

            // ── 1) 토크 축별 평균 (detrending 기준) ──────────────────────────
            double[] axMeans = null;
            if (hasTorque)
            {
                axMeans = new double[torqueChIdx.Length];
                for (int ci = 0; ci < torqueChIdx.Length; ci++)
                {
                    double sum = 0;
                    for (int t = 0; t < windowSize; t++)
                        sum += window[t * nChannels + torqueChIdx[ci]];
                    axMeans[ci] = sum / windowSize;
                }
            }

            // ── 2) 단일 시간 루프: 모든 특징 동시 계산 ──────────────────────
            double accRmsSumSq  = 0, accPeak = 0;
            var    trqDetSumSq  = hasTorque ? new double[torqueChIdx.Length] : null;
            var    trqSumSq     = hasTorque ? new double[torqueChIdx.Length] : null;
            var    trqMin       = hasTorque ? new double[torqueChIdx.Length] : null;
            var    trqMax       = hasTorque ? new double[torqueChIdx.Length] : null;
            if (hasTorque)
                for (int ci = 0; ci < torqueChIdx.Length; ci++)
                { trqMin[ci] = double.MaxValue; trqMax[ci] = double.MinValue; }

            double zeroCombinedCnt = 0;

            for (int t = 0; t < windowSize; t++)
            {
                int b = t * nChannels;

                // 가속도
                double accMag = 0;
                if (hasAccel)
                {
                    double x = window[b + accelChIdx[0]];
                    double y = window[b + accelChIdx[1]];
                    double z = window[b + accelChIdx[2]];
                    accMag = Math.Sqrt(x * x + y * y + z * z);
                    accRmsSumSq += accMag * accMag;
                    if (accMag > accPeak) accPeak = accMag;
                }

                // 토크 detrended
                double trqDynMag = 0;
                if (hasTorque)
                {
                    double dynSumSq = 0;
                    for (int ci = 0; ci < torqueChIdx.Length; ci++)
                    {
                        double v = window[b + torqueChIdx[ci]];
                        double d = v - axMeans[ci];
                        trqDetSumSq[ci] += d * d;
                        trqSumSq[ci]    += v * v;   // raw sum for std
                        dynSumSq        += d * d;
                        if (v < trqMin[ci]) trqMin[ci] = v;
                        if (v > trqMax[ci]) trqMax[ci] = v;
                    }
                    trqDynMag = Math.Sqrt(dynSumSq);
                }

                // zero_ratio 조건 (두 채널이 모두 있으면 AND 조건)
                bool accNearZero = !hasAccel  || accMag   < thr.AccZeroThreshold;
                bool trqNearZero = !hasTorque || trqDynMag < thr.TrqDynZeroThreshold;
                if (accNearZero && trqNearZero) zeroCombinedCnt++;
            }

            // ── 3) 집계 ────────────────────────────────────────────────────────
            if (hasAccel)
            {
                f.AccMagRms  = Math.Sqrt(accRmsSumSq / windowSize);
                f.AccMagPeak = accPeak;
            }

            if (hasTorque)
            {
                int    n         = torqueChIdx.Length;
                double sumDetRms = 0, sumStd = 0, maxP2P = 0;
                for (int ci = 0; ci < n; ci++)
                {
                    sumDetRms += Math.Sqrt(trqDetSumSq[ci] / windowSize);
                    // std = sqrt(E[x²] - mean²)
                    double mean = axMeans[ci];
                    double varV = trqSumSq[ci] / windowSize - mean * mean;
                    sumStd += Math.Sqrt(varV < 0 ? 0 : varV);
                    double p2p = trqMax[ci] - trqMin[ci];
                    if (p2p > maxP2P) maxP2P = p2p;
                }
                f.TrqDetrendedRms  = sumDetRms / n;
                f.TrqStdMean       = sumStd    / n;
                f.TrqPeakToPeakMax = maxP2P;
            }

            f.ZeroRatio = zeroCombinedCnt / windowSize;

            // ── 4) robust_scale_01 정규화 후 가중합 ───────────────────────────
            double sAccRms  = thr.AccMagRms.Normalize(f.AccMagRms);
            double sAccPeak = thr.AccMagPeak.Normalize(f.AccMagPeak);
            double sTrqRms  = thr.TrqDetrendedRms.Normalize(f.TrqDetrendedRms);
            double sTrqStd  = thr.TrqStdMean.Normalize(f.TrqStdMean);
            double sTrqP2p  = thr.TrqPeakToPeakMax.Normalize(f.TrqPeakToPeakMax);

            f.MotionScore = thr.WeightAccMagRms        * sAccRms
                          + thr.WeightAccMagPeak       * sAccPeak
                          + thr.WeightTrqDetrendedRms  * sTrqRms
                          + thr.WeightTrqStdMean       * sTrqStd
                          + thr.WeightTrqPeakToPeakMax * sTrqP2p;

            return f;
        }

        /// <summary>
        /// motion_score + zero_ratio 기반 윈도우 상태 분류.
        /// zero_ratio 가 ZeroRatioIdleThreshold 초과 시 즉시 Idle.
        /// </summary>
        public static WindowState Classify(WindowFeatures f, WindowStateThresholds thr)
        {
            if (f.ZeroRatio   >= thr.ZeroRatioIdleThreshold) return WindowState.Idle;
            if (f.MotionScore >= thr.MotionScoreThreshold)   return WindowState.Motion;
            if (f.MotionScore <= thr.IdleScoreThreshold)     return WindowState.Idle;
            return WindowState.Ambiguous;
        }

        /// <summary>
        /// sensor_type 문자열에서 채널 역할(accel / torque 인덱스)을 추론합니다.
        ///   "accel"    → accelIdx=[0,1,2],     torqueIdx=[]
        ///   "torque"   → accelIdx=[],           torqueIdx=[0..nCh-1]
        ///   "combined" → accelIdx=[0,1,2],      torqueIdx=[3..nCh-1]
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

        /// <summary>
        /// 서버 /model_info 의 window_state 블록에서 WindowStateThresholds 를 생성합니다.
        /// </summary>
        public static WindowStateThresholds FromServerInfo(WindowStateInfo info)
        {
            if (info == null) return null;

            var thr = new WindowStateThresholds
            {
                MotionScoreThreshold   = info.MotionScoreThreshold,
                IdleScoreThreshold     = info.IdleScoreThreshold,
                AccZeroThreshold       = info.ZeroRatioAccThreshold,
                TrqDynZeroThreshold    = info.ZeroRatioTrqThreshold,
                ZeroRatioIdleThreshold = info.ZeroRatioIdleThreshold,
            };

            // scaler
            if (info.Scaler != null)
            {
                WindowStateScalerEntry e;
                if (info.Scaler.TryGetValue("acc_mag_rms",          out e) && e != null)
                { thr.AccMagRms.Q05        = e.Q05; thr.AccMagRms.Q95        = e.Q95; }
                if (info.Scaler.TryGetValue("acc_mag_peak",         out e) && e != null)
                { thr.AccMagPeak.Q05       = e.Q05; thr.AccMagPeak.Q95       = e.Q95; }
                if (info.Scaler.TryGetValue("trq_detrended_rms",    out e) && e != null)
                { thr.TrqDetrendedRms.Q05  = e.Q05; thr.TrqDetrendedRms.Q95  = e.Q95; }
                if (info.Scaler.TryGetValue("trq_std_mean",         out e) && e != null)
                { thr.TrqStdMean.Q05       = e.Q05; thr.TrqStdMean.Q95       = e.Q95; }
                if (info.Scaler.TryGetValue("trq_peak_to_peak_max", out e) && e != null)
                { thr.TrqPeakToPeakMax.Q05 = e.Q05; thr.TrqPeakToPeakMax.Q95 = e.Q95; }
            }

            // weights
            if (info.Weights != null)
            {
                double w;
                if (info.Weights.TryGetValue("acc_mag_rms",          out w)) thr.WeightAccMagRms        = w;
                if (info.Weights.TryGetValue("acc_mag_peak",         out w)) thr.WeightAccMagPeak       = w;
                if (info.Weights.TryGetValue("trq_detrended_rms",    out w)) thr.WeightTrqDetrendedRms  = w;
                if (info.Weights.TryGetValue("trq_std_mean",         out w)) thr.WeightTrqStdMean       = w;
                if (info.Weights.TryGetValue("trq_peak_to_peak_max", out w)) thr.WeightTrqPeakToPeakMax = w;
            }

            return thr;
        }
    }
}
