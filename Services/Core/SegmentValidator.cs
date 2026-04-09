using System;
using System.Collections.Generic;
using PHM_Project_DockPanel.Services.DAQ;

namespace PHM_Project_DockPanel.Services.Core
{
    /// <summary>
    /// Outlier detection result codes — used for logging and diagnostics.
    /// </summary>
    public enum SegmentRejectReason
    {
        None,
        TooShort,               // SampleCount < MinSamples
        TooLong,                // SampleCount > MaxSamples
        HardLimitExceeded,      // |value| > physical hard limit (ADC overflow, sensor disconnect)
        FlatSignal,             // StdDev < FlatStdDevThresholdG — dead / disconnected sensor
        TooManySpikes           // > MaxSpikeFraction of samples exceed MAD-based spike threshold
    }

    /// <summary>
    /// Central outlier filtering rules applied consistently across all pipeline stages:
    ///   • Ingestion    — AccelInfluxPublisher.Feed() before writing to InfluxDB
    ///   • DB queries   — InfluxDbDataSource.Segmentize() / SegmentizeTorque() after parsing
    ///   • Real-time    — DashboardForm.ProcessInfluxSegment() before scoring
    ///   • CSV scoring  — SignalFeatures.BuildFeatureVectorFromCsv() before feature extraction
    ///
    /// All thresholds are public static fields and can be tuned at startup via AppSettings.
    /// </summary>
    public static class SegmentValidator
    {
        // ── Length thresholds ─────────────────────────────────────────────────

        /// <summary>
        /// Minimum samples for a usable segment.
        /// At 1 kHz this is 50 ms — the shortest window that still produces meaningful FFT features.
        /// </summary>
        public static int MinSamples = 50;

        /// <summary>
        /// Maximum samples per segment (safety cap against runaway Flux queries).
        /// At 1 kHz this is 100 seconds; anything longer is almost certainly a pipeline error.
        /// </summary>
        public static int MaxSamples = 100_000;

        // ── Value-range thresholds ─────────────────────────────────────────────

        /// <summary>
        /// Hard absolute limit for acceleration channels (g).
        /// Most industrial MEMS sensors clip or saturate below ±50 g.
        /// A reading beyond this threshold indicates ADC overflow or a disconnected sensor.
        /// </summary>
        public static double AccelHardLimitG = 50.0;

        /// <summary>
        /// Hard absolute limit for torque channels (%).
        /// Physically possible range is typically 0–100 %; values beyond ±200 % are invalid.
        /// </summary>
        public static double TorqueHardLimitPct = 200.0;

        // ── Signal-quality thresholds ─────────────────────────────────────────

        /// <summary>
        /// Standard deviation below which an acceleration signal is considered flat/dead.
        /// Even a lightly vibrating axis will exceed 0.001 g RMS due to sensor noise floor.
        /// </summary>
        public static double FlatStdDevThresholdG = 0.001;

        /// <summary>
        /// Spike multiplier for MAD-based detection.
        /// A sample is flagged as a spike if |value − median| > SpikeMADMultiplier × MAD.
        /// 15 × MAD ≈ 22 σ for Gaussian data — only catches extreme impulse artifacts.
        /// </summary>
        public static double SpikeMADMultiplier = 15.0;

        /// <summary>
        /// Maximum allowed fraction of spike samples in a segment.
        /// Segments where > 5 % of samples are spikes are rejected.
        /// </summary>
        public static double MaxSpikeFraction = 0.05;

        // ── Public validation API ─────────────────────────────────────────────

        /// <summary>
        /// Full validation of a <see cref="SignalSegment"/>.
        /// Checks: length, hard limits per channel, flat-signal (accel only), spike ratio.
        /// Returns <c>true</c> if the segment is safe to use for feature extraction and inference.
        /// </summary>
        public static bool IsValid(SignalSegment seg, out SegmentRejectReason reason)
        {
            reason = SegmentRejectReason.None;
            if (seg == null) { reason = SegmentRejectReason.TooShort; return false; }

            int n = seg.SampleCount;
            if (n < MinSamples) { reason = SegmentRejectReason.TooShort; return false; }
            if (n > MaxSamples) { reason = SegmentRejectReason.TooLong;  return false; }

            // Acceleration channels — hard limit + flat signal + spike ratio
            if (seg.X != null && seg.X.Length > 0)
                if (!CheckChannel(seg.X, AccelHardLimitG, checkFlat: true, out reason)) return false;
            if (seg.Y != null && seg.Y.Length > 0)
                if (!CheckChannel(seg.Y, AccelHardLimitG, checkFlat: true, out reason)) return false;
            if (seg.Z != null && seg.Z.Length > 0)
                if (!CheckChannel(seg.Z, AccelHardLimitG, checkFlat: true, out reason)) return false;

            // Torque channel — hard limit + spike ratio (flat check skipped: zero torque is valid)
            if (seg.Torque != null && seg.Torque.Length > 0)
                if (!CheckChannel(seg.Torque, TorqueHardLimitPct, checkFlat: false, out reason)) return false;

            return true;
        }

        /// <summary>
        /// Validates a single channel array before feature extraction (e.g., parsed from CSV).
        /// Applies the same length + hard-limit + flat-signal + spike-ratio rules.
        /// </summary>
        /// <param name="arr">Raw sample array.</param>
        /// <param name="isTorque">
        ///   <c>true</c> to use <see cref="TorqueHardLimitPct"/> and skip the flat-signal check.
        ///   <c>false</c> to use <see cref="AccelHardLimitG"/> with flat-signal check enabled.
        /// </param>
        public static bool IsValidChannel(IList<double> arr, bool isTorque, out SegmentRejectReason reason)
        {
            reason = SegmentRejectReason.None;
            if (arr == null || arr.Count < MinSamples) { reason = SegmentRejectReason.TooShort; return false; }
            if (arr.Count > MaxSamples)                { reason = SegmentRejectReason.TooLong;  return false; }
            double limit = isTorque ? TorqueHardLimitPct : AccelHardLimitG;
            return CheckChannel(arr, limit, checkFlat: !isTorque, out reason);
        }

        /// <summary>
        /// Lightweight ingestion check for real-time write paths (no spike-ratio computation).
        /// Only checks hard limits and flat signal — designed for low-latency use in Feed().
        /// </summary>
        public static bool IsValidForIngestion(double[] arr, bool isTorque, out SegmentRejectReason reason)
        {
            reason = SegmentRejectReason.None;
            if (arr == null || arr.Length == 0) { reason = SegmentRejectReason.TooShort; return false; }
            double limit = isTorque ? TorqueHardLimitPct : AccelHardLimitG;

            foreach (var v in arr)
            {
                if (double.IsNaN(v) || double.IsInfinity(v) || Math.Abs(v) > limit)
                { reason = SegmentRejectReason.HardLimitExceeded; return false; }
            }

            if (!isTorque && StdDev(arr, arr.Length) < FlatStdDevThresholdG)
            { reason = SegmentRejectReason.FlatSignal; return false; }

            return true;
        }

        // ── Private helpers ───────────────────────────────────────────────────

        private static bool CheckChannel(IList<double> arr, double hardLimit, bool checkFlat,
                                         out SegmentRejectReason reason)
        {
            reason = SegmentRejectReason.None;
            int n = arr.Count;

            // 1. NaN / Inf / hard limit
            foreach (var v in arr)
            {
                if (double.IsNaN(v) || double.IsInfinity(v) || Math.Abs(v) > hardLimit)
                { reason = SegmentRejectReason.HardLimitExceeded; return false; }
            }

            // 2. Flat signal (accel only)
            double mean = Mean(arr, n);
            if (checkFlat)
            {
                double sd = StdDevFromMean(arr, n, mean);
                if (sd < FlatStdDevThresholdG) { reason = SegmentRejectReason.FlatSignal; return false; }
            }

            // 3. Spike ratio via MAD
            double[] sorted = ToSortedArray(arr, n);
            double median = MedianOfSorted(sorted);
            double mad    = MadOfSorted(sorted, median);
            if (mad < 1e-12) return true;   // all values equal — uniform signal (caught by flat above)

            double spikeThreshold = SpikeMADMultiplier * mad;
            int spikeCount = 0;
            foreach (var v in arr)
                if (Math.Abs(v - median) > spikeThreshold) spikeCount++;

            if ((double)spikeCount / n > MaxSpikeFraction)
            { reason = SegmentRejectReason.TooManySpikes; return false; }

            return true;
        }

        private static double Mean(IList<double> arr, int n)
        {
            double sum = 0;
            foreach (var v in arr) sum += v;
            return sum / n;
        }

        private static double StdDev(double[] arr, int n)
        {
            if (n < 2) return 0;
            double mean = 0;
            foreach (var v in arr) mean += v;
            mean /= n;
            return StdDevFromMean(arr, n, mean);
        }

        private static double StdDevFromMean(IList<double> arr, int n, double mean)
        {
            if (n < 2) return 0;
            double sum = 0;
            foreach (var v in arr) { double d = v - mean; sum += d * d; }
            return Math.Sqrt(sum / n);
        }

        private static double[] ToSortedArray(IList<double> arr, int n)
        {
            var copy = new double[n];
            for (int i = 0; i < n; i++) copy[i] = arr[i];
            Array.Sort(copy);
            return copy;
        }

        private static double MedianOfSorted(double[] sorted)
        {
            int n = sorted.Length;
            return (n % 2 == 0)
                ? (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
                : sorted[n / 2];
        }

        private static double MadOfSorted(double[] sorted, double median)
        {
            int n = sorted.Length;
            var devs = new double[n];
            for (int i = 0; i < n; i++) devs[i] = Math.Abs(sorted[i] - median);
            Array.Sort(devs);
            return MedianOfSorted(devs);
        }
    }
}
