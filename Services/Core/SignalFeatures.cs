using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;

namespace PHM_Project_DockPanel.Services.Core
{
    /// <summary>
    /// 신호 기반 특징 추출/스코어링 유틸 (학습/추론 공용)
    /// </summary>
    public static class SignalFeatures
    {
        // ---- 공통 상수/도우미 ----
        private static readonly char[] Delimiters = new[] { ',', ';', '\t' };
        private static readonly IFormatProvider Invariant = CultureInfo.InvariantCulture;

        public class FeatureRow
        {
            public string FileName { get; set; }
            /// <summary>InfluxDB 세그먼트에서 가져온 레이블 (CSV 모드에서는 빈 문자열)</summary>
            public string Label { get; set; } = "";
            public double AbsMax { get; set; }
            public double AbsMean { get; set; }
            public double P2P { get; set; }
            public double RMS { get; set; }
            public double Skewness { get; set; }
            public double Kurtosis { get; set; }
            public double Crest { get; set; }
            public double Shape { get; set; }
            public double Impulse { get; set; }
            public double Peak1Freq { get; set; }
            public double Peak1Amp { get; set; }
            public double Peak2Freq { get; set; }
            public double Peak2Amp { get; set; }
            public double Peak3Freq { get; set; }
            public double Peak3Amp { get; set; }
            public double Peak4Freq { get; set; }
            public double Peak4Amp { get; set; }
        }

        // ---------- CSV 유틸 ----------
        public static string[] GetCsvHeaders(string filePath)
        {
            try
            {
                return RetryIo(() =>
                {
                    using (var sr = OpenSharedReader(filePath))
                    {
                        var line = sr.ReadLine();
                        if (string.IsNullOrWhiteSpace(line)) return null;
                        return line.Split(Delimiters, StringSplitOptions.None);
                    }
                });
            }
            catch
            {
                return null;
            }
        }

        private static StreamReader OpenSharedReader(string path)
        {
            var fs = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete,   // ★ 다른 프로세스가 쓰는 중이어도 읽기 허용
                4096,
                FileOptions.SequentialScan);              // 순차 읽기 힌트
            return new StreamReader(fs, detectEncodingFromByteOrderMarks: true);
        }

        private static T RetryIo<T>(Func<T> action, int maxTry = 5, int delayMs = 150)
        {
            int attempt = 0;
            for (; ; )
            {
                try { return action(); }
                catch (IOException)
                {
                    attempt++;
                    if (attempt >= maxTry) throw;
                    System.Threading.Thread.Sleep(delayMs);
                }
            }
        }

        public static bool IsTimeColumn(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return false;
            name = name.Trim();
            return name.Equals("CYCLE", StringComparison.OrdinalIgnoreCase) ||
                   name.Equals("time_s", StringComparison.OrdinalIgnoreCase);
        }

        public static int FindTimeColumnIndex(string[] headers)
        {
            if (headers == null) return -1;
            for (int i = 0; i < headers.Length; i++)
                if (IsTimeColumn(headers[i])) return i;
            return -1;
        }

        /// <summary>
        /// CSV에서 특정 y컬럼 시계열을 파싱합니다. (시간축은 있어도 무시)
        /// </summary>
        public static bool TryParseCsvColumn(string filePath, string yColumn, out List<double> ys)
        {
            ys = null;
            try
            {
                var headers = GetCsvHeaders(filePath);
                if (headers == null) return false;

                int yIndex = Array.FindIndex(headers, h => h.Equals(yColumn, StringComparison.OrdinalIgnoreCase));
                if (yIndex < 0) return false;

                var list = new List<double>(4096);
                using (var sr = new StreamReader(filePath))
                {
                    sr.ReadLine(); // skip header
                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        var parts = line.Split(Delimiters, StringSplitOptions.None);
                        if (parts.Length <= yIndex) continue;
                        if (double.TryParse(parts[yIndex], NumberStyles.Float, Invariant, out double v))
                            list.Add(v);
                    }
                }
                if (list.Count < 4) return false;
                ys = list;
                return true;
            }
            catch
            {
                return false;
            }
        }

        // ---------- 특징 추출 ----------
        /// <summary>
        /// 타임도메인 특징 계산
        /// </summary>
        public static void FillTimeDomainFeatures(IList<double> y, FeatureRow row)
        {
            int n = y.Count;
            double absMax = 0, absSum = 0, mean = 0, rmsSum = 0;
            double yMin = double.MaxValue, yMax = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                double v = y[i];
                double av = (v >= 0) ? v : -v;
                if (av > absMax) absMax = av;
                absSum += av;
                mean += v;
                rmsSum += v * v;

                if (v < yMin) yMin = v;
                if (v > yMax) yMax = v;
            }
            mean /= n;
            double absMean = absSum / n;
            double rms = Math.Sqrt(rmsSum / n);
            double p2p = yMax - yMin;

            // 중앙 모멘트
            double m2 = 0, m3 = 0, m4 = 0;
            for (int i = 0; i < n; i++)
            {
                double d = y[i] - mean;
                double d2 = d * d;
                m2 += d2;
                m3 += d2 * d;
                m4 += d2 * d2;
            }
            m2 /= n; m3 /= n; m4 /= n;

            double std = Math.Sqrt(m2);
            double skew = (std > 0) ? (m3 / (std * std * std)) : 0.0;
            double kurt = (m2 > 0) ? (m4 / (m2 * m2)) : 0.0;

            // 비율형 지표
            double crest = (rms > 0) ? (absMax / rms) : double.NaN;
            double shape = (absMean > 0) ? (rms / absMean) : double.NaN;
            double impulse = (absMean > 0) ? (absMax / absMean) : double.NaN;

            row.AbsMax = absMax; row.AbsMean = absMean; row.P2P = p2p; row.RMS = rms;
            row.Skewness = skew; row.Kurtosis = kurt; row.Crest = crest; row.Shape = shape; row.Impulse = impulse;
        }

        /// <summary>
        /// 크기 스펙트럼(0~Nyquist)과 주파수 벡터를 계산합니다.
        /// </summary>
        public static double[] ComputeMagnitudeSpectrum(double[] y, double fs, out double[] freq)
        {
            int n0 = y.Length;
            int N = NextPow2(n0);
            var buf = new Complex[N];

            // Hann window + zero padding
            for (int i = 0; i < n0; i++)
            {
                double w = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (n0 - 1)));
                buf[i] = new Complex(y[i] * w, 0);
            }
            for (int i = n0; i < N; i++) buf[i] = Complex.Zero;

            FFT(buf);

            int bins = N / 2;
            var mag = new double[bins];
            freq = new double[bins];
            double norm = 2.0 / N;
            for (int k = 0; k < bins; k++)
            {
                mag[k] = buf[k].Magnitude * norm;
                freq[k] = k * fs / N;
            }
            return mag;
        }

        /// <summary>
        /// 상위 4개 peak (≈1Hz 최소 간격) 특징 추출
        /// </summary>
        public static void FillPeakFeatures(double[] freq, double[] mag, FeatureRow row)
        {
            int n = (freq == null || mag == null) ? 0 : Math.Min(freq.Length, mag.Length);
            if (n < 8)
            {
                row.Peak1Freq = row.Peak2Freq = row.Peak3Freq = row.Peak4Freq = double.NaN;
                row.Peak1Amp = row.Peak2Amp = row.Peak3Amp = row.Peak4Amp = double.NaN;
                return;
            }

            double df = freq[1] - freq[0];
            if (df <= 0) df = 1; // 안전장치
            int minSep = Math.Max(2, (int)Math.Round(1.0 / df)); // ~1Hz 최소 간격

            // local maxima 후보 수집
            var cand = new List<(int idx, double amp)>(n / 4);
            for (int i = 1; i < n - 1; i++)
            {
                if (freq[i] <= 1e-9) continue; // DC 근처 제외
                double mi = mag[i];
                if (mi > mag[i - 1] && mi >= mag[i + 1]) cand.Add((i, mi));
            }
            if (cand.Count == 0)
            {
                row.Peak1Freq = row.Peak2Freq = row.Peak3Freq = row.Peak4Freq = double.NaN;
                row.Peak1Amp = row.Peak2Amp = row.Peak3Amp = row.Peak4Amp = double.NaN;
                return;
            }

            cand.Sort((a, b) => b.amp.CompareTo(a.amp));
            var picked = new List<int>(4);
            foreach (var (idx, _) in cand)
            {
                if (picked.Count >= 4) break;
                bool tooClose = false;
                for (int j = 0; j < picked.Count; j++)
                    if (Math.Abs(picked[j] - idx) < minSep) { tooClose = true; break; }
                if (!tooClose) picked.Add(idx);
            }
            picked.Sort();

            double[] pf = new double[4], pa = new double[4];
            for (int k = 0; k < 4; k++)
            {
                if (k < picked.Count) { pf[k] = freq[picked[k]]; pa[k] = mag[picked[k]]; }
                else { pf[k] = double.NaN; pa[k] = double.NaN; }
            }
            row.Peak1Freq = pf[0]; row.Peak1Amp = pa[0];
            row.Peak2Freq = pf[1]; row.Peak2Amp = pa[1];
            row.Peak3Freq = pf[2]; row.Peak3Amp = pa[2];
            row.Peak4Freq = pf[3]; row.Peak4Amp = pa[3];
        }

        /// <summary>
        /// 시계열에서 FeatureRow를 추출(타임 + 주파수 피처)
        /// </summary>
        public static FeatureRow ExtractFeatures(IList<double> y, double sampleRate, int maxSamples = 16384)
        {
            if (y == null || y.Count < 4) return null;

            var fr = new FeatureRow();
            FillTimeDomainFeatures(y, fr);

            int avail = Math.Min(y.Count, Math.Max(8, maxSamples));
            var yarr = new double[avail];
            for (int i = 0; i < avail; i++) yarr[i] = y[i];

            var mag = ComputeMagnitudeSpectrum(yarr, sampleRate, out var freq);
            FillPeakFeatures(freq, mag, fr);
            return fr;
        }

        /// <summary>
        /// 지정된 feature key 배열 순서대로 특징 벡터 생성 (표준화 없음)
        /// </summary>
        public static double[] BuildFeatureVectorFromSeries(IList<double> y, string[] featureKeys, double sampleRate)
        {
            if (featureKeys == null || featureKeys.Length == 0) return null;
            var fr = ExtractFeatures(y, sampleRate);
            if (fr == null) return null;

            var fv = new double[featureKeys.Length];
            for (int j = 0; j < featureKeys.Length; j++)
            {
                var v = GetFeatureValue(fr, featureKeys[j]);
                if (!v.HasValue || double.IsNaN(v.Value) || double.IsInfinity(v.Value)) return null;
                fv[j] = v.Value;
            }
            return fv;
        }

        /// <summary>
        /// CSV 파일의 time_s 컬럼 간격으로 실제 샘플레이트를 감지합니다.
        /// time_s 컬럼이 없거나 간격이 비정상이면 fallbackSr를 반환합니다.
        /// </summary>
        public static double DetectSampleRateFromCsv(string filePath, double fallbackSr)
        {
            try
            {
                return RetryIo(() =>
                {
                    using (var reader = OpenSharedReader(filePath))
                    {
                        string headerLine = reader.ReadLine();
                        if (headerLine == null) return fallbackSr;
                        var headers = headerLine.Split(Delimiters, StringSplitOptions.None);

                        // time_s 컬럼만 사용 (CYCLE 등 다른 시간 컬럼은 제외)
                        int timeIdx = -1;
                        for (int i = 0; i < headers.Length; i++)
                            if (headers[i].Trim().Equals("time_s", StringComparison.OrdinalIgnoreCase))
                            { timeIdx = i; break; }
                        if (timeIdx < 0) return fallbackSr;

                        string line1 = reader.ReadLine();
                        string line2 = reader.ReadLine();
                        if (line1 == null || line2 == null) return fallbackSr;

                        var p1 = line1.Split(Delimiters, StringSplitOptions.None);
                        var p2 = line2.Split(Delimiters, StringSplitOptions.None);
                        if (p1.Length <= timeIdx || p2.Length <= timeIdx) return fallbackSr;

                        double t1, t2;
                        if (!double.TryParse(p1[timeIdx], NumberStyles.Float, Invariant, out t1)) return fallbackSr;
                        if (!double.TryParse(p2[timeIdx], NumberStyles.Float, Invariant, out t2)) return fallbackSr;

                        double dt = t2 - t1;
                        // 1 μs ~ 1 s 범위만 유효 (sr: 1 Hz ~ 1 MHz)
                        if (dt <= 0 || dt > 1.0) return fallbackSr;
                        return Math.Round(1.0 / dt);
                    }
                });
            }
            catch
            {
                return fallbackSr;
            }
        }

        /// <summary>
        /// CSV 파일에서 y컬럼을 읽어 즉시 특징 벡터 생성 (표준화 없음)
        /// time_s 컬럼이 있으면 실제 샘플레이트를 자동 감지합니다.
        /// </summary>
        public static double[] BuildFeatureVectorFromCsv(string filePath, string yColumn, string[] featureKeysInOrder)
        {
            if (featureKeysInOrder == null || featureKeysInOrder.Length == 0) return null;
            if (!TryParseCsvColumn(filePath, yColumn, out var ys)) return null;

            double fallbackSr = AppState.GetForColumn(yColumn);
            double sr = DetectSampleRateFromCsv(filePath, fallbackSr);

            return BuildFeatureVectorFromSeries(ys, featureKeysInOrder, sr);
        }

        public static double? GetFeatureValue(FeatureRow r, string key)
        {
            switch (key)
            {
                case "AbsMax": return r.AbsMax;
                case "AbsMean": return r.AbsMean;
                case "P2P": return r.P2P;
                case "RMS": return r.RMS;
                case "Skewness": return r.Skewness;
                case "Kurtosis": return r.Kurtosis;
                case "Crest": return r.Crest;
                case "Shape": return r.Shape;
                case "Impulse": return r.Impulse;
                case "Peak1Freq": return r.Peak1Freq;
                case "Peak1Amp": return r.Peak1Amp;
                case "Peak2Freq": return r.Peak2Freq;
                case "Peak2Amp": return r.Peak2Amp;
                case "Peak3Freq": return r.Peak3Freq;
                case "Peak3Amp": return r.Peak3Amp;
                case "Peak4Freq": return r.Peak4Freq;
                case "Peak4Amp": return r.Peak4Amp;
                default: return null;
            }
        }

        // ---------- 표준화 ----------
        public static double[] Standardize(double[] v, double[] mean, double[] std)
        {
            if (v == null) return null;
            if (mean == null || std == null) return (double[])v.Clone();

            int d = v.Length;
            var r = new double[d];
            for (int j = 0; j < d; j++)
            {
                double mu = (j < mean.Length) ? mean[j] : 0.0;
                double sd = (j < std.Length && std[j] != 0) ? std[j] : 1.0;
                r[j] = (v[j] - mu) / sd;
            }
            return r;
        }

        public static double ScoreKnn(
            double[] rawX,         // 비표준화 특징 벡터
            double[][] train,      // 학습 벡터 (비표준화)
            int k,
            bool standardize,      // true면 mean/std 기준으로 동일 척도화
            double[] mean = null,
            double[] std = null,
            bool leaveOneOut = false)  // 학습 샘플 자체 평가 시 자기 자신 제외 옵션
        {
            if (rawX == null || train == null || train.Length == 0) return double.NaN;

            var dists = new List<double>(train.Length);

            for (int i = 0; i < train.Length; i++)
            {
                var t = train[i];
                if (t == null || t.Length != rawX.Length) return double.NaN;

                if (leaveOneOut && ReferenceEquals(rawX, t))
                    continue;

                // 공통 거리 함수 사용
                dists.Add(EuclidDist(rawX, t, standardize, mean, std));
            }

            if (dists.Count == 0) return double.NaN;

            dists.Sort();
            int kEff = Math.Min(Math.Max(k, 1), dists.Count);

            double sum = 0;
            for (int i = 0; i < kEff; i++) sum += dists[i];
            return sum / kEff; // ★ 학습 Score()와 동일
        }

        // ===== 내부 거리(표준화 고려) =====
        private static double EuclidDist(double[] a, double[] b, bool useStd, double[] mean, double[] std)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double ax = a[i], bx = b[i];
                if (useStd && std != null && mean != null)
                {
                    double denom = (std[i] == 0 ? 1 : std[i]);
                    ax = (ax - mean[i]) / denom;
                    bx = (bx - mean[i]) / denom;
                }
                double d = ax - bx;
                s += d * d;
            }
            return Math.Sqrt(s);
        }
        // ---------- 내부 FFT/수학 유틸 ----------
        private static int NextPow2(int n)
        {
            int p = 1; while (p < n) p <<= 1; return p;
        }

        private static void FFT(Complex[] a)
        {
            int n = a.Length, j = 0;
            for (int i = 1; i < n; i++)
            {
                int bit = n >> 1;
                for (; (j & bit) != 0; bit >>= 1) j &= ~bit;
                j |= bit;
                if (i < j) { var t = a[i]; a[i] = a[j]; a[j] = t; }
            }
            for (int len = 2; len <= n; len <<= 1)
            {
                double ang = -2 * Math.PI / len;
                var wlen = new Complex(Math.Cos(ang), Math.Sin(ang));
                for (int i = 0; i < n; i += len)
                {
                    var w = Complex.One; int half = len >> 1;
                    for (int k = 0; k < half; k++)
                    {
                        var u = a[i + k];
                        var v = w * a[i + k + half];
                        a[i + k] = u + v;
                        a[i + k + half] = u - v;
                        w *= wlen;
                    }
                }
            }
        }

        private static double SqDist(double[] a, double[] b)
        {
            double s = 0;
            for (int j = 0; j < a.Length; j++)
            {
                double d = a[j] - b[j];
                s += d * d;
            }
            return s;
        }
    }
}
