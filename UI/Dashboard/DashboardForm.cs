using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using WeifenLuo.WinFormsUI.Docking;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// PHM 프로젝트 내부 기능
using PHM_Project_DockPanel.Services.Core;
using PHM_Project_DockPanel.Services; // SignalFeatures
using PHM_Project_DockPanel.Services.DAQ;

namespace PHM_Project_DockPanel.UI.Dashboard
{
    /// <summary>
    /// InferenceForm를 완전히 대체하는 C# 7.3 호환 실시간 대시보드 폼.
    /// - 모델 로드(축별)
    /// - CSV 폴더 감시 및 스코어링
    /// - KPI 카드/라인차트/도넛차트/이벤트 테이블 표시
    /// </summary>
    public class DashboardForm : DockContent
    {
        #region 모델 포맷
        private class PersistedKnnModel
        {
            public string ModelType { get; set; }        // "KNN" or "KNN_AD"
            public int K { get; set; }
            public bool Standardize { get; set; }
            public string[] Features { get; set; }       // Feature keys in order
            public double Threshold { get; set; }        // 축별 임계값 (<=0이면 기본값 사용)
            public double[] Mean { get; set; }           // null if !Standardize
            public double[] Std { get; set; }            // null if !Standardize
            public double[][] Train { get; set; }        // NxD training vectors (RAW)
            public string YColumn { get; set; }          // 특징 추출에 사용한 Y 컬럼명
        }

        private class AxisModel
        {
            public int AxisId { get; set; }
            public string ModelPath { get; set; }
            public PersistedKnnModel Model { get; set; }
        }
        #endregion

        private struct AxisCol { public int AxisId; public int ColIndex; }

        #region 상수/필드
        private const bool WatchSubdirectories = true;
        private const int ChartKeepPoints = 3000;  // ~6분 분량 (7.8 pt/s × 360s)
        private double MotionEps { get; set; } = 0.010;  // 움직임 판정(Δpos) 기본값
        private double DefaultThreshold { get; set; } = 1.000; // 기본 임계값(필요 시)
        private const string DefaultLogsPath = @"C:\Data\PHM_Logs";
        private const int SampleMinTileWidth = 320;   // 한 타일의 최소 가로폭(px)
        private const int SampleMaxColumns = 3;     // 최대 열 수(원하면 4 등으로 조절)

        private static readonly Regex[] AxisPosRegexes =
        {
            new Regex("^CMDPOS(?<id>\\d+)$", RegexOptions.IgnoreCase | RegexOptions.Compiled),
            new Regex("^POS(?<id>\\d+)$",    RegexOptions.IgnoreCase | RegexOptions.Compiled),
        };

        #region ONNX Inference
        private class OnnxAxisModel
        {
            public int AxisId;
            public string ModelPath;
            public string Kind;                 // "LSTM", "CNN1D", "AE-CNN1D" 등 (표시용)
            public string YColumn;              // CSV Y 컬럼 (예: FBTRQ0)
            public int C = 1;                   // 입력 채널 수
            public string InputName = "input";
            public string OutputName = "logits"; // 분류기일 때
            public bool StandardizePerSample = true;

            public bool IsAutoencoder;          // AE 여부
            public string ReconOutputName = "recon"; // (T,C) 또는 (1,T,C)
            public double Threshold = 1.0;      // AE 스코어 임계값 (없으면 DashboardForm.DefaultThreshold)
            public double   RmsMean    = -1;    // AE 정상 데이터 RMS 평균 (-1=없음)
            public double   RmsThr     = -1;    // AE RMS 임계값 (mean+2σ)
            public int      WindowSize = 256;   // AE 학습 윈도우 크기 — 추론 시 슬라이딩 윈도우에 사용
            public InferenceSession Session;
        }


        #region ===== High-pass Filter with cutoff/order =====
        private static class DspUtils
        {
            /// <summary>
            /// seq: (T x C) 에 대해 채널별 Butterworth IIR 계수(b,a)를 사용하여
            /// zero-phase(=filtfilt) 필터링을 수행. In-place.
            /// </summary>
            public static void HighpassFilterInPlace(float[,] seq, double[] b, double[] a)
            {
                if (seq == null) throw new ArgumentNullException(nameof(seq));
                if (b == null || a == null) throw new ArgumentNullException("b/a");
                if (a.Length == 0 || Math.Abs(a[0]) < 1e-20) throw new ArgumentException("a[0] must be non-zero");

                int T = seq.GetLength(0);
                int C = seq.GetLength(1);
                if (T <= 2) return;

                // a[0] == 1로 정규화
                var bb = (double[])b.Clone();
                var aa = (double[])a.Clone();
                if (Math.Abs(aa[0] - 1.0) > 1e-12)
                {
                    for (int i = 0; i < bb.Length; i++) bb[i] /= aa[0];
                    for (int i = 1; i < aa.Length; i++) aa[i] /= aa[0];
                    aa[0] = 1.0;
                }

                int nfilt = Math.Max(bb.Length, aa.Length);
                int padlen = 3 * (nfilt - 1);
                if (padlen < 0) padlen = 0;
                if (T <= padlen + 1) padlen = Math.Max(0, T - 2);

                var tmp = new double[T + 2 * padlen];
                var buf = new double[T];

                for (int c = 0; c < C; c++)
                {
                    // 입력 -> buf
                    for (int t = 0; t < T; t++) buf[t] = seq[t, c];

                    // odd reflection padding
                    if (padlen > 0)
                    {
                        Array.Copy(buf, 0, tmp, padlen, T);
                        for (int i = 0; i < padlen; i++)
                            tmp[padlen - 1 - i] = 2 * buf[0] - buf[i + 1];
                        for (int i = 0; i < padlen; i++)
                            tmp[padlen + T + i] = 2 * buf[T - 1] - buf[T - 2 - i];
                    }
                    else
                    {
                        Array.Copy(buf, tmp, T);
                    }

                    // forward IIR
                    IirDf2tInPlace(tmp, bb, aa);
                    // backward IIR
                    Array.Reverse(tmp);
                    IirDf2tInPlace(tmp, bb, aa);
                    Array.Reverse(tmp);

                    // pad 제거 -> seq
                    if (padlen > 0)
                    {
                        for (int t = 0; t < T; t++)
                            seq[t, c] = (float)tmp[padlen + t];
                    }
                    else
                    {
                        for (int t = 0; t < T; t++)
                            seq[t, c] = (float)tmp[t];
                    }
                }
            }

            /// <summary>
            /// Direct Form II Transposed IIR (a[0]=1 가정). 입력 배열을 제자리에서 필터링.
            /// </summary>
            private static void IirDf2tInPlace(double[] x, double[] b, double[] a)
            {
                int nb = b.Length;
                int na = a.Length;
                int M = Math.Max(nb, na) - 1;
                if (M <= 0) return;

                var z = new double[M]; // 상태: z[0..M-1]

                for (int n = 0; n < x.Length; n++)
                {
                    double w = x[n] + z[0];
                    double y = b[0] * w;

                    for (int i = 0; i < M - 1; i++)
                    {
                        double bi1 = (i + 1 < nb) ? b[i + 1] : 0.0;
                        double ai1 = (i + 1 < na) ? a[i + 1] : 0.0;
                        double zi = z[i + 1] + bi1 * w - ai1 * y;
                        z[i] = zi;
                    }

                    {
                        double bM = (M < nb) ? b[M] : 0.0;
                        double aM = (M < na) ? a[M] : 0.0;
                        z[M - 1] = bM * w - aM * y;
                    }

                    x[n] = y;
                }
            }
        }

        public static void HighpassFilterInPlace(float[,] seq, double cutoff, double fs, int order)
        {
            // 1) 설계 (Butterworth high-pass)
            (double[] b, double[] a) = DesignButterworthHighpass(order, cutoff, fs);

            // 2) 적용 (filtfilt)
            DspUtils.HighpassFilterInPlace(seq, b, a);
        }

        /// <summary>
        /// Bilinear transform으로 Butterworth high-pass filter 계수를 설계
        /// (scipy.signal.butter(order, Wn, 'high')와 동일한 결과)
        /// </summary>
        private static (double[] b, double[] a) DesignButterworthHighpass(int order, double cutoff, double fs)
        {
            // normalize cutoff to Nyquist
            double Wn = cutoff / (fs / 2.0);

            // 아날로그 저역통과 프로토타입 폴 생성
            var poles = new List<System.Numerics.Complex>();
            for (int k = 0; k < order; k++)
            {
                double theta = Math.PI * (2.0 * k + 1 + order) / (2.0 * order);
                poles.Add(System.Numerics.Complex.Exp(System.Numerics.Complex.ImaginaryOne * theta));
            }

            // 저역통과 -> 고역통과 변환 (s -> wc/s)
            double wc = Math.Tan(Math.PI * Wn / 2.0); // pre-warp
            for (int i = 0; i < poles.Count; i++)
                poles[i] = wc / poles[i];

            // z-plane 변환 (bilinear: s = (1-z^-1)/(1+z^-1))
            var pz = new List<System.Numerics.Complex>();
            foreach (var p in poles)
            {
                var num = 1.0 + p / 2.0;
                var den = 1.0 - p / 2.0;
                pz.Add(num / den);
            }

            // 다항식 계수 구하기 (분모)
            double[] a = PolyFromRoots(pz.ToArray());

            // 분자 계수 b: high-pass 특성에 맞게 alternating sign
            double[] b = new double[a.Length];
            for (int i = 0; i < b.Length; i++)
                b[i] = (i % 2 == 0 ? 1 : -1);

            // gain normalize (a[0]=1)
            for (int i = 0; i < b.Length; i++) b[i] /= a[0];
            for (int i = 0; i < a.Length; i++) a[i] /= a[0];
            return (b, a);
        }

        /// <summary>
        /// 폴 배열 -> 다항식 계수
        /// </summary>
        private static double[] PolyFromRoots(System.Numerics.Complex[] roots)
        {
            // 시작: 1
            var coeffs = new System.Numerics.Complex[roots.Length + 1];
            coeffs[0] = System.Numerics.Complex.One;

            foreach (var r in roots)
            {
                // 현재 coeffs * (1 - r z^-1)
                var next = new System.Numerics.Complex[coeffs.Length];
                for (int i = 0; i < coeffs.Length - 1; i++)
                {
                    next[i] += coeffs[i];            // * 1
                    next[i + 1] -= coeffs[i] * r;        // * (-r)
                }
                coeffs = next;
            }

            // 결과는 실수여야 함(켤레쌍 보장). 수치 오차만 허용.
            var real = new double[coeffs.Length];
            for (int i = 0; i < coeffs.Length; i++)
                real[i] = coeffs[i].Real; // Imag는 ~1e-12 수준이어야 함

            return real;
        }
        #endregion

        private static double MeanAbsoluteError(float[,] a, float[,] b)
        {
            int T = a.GetLength(0), C = a.GetLength(1);
            if (b.GetLength(0) != T || b.GetLength(1) != C)
                return double.PositiveInfinity;

            double s = 0;
            long n = 0;
            for (int t = 0; t < T; t++)
            {
                for (int c = 0; c < C; c++)
                {
                    double d = a[t, c] - b[t, c];
                    s += Math.Abs(d);
                    n++;
                }
            }
            return (n > 0) ? s / n : double.PositiveInfinity;
        }

        // 기존: AE/일반 통합
        private readonly Dictionary<int, OnnxAxisModel> _axisOnnx = new Dictionary<int, OnnxAxisModel>();

        // 추가: 축별 "분류" ONNX
        private readonly Dictionary<int, OnnxAxisModel> _axisOnnxCls = new Dictionary<int, OnnxAxisModel>();

        // sklearn 피처 기반 ONNX (AIForm에서 학습·저장한 모델)
        private class OnnxSklModel
        {
            public int AxisId;
            public string ModelPath;
            public string Session;            // "AD" or "FD"
            public string ModelType;          // "knn", "isoforest", "ocsvm", "svm", "rf", "gbm", "mlp"
            public string[] Features;         // 피처 키 목록 (추출 순서)
            public string YColumn;            // CSV Y 컬럼명
            public double Threshold;          // C# kNN 임계값
            public double ScoreThreshold;     // decision_function 기반 임계값 (0이면 미산출 → label만 사용)
            public string[] ClassNames;       // FD 클래스명
            public InferenceSession OnnxSession;
            // knn AD 전용: C# kNN 거리 스코어링 (AI Form 평가와 동일한 값)
            public double[][] TrainVectors;   // 학습 벡터 (raw, 표준화 전)
            public int K = 5;
            public bool Standardize;
            public double[] Mean;
            public double[] Std;
        }
        private readonly Dictionary<int, OnnxSklModel> _axisSklModels = new Dictionary<int, OnnxSklModel>();
        #endregion

        // UI
        private KpiCard cardDanger, cardWarning, cardCycles;
        private Chart chartLine;
        private DataGridView grid;
        private Button btnLoadSklModel, btnLoadOnnxModelSingle, btnLoadModelFolder, btnSelectFolder, btnQuickFolder, btnStart, btnStop;
        private Label lblFolder, lblStatus;
        private static readonly string MruFile = Path.Combine(DefaultLogsPath, "recent_watch_folders.txt");
        private const int MruMaxCount = 5;
        private DataGridView gridModelPaths;
        private TableLayoutPanel sampleGrid;                       // rightBottom 안에서 그리드 역할
        private readonly Dictionary<int, Chart> sampleCharts =     // 축별 Chart 캐시
            new Dictionary<int, Chart>();
        private TextBox txtEventLog;

        // 상태
        private string _watchFolder;
        private FileSystemWatcher _watcher;
        private readonly Dictionary<int, AxisModel> _axisModels = new Dictionary<int, AxisModel>();

        // 전역 모델 — 축별 모델이 없는 축에 폴백으로 적용
        private PersistedKnnModel _globalKnnModel;
        private string _globalKnnModelPath;
        private OnnxAxisModel _globalOnnxAe;

        private readonly HashSet<string> _processing = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, CancellationTokenSource> _debouncers = new Dictionary<string, CancellationTokenSource>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, long> _lastProcessedLen = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        private readonly object _sync = new object();

        // DB 모니터링 상태
        private bool _isDbMode = false;
        private InfluxDbDataSource _influxSource;
        private CancellationTokenSource _influxPollCts;

        // 통계
        private int cntDanger, cntWarning, cycles;
        private readonly ConcurrentQueue<Tuple<int, DateTime, double>> scoreSeries = new ConcurrentQueue<Tuple<int, DateTime, double>>();
        private bool _lineFirstFlush = true;
        private const string SkeletonSeriesName = "_skeleton_";

        // 데이터 테이블
        private BindingList<EventRow> rows = new BindingList<EventRow>();
        private enum XMode { Numeric, DateTime, Index }
        private FlowLayoutPanel axisGaugeFlow;
        private readonly Dictionary<int, ProbGaugeControl> _axisGauges = new Dictionary<int, ProbGaugeControl>();
        private string[] _clsLabels = { "0. 정상", "1. 벨트 결함", "2. 볼트 풀림", "3. 바디 불평형" };
        private readonly Dictionary<int, float[]> _axisGaugePrev = new Dictionary<int, float[]>();
        private const bool GAUGE_SORT_DESC = false;   // true면 확률 내림차순으로 정렬해 보여줌
        private Button btnTestProbs;              // ← 추가
        private readonly Random _rng = new Random(); // ← 추가

        // ── 서버 실시간 추론 UI ──────────────────────────────────────────────────
        // key = "{sensorType}_ax{n}" (예: "accel_ax0", "torque_ax2", "combined_ax0")
        private Chart _chartAccel;                  // 가속도 서버 추론 스코어 차트
        private Chart _chartTorque;                 // 토크 서버 추론 스코어 차트
        private bool  _liveChartFirstData = true;   // 첫 실데이터 수신 시 스켈레톤 제거 플래그
        private Label _lblAccelModelInfo;           // 가속도 차트 모델 정보 라벨
        private Label _lblTorqueModelInfo;          // 토크 차트 모델 정보 라벨
        private TableLayoutPanel _dualChartPanel;   // 이상 스코어 차트 컨테이너 (동적 열 너비 조정용)
        private Panel _accelChartWrap;              // 가속도 차트 래퍼 패널
        private Panel _torqueChartWrap;             // 토크 차트 래퍼 패널
        private readonly System.Collections.Generic.HashSet<string> _seenAccelModels  = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private readonly System.Collections.Generic.HashSet<string> _seenTorqueModels = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private FlowLayoutPanel _statusFlow;        // 상단 상태 바 칩 컨테이너

        // ── 센서 표시 필터 체크박스 ──────────────────────────────────────────
        private CheckBox _chkShowAccel;             // 가속도 결과 표시 여부
        private CheckBox _chkShowTorque;            // 토크 결과 표시 여부
        private CheckBox _chkShowCombined;          // 결합 결과 표시 여부

        // ── 실시간 분류 현황 매트릭스 ─────────────────────────────────────────
        // 행=토크 축, 열=AE 토크 점수/상태 + CLS 결함/신뢰도 (가속도 AE는 전역 단일 패널로 별도 표시)
        private DataGridView _classMatrixDgv;
        private readonly Dictionary<int, int> _classMatrixAxisRow = new Dictionary<int, int>(); // axis → rowIdx
        // ── AE 이상탐지 열 (토크 per-axis 전용) ─────────
        private const int CMG_COL_AXIS     = 0;
        private const int CMG_COL_TS       = 1;  // [AE] 토크 점수
        private const int CMG_COL_TSTATE   = 2;  // [AE] 토크 판정
        // ── CLS 결함진단 열 ─────────────────────────────
        private const int CMG_COL_TCLS     = 3;  // [CLS] 토크 결함명
        private const int CMG_COL_TCLSCONF = 4;  // [CLS] 토크 신뢰도
        private const int CMG_COL_CCLS     = 5;  // [CLS] 결합 결함명
        private const int CMG_COL_CCLSCONF = 6;  // [CLS] 결합 신뢰도

        // ── 가속도 AE 전역 표시 패널 (단일 모델 — per-axis 없음) ─────────
        private Panel  _pnlAccelAeBar;
        private Label  _lblAccelAeScore;
        private Label  _lblAccelAeState;
        private double _accelAeNormScore;
        // 마지막 스코어 보관 (색 재계산용)
        private readonly Dictionary<string, double> _classMatrixLastScore = new Dictionary<string, double>();
        private readonly Dictionary<string, Panel>  _statusChips      = new Dictionary<string, Panel>();
        private readonly Dictionary<string, Label>  _liveStatusLabels = new Dictionary<string, Label>();
        private readonly Dictionary<string, Label>  _liveScoreLabels  = new Dictionary<string, Label>();
        private readonly ToolTip                    _chipToolTip      = new ToolTip();
        // 축별 임계값: key → threshold (서버값으로 자동 초기화, UI에서 재설정 가능)
        private readonly System.Collections.Concurrent.ConcurrentDictionary<string, double>
            _axisThresholds = new System.Collections.Concurrent.ConcurrentDictionary<string, double>();
        private static readonly string ThresholdsFile =
            Path.Combine(DefaultLogsPath, "axis_thresholds.txt");
        private readonly ConcurrentQueue<Tuple<string, DateTime, double>> _liveScoreQueue
            = new ConcurrentQueue<Tuple<string, DateTime, double>>();

        // ── 외력 감지: EMA 베이스라인 대비 급증(spike) 감지 ─────────────────────
        // key → (EMA 베이스라인, 누적 샘플 수)
        // 샘플이 충분히 쌓인 후(>= SpikeWarmup) 현재 스코어가 EMA의 SpikeFactor배 이상이면 anomaly
        private readonly System.Collections.Concurrent.ConcurrentDictionary<string, (double ema, int count)>
            _scoreBaseline = new System.Collections.Concurrent.ConcurrentDictionary<string, (double, int)>();
        private const double SpikeEmaAlpha  = 0.1;  // EMA 감쇠율 (느릴수록 베이스라인 안정적)
        private const double SpikeFactor    = 2.5;  // EMA 대비 이 배수 이상이면 spike 판정
                                                    // 서버가 MAE+peak 혼합 스코어 반환하므로 충격 시 spike 폭 확대됨
        private const int    SpikeWarmup    = 30;   // 워밍업 후 spike 판정 시작 (충분한 베이스라인 수집)

        // ── 위험/경고 등급 임계 배수 (normScore = rawScore / threshold 기준) ──
        // normScore ∈ [WarnMultiplier, DangerMultiplier) → 경고
        // normScore ≥ DangerMultiplier                   → 위험
        private const double WarnMultiplier   = 1.0;  // 임계값 초과 즉시 경고
        private const double DangerMultiplier = 2.0;  // 임계값의 2배 이상이면 위험

        // ── 차트 표시용 EMA 평활화 ────────────────────────────────────────────
        // 128ms 간격의 per-window 스코어 노이즈를 줄여 차트를 부드럽게 표시.
        // 이상 판정(threshAnomaly/spikeAnomaly)은 raw score 기반으로 유지.
        // alpha=0.15: 약 12샘플(~1.5초) 수렴, 차트 노이즈 감소
        private readonly System.Collections.Concurrent.ConcurrentDictionary<string, double>
            _chartEma = new System.Collections.Concurrent.ConcurrentDictionary<string, double>();
        private const double ChartEmaAlpha = 0.15;

        // ── 이동평균선 (MA선) ──────────────────────────────────────────────────
        // alpha=0.05: 약 40샘플(~10초) 수렴 — 단기 노이즈 제거, 추세 확인용
        private readonly ConcurrentQueue<Tuple<string, DateTime, double>> _liveMaQueue
            = new ConcurrentQueue<Tuple<string, DateTime, double>>();
        private readonly System.Collections.Concurrent.ConcurrentDictionary<string, double>
            _chartMaEma = new System.Collections.Concurrent.ConcurrentDictionary<string, double>();
        private const double ChartMaAlpha = 0.05;

        // ── 이상 이벤트 로그 중복 방지 ──────────────────────────────────────
        // 상태 전환(정상→이상) 시 즉시, 이상 지속 시 쿨다운마다 1회씩만 로그
        private readonly Dictionary<string, bool>     _prevAnomalyState   = new Dictionary<string, bool>();
        private readonly Dictionary<string, DateTime> _lastAnomalyLogTime = new Dictionary<string, DateTime>();
        private static readonly TimeSpan AnomalyLogCooldown = TimeSpan.FromSeconds(30);

        // ── 연속 이상 카운터 (N회 연속 threshold 초과 시 경보 확정) ──────────
        // 256ms 간격 × 5회 = 약 1.3초 연속 초과 시 확정 → 단발성 노이즈 무시
        private const int AnomalyConfirmCount = 5;
        private readonly Dictionary<string, int> _consecutiveAnomalyCount
            = new Dictionary<string, int>();

        // 프로파일 관리
        private ComboBox _cmbProfile;
        private Button   _btnProfileApply, _btnProfileRefresh;
        private PHM_Project_DockPanel.Services.Core.InferenceServerClient _profileClient;

        // DB 모드 UI 컨트롤
        private RadioButton rbtnCsvMode, rbtnDbMode;
        private Panel pnlCsvSource, pnlDbSource;
        private ComboBox cmbDbDevice, cmbDbLabel;
        private DateTimePicker dtpDbFrom, dtpDbTo;
        private Button btnDbRefresh, btnDbFullRange;

        // Preprocessing과 동일한 시간 컬럼 후보
        private static readonly string[] TimeColumnCandidates = { "time_s", "cycle" };
        private static readonly string[][] AccColumnSets =
        {
            new[] { "x", "y", "z" }
        };

        private static bool IsTimeColumn(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return false;
            var n = name.Trim();
            return TimeColumnCandidates.Any(tc => n.Equals(tc, StringComparison.OrdinalIgnoreCase));
        }
        private static HashSet<int> AxesFromFilename(string path)
        {
            var axes = new HashSet<int>();
            var name = Path.GetFileNameWithoutExtension(path) ?? "";
            var m = Regex.Matches(name, @"Axis(?<id>\d+)", RegexOptions.IgnoreCase);
            foreach (Match mm in m)
                if (int.TryParse(mm.Groups["id"].Value, out int ax)) axes.Add(ax);
            return axes;
        }
        private static bool TryResolveAccelColumns(string[] headers, string baseNameOrX,
    out int idxX, out int idxY, out int idxZ)
        {
            idxX = idxY = idxZ = -1;
            if (headers == null || headers.Length == 0) return false;

            // 1) baseNameOrX가 "x" 류라면, 셋 중 하나로 간주
            var headerNorm = headers.Select(h => h?.Trim()).ToArray();
            int simpleX = Array.FindIndex(headerNorm, h => string.Equals(h, baseNameOrX, StringComparison.OrdinalIgnoreCase));
            if (simpleX >= 0)
            {
                // 같은 집합에서 y/z 찾기
                foreach (var set in AccColumnSets)
                {
                    int ix = Array.FindIndex(headerNorm, h => string.Equals(h, set[0], StringComparison.OrdinalIgnoreCase));
                    int iy = Array.FindIndex(headerNorm, h => string.Equals(h, set[1], StringComparison.OrdinalIgnoreCase));
                    int iz = Array.FindIndex(headerNorm, h => string.Equals(h, set[2], StringComparison.OrdinalIgnoreCase));
                    if (ix >= 0 && iy >= 0 && iz >= 0) { idxX = ix; idxY = iy; idxZ = iz; return true; }
                }
            }

            // 2) 접두사 방식: "{base}_x", "{base}_y", "{base}_z" 등
            if (!string.IsNullOrWhiteSpace(baseNameOrX))
            {
                string b = baseNameOrX.Trim();
                string[] tryNamesX = { $"{b}_x", $"{b}x", $"{b}X" };
                string[] tryNamesY = { $"{b}_y", $"{b}y", $"{b}Y" };
                string[] tryNamesZ = { $"{b}_z", $"{b}z", $"{b}Z" };

                idxX = headerNorm.ToList().FindIndex(h => tryNamesX.Any(c => string.Equals(h, c, StringComparison.OrdinalIgnoreCase)));
                idxY = headerNorm.ToList().FindIndex(h => tryNamesY.Any(c => string.Equals(h, c, StringComparison.OrdinalIgnoreCase)));
                idxZ = headerNorm.ToList().FindIndex(h => tryNamesZ.Any(c => string.Equals(h, c, StringComparison.OrdinalIgnoreCase)));
                if (idxX >= 0 && idxY >= 0 && idxZ >= 0) return true;
            }

            // 3) 완전 일반형(헤더에 x/y/z 셋이 있으면 채택)
            foreach (var set in AccColumnSets)
            {
                int ix = Array.FindIndex(headerNorm, h => string.Equals(h, set[0], StringComparison.OrdinalIgnoreCase));
                int iy = Array.FindIndex(headerNorm, h => string.Equals(h, set[1], StringComparison.OrdinalIgnoreCase));
                int iz = Array.FindIndex(headerNorm, h => string.Equals(h, set[2], StringComparison.OrdinalIgnoreCase));
                if (ix >= 0 && iy >= 0 && iz >= 0) { idxX = ix; idxY = iy; idxZ = iz; return true; }
            }
            return false;
        }

        // ★ NEW: axis-aware overload
        private static bool TryResolveAccelColumns(string[] headers, string baseNameOrX, int axis,
            out int idxX, out int idxY, out int idxZ)
        {
            idxX = idxY = idxZ = -1;
            if (headers == null || headers.Length == 0) return false;
            var H = headers.Select(h => h?.Trim()).ToArray();

            // 축 번호가 붙는 흔한 패턴들(프로젝트 상황에 맞게 늘려도 됨)
            string[] xCands = { $"x{axis}", $"acc{axis}_x", $"axis{axis}_x", $"{baseNameOrX}{axis}_x", $"{baseNameOrX}{axis}x" };
            string[] yCands = { $"y{axis}", $"acc{axis}_y", $"axis{axis}_y", $"{baseNameOrX}{axis}_y", $"{baseNameOrX}{axis}y" };
            string[] zCands = { $"z{axis}", $"acc{axis}_z", $"axis{axis}_z", $"{baseNameOrX}{axis}_z", $"{baseNameOrX}{axis}z" };

            int find(string[] c) => Array.FindIndex(H, h => c.Any(s => string.Equals(h, s, StringComparison.OrdinalIgnoreCase)));

            int ix = find(xCands);
            int iy = find(yCands);
            int iz = find(zCands);
            if (ix >= 0 && iy >= 0 && iz >= 0) { idxX = ix; idxY = iy; idxZ = iz; return true; }

            // 못 찾으면 기존 일반형으로 폴백
            return TryResolveAccelColumns(headers, baseNameOrX, out idxX, out idxY, out idxZ);
        }

        private static float[,] BuildSequenceFromCsvAccel3(string filePath, string baseOrXCol, int axis)
        {
            // 1) 헤더
            string header;
            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var sr = new StreamReader(fs))
                header = sr.ReadLine();
            if (string.IsNullOrEmpty(header)) return null;

            var headers = SplitCsvLine(header);
            if (!TryResolveAccelColumns(headers, baseOrXCol, axis, out int ix, out int iy, out int iz))
                return null;

            // 2) 본문
            var xs = new List<float>(8192);
            var ys = new List<float>(8192);
            var zs = new List<float>(8192);
            var cult = CultureInfo.InvariantCulture;

            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); // skip header
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = SplitCsvLine(line);
                    if (parts.Length <= Math.Max(ix, Math.Max(iy, iz))) continue;

                    if (!float.TryParse(parts[ix], NumberStyles.Float, cult, out float vx) &&
                        !float.TryParse(parts[ix], NumberStyles.Float, CultureInfo.CurrentCulture, out vx)) continue;
                    if (!float.TryParse(parts[iy], NumberStyles.Float, cult, out float vy) &&
                        !float.TryParse(parts[iy], NumberStyles.Float, CultureInfo.CurrentCulture, out vy)) continue;
                    if (!float.TryParse(parts[iz], NumberStyles.Float, cult, out float vz) &&
                        !float.TryParse(parts[iz], NumberStyles.Float, CultureInfo.CurrentCulture, out vz)) continue;

                    if (float.IsNaN(vx) || float.IsNaN(vy) || float.IsNaN(vz) ||
                        float.IsInfinity(vx) || float.IsInfinity(vy) || float.IsInfinity(vz)) continue;

                    xs.Add(vx); ys.Add(vy); zs.Add(vz);
                }
            }
            int T = Math.Min(xs.Count, Math.Min(ys.Count, zs.Count));
            if (T <= 0) return null;

            var seq = new float[T, 3];
            for (int t = 0; t < T; t++) { seq[t, 0] = xs[t]; seq[t, 1] = ys[t]; seq[t, 2] = zs[t]; }
            return seq;
        }

        // AppState 래퍼 (PreprocessingForm과 동일 시그니처)
        private static double SampleRateFor(string yColumn) => AppState.GetForColumn(yColumn);
        private static double SamplePeriodFor(string yColumn) => AppState.GetPeriodForColumn(yColumn);

        private struct Snapshot { public DateTime T; public int D; public int W; public int C; } // Danger, Warning, Cycles
        private readonly LinkedList<Snapshot> _history = new LinkedList<Snapshot>();
        private DateTime _lastSnap = DateTime.MinValue;

        private NotifyIcon _notifier;
        #endregion

        public DashboardForm()
        {
            this.Text = "실시간 대시보드";
            this.MinimumSize = new Size(1000, 600);
            LoadAxisThresholds();
            BuildUI();
            _notifier = new NotifyIcon
            {
                Visible = true,
                Icon = SystemIcons.Warning,
                Text = "PHM 알림"
            };

            // 서버 실시간 추론 결과 구독 (AE 이상탐지 + CLS 결함진단)
            AppEvents.InferenceResultReceived    += OnLiveInferenceResult;
            AppEvents.ClsInferenceResultReceived += OnLiveClsInferenceResult;
            AppEvents.LoopCompleted              += OnLoopCompleted;

            // 프로파일 클라이언트 초기화 + 목록 로드
            string inferUrl = PHM_Project_DockPanel.Services.ServerSettings.Current.InferenceServerUrl;
            if (!string.IsNullOrWhiteSpace(inferUrl))
            {
                _profileClient = new PHM_Project_DockPanel.Services.Core.InferenceServerClient(inferUrl);
                // 비동기 로드 (UI 스레드 블로킹 없이)
                _ = RefreshProfilesAsync();
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            AppEvents.InferenceResultReceived    -= OnLiveInferenceResult;
            AppEvents.ClsInferenceResultReceived -= OnLiveClsInferenceResult;
            AppEvents.LoopCompleted              -= OnLoopCompleted;
            SaveAxisThresholds();
            try { StopWatch(); _notifier?.Dispose(); } catch { }
            try { DisposeOnnxSessions(); } catch { }
            try { _profileClient?.Dispose(); } catch { }
            base.OnFormClosing(e);
        }

        // ── 프로파일 관리 ─────────────────────────────────────────────────────
        private async System.Threading.Tasks.Task RefreshProfilesAsync()
        {
            if (_profileClient == null) return;
            var result = await _profileClient.GetProfilesAsync().ConfigureAwait(false);
            if (result == null) return;

            if (!IsHandleCreated || IsDisposed) return;
            BeginInvoke(new Action(() =>
            {
                string prevSelected = _cmbProfile.SelectedItem?.ToString();
                _cmbProfile.Items.Clear();
                foreach (var p in result.Profiles)
                {
                    string display = string.IsNullOrEmpty(p.Label) || p.Label == p.Name
                        ? p.Name
                        : $"{p.Label} ({p.Name})";
                    _cmbProfile.Items.Add(new ProfileComboItem(p.Name, display, p.ModelCount));
                }

                // 현재 활성 프로파일 선택
                string active = result.Active ?? "default";
                int idx = -1;
                for (int i = 0; i < _cmbProfile.Items.Count; i++)
                {
                    if (_cmbProfile.Items[i] is ProfileComboItem pi && pi.Name == active)
                    { idx = i; break; }
                }
                if (idx >= 0) _cmbProfile.SelectedIndex = idx;
                else if (_cmbProfile.Items.Count > 0) _cmbProfile.SelectedIndex = 0;
            }));
        }

        private async System.Threading.Tasks.Task ApplyProfileAsync()
        {
            if (_profileClient == null || _cmbProfile.SelectedItem == null) return;
            if (!(_cmbProfile.SelectedItem is ProfileComboItem item)) return;

            _btnProfileApply.Enabled = false;
            _btnProfileApply.Text    = "...";
            try
            {
                bool ok = await _profileClient.ActivateProfileAsync(item.Name).ConfigureAwait(false);
                if (!IsHandleCreated || IsDisposed) return;
                BeginInvoke(new Action(() =>
                {
                    if (ok)
                    {
                        AppEvents.RaiseLog($"[프로파일] 전환 완료: {item.Name}  ({item.ModelCount}개 모델)");
                        // 칩/EMA/차트 상태 리셋
                        _chartEma.Clear();
                        _chartMaEma.Clear();
                        _scoreBaseline.Clear();
                        _consecutiveAnomalyCount.Clear();
                        _seenAccelModels.Clear();
                        _seenTorqueModels.Clear();
                        if (_lblAccelModelInfo != null) _lblAccelModelInfo.Text = "모델: 수신 대기 중...";
                        if (_lblTorqueModelInfo != null) _lblTorqueModelInfo.Text = "모델: 수신 대기 중...";
                    }
                    else
                    {
                        AppEvents.RaiseLog($"[프로파일] 전환 실패: {item.Name}");
                        MessageBox.Show($"프로파일 '{item.Name}' 전환에 실패했습니다.\n서버 로그를 확인하세요.",
                            "프로파일 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }));
            }
            finally
            {
                if (IsHandleCreated && !IsDisposed)
                    BeginInvoke(new Action(() =>
                    {
                        _btnProfileApply.Enabled = true;
                        _btnProfileApply.Text    = "적용";
                    }));
            }
        }

        /// <summary>ComboBox 아이템 — 프로파일 이름과 표시 텍스트를 분리합니다.</summary>
        private sealed class ProfileComboItem
        {
            public string Name       { get; }
            public int    ModelCount { get; }
            private readonly string _display;
            public ProfileComboItem(string name, string display, int modelCount)
            { Name = name; _display = display; ModelCount = modelCount; }
            public override string ToString() => _display;
        }

        private static void DownsampleMinMax(IList<double> xs, IList<double> ys, int maxPoints,
            out double[] dx, out double[] dy)
        {
            int n = Math.Min(xs.Count, ys.Count);
            if (n <= maxPoints || maxPoints < 4)
            {
                dx = xs.Take(n).ToArray();
                dy = ys.Take(n).ToArray();
                return;
            }

            int buckets = maxPoints / 2; // min+max 두 점
            double bucketSize = (double)n / buckets;

            var xList = new List<double>(maxPoints);
            var yList = new List<double>(maxPoints);

            for (int b = 0; b < buckets; b++)
            {
                int start = (int)Math.Floor(b * bucketSize);
                int end = (int)Math.Min(n, Math.Floor((b + 1) * bucketSize));
                if (end <= start) continue;

                int minIdx = start, maxIdx = start;
                double minV = ys[start], maxV = ys[start];

                for (int i = start + 1; i < end; i++)
                {
                    double v = ys[i];
                    if (v < minV) { minV = v; minIdx = i; }
                    if (v > maxV) { maxV = v; maxIdx = i; }
                }

                if (minIdx <= maxIdx)
                {
                    xList.Add(xs[minIdx]); yList.Add(ys[minIdx]);
                    xList.Add(xs[maxIdx]); yList.Add(ys[maxIdx]);
                }
                else
                {
                    xList.Add(xs[maxIdx]); yList.Add(ys[maxIdx]);
                    xList.Add(xs[minIdx]); yList.Add(ys[minIdx]);
                }
            }

            dx = xList.ToArray();
            dy = yList.ToArray();
        }

        #region UI 구성
        private void BuildUI()
        {
            SuspendLayout();
            const int BtnH = 28;
            BackColor = Color.FromArgb(245, 247, 250);

            // ══════════════════════════════════════════════════
            //  콘텐츠 패널 (전체 — 사이드바 없음)
            // ══════════════════════════════════════════════════
            // ══════════════════════════════════════════════════
            //  Content Area (툴바 + 상태바 + 2×2 메인 그리드)
            // ══════════════════════════════════════════════════
            var contentPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3,
                Padding = Padding.Empty, Margin = Padding.Empty
            };
            // DPI 스케일을 먼저 계산해서 행 높이에 반영
            float _dpiScaleEarly;
            using (var _dg = System.Drawing.Graphics.FromHwnd(IntPtr.Zero))
                _dpiScaleEarly = _dg.DpiX / 96.0f;
            int statusBarH = Math.Max(76, (int)(76 * _dpiScaleEarly));  // 상태 바 (칩) 높이

            contentPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            contentPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));          // 툴바
            contentPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, statusBarH));  // 상태 바 (칩)
            contentPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));          // 2×2 메인 그리드

            // ── Row 0: 상단 툴바 ──────────────────────────────────────────────
            var toolbar = new Panel
            {
                Dock = DockStyle.Fill, BackColor = Color.White,
                Padding = new Padding(8, 0, 8, 0)
            };
            toolbar.Paint += (s, e) =>
            {
                using (var pen = new Pen(Color.FromArgb(215, 215, 222)))
                    e.Graphics.DrawLine(pen, 0, toolbar.Height - 1, toolbar.Width, toolbar.Height - 1);
            };

            // 데이터 소스 모드
            rbtnCsvMode = new RadioButton { Text = "CSV 폴더 감시", Checked = true, AutoSize = true };
            rbtnDbMode  = new RadioButton { Text = "DB 모니터링",   Checked = false, AutoSize = true };
            rbtnCsvMode.CheckedChanged += (s, e) => { if (rbtnCsvMode.Checked) SwitchSourceMode(false); };
            rbtnDbMode.CheckedChanged  += (s, e) => { if (rbtnDbMode.Checked)  SwitchSourceMode(true);  };

            // CSV 폴더 컨트롤
            pnlCsvSource = new Panel { AutoSize = true, Visible = true };
            btnSelectFolder = new Button { Text = "📁 폴더 선택", Width = 110, Height = BtnH };
            btnSelectFolder.Click += (s, e) => SelectFolder();
            btnQuickFolder = new Button { Text = "▾", Width = 24, Height = BtnH };
            btnQuickFolder.Click += (s, e) => ShowFolderQuickMenu();
            lblFolder = new Label
            {
                AutoSize = true, ForeColor = Color.Gray,
                TextAlign = ContentAlignment.MiddleLeft,
                MaximumSize = new Size(400, 0)
            };
            var folderFlow = new FlowLayoutPanel
            {
                AutoSize = true, FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false, Padding = Padding.Empty
            };
            folderFlow.Controls.AddRange(new Control[] { btnSelectFolder, btnQuickFolder, lblFolder });
            pnlCsvSource.Controls.Add(folderFlow);

            // DB 소스 컨트롤 (인라인)
            pnlDbSource = new Panel { AutoSize = true, Visible = false };
            var dbFlow = new FlowLayoutPanel
            {
                AutoSize = true, FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false, Padding = Padding.Empty
            };
            var lblDevice  = new Label  { Text = "장치:",  AutoSize = false, Width = 38, Height = BtnH, TextAlign = ContentAlignment.MiddleLeft };
            cmbDbDevice    = new ComboBox { Width = 160, Height = BtnH, DropDownStyle = ComboBoxStyle.DropDown };
            btnDbRefresh   = new Button { Text = "↺", Width = 28, Height = BtnH };
            btnDbRefresh.Click += (s, e) => RefreshDbDevices();
            var lblLabelDb = new Label { Text = "레이블:", AutoSize = false, Width = 48, Height = BtnH, TextAlign = ContentAlignment.MiddleLeft };
            cmbDbLabel     = new ComboBox { Width = 120, Height = BtnH, DropDownStyle = ComboBoxStyle.DropDown };
            var lblFrom    = new Label { Text = "시작:", AutoSize = false, Width = 36, Height = BtnH, TextAlign = ContentAlignment.MiddleLeft };
            dtpDbFrom      = new DateTimePicker { Width = 140, Height = BtnH, Format = DateTimePickerFormat.Custom, CustomFormat = "yyyy-MM-dd HH:mm", Value = DateTime.Now.AddHours(-1) };
            btnDbFullRange = new Button { Text = "↔", Width = 28, Height = BtnH };
            btnDbFullRange.Click += async (s, e) => await FillDbFullRangeAsync();
            var lblTo      = new Label { Text = "종료:", AutoSize = false, Width = 36, Height = BtnH, TextAlign = ContentAlignment.MiddleLeft };
            dtpDbTo        = new DateTimePicker { Width = 140, Height = BtnH, Format = DateTimePickerFormat.Custom, CustomFormat = "yyyy-MM-dd HH:mm", Value = DateTime.Now };
            dbFlow.Controls.AddRange(new Control[] {
                lblDevice, cmbDbDevice, btnDbRefresh,
                lblLabelDb, cmbDbLabel,
                lblFrom, dtpDbFrom, btnDbFullRange,
                lblTo, dtpDbTo
            });
            pnlDbSource.Controls.Add(dbFlow);

            // 시작/중지 버튼 (오른쪽 앵커)
            btnStop = new Button
            {
                Text = "■  중지", Width = 90, Height = BtnH,
                Anchor = AnchorStyles.Top | AnchorStyles.Right, Enabled = false,
                BackColor = Color.FromArgb(196, 43, 28), ForeColor = Color.White, FlatStyle = FlatStyle.Flat
            };
            btnStart = new Button
            {
                Text = "▶  진단 시작", Width = 110, Height = BtnH,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                BackColor = Color.FromArgb(0, 120, 212), ForeColor = Color.White, FlatStyle = FlatStyle.Flat,
                Font = new Font(this.Font, FontStyle.Bold)
            };
            lblStatus = new Label
            {
                AutoSize = true, Anchor = AnchorStyles.Top | AnchorStyles.Right,
                ForeColor = Color.Gray, MaximumSize = new Size(200, 0)
            };
            btnStart.Click += (s, e) => StartWatch();
            btnStop.Click  += (s, e) => StopWatch();

            // ── 프로파일 선택 컨트롤 ─────────────────────────────────────────
            _cmbProfile = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDownList,
                Width = 160, Height = BtnH,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                Font = new Font("Segoe UI", 8.5f),
            };
            _btnProfileRefresh = new Button
            {
                Text = "↺", Width = 26, Height = BtnH,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                FlatStyle = FlatStyle.Flat, Font = new Font("Segoe UI", 9f),
                ForeColor = Color.FromArgb(60, 60, 60),
            };
            _btnProfileApply = new Button
            {
                Text = "적용", Width = 46, Height = BtnH,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(0, 120, 212), ForeColor = Color.White,
            };
            _btnProfileRefresh.Click += async (s, e) => await RefreshProfilesAsync();
            _btnProfileApply.Click   += async (s, e) => await ApplyProfileAsync();

            // 왼쪽 소스 Flow
            var toolbarLeft = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false, Padding = Padding.Empty, Margin = Padding.Empty
            };
            toolbarLeft.Controls.AddRange(new Control[] { rbtnCsvMode, rbtnDbMode, pnlCsvSource, pnlDbSource });

            toolbar.Controls.Add(btnStop);
            toolbar.Controls.Add(btnStart);
            toolbar.Controls.Add(_btnProfileApply);
            toolbar.Controls.Add(_btnProfileRefresh);
            toolbar.Controls.Add(_cmbProfile);
            toolbar.Controls.Add(lblStatus);
            toolbar.Controls.Add(toolbarLeft);

            toolbar.Resize += (s, e) =>
            {
                int cy    = (toolbar.ClientSize.Height - BtnH) / 2;
                int right = toolbar.ClientSize.Width - toolbar.Padding.Right;
                btnStop.Top  = btnStart.Top  = cy;
                btnStop.Left  = right - btnStop.Width;
                btnStart.Left = btnStop.Left - btnStart.Width - 4;

                // 프로파일 컨트롤: 진단 시작 버튼 왼쪽
                _btnProfileApply.Top   = cy;
                _btnProfileApply.Left  = btnStart.Left - _btnProfileApply.Width - 6;
                _btnProfileRefresh.Top  = cy;
                _btnProfileRefresh.Left = _btnProfileApply.Left - _btnProfileRefresh.Width - 2;
                _cmbProfile.Top  = cy;
                _cmbProfile.Left = _btnProfileRefresh.Left - _cmbProfile.Width - 4;

                lblStatus.Top  = cy + 2;
                lblStatus.Left = _cmbProfile.Left - lblStatus.PreferredWidth - 8;
            };

            // ── 모드 전환 시 소스 패널 교체 ──────────────────────────────────
            void SwitchSourcePanel(bool isDb)
            {
                pnlCsvSource.Visible = !isDb;
                pnlDbSource.Visible  =  isDb;
                toolbarLeft.PerformLayout();
            }
            rbtnCsvMode.CheckedChanged += (s, e) => { if (rbtnCsvMode.Checked) SwitchSourcePanel(false); };
            rbtnDbMode.CheckedChanged  += (s, e) => { if (rbtnDbMode.Checked)  SwitchSourcePanel(true);  };

            // ── Row 0: Live Status Bar ─────────────────────────────────────────
            var statusBarPanel = new Panel
            {
                Dock = DockStyle.Fill, BackColor = Color.White,
                Padding = new Padding(10, 7, 8, 5)
            };
            statusBarPanel.Paint += (s, e) =>
            {
                using (var pen = new Pen(Color.FromArgb(215, 215, 222)))
                    e.Graphics.DrawLine(pen, 0, statusBarPanel.Height - 1,
                                        statusBarPanel.Width, statusBarPanel.Height - 1);
            };
            var lblStatusTitle = new Label
            {
                Text = "실시간 추론 상태",
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                AutoSize = true, Dock = DockStyle.Left,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(0, 2, 12, 2),
                ForeColor = Color.FromArgb(55, 55, 65)
            };
            _statusFlow = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false, AutoScroll = false, Padding = Padding.Empty
            };
            // 칩이 많아 넘칠 때 마우스휠로 좌우 스크롤 (스크롤바 없이)
            _statusFlow.MouseWheel += (s, e) =>
            {
                var hScroll = _statusFlow.HorizontalScroll;
                int delta   = -e.Delta / 3;
                int newVal  = Math.Max(hScroll.Minimum, Math.Min(hScroll.Maximum, hScroll.Value + delta));
                _statusFlow.HorizontalScroll.Value = newVal;
                _statusFlow.PerformLayout();
            };
            // ── 센서 표시 필터 체크박스 (상태 바 오른쪽) ──────────────────────
            _chkShowAccel    = new CheckBox { Text = "가속도", Checked = true,  AutoSize = true,
                Font = new Font("Segoe UI", 8f), ForeColor = Color.FromArgb(30, 100, 200),
                Padding = new Padding(0, 0, 4, 0) };
            _chkShowTorque   = new CheckBox { Text = "토크",   Checked = false, AutoSize = true,
                Font = new Font("Segoe UI", 8f), ForeColor = Color.FromArgb(160, 80, 0),
                Padding = new Padding(0, 0, 4, 0) };
            _chkShowCombined = new CheckBox { Text = "결합",   Checked = false, AutoSize = true,
                Font = new Font("Segoe UI", 8f), ForeColor = Color.FromArgb(60, 140, 60),
                Padding = new Padding(0, 0, 4, 0) };
            // 체크박스 변경 시 차트 레이아웃 + 상태 칩/컨트롤 동시 갱신
            _chkShowAccel.CheckedChanged    += (s, e) => { ApplySensorVisibility("accel",    _chkShowAccel.Checked);    _UpdateChartLayout(); };
            _chkShowTorque.CheckedChanged   += (s, e) => { ApplySensorVisibility("torque",   _chkShowTorque.Checked);   _UpdateChartLayout(); };
            _chkShowCombined.CheckedChanged += (s, e) => { ApplySensorVisibility("combined", _chkShowCombined.Checked); /* combined 차트 없음 */ };

            var lblFilter = new Label { Text = "표시:", AutoSize = true,
                Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                ForeColor = Color.FromArgb(55, 55, 65),
                Padding = new Padding(8, 0, 4, 0) };

            var filterFlow = new FlowLayoutPanel
            {
                Dock = DockStyle.Right, AutoSize = true, AutoSizeMode = AutoSizeMode.GrowAndShrink,
                FlowDirection = FlowDirection.LeftToRight, WrapContents = false,
                Padding = new Padding(0, 3, 6, 0)
            };
            filterFlow.Controls.AddRange(new Control[] { lblFilter, _chkShowAccel, _chkShowTorque, _chkShowCombined });

            // Right dock → Fill 순으로 Controls.Add (Right이 먼저여야 Fill이 남은 공간 차지)
            statusBarPanel.Controls.Add(filterFlow);
            statusBarPanel.Controls.Add(_statusFlow);
            statusBarPanel.Controls.Add(lblStatusTitle);

            // ── DPI 스케일 (앞서 계산한 값 재사용) ───────────────────────────
            float _dpiScale = _dpiScaleEarly;
            int kpiRowH   = Math.Max(140, (int)(140 * _dpiScale));
            int accelBarH = Math.Max(38,  (int)(38  * _dpiScale));

            // ── 2×2 메인 그리드 ───────────────────────────────────────────────
            // ┌──────────────────────┬──────────────────────┐
            // │ (0,0) 설비 상태 현황  │ (1,0) 실시간 분류 현황│
            // ├──────────────────────┼──────────────────────┤
            // │ (0,1) 이상 스코어 차트│ (1,1) 발생 이벤트    │
            // └──────────────────────┴──────────────────────┘
            var mainPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2,
                Padding = new Padding(6, 4, 6, 6), Margin = Padding.Empty,
                BackColor = Color.FromArgb(245, 247, 250)
            };
            mainPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 44));
            mainPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 56));
            mainPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 38));   // 상단 행 (KPI + 분류 현황)
            mainPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 62));   // 하단 행 (차트 + 이벤트)

            // ── (0,0) 상단-좌: 설비 상태 현황 ────────────────────────────────
            var leftTopPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3,
                Margin = new Padding(0, 0, 4, 4), Padding = Padding.Empty
            };
            leftTopPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            leftTopPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));         // 섹션 제목
            leftTopPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));         // KPI 카드 (flex)
            leftTopPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, accelBarH));  // 가속도 AE 바

            var lblKpiTitle = new Label
            {
                Text = "설비 상태 현황",
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(4, 0, 0, 0), ForeColor = Color.FromArgb(30, 30, 40)
            };
            var kpiPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 1,
                Padding = new Padding(2), Margin = new Padding(0, 0, 0, 4), MinimumSize = new Size(60, 60)
            };
            kpiPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33f));
            kpiPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33f));
            kpiPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33f));
            int cardMinH = Math.Max(110, (int)(110 * _dpiScale));
            cardDanger  = new KpiCard { Title = "위험 건수",   ValueText = "0 건", DeltaText = "—", Footnote = "2시간 전 대비", Dock = DockStyle.Fill, Margin = new Padding(3, 3, 3, 8), MinimumSize = new Size(50, cardMinH) };
            cardWarning = new KpiCard { Title = "경고 건수",   ValueText = "0 건", DeltaText = "—", Footnote = "2시간 전 대비", Dock = DockStyle.Fill, Margin = new Padding(3, 3, 3, 8), MinimumSize = new Size(50, cardMinH) };
            cardCycles  = new KpiCard { Title = "설비 사용률", ValueText = "0 회", DeltaText = "0.0%", Footnote = "2시간 전 대비", Dock = DockStyle.Fill, Margin = new Padding(3, 3, 3, 8), MinimumSize = new Size(50, cardMinH) };
            kpiPanel.Controls.Add(cardDanger,  0, 0);
            kpiPanel.Controls.Add(cardWarning, 1, 0);
            kpiPanel.Controls.Add(cardCycles,  2, 0);

            // 가속도 AE 전역 스코어 패널
            _pnlAccelAeBar = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(245, 245, 248),
                Padding = new Padding(10, 0, 10, 0)
            };
            _pnlAccelAeBar.Paint += (s, e) =>
            {
                using (var pen = new Pen(Color.FromArgb(210, 210, 220)))
                    e.Graphics.DrawRectangle(pen, 0, 0, _pnlAccelAeBar.Width - 1, _pnlAccelAeBar.Height - 1);
            };
            var lblAccelAeTitle = new Label
            {
                Text = "가속도 AE (전역 단일 모델):",
                AutoSize = true, Dock = DockStyle.Left,
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                ForeColor = Color.FromArgb(30, 80, 180),
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(0, 0, 12, 0)
            };
            _lblAccelAeScore = new Label
            {
                Text = "—", AutoSize = true, Dock = DockStyle.Left,
                Font = new Font("Segoe UI", 11f, FontStyle.Bold),
                ForeColor = Color.FromArgb(50, 50, 60),
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(0, 0, 10, 0)
            };
            _lblAccelAeState = new Label
            {
                Text = "—", AutoSize = true, Dock = DockStyle.Left,
                Font = new Font("Segoe UI", 9f),
                ForeColor = Color.Gray,
                TextAlign = ContentAlignment.MiddleLeft
            };
            _pnlAccelAeBar.Controls.Add(_lblAccelAeState);
            _pnlAccelAeBar.Controls.Add(_lblAccelAeScore);
            _pnlAccelAeBar.Controls.Add(lblAccelAeTitle);

            leftTopPanel.SuspendLayout();
            leftTopPanel.Controls.Add(lblKpiTitle,    0, 0);
            leftTopPanel.Controls.Add(kpiPanel,       0, 1);
            leftTopPanel.Controls.Add(_pnlAccelAeBar, 0, 2);
            leftTopPanel.ResumeLayout(false);

            // ── (1,0) 상단-우: 실시간 분류 현황 ──────────────────────────────
            var classMatrixPanel = BuildClassMatrixPanel();

            // ── (0,1) 하단-좌: 이상 스코어 차트 ─────────────────────────────
            _dualChartPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1,
                Padding = new Padding(0, 4, 4, 0), Margin = Padding.Empty,
                BackColor = Color.FromArgb(245, 247, 250)
            };
            _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            _dualChartPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            _chartAccel  = BuildLiveInferenceChart(isAccel: true);
            _chartTorque = BuildLiveInferenceChart(isAccel: false);
            _dualChartPanel.SuspendLayout();
            _accelChartWrap  = WrapChartInPanel("가속도 이상 스코어  [서버 추론]", _chartAccel,  Color.FromArgb(0, 84, 166),  out _lblAccelModelInfo);
            _torqueChartWrap = WrapChartInPanel("토크 이상 스코어  [서버 추론]",   _chartTorque, Color.FromArgb(165, 45, 15), out _lblTorqueModelInfo);
            _dualChartPanel.Controls.Add(_accelChartWrap,  0, 0);
            _dualChartPanel.Controls.Add(_torqueChartWrap, 1, 0);
            _dualChartPanel.ResumeLayout(false);

            // ── (1,1) 하단-우: 발생 이벤트 ───────────────────────────────────
            var rightColPanel = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2,
                Margin = new Padding(4, 0, 0, 0), Padding = Padding.Empty
            };
            rightColPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            rightColPanel.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));    // 섹션 제목
            rightColPanel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));    // 이벤트 로그 + 그리드

            var lblEvents = new Label
            {
                Text = "발생 이벤트",
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(4, 0, 0, 0), ForeColor = Color.FromArgb(30, 30, 40)
            };
            var eventsLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 2,
                Margin = Padding.Empty, Padding = Padding.Empty
            };
            eventsLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            eventsLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 35));
            eventsLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 65));

            txtEventLog = new TextBox
            {
                Multiline = true, ReadOnly = true, Dock = DockStyle.Fill,
                ScrollBars = ScrollBars.Vertical,
                BackColor = Color.FromArgb(251, 252, 253),
                BorderStyle = BorderStyle.FixedSingle,
                Font = new Font("Consolas", 8.5f), WordWrap = false
            };
            grid = new DataGridView
            {
                Dock = DockStyle.Fill, ReadOnly = true,
                AllowUserToAddRows = false, AllowUserToDeleteRows = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                DataSource = rows
            };
            eventsLayout.Controls.Add(txtEventLog, 0, 0);
            eventsLayout.Controls.Add(grid,        0, 1);

            rightColPanel.SuspendLayout();
            rightColPanel.Controls.Add(lblEvents,    0, 0);
            rightColPanel.Controls.Add(eventsLayout, 0, 1);
            rightColPanel.ResumeLayout(false);

            // 2×2 조립
            mainPanel.SuspendLayout();
            mainPanel.Controls.Add(leftTopPanel,    0, 0);
            mainPanel.Controls.Add(classMatrixPanel, 1, 0);
            mainPanel.Controls.Add(_dualChartPanel, 0, 1);
            mainPanel.Controls.Add(rightColPanel,   1, 1);
            mainPanel.ResumeLayout(false);

            contentPanel.SuspendLayout();
            contentPanel.Controls.Add(toolbar,        0, 0);
            contentPanel.Controls.Add(statusBarPanel, 0, 1);
            contentPanel.Controls.Add(mainPanel,      0, 2);
            contentPanel.ResumeLayout(false);

            Controls.Add(contentPanel);

            // ── 초기 센서 visibility 적용 (기본: 토크/결합 숨김) ─────────────
            _UpdateChartLayout();
            ApplySensorVisibility("torque",   false);
            ApplySensorVisibility("combined", false);

            // ── 타이머 ────────────────────────────────────────────────────────
            var timer = new System.Windows.Forms.Timer { Interval = 300 };
            timer.Tick += (s, e) =>
            {
                FlushLineChart();

                if ((DateTime.Now - _lastSnap).TotalSeconds >= 60)
                {
                    _lastSnap = DateTime.Now;
                    _history.AddLast(new Snapshot { T = _lastSnap, D = cntDanger, W = cntWarning, C = cycles });
                    while (_history.First != null && (DateTime.Now - _history.First.Value.T).TotalHours > 3)
                        _history.RemoveFirst();
                }
                DateTime anchor = DateTime.Now.AddHours(-2);
                Snapshot? baseSnap = null;
                for (var node = _history.Last; node != null; node = node.Previous)
                    if (node.Value.T <= anchor) { baseSnap = node.Value; break; }

                if (baseSnap.HasValue)
                {
                    int dD = cntDanger  - baseSnap.Value.D;
                    int dW = cntWarning - baseSnap.Value.W;
                    int dC = cycles     - baseSnap.Value.C;
                    cardDanger.DeltaText  = (dD >= 0 ? "+" : "") + dD + "건";
                    cardWarning.DeltaText = (dW >= 0 ? "+" : "") + dW + "건";
                    cardCycles.DeltaText  = (dC >= 0 ? "+" : "") + dC + " 회";
                    cardDanger.Footnote = cardWarning.Footnote = cardCycles.Footnote = "2시간 전 대비";
                }
                else
                {
                    cardDanger.DeltaText = cardWarning.DeltaText = "—";
                    cardCycles.DeltaText = "—";
                    cardDanger.Footnote = cardWarning.Footnote = cardCycles.Footnote = "2시간 전 대비";
                }
            };
            timer.Start();

            this.ClientSize = new Size(1280, 840);
            _watchFolder = Path.Combine(DefaultLogsPath, "Signals");
            try { Directory.CreateDirectory(_watchFolder); } catch { }
            lblFolder.Text = "폴더: " + _watchFolder;

            this.Shown += (s, e) =>
            {
                if (axisGaugeFlow != null && !axisGaugeFlow.IsDisposed)
                {
                    foreach (Control c in axisGaugeFlow.Controls)
                        if (c is GroupBox gb)
                        {
                            int w = Math.Max(axisGaugeFlow.ClientSize.Width
                                             - axisGaugeFlow.Padding.Horizontal
                                             - gb.Margin.Horizontal, 160);
                            gb.Width = w;
                        }
                    axisGaugeFlow.PerformLayout();
                    axisGaugeFlow.Refresh();
                }
            };

            ResumeLayout(false);
        }

        /// <summary>서버 추론 전용 차트를 생성합니다 (임계값 1.0 점선 포함).</summary>
        private Chart BuildLiveInferenceChart(bool isAccel)
        {
            var chart = new Chart { Dock = DockStyle.Fill, BackColor = Color.White, MinimumSize = new Size(1, 1) };
            var ca = new ChartArea("a") { BackColor = Color.White };
            ca.AxisX.LabelStyle.Format   = "HH:mm:ss";
            ca.AxisX.IntervalAutoMode    = IntervalAutoMode.VariableCount;
            ca.AxisX.MajorGrid.Enabled   = true;
            ca.AxisX.MajorGrid.LineColor = Color.FromArgb(228, 228, 234);
            ca.AxisY.MajorGrid.Enabled   = true;
            ca.AxisY.MajorGrid.LineColor = Color.FromArgb(228, 228, 234);
            ca.AxisY.LabelStyle.Format   = "0.00";
            ca.AxisY.Minimum             = 0;
            // 임계값 점선 (서버 결과 수신 시 IntervalOffset 동적 갱신)
            ca.AxisY.StripLines.Add(new StripLine
            {
                IntervalOffset  = 1.0,
                StripWidth      = 0,
                BorderColor     = Color.FromArgb(215, 50, 60),
                BorderWidth     = 1,
                BorderDashStyle = ChartDashStyle.Dash,
                Text            = "임계값",
                ForeColor       = Color.FromArgb(215, 50, 60),
                Font            = new Font("Segoe UI", 7f)
            });
            chart.ChartAreas.Add(ca);
            chart.Legends.Clear();
            chart.Legends.Add(new Legend
            {
                Docking   = Docking.Top,
                Alignment = StringAlignment.Near,
                Font      = new Font("Segoe UI", 7.5f)
            });
            // 스켈레톤 (X축 초기화용)
            var now3 = DateTime.Now;
            var sk = new Series("_sk_")
            {
                ChartType = SeriesChartType.FastLine, XValueType = ChartValueType.DateTime,
                IsVisibleInLegend = false, Color = Color.Transparent
            };
            sk.Points.AddXY(now3.AddMinutes(-5), 0);
            sk.Points.AddXY(now3, 0);
            chart.Series.Add(sk);
            return chart;
        }

        /// <summary>차트를 제목 라벨과 함께 패널에 감쌉니다.</summary>
        private Panel WrapChartInPanel(string title, Chart chart, Color titleColor, out Label modelInfoLabel)
        {
            var panel = new Panel
            {
                Dock = DockStyle.Fill, BackColor = Color.White,
                Margin = new Padding(3), Padding = Padding.Empty,
                MinimumSize = new Size(1, 10)
            };
            panel.Paint += (s, e) =>
            {
                using (var pen = new Pen(Color.FromArgb(210, 210, 220)))
                    e.Graphics.DrawRectangle(pen, 0, 0, panel.Width - 1, panel.Height - 1);
            };
            var lbl = new Label
            {
                Text      = title,
                Font      = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                Dock      = DockStyle.Top, Height = 22,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding   = new Padding(8, 0, 0, 0),
                ForeColor = titleColor, BackColor = Color.White
            };
            // 모델 정보 라벨 (제목 바로 아래, 추론 결과 수신 시 갱신)
            modelInfoLabel = new Label
            {
                Text      = "모델: 수신 대기 중...",
                Font      = new Font("Segoe UI", 7.5f),
                Dock      = DockStyle.Top, Height = 17,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding   = new Padding(10, 0, 0, 0),
                ForeColor = Color.FromArgb(120, 120, 140), BackColor = Color.White
            };
            panel.Controls.Add(chart);
            panel.Controls.Add(modelInfoLabel);
            panel.Controls.Add(lbl);
            return panel;
        }
        #endregion

        private void SeedAxisGauge(int axis, int k)
        {
            if (InvokeRequired) { BeginInvoke(new Action(() => SeedAxisGauge(axis, k))); return; }
            var g = EnsureAxisProbGauge(axis, k);
            var labels = (_clsLabels?.Length == k) ? _clsLabels : Enumerable.Range(0, k).Select(i => $"Class {i}").ToArray();
            g.SetData(labels, Enumerable.Repeat(0f, k).ToArray(), -1);
            if (g.Parent is GroupBox gb) gb.Text = $"Axis {axis} 분류 확률";

            ReflowGaugeHeights(); // ★ 추가
        }

        private static int GetNumClassesFromOnnx(InferenceSession session, string preferredOutputName, int fallback = 4)
        {
            if (session == null) return fallback;

            // 출력 이름이 다를 수 있으므로 우선 시도 후, 없으면 첫 번째 출력으로 대체
            if (!string.IsNullOrEmpty(preferredOutputName) &&
                session.OutputMetadata.TryGetValue(preferredOutputName, out var meta))
            {
                var shape = meta.Dimensions;
                int k = shape?.LastOrDefault(d => d > 0) ?? 0;
                return (k > 0) ? k : fallback;
            }

            // 첫 번째 출력 사용
            var first = session.OutputMetadata.FirstOrDefault();
            if (!string.IsNullOrEmpty(first.Key))
            {
                var shape = first.Value.Dimensions;
                int k = shape?.LastOrDefault(d => d > 0) ?? 0;
                return (k > 0) ? k : fallback;
            }
            return fallback;
        }

        private void RandomizeClassProbsForAllAxes()
        {
            if (_axisGauges.Count == 0)
            {
                var axes = new HashSet<int>(_axisOnnx.Keys.Concat(_axisModels.Keys));
                if (axes.Count == 0) axes.Add(0);
                foreach (var ax in axes)
                    SeedAxisGauge(ax, (_clsLabels?.Length ?? 4));
            }

            foreach (var kv in _axisGauges.ToList())
            {
                int axis = kv.Key;
                int k = Math.Max(1, kv.Value.Labels?.Length ?? (_clsLabels?.Length ?? 4));

                var logits = new double[k];
                for (int i = 0; i < k; i++) logits[i] = _rng.NextDouble() * 2 - 1;
                double mx = logits.Max(), sum = 0;
                for (int i = 0; i < k; i++) { logits[i] = Math.Exp(logits[i] - mx); sum += logits[i]; }
                var probs = new float[k];
                for (int i = 0; i < k; i++) probs[i] = (float)(logits[i] / Math.Max(sum, 1e-12));

                int pred = 0; float best = probs[0];
                for (int i = 1; i < k; i++) if (probs[i] > best) { best = probs[i]; pred = i; }

                UpdateAxisClassGauge(axis, probs, pred);
            }

            AppendEventLog("[TEST] 확률을 무작위로 갱신했습니다.");
        }

        private ProbGaugeControl EnsureAxisProbGauge(int axis, int k)
        {
            if (_axisGauges.TryGetValue(axis, out var existed) && existed != null && !existed.IsDisposed)
                return existed;

            var box = new GroupBox
            {
                Text = $"Axis {axis} 분류 확률",
                // ↓↓↓ 컴팩트하게
                Padding = new Padding(6, 4, 6, 4),
                Margin = new Padding(4),
                Width = Math.Max((axisGaugeFlow?.DisplayRectangle.Width ?? 360) - 10, 240),
                Height = 120,                     // 기본 높이(자동 분배에서 다시 세팅됨)
                AutoSize = false,
                MinimumSize = new Size(240, 80)
            };

            var gauge = new ProbGaugeControl
            {
                Dock = DockStyle.Fill,
                Labels = (_clsLabels?.Length == k) ? _clsLabels : Enumerable.Range(0, k).Select(i => $"Class {i}").ToArray(),
                Probs = Enumerable.Repeat(0f, k).ToArray(),
                PredIndex = -1,
                SortDesc = GAUGE_SORT_DESC,
                // ↓↓↓ (2)에서 추가할 컴팩트 스타일 호출 (기존 컨트롤에 속성 추가했다면 세팅)
                // BarHeight = 12, BarGap = 4, LeftLabelWidth = 96, RightValueWidth = 48, InnerPadding = new Padding(6,2,6,2)
            };

            box.Controls.Add(gauge);
            axisGaugeFlow?.Controls.Add(box);

            _axisGauges[axis] = gauge;

            // 추가: 생성 시점에 레이아웃 재분배
            ReflowGaugeHeights();
            return gauge;
        }

        private void ReflowGaugeHeights()
        {
            if (axisGaugeFlow == null || axisGaugeFlow.IsDisposed) return;

            // 그룹박스들만 대상
            var boxes = axisGaugeFlow.Controls.OfType<GroupBox>().Where(g => g.Visible).ToList();
            if (boxes.Count == 0) return;

            // 사용 가능한 높이(패딩/스크롤바 여유 약간 제외)
            int availH = Math.Max(0, axisGaugeFlow.ClientSize.Height - axisGaugeFlow.Padding.Vertical - 6);
            int n = boxes.Count;

            // 마진 합(간격) 추정: 각 박스 마진 위/아래 4px씩 → 박스 사이 n-1개 + 위/아래 1개씩
            int perMargin = 8; // 위4 + 아래4
            int totalMargin = perMargin * (n + 1);

            // 박스 목표 높이 (최소 80, 최대 200 정도로 클램프)
            int h = (availH > totalMargin)
                ? (availH - totalMargin) / n
                : 80;

            h = Math.Max(80, Math.Min(200, h));

            // 각 박스에 세팅
            foreach (var gb in boxes)
            {
                if (gb.MinimumSize.Height > 0) gb.MinimumSize = new Size(gb.MinimumSize.Width, 0);
                gb.Height = h;
            }

            // 게이지 내부도 축소(ProbGaugeControl에 축소 속성/메서드가 있다면 여기서 스케일)
            float scale = Math.Min(1f, h / 140f); // 140을 '기본 기준 높이'로 가정
            foreach (var g in boxes.Select(b => b.Controls.OfType<ProbGaugeControl>().FirstOrDefault()).Where(x => x != null))
                ApplyGaugeCompactScale(g, scale);
        }

        private void ApplyGaugeCompactScale(ProbGaugeControl g, float scale)
        {
            try
            {
                dynamic dg = g;  // 컴파일 에러 싫으면 속성 있나 검사해서 세팅하세요.
                if (HasProp(dg, "BarHeight")) dg.BarHeight = Math.Max(8, (int)(14 * scale));
                if (HasProp(dg, "BarGap")) dg.BarGap = Math.Max(2, (int)(4 * scale));
                if (HasProp(dg, "LeftLabelWidth")) dg.LeftLabelWidth = Math.Max(72, (int)(96 * scale));
                if (HasProp(dg, "RightValueWidth")) dg.RightValueWidth = Math.Max(36, (int)(48 * scale));
                if (HasProp(dg, "InnerPadding")) dg.InnerPadding = new Padding(6, Math.Max(2, (int)(4 * scale)), 6, Math.Max(2, (int)(4 * scale)));
                if (HasProp(dg, "Font"))
                {
                    var f = g.Font;
                    g.Font = new Font(f.FontFamily, Math.Max(8f, f.Size * Clamp(scale, 0.75f, 1f)), f.Style);
                }
                g.Invalidate();
            }
            catch { /* ProbGaugeControl에 해당 속성이 없으면 무시 */ }

            bool HasProp(dynamic obj, string name)
            {
                try { var _ = obj.GetType().GetProperty(name); return _ != null; } catch { return false; }
            }
        }

        private static float Clamp(float value, float min, float max)
        {
            if (value < min) return min;
            if (value > max) return max;
            return value;
        }

        private void UpdateAxisClassGauge(int axis, float[] probs, int pred = -1)
        {
            if (probs == null || probs.Length == 0) return;
            if (InvokeRequired) { BeginInvoke(new Action(() => UpdateAxisClassGauge(axis, probs, pred))); return; }

            var g = EnsureAxisProbGauge(axis, probs.Length);
            var labels = (_clsLabels != null && _clsLabels.Length == probs.Length)
                         ? _clsLabels
                         : Enumerable.Range(0, probs.Length).Select(i => $"Class {i}").ToArray();
            g.SetData(labels, probs, pred);

            if (g.Parent is GroupBox gb)
            {
                string predLabel = (pred >= 0 && pred < labels.Length) ? labels[pred] : "-";
                float pv = (pred >= 0 && pred < probs.Length) ? probs[pred] : 0f;
                gb.Text = $"Axis {axis} 분류 확률 · pred: {predLabel} (p={pv:0.000})";
            }

            ReflowGaugeHeights(); // ★ 추가
        }

        // ── DL ONNX 단일 등록 ────────────────────────────────────────────────────
        // _meta.json 자동 파싱 → 단일 확인 폼 → 등록 (기존 7단계 InputBox 대체)

        private sealed class OnnxMeta
        {
            public string   Kind;
            public string   YColumn;
            public string[] Channels;
            public int      NChannels   = 1;
            public string[] ClassNames;
            public bool     IsAe;
            public double   Threshold   = -1;   // -1 = 없음(기본값 사용)
            public double   RmsMean     = -1;   // AE 정상 RMS 평균
            public double   RmsThr      = -1;   // AE RMS 임계값
            public int      WindowSize  = 256;   // 학습 시 슬라이딩 윈도우 크기
        }

        private static OnnxMeta TryParseOnnxMeta(string onnxPath)
        {
            string metaPath = Path.Combine(
                Path.GetDirectoryName(onnxPath) ?? ".",
                Path.GetFileNameWithoutExtension(onnxPath) + "_meta.json");
            if (!File.Exists(metaPath)) return null;
            try
            {
                using (var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(metaPath)))
                {
                    var root = doc.RootElement;
                    var m = new OnnxMeta();
                    if (root.TryGetProperty("kind",      out var kp)) m.Kind    = kp.GetString();
                    if (root.TryGetProperty("y_column",  out var yp)) m.YColumn = yp.GetString();
                    if (root.TryGetProperty("n_channels",out var nc)) m.NChannels = nc.GetInt32();
                    if (root.TryGetProperty("channels",  out var cp) &&
                        cp.ValueKind == System.Text.Json.JsonValueKind.Array)
                    {
                        var list = new System.Collections.Generic.List<string>();
                        foreach (var c in cp.EnumerateArray()) list.Add(c.GetString() ?? "");
                        m.Channels  = list.ToArray();
                        m.NChannels = list.Count;
                    }
                    if (root.TryGetProperty("class_names", out var cn) &&
                        cn.ValueKind == System.Text.Json.JsonValueKind.Array)
                    {
                        var list = new System.Collections.Generic.List<string>();
                        foreach (var n in cn.EnumerateArray()) list.Add(n.GetString() ?? "");
                        m.ClassNames = list.ToArray();
                    }
                    string session = "";
                    if (root.TryGetProperty("session", out var sp)) session = sp.GetString() ?? "";
                    m.IsAe = session.ToUpper() == "AD"
                          || (m.Kind?.ToUpper().Contains("AD") ?? false)
                          || (m.ClassNames == null || m.ClassNames.Length == 0);
                    if (root.TryGetProperty("threshold", out var tp))
                    {
                        double tv;
                        if (tp.TryGetDouble(out tv)) m.Threshold = tv;
                    }
                    if (root.TryGetProperty("rms_mean", out var rmp))
                    {
                        double rv; if (rmp.TryGetDouble(out rv)) m.RmsMean = rv;
                    }
                    if (root.TryGetProperty("rms_thr", out var rtp))
                    {
                        double rv; if (rtp.TryGetDouble(out rv)) m.RmsThr = rv;
                    }
                    if (root.TryGetProperty("window_size", out var wp))
                    {
                        int wv;
                        if (wp.TryGetInt32(out wv) && wv > 0) m.WindowSize = wv;
                    }
                    return m;
                }
            }
            catch { return null; }
        }

        private void LoadOnnxModelSingle()
        {
            // 1) ONNX 파일 선택
            string onnxPath;
            using (var ofd = new OpenFileDialog
            {
                Title = "DL ONNX 모델 선택",
                Filter = "ONNX 모델 (*.onnx)|*.onnx|모든 파일 (*.*)|*.*",
            })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;
                onnxPath = ofd.FileName;
            }

            // 2) _meta.json 자동 파싱
            var meta = TryParseOnnxMeta(onnxPath);

            // 3) 파일명 + 상위 폴더에서 축 번호 추론 (axis0, axis_0, axis-0, ...)
            int inferAxis = 0;
            var axisMatch = System.Text.RegularExpressions.Regex.Match(
                Path.GetFileNameWithoutExtension(onnxPath), @"(?i)axis[\s_-]?(\d+)");
            if (axisMatch.Success)
                int.TryParse(axisMatch.Groups[1].Value, out inferAxis);
            else
            {
                // 상위 디렉토리명에서 재시도
                string dirName = Path.GetFileName(Path.GetDirectoryName(onnxPath) ?? "") ?? "";
                var dirMatch = System.Text.RegularExpressions.Regex.Match(dirName, @"(?i)axis[\s_-]?(\d+)");
                if (dirMatch.Success) int.TryParse(dirMatch.Groups[1].Value, out inferAxis);
            }

            // 4) AE 추론
            bool inferAe = meta != null ? meta.IsAe
                : Path.GetFileNameWithoutExtension(onnxPath).IndexOf("ae", StringComparison.OrdinalIgnoreCase) >= 0;

            // 5) Y 컬럼 추론 — 다채널이면 channels 전체, 단채널이면 y_column
            string inferYCol;
            if (meta?.Channels != null && meta.Channels.Length > 1)
                inferYCol = string.Join(", ", meta.Channels);
            else if (meta?.YColumn != null)
                inferYCol = meta.YColumn;
            else if (Path.GetFileNameWithoutExtension(onnxPath).IndexOf("torque", StringComparison.OrdinalIgnoreCase) >= 0)
                inferYCol = "Trq(%)";
            else
                inferYCol = "x";

            // 6) 채널 수 추론
            int inferC = meta?.NChannels > 0 ? meta.NChannels : 1;

            // ── 단일 등록 폼 (모든 필드 읽기 전용, AE 임계값만 편집 가능) ────
            NumericUpDown numThr = null;
            Label         lblThrLabel;

            // 읽기 전용 값 레이블 헬퍼
            Label ValLbl(string text, bool highlight = false) => new Label
            {
                Text = text, Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                ForeColor = highlight ? Color.FromArgb(0, 100, 180) : Color.FromArgb(30, 30, 30),
                Font = highlight ? new Font(Font.FontFamily, Font.Size, FontStyle.Bold) : Font,
            };

            var form = new Form
            {
                Text = "DL ONNX 모델 등록",
                Size = new Size(420, 300),
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition   = FormStartPosition.CenterParent,
                MaximizeBox = false, MinimizeBox = false,
            };
            int rowCount = inferAe ? 7 : 6;
            var tl = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 2, RowCount = rowCount,
                Padding = new Padding(14, 10, 14, 8),
            };
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 90));
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            for (int i = 0; i < rowCount - 1; i++) tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 30));
            tl.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

            int r = 0;

            // 파일명
            tl.Controls.Add(Lbl("파일:"), 0, r);
            tl.Controls.Add(ValLbl(Path.GetFileName(onnxPath)), 1, r++);

            // 축 번호 (읽기 전용)
            tl.Controls.Add(Lbl("축 번호:"), 0, r);
            tl.Controls.Add(ValLbl(inferAxis.ToString(), highlight: true), 1, r++);

            // 역할 (읽기 전용)
            tl.Controls.Add(Lbl("역할:"), 0, r);
            tl.Controls.Add(ValLbl(inferAe ? "오토인코더 (AE)" : "분류 (CLS)", highlight: true), 1, r++);

            // Y 컬럼 (읽기 전용)
            tl.Controls.Add(Lbl("Y 컬럼:"), 0, r);
            tl.Controls.Add(ValLbl(inferYCol, highlight: true), 1, r++);

            // 채널 수 (읽기 전용)
            tl.Controls.Add(Lbl("채널 수(C):"), 0, r);
            tl.Controls.Add(ValLbl(inferC.ToString(), highlight: true), 1, r++);

            // AE 임계값 (AE 모드에서만 표시, 유일한 편집 가능 필드)
            if (inferAe)
            {
                double initThr = (meta != null && meta.Threshold > 0) ? meta.Threshold : DefaultThreshold;
                lblThrLabel = Lbl("임계값(AE):");
                tl.Controls.Add(lblThrLabel, 0, r);
                numThr = new NumericUpDown { Dock = DockStyle.Fill, Minimum = 0, Maximum = 100000,
                    DecimalPlaces = 6, Value = (decimal)initThr, Increment = 0.001m };
                tl.Controls.Add(numThr, 1, r++);
            }

            // meta 정보 + 버튼
            string metaText = meta != null
                ? $"✔ _meta.json 로드 — {(meta.IsAe ? "AE" : "분류")} | 클래스: {string.Join(", ", meta.ClassNames ?? new string[0])} | 채널: {string.Join(",", meta.Channels ?? new string[0])}"
                : "⚠ _meta.json 없음 — 파일명으로 자동 추론";
            var lblMeta = new Label { Text = metaText, AutoSize = false, Dock = DockStyle.Fill,
                ForeColor = meta != null ? Color.DarkGreen : Color.DarkOrange,
                TextAlign = ContentAlignment.TopLeft };

            var btnOk     = new Button { Text = "등록", Width = 80, Height = 26, DialogResult = DialogResult.OK };
            var btnCancel = new Button { Text = "취소", Width = 80, Height = 26, DialogResult = DialogResult.Cancel };
            var btnFlow   = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 30, FlowDirection = FlowDirection.RightToLeft };
            btnFlow.Controls.Add(btnCancel);
            btnFlow.Controls.Add(btnOk);
            var bottomPanel = new Panel { Dock = DockStyle.Fill };
            bottomPanel.Controls.Add(btnFlow);
            bottomPanel.Controls.Add(lblMeta);
            tl.SetColumnSpan(bottomPanel, 2);
            tl.Controls.Add(bottomPanel, 0, r);

            form.Controls.Add(tl);
            form.AcceptButton = btnOk;
            form.CancelButton = btnCancel;

            if (form.ShowDialog(this) != DialogResult.OK) return;

            // 7) 등록 — 모든 값은 자동 추론된 값 사용
            int    axis      = inferAxis;
            bool   isAe      = inferAe;
            // inferYCol이 "x, y, z" 처럼 다채널인 경우 YColumn에는 첫 번째 채널명만 저장
            string yCol      = inferYCol.Contains(",")
                ? inferYCol.Split(',')[0].Trim()
                : inferYCol;
            int    C         = inferC;
            string kind      = meta?.Kind ?? (isAe ? "AE-CNN1D" : "CNN1D-CLS");
            double threshold = isAe && numThr != null ? (double)numThr.Value : DefaultThreshold;

            try
            {
                InferenceSession session;
                try { session = new InferenceSession(onnxPath); }
                catch (Exception ex) { MessageBox.Show("ONNX 로드 실패: " + ex.Message); return; }

                var om = new OnnxAxisModel
                {
                    AxisId              = axis,
                    ModelPath           = onnxPath,
                    YColumn             = yCol,
                    C                   = C,
                    Kind                = kind,
                    InputName           = "input",
                    OutputName          = isAe ? null    : "logits",
                    ReconOutputName     = isAe ? "recon" : null,
                    IsAutoencoder       = isAe,
                    StandardizePerSample = true,   // AE도 per-sample z-score 사용 (형태 학습)
                    RmsMean             = (isAe && meta != null) ? meta.RmsMean : -1,
                    RmsThr              = (isAe && meta != null) ? meta.RmsThr  : -1,
                    WindowSize          = (isAe && meta != null && meta.WindowSize > 0) ? meta.WindowSize : 256,
                    Threshold           = threshold,
                    Session             = session,
                };

                if (isAe)
                {
                    OnnxAxisModel old;
                    if (_axisOnnx.TryGetValue(axis, out old) && old?.Session != null)
                        try { old.Session.Dispose(); } catch { }
                    _axisOnnx[axis] = om;
                    AppendEventLog($"[ONNX-AE] 축 {axis}: {Path.GetFileName(onnxPath)}  Y={yCol}  C={C}  thr={threshold:0.###}");
                }
                else
                {
                    OnnxAxisModel oldCls;
                    if (_axisOnnxCls.TryGetValue(axis, out oldCls) && oldCls?.Session != null)
                        try { oldCls.Session.Dispose(); } catch { }
                    _axisOnnxCls[axis] = om;

                    int k = GetNumClassesFromOnnx(session, om.OutputName ?? "logits",
                        meta?.ClassNames?.Length ?? _clsLabels?.Length ?? 4);
                    if (meta?.ClassNames != null && meta.ClassNames.Length == k)
                        _clsLabels = meta.ClassNames;
                    SeedAxisGauge(axis, k);
                    AppendEventLog($"[ONNX-CLS] 축 {axis}: {Path.GetFileName(onnxPath)}  클래스=[{string.Join(",", meta?.ClassNames ?? new[] { "?" })}]  C={C}");
                }

                RefreshModelPathList();
            }
            catch (Exception ex)
            {
                MessageBox.Show("모델 등록 오류: " + ex.Message);
            }
        }

        private static Label Lbl(string text) =>
            new Label { Text = text, AutoSize = false, Dock = DockStyle.Fill,
                        TextAlign = ContentAlignment.MiddleLeft };

        private void DisposeOnnxSessions()
        {
            foreach (var kv in _axisOnnx.ToList())
                try { kv.Value?.Session?.Dispose(); } catch { }
            _axisOnnx.Clear();

            foreach (var kv in _axisOnnxCls.ToList())
                try { kv.Value?.Session?.Dispose(); } catch { }
            _axisOnnxCls.Clear();

            foreach (var kv in _axisSklModels.ToList())
                try { kv.Value?.OnnxSession?.Dispose(); } catch { }
            _axisSklModels.Clear();
        }

        private static float[,] BuildSequenceFromCsvSingleChannel(string filePath, string yColumn)
        {
            // CSV에서 헤더 읽기
            string header;
            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var sr = new StreamReader(fs))
                header = sr.ReadLine();

            if (string.IsNullOrEmpty(header)) return null;
            var headers = SplitCsvLine(header);
            int yIdx = Array.FindIndex(headers, h => string.Equals(h?.Trim(), yColumn, StringComparison.OrdinalIgnoreCase));
            if (yIdx < 0) return null;

            var ys = new List<float>(8192);
            var cult = CultureInfo.InvariantCulture;

            using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var sr = new StreamReader(fs))
            {
                sr.ReadLine(); // skip header
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = SplitCsvLine(line);
                    if (yIdx >= parts.Length) continue;

                    if (!float.TryParse(parts[yIdx], NumberStyles.Float, cult, out float yv))
                    {
                        if (!float.TryParse(parts[yIdx], NumberStyles.Float, CultureInfo.CurrentCulture, out yv))
                            continue;
                    }
                    if (float.IsNaN(yv) || float.IsInfinity(yv)) continue;
                    ys.Add(yv);
                }
            }
            if (ys.Count == 0) return null;

            // (T,C=1)
            var seq = new float[ys.Count, 1];
            for (int t = 0; t < ys.Count; t++) seq[t, 0] = ys[t];
            return seq;
        }

        private static void ZScoreInPlace(float[,] seq)
        {
            // seq: (T,C)
            int T = seq.GetLength(0);
            int C = seq.GetLength(1);
            if (T <= 1) return;

            for (int c = 0; c < C; c++)
            {
                double m = 0;
                for (int t = 0; t < T; t++) m += seq[t, c];
                m /= T;

                double v = 0;
                for (int t = 0; t < T; t++)
                {
                    double d = seq[t, c] - m;
                    v += d * d;
                }
                v = Math.Sqrt(v / Math.Max(1, T - 1));
                float std = (float)(v > 1e-8 ? v : 1e-8);

                for (int t = 0; t < T; t++)
                    seq[t, c] = (float)((seq[t, c] - m) / std);
            }
        }

        /// <summary>AE 전역 정규화: 학습 시 저장한 채널별 mean/std 로 정규화합니다.</summary>
        private static void GlobalNormalizeInPlace(float[,] seq, double[] mean, double[] std)
        {
            int T = seq.GetLength(0);
            int C = seq.GetLength(1);
            for (int c = 0; c < C; c++)
            {
                double m = (mean != null && c < mean.Length) ? mean[c] : 0.0;
                double s = (std  != null && c < std.Length  && std[c] > 1e-8) ? std[c] : 1.0;
                for (int t = 0; t < T; t++)
                    seq[t, c] = (float)((seq[t, c] - m) / s);
            }
        }

        private bool TryOnnxInferOnce(int axis, string csvPath, out int predClass, out float[] probs, out string info)
        {
            predClass = -1; probs = null; info = null;

            if (!_axisOnnxCls.TryGetValue(axis, out var om) || om?.Session == null) { info = "no onnx"; return false; }

            // 1) 시퀀스 만들기
            float[,] seq = (om.C > 1)
                ? BuildSequenceFromCsvAccel3(csvPath, om.YColumn, axis)  // ★ axis 전달
                : BuildSequenceFromCsvSingleChannel(csvPath, om.YColumn);
            if (seq == null) { info = "no sequence"; return false; }

            int T = seq.GetLength(0);
            int C = seq.GetLength(1);
            if (C != om.C) { info = $"channel mismatch: model C={om.C}, csv C={C}"; return false; }

            // 2) High-pass (SciPy 계수와 동일 사용)
            // HighpassFilterInPlace(seq, cutoff: 128, fs: 1000, order: 5);

            // 2) z-score (학습과 동일)
            if (om.StandardizePerSample) ZScoreInPlace(seq);

            // 3) DenseTensor(1,T,C)
            var tensor = new DenseTensor<float>(new[] { 1, T, C });
            for (int t = 0; t < T; t++)
                for (int c = 0; c < C; c++)
                    tensor[0, t, c] = seq[t, c];

            // 4) ONNX 실행
            var inputs = new List<NamedOnnxValue>(1);
            inputs.Add(NamedOnnxValue.CreateFromTensor(om.InputName, tensor));

            using (var results = om.Session.Run(inputs))   // ✅ 전통적 using
            {
                var outNv = results.FirstOrDefault(v => v.Name == om.OutputName) ?? results.First();
                float[] logits = outNv.AsEnumerable<float>().ToArray();

                // softmax & argmax
                probs = Softmax(logits);
                predClass = Array.IndexOf(probs, probs.Max());
                info = string.Format("{0}({1})", om.Kind, Path.GetFileNameWithoutExtension(om.ModelPath));
            }
            return true;
        }

        // 전역 모델(또는 임의 OnnxAxisModel)을 직접 지정해서 AE 스코어 계산
        private bool TryOnnxAeScoreOnce(int axis, string csvPath, OnnxAxisModel om, out double score, out string info)
        {
            score = 0; info = null;
            if (om == null || om.Session == null || !om.IsAutoencoder) { info = "no ae"; return false; }
            return TryOnnxAeScoreCore(axis, csvPath, om, out score, out info);
        }

        private bool TryOnnxAeScoreOnce(int axis, string csvPath, out double score, out string info)
        {
            score = 0; info = null;
            OnnxAxisModel om;
            if (!_axisOnnx.TryGetValue(axis, out om) || om == null || om.Session == null || !om.IsAutoencoder)
            { info = "no ae"; return false; }
            return TryOnnxAeScoreCore(axis, csvPath, om, out score, out info);
        }

        private bool TryOnnxAeScoreCore(int axis, string csvPath, OnnxAxisModel om, out double score, out string info)
        {
            score = 0; info = null;

            // 1) 입력 시퀀스 생성
            float[,] seq = (om.C > 1)
                ? BuildSequenceFromCsvAccel3(csvPath, om.YColumn, axis)  // ★ axis 전달
                : BuildSequenceFromCsvSingleChannel(csvPath, om.YColumn);
            if (seq == null) { info = "no sequence"; return false; }

            int T = seq.GetLength(0), C = seq.GetLength(1);
            if (C != om.C) { info = "channel mismatch"; return false; }

            // 2) 슬라이딩 윈도우 추론
            //    학습과 동일한 window_size로 분할 → 각 윈도우별 (z-score 후) MAE 계산 → 최대값 사용
            int W      = om.WindowSize > 0 ? om.WindowSize : 256;
            int stride = Math.Max(1, W / 2);  // 50% 오버랩

            // RMS 계산은 z-score 전 원본 값으로 (진폭 보존)
            double windowRmsMax = 0.0;

            if (T < W)
            {
                double rawRms = ComputeRms(seq, 0, T);
                if (om.StandardizePerSample) ZScoreInPlace(seq);
                double mae = AeInferWindow(om, seq, 0, T);
                score = CompositeAeScore(mae, rawRms, om);
                info  = string.Format("AE W={0} single", T);
                return score >= 0;
            }

            double maxMae = 0.0;
            int    wCount = 0;
            for (int start = 0; start <= T - W; start += stride)
            {
                // raw RMS (정규화 전)
                double rawRms = ComputeRms(seq, start, W);
                if (rawRms > windowRmsMax) windowRmsMax = rawRms;

                // z-score 슬라이스 복사 후 추론
                double mae = AeInferWindowZScore(om, seq, start, W);
                if (mae < 0) continue;
                if (mae > maxMae) maxMae = mae;
                wCount++;
            }

            if (wCount == 0) { info = "no valid window"; return false; }

            score = CompositeAeScore(maxMae, windowRmsMax, om);
            info  = string.Format("AE W={0} n={1} mae={2:F4} rms={3:F4}",
                W, wCount, maxMae, windowRmsMax);
            return true;
        }

        /// <summary>RAW 슬라이스의 전채널 RMS를 계산합니다.</summary>
        private static double ComputeRms(float[,] seq, int start, int length)
        {
            int C = seq.GetLength(1);
            double sum = 0.0; long n = 0;
            for (int t = 0; t < length; t++)
                for (int c = 0; c < C; c++)
                {
                    double v = seq[start + t, c];
                    sum += v * v; n++;
                }
            return n > 0 ? Math.Sqrt(sum / n) : 0.0;
        }

        /// <summary>복합 스코어 = max(mae_ratio, rms_ratio) — 각 비율이 1을 넘으면 이상.</summary>
        private static double CompositeAeScore(double mae, double rms, OnnxAxisModel om)
        {
            if (mae < 0) return -1;
            double maeRatio = om.Threshold > 0 ? mae / om.Threshold : mae;
            double rmsRatio = (om.RmsThr > 0 && rms > 0)
                ? Math.Max(0.0, (rms - om.RmsMean) / (om.RmsThr - om.RmsMean + 1e-9))
                : 0.0;
            // 최종 스코어: mae 직접 반환, rms 이상은 mae를 스케일업하여 표현
            // score = mae * (1 + rms_excess) — threshold와 같은 단위 유지
            double rmsExcess = Math.Max(0.0, rmsRatio - 1.0);
            return mae * (1.0 + rmsExcess * 2.0);
        }

        /// <summary>슬라이스를 z-score 후 AE에 통과해 MAE를 반환합니다. 실패 시 -1.</summary>
        private double AeInferWindowZScore(OnnxAxisModel om, float[,] seq, int start, int length)
        {
            int C = seq.GetLength(1);
            // 슬라이스 복사 + z-score
            float[,] win = new float[length, C];
            for (int t = 0; t < length; t++)
                for (int c = 0; c < C; c++)
                    win[t, c] = seq[start + t, c];
            ZScoreInPlace(win);

            return AeInferWindow(om, win, 0, length);
        }

        /// <summary>이미 정규화된 seq의 [start, start+length) 슬라이스를 AE에 통과해 MAE를 반환합니다. 실패 시 -1.</summary>
        private double AeInferWindow(OnnxAxisModel om, float[,] seq, int start, int length)
        {
            int C = seq.GetLength(1);
            var tensor = new DenseTensor<float>(new[] { 1, length, C });
            for (int t = 0; t < length; t++)
                for (int c = 0; c < C; c++)
                    tensor[0, t, c] = seq[start + t, c];

            var inputs = new List<NamedOnnxValue>
                { NamedOnnxValue.CreateFromTensor(om.InputName, tensor) };
            try
            {
                using (var results = om.Session.Run(inputs))
                {
                    var outNv     = results.FirstOrDefault(v => v.Name == om.ReconOutputName) ?? results.First();
                    var reconFlat = outNv.AsEnumerable<float>().ToArray();
                    if (reconFlat.Length < length * C) return -1;

                    float[,] recon = new float[length, C];
                    int idx = 0;
                    for (int t = 0; t < length; t++)
                        for (int c = 0; c < C; c++)
                            recon[t, c] = reconFlat[idx++];
                    return MeanAbsoluteError(seq.GetLength(0) == length ? seq
                        : ExtractSlice(seq, start, length), recon);
                }
            }
            catch { return -1; }
        }

        private static float[,] ExtractSlice(float[,] seq, int start, int length)
        {
            int C = seq.GetLength(1);
            var s = new float[length, C];
            for (int t = 0; t < length; t++)
                for (int c = 0; c < C; c++)
                    s[t, c] = seq[start + t, c];
            return s;
        }

        private static float[] Softmax(IReadOnlyList<float> logits)
        {
            if (logits == null || logits.Count == 0) return Array.Empty<float>();
            float max = logits.Max();
            double sum = 0; var exps = new double[logits.Count];
            for (int i = 0; i < logits.Count; i++) { double e = Math.Exp(logits[i] - max); exps[i] = e; sum += e; }
            var p = new float[logits.Count];
            for (int i = 0; i < logits.Count; i++) p[i] = (float)(exps[i] / sum);
            return p;
        }

        private void RenderSampleChartSafe(string csvPath, string yColumn, int axis)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(csvPath) || string.IsNullOrWhiteSpace(yColumn)) return;
                if (!File.Exists(csvPath)) return;

                // 1) 헤더
                string headerLine;
                using (var fsH = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var srH = new StreamReader(fsH))
                    headerLine = srH.ReadLine();
                if (string.IsNullOrEmpty(headerLine)) return;

                var headers = SplitCsvLine(headerLine);
                if (headers == null || headers.Length == 0) return;

                // ---- X 인덱스/정보 결정 ----
                int xIdx = 0;
                string xName = headers[0]?.Trim();
                bool xIsTimeCol = IsTimeColumn(xName);
                bool xIsCycle = xIsTimeCol && xName.Equals("cycle", StringComparison.OrdinalIgnoreCase);

                // Y 인덱스 (단일 채널용)
                int yIdx = Array.FindIndex(headers, h => string.Equals(h?.Trim(), yColumn, StringComparison.OrdinalIgnoreCase));

                // 가속도 컬럼 유무 (축 인지)
                bool hasAccel = TryResolveAccelColumns(headers, yColumn, axis, out int accX, out int accY, out int accZ);

                double sp = SamplePeriodFor(yColumn);
                var culture = CultureInfo.InvariantCulture;

                // 2) CSV -> 데이터 읽기
                var xsSec = new List<double>(8192); // X(초)
                var yMain = new List<double>(8192); // 메인 표시 (가속도면 magnitude, 토크면 FBTRQ)

                // 가속도 채널 개별(옵션 표시)
                List<double> chX = null, chY = null, chZ = null;

                using (var fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var str = new StreamReader(fs))
                {
                    str.ReadLine(); // skip header
                    string line;
                    int i = 0; double? firstX = null;

                    while ((line = str.ReadLine()) != null)
                    {
                        if (string.IsNullOrWhiteSpace(line)) { i++; continue; }
                        var parts = SplitCsvLine(line);

                        // X → seconds
                        double xSec;
                        if (xIsTimeCol)
                        {
                            if (double.TryParse(parts[xIdx], NumberStyles.Float, culture, out var xv) ||
                                double.TryParse(parts[xIdx], NumberStyles.Float, CultureInfo.CurrentCulture, out xv))
                            { if (!firstX.HasValue) firstX = xv; xSec = xIsCycle ? (xv - firstX.Value) * Math.Max(sp, 0) : (xv - firstX.Value); }
                            else xSec = i * Math.Max(sp, 0);
                        }
                        else if (double.TryParse(parts[xIdx], NumberStyles.Float, culture, out var xvNum) ||
                                 double.TryParse(parts[xIdx], NumberStyles.Float, CultureInfo.CurrentCulture, out xvNum))
                        { if (!firstX.HasValue) firstX = xvNum; xSec = (xvNum - firstX.Value); }
                        else if (DateTime.TryParse(parts[xIdx], CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out var xd) ||
                                 DateTime.TryParse(parts[xIdx], CultureInfo.CurrentCulture, DateTimeStyles.AssumeLocal, out xd))
                        { if (!firstX.HasValue) firstX = xd.ToOADate(); xSec = (xd.ToOADate() - firstX.Value) * 24.0 * 3600.0; }
                        else xSec = i * Math.Max(sp, 0);

                        if (hasAccel)
                        {
                            if (parts.Length <= Math.Max(xIdx, Math.Max(accX, Math.Max(accY, accZ)))) { i++; continue; }

                            if (!double.TryParse(parts[accX], NumberStyles.Float, culture, out var vx) &&
                                !double.TryParse(parts[accX], NumberStyles.Float, CultureInfo.CurrentCulture, out vx)) { i++; continue; }
                            if (!double.TryParse(parts[accY], NumberStyles.Float, culture, out var vy) &&
                                !double.TryParse(parts[accY], NumberStyles.Float, CultureInfo.CurrentCulture, out vy)) { i++; continue; }
                            if (!double.TryParse(parts[accZ], NumberStyles.Float, culture, out var vz) &&
                                !double.TryParse(parts[accZ], NumberStyles.Float, CultureInfo.CurrentCulture, out vz)) { i++; continue; }

                            double mag = Math.Sqrt(vx * vx + vy * vy + vz * vz);
                            xsSec.Add(xSec);
                            yMain.Add(mag);

                            if (chX == null) { chX = new List<double>(8192); chY = new List<double>(8192); chZ = new List<double>(8192); }
                            chX.Add(vx); chY.Add(vy); chZ.Add(vz);
                        }
                        else
                        {
                            if (yIdx < 0 || parts.Length <= Math.Max(xIdx, yIdx)) { i++; continue; }
                            if (!double.TryParse(parts[yIdx], NumberStyles.Float, culture, out var yv) &&
                                !double.TryParse(parts[yIdx], NumberStyles.Float, CultureInfo.CurrentCulture, out yv)) { i++; continue; }

                            xsSec.Add(xSec);
                            yMain.Add(yv);
                        }

                        i++;
                    }
                }
                if (yMain.Count == 0) return;

                // 3) 다운샘플
                const int MaxDisplayPoints = 4000;
                DownsampleMinMax(xsSec, yMain, MaxDisplayPoints, out var dx, out var dy);
                if (dy == null || dy.Length == 0) return; // 안전장치

                double[] dx1 = null, dy1 = null, dx2 = null, dy2 = null, dx3 = null, dy3 = null;
                if (hasAccel && chX != null)
                {
                    DownsampleMinMax(xsSec, chX, MaxDisplayPoints, out dx1, out dy1);
                    DownsampleMinMax(xsSec, chY, MaxDisplayPoints, out dx2, out dy2);
                    DownsampleMinMax(xsSec, chZ, MaxDisplayPoints, out dx3, out dy3);
                }

                // 4) 차트 바인딩 (UI 스레드)
                BeginInvoke(new Action(() =>
                {
                    var chart = EnsureSampleChartForAxis(axis);
                    if (chart == null || chart.IsDisposed) return;

                    chart.BeginInit();
                    try
                    {
                        var area = chart.ChartAreas["s"];

                        // 메인 시리즈: magnitude or torque
                        var sMain = chart.Series["Sample"];
                        sMain.Points.DataBindXY(dx, dy);

                        area.AxisX.Title = xName ?? "X";
                        area.AxisY.Title = hasAccel ? "Accel |a|" : yColumn;
                        area.AxisY.LabelStyle.Format = "0.0";
                        if (xIsTimeCol) area.AxisX.LabelStyle.Format = "0.###";
                        area.AxisX.Minimum = double.NaN; area.AxisX.Maximum = double.NaN;
                        area.AxisY.Minimum = double.NaN; area.AxisY.Maximum = double.NaN;
                        area.RecalculateAxesScale();

                        chart.Titles.Clear();
                        chart.Titles.Add(hasAccel
                            ? $"X:{(xName ?? "col0")} · Y:|{yColumn}| (magnitude)"
                            : $"X:{(xName ?? "col0")} · Y:{yColumn}");

                        // 보조 라인(가속도 개별 채널)
                        if (hasAccel)
                        {
                            var sX = EnsureSeries(chart, "ax");
                            var sY = EnsureSeries(chart, "ay");
                            var sZ = EnsureSeries(chart, "az");
                            sX.Points.DataBindXY(dx1 ?? Array.Empty<double>(), dy1 ?? Array.Empty<double>());
                            sY.Points.DataBindXY(dx2 ?? Array.Empty<double>(), dy2 ?? Array.Empty<double>());
                            sZ.Points.DataBindXY(dx3 ?? Array.Empty<double>(), dy3 ?? Array.Empty<double>());
                        }
                        else
                        {
                            // 토크 모드에선 ax/ay/az 숨김(있다면)
                            foreach (var name in new[] { "ax", "ay", "az" })
                            {
                                var s = chart.Series.FindByName(name);
                                if (s != null) s.Points.Clear();
                            }
                        }
                    }
                    finally { chart.EndInit(); }
                }));
            }
            catch { /* swallow */ }
        }

        private Series EnsureSeries(Chart ch, string name)
        {
            var s = ch.Series.FindByName(name);
            if (s != null) return s;
            s = new Series(name)
            {
                ChartType = SeriesChartType.FastLine,
                XValueType = ChartValueType.Double,
                BorderWidth = 2,
                ChartArea = "s"
            };
            ch.Series.Add(s);
            return s;
        }

        // ── 인메모리 배열로 샘플 차트 렌더링 (DB 모드용) ─────────────────────
        private void RenderSampleChartFromArrays(
            int axis, string yLabel,
            IList<double> xsSec, IList<double> yMain,
            IList<double> chX, IList<double> chY, IList<double> chZ,
            bool hasAccel)
        {
            try
            {
                if (yMain == null || yMain.Count == 0) return;

                const int MaxDisplayPoints = 4000;
                DownsampleMinMax(xsSec, yMain, MaxDisplayPoints, out var dx, out var dy);
                if (dy == null || dy.Length == 0) return;

                double[] dx1 = null, dy1 = null, dx2 = null, dy2 = null, dx3 = null, dy3 = null;
                if (hasAccel && chX != null && chX.Count > 0)
                {
                    DownsampleMinMax(xsSec, chX, MaxDisplayPoints, out dx1, out dy1);
                    DownsampleMinMax(xsSec, chY, MaxDisplayPoints, out dx2, out dy2);
                    DownsampleMinMax(xsSec, chZ, MaxDisplayPoints, out dx3, out dy3);
                }

                BeginInvoke(new Action(() =>
                {
                    var chart = EnsureSampleChartForAxis(axis);
                    if (chart == null || chart.IsDisposed) return;

                    chart.BeginInit();
                    try
                    {
                        var area = chart.ChartAreas["s"];

                        var sMain = chart.Series["Sample"];
                        sMain.Points.DataBindXY(dx, dy);

                        area.AxisX.Title = "Time (s)";
                        area.AxisY.Title = hasAccel ? "Accel |a|" : yLabel;
                        area.AxisY.LabelStyle.Format = "0.0";
                        area.AxisX.LabelStyle.Format = "0.###";
                        area.AxisX.Minimum = double.NaN; area.AxisX.Maximum = double.NaN;
                        area.AxisY.Minimum = double.NaN; area.AxisY.Maximum = double.NaN;
                        area.RecalculateAxesScale();

                        chart.Titles.Clear();
                        chart.Titles.Add(hasAccel
                            ? $"Axis {axis} · |{yLabel}| (magnitude) [DB]"
                            : $"Axis {axis} · {yLabel} [DB]");

                        if (hasAccel)
                        {
                            var sX = EnsureSeries(chart, "ax");
                            var sY = EnsureSeries(chart, "ay");
                            var sZ = EnsureSeries(chart, "az");
                            sX.Points.DataBindXY(dx1 ?? Array.Empty<double>(), dy1 ?? Array.Empty<double>());
                            sY.Points.DataBindXY(dx2 ?? Array.Empty<double>(), dy2 ?? Array.Empty<double>());
                            sZ.Points.DataBindXY(dx3 ?? Array.Empty<double>(), dy3 ?? Array.Empty<double>());
                        }
                        else
                        {
                            foreach (var name in new[] { "ax", "ay", "az" })
                            {
                                var s = chart.Series.FindByName(name);
                                if (s != null) s.Points.Clear();
                            }
                        }
                    }
                    finally { chart.EndInit(); }
                }));
            }
            catch { /* swallow */ }
        }

        // ── InfluxDB 세그먼트를 샘플 차트에 렌더링 ───────────────────────────
        private void RenderSegmentChart(SignalSegment seg, string yColumn, int axis)
        {
            if (seg == null || seg.Time == null || seg.Time.Length == 0) return;

            var xsSec = (IList<double>)seg.Time;

            bool isAccel = yColumn != null &&
                           (yColumn.Equals("x", StringComparison.OrdinalIgnoreCase) ||
                            yColumn.Equals("y", StringComparison.OrdinalIgnoreCase) ||
                            yColumn.Equals("z", StringComparison.OrdinalIgnoreCase));

            IList<double> yMain, chX = null, chY = null, chZ = null;
            if (isAccel)
            {
                // magnitude  |a| = sqrt(X²+Y²+Z²)
                double[] mag = new double[seg.Time.Length];
                double[] xa = seg.X ?? Array.Empty<double>();
                double[] ya = seg.Y ?? Array.Empty<double>();
                double[] za = seg.Z ?? Array.Empty<double>();
                for (int i = 0; i < mag.Length; i++)
                {
                    double vx = i < xa.Length ? xa[i] : 0;
                    double vy = i < ya.Length ? ya[i] : 0;
                    double vz = i < za.Length ? za[i] : 0;
                    mag[i] = Math.Sqrt(vx * vx + vy * vy + vz * vz);
                }
                yMain = mag;
                chX = seg.X; chY = seg.Y; chZ = seg.Z;
            }
            else
            {
                yMain = seg.GetChannel(yColumn) ?? Array.Empty<double>();
            }

            RenderSampleChartFromArrays(axis, yColumn ?? "signal",
                xsSec, yMain, chX, chY, chZ, isAccel);
        }

        private const int MaxEventLogLines = 400;

        private void AppendEventLog(string line)
        {
            if (txtEventLog == null || txtEventLog.IsDisposed) return;

            try
            {
                // 줄 추가
                txtEventLog.AppendText(line + Environment.NewLine);

                // 맨 아래로 스크롤
                txtEventLog.SelectionStart = txtEventLog.TextLength;
                txtEventLog.ScrollToCaret();

                // 라인 수 제한 (오래된 로그 제거)
                var ln = txtEventLog.Lines;
                if (ln.Length > MaxEventLogLines)
                {
                    txtEventLog.Lines = ln.Skip(ln.Length - MaxEventLogLines).ToArray();
                    txtEventLog.SelectionStart = txtEventLog.TextLength;
                    txtEventLog.ScrollToCaret();
                }
            }
            catch { /* ignore */ }
        }

        /// <summary>
        /// 축별 샘플 차트를 (없으면) 생성하고 반환.
        /// rightBottom의 sampleGrid에 배치하며, 축 개수에 따라 자동으로 그리드 재구성.
        /// </summary>
        private Chart EnsureSampleChartForAxis(int axis)
        {
            if (sampleCharts.TryGetValue(axis, out var existed))
                return existed;

            var box = new GroupBox
            {
                Text = $"Axis {axis} 최근 감지 샘플",
                Dock = DockStyle.Fill,
                Padding = new Padding(6, 8, 6, 6),
                Margin = new Padding(6),
                MinimumSize = Size.Empty,
                AutoSize = false
            };

            var ch = new Chart { Dock = DockStyle.Fill, Margin = Padding.Empty, MinimumSize = new Size(1, 1) };
            var caS = new ChartArea("s");
            // …(차트 설정 동일)…
            ch.ChartAreas.Add(caS);
            ch.Series.Add(new Series("Sample") { ChartType = SeriesChartType.FastLine, XValueType = ChartValueType.Double, BorderWidth = 2, ChartArea = "s" });

            box.Controls.Add(ch);
            sampleCharts[axis] = ch;

            // 기존 + 새 컨트롤로 재배치 (sampleGrid 없으면 무시)
            if (sampleGrid != null)
            {
                var list = sampleGrid.Controls.Cast<Control>().ToList();
                list.Add(box);
                RebuildSampleGrid(list);
            }

            return ch;
        }

        private void RebuildSampleGrid(IList<Control> boxes)
        {
            if (sampleGrid == null || boxes == null || boxes.Count == 0) return;

            int n = boxes.Count;
            int availW = Math.Max(1, sampleGrid.ClientSize.Width - sampleGrid.Padding.Horizontal);
            int colsByWidth = Math.Max(1, availW / Math.Max(1, SampleMinTileWidth));
            int cols = Math.Min(n, Math.Min(SampleMaxColumns, colsByWidth));
            int rows = (int)Math.Ceiling(n / (double)cols);

            sampleGrid.SuspendLayout();
            try
            {
                sampleGrid.Controls.Clear();
                sampleGrid.ColumnStyles.Clear();
                sampleGrid.RowStyles.Clear();
                sampleGrid.ColumnCount = cols;
                sampleGrid.RowCount = rows;

                for (int c = 0; c < cols; c++)
                    sampleGrid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / cols));
                for (int r = 0; r < rows; r++)
                    sampleGrid.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / rows));

                for (int i = 0; i < n; i++)
                {
                    int r = i / cols;
                    int c = i % cols;
                    var ctrl = boxes[i];
                    ctrl.MinimumSize = Size.Empty;
                    sampleGrid.Controls.Add(ctrl, c, r);
                }
            }
            finally { sampleGrid.ResumeLayout(); }
        }

        private void UpdateSampleGridLayout()
        {
            if (sampleGrid == null) return;
            RebuildSampleGrid(sampleGrid.Controls.Cast<Control>().ToList());
        }

        private void ShowToast(AlarmLevel level, int axis, double score)
            => ShowToast(level, $"Ax{axis}", score);

        /// <summary>센서 표시명(예: "가속도", "Ax0 토크")으로 토스트를 띄웁니다.</summary>
        private void ShowToast(AlarmLevel level, string sensorLabel, double score)
        {
            if (_notifier == null) return;
            if (level != AlarmLevel.Danger) return;

            _notifier.BalloonTipTitle = "⚠ 위험 감지";
            _notifier.BalloonTipText  = $"{sensorLabel}  {DateTime.Now:HH:mm:ss}  Score {score:F2}";
            _notifier.ShowBalloonTip(3000);
        }
        #region KPI/이벤트
        private class EventRow
        {
            public string TimeLine    { get; set; }
            public string Sensor      { get; set; }   // "가속도", "토크 Ax0" 등
            public double AnomalyScore { get; set; }
            public double Threshold   { get; set; }
            public string Alarm       { get; set; }
        }

        private enum AlarmLevel { Normal, Warning, Danger }
        #endregion

        #region 모델 로드/표시
        private void LoadAxisModelSingle()
        {
            using (OpenFileDialog ofd = new OpenFileDialog { Filter = "PHM Model (*.json)|*.json|All files (*.*)|*.*" })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;

                PersistedKnnModel model; string err;
                if (!TryLoadModelFromPath(ofd.FileName, out model, out err))
                { MessageBox.Show("모델 로드 실패: " + err); return; }

                const int GlobalKey = 0;
                _axisModels[GlobalKey] = new AxisModel { AxisId = GlobalKey, ModelPath = ofd.FileName, Model = model };
                RefreshModelPathList();
                MessageBox.Show("KNN 모델 로드 완료 (전체 축 적용)\n" + Path.GetFileName(ofd.FileName));
            }
        }

        /// <summary>
        /// AIForm에서 학습·저장한 sklearn ONNX 모델(.onnx + _meta.json)을 로드합니다.
        /// </summary>
        private void LoadSklOnnxModel()
        {
            using (var ofd = new OpenFileDialog
            {
                Title = "SKL ONNX 모델 선택 (AIForm 저장)",
                Filter = "ONNX 모델 (*.onnx)|*.onnx|모든 파일 (*.*)|*.*",
                InitialDirectory = Path.Combine(DefaultLogsPath, "Models")
            })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;
                string onnxPath = ofd.FileName;

                // 1) 사이드카 _meta.json 읽기
                string metaPath = Path.Combine(
                    Path.GetDirectoryName(onnxPath) ?? "",
                    Path.GetFileNameWithoutExtension(onnxPath) + "_meta.json");

                string session = "AD";
                string modelType = "";
                string[] features = Array.Empty<string>();
                string yColumn = "";
                double threshold = 0.5;
                double scoreThreshold = 0.0;
                int knn_k = 5;
                bool knn_standardize = false;
                double[] knn_mean = null;
                double[] knn_std = null;
                double[][] knn_trainVectors = null;
                string[] classNames = new[] { "Normal", "Anomaly" };

                if (File.Exists(metaPath))
                {
                    try
                    {
                        var metaDoc = JsonDocument.Parse(File.ReadAllText(metaPath));
                        var root = metaDoc.RootElement;
                        if (root.TryGetProperty("session", out var sProp)) session = sProp.GetString() ?? "AD";
                        if (root.TryGetProperty("model_type", out var mtProp)) modelType = mtProp.GetString() ?? "";
                        if (root.TryGetProperty("y_column", out var ycProp)) yColumn = ycProp.GetString() ?? "";
                        if (root.TryGetProperty("threshold", out var thrProp) && thrProp.TryGetDouble(out double thrVal)) threshold = thrVal;
                        if (root.TryGetProperty("score_threshold", out var stProp) && stProp.ValueKind != JsonValueKind.Null && stProp.TryGetDouble(out double stVal)) scoreThreshold = stVal;
                        if (root.TryGetProperty("k", out var kProp) && kProp.TryGetInt32(out int kVal)) knn_k = kVal;
                        if (root.TryGetProperty("standardize", out var szProp)) knn_standardize = szProp.GetBoolean();
                        if (root.TryGetProperty("mean", out var meanProp) && meanProp.ValueKind == JsonValueKind.Array)
                            knn_mean = meanProp.EnumerateArray().Select(e => e.GetDouble()).ToArray();
                        if (root.TryGetProperty("std", out var stdProp) && stdProp.ValueKind == JsonValueKind.Array)
                            knn_std = stdProp.EnumerateArray().Select(e => e.GetDouble()).ToArray();
                        if (root.TryGetProperty("train_vectors", out var tvProp) && tvProp.ValueKind == JsonValueKind.Array)
                            knn_trainVectors = tvProp.EnumerateArray()
                                .Select(row => row.EnumerateArray().Select(e => e.GetDouble()).ToArray())
                                .ToArray();
                        if (root.TryGetProperty("features", out var fProp) && fProp.ValueKind == JsonValueKind.Array)
                            features = fProp.EnumerateArray().Select(e => e.GetString() ?? "").Where(s => s != "").ToArray();
                        if (root.TryGetProperty("class_names", out var cnProp) && cnProp.ValueKind == JsonValueKind.Array)
                            classNames = cnProp.EnumerateArray().Select(e => e.GetString() ?? "").Where(s => s != "").ToArray();
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"_meta.json 읽기 오류: {ex.Message}\n메타 정보를 수동으로 입력합니다.");
                    }
                }
                else
                {
                    MessageBox.Show($"_meta.json 파일이 없습니다 ({Path.GetFileName(metaPath)}).\n기본값(AD, 피처 없음)으로 로드합니다.\nAIForm에서 저장된 모델인지 확인하세요.");
                }

                // 2) 전체 축 공용 모델 — 축 선택 없이 GlobalKey=0에 저장
                const int axis = 0;

                // 3) 피처가 없으면 경고
                if (features.Length == 0)
                {
                    MessageBox.Show("피처 목록이 비어 있습니다. _meta.json을 확인하세요.\n모델을 등록하지만 스코어링이 작동하지 않을 수 있습니다.", "경고", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }

                // 4) ONNX 세션 생성
                InferenceSession sess;
                try { sess = new InferenceSession(onnxPath); }
                catch (Exception ex)
                { MessageBox.Show("ONNX 세션 생성 실패: " + ex.Message); return; }

                // 기존 세션 교체
                if (_axisSklModels.TryGetValue(axis, out var old) && old?.OnnxSession != null)
                    try { old.OnnxSession.Dispose(); } catch { }

                _axisSklModels[axis] = new OnnxSklModel
                {
                    AxisId         = axis,
                    ModelPath      = onnxPath,
                    Session        = session.ToUpperInvariant(),
                    ModelType      = modelType,
                    Features       = features,
                    YColumn        = yColumn,
                    Threshold      = threshold > 0 ? threshold : 0.5,
                    ScoreThreshold = scoreThreshold,
                    ClassNames     = classNames,
                    OnnxSession    = sess,
                    TrainVectors   = knn_trainVectors,
                    K              = knn_k,
                    Standardize    = knn_standardize,
                    Mean           = knn_mean,
                    Std            = knn_std,
                };

                RefreshModelPathList();
                string featStr = features.Length > 0 ? string.Join(", ", features) : "(없음)";
                MessageBox.Show(
                    $"SKL ONNX 모델 등록 완료 (전체 축 적용)\n" +
                    $"파일: {Path.GetFileName(onnxPath)}\n" +
                    $"세션: {session}  알고리즘: {modelType}\n" +
                    $"YColumn: {yColumn}  피처({features.Length}): {featStr}");
            }
        }

        /// <summary>
        /// sklearn ONNX 모델로 스코어링합니다.
        /// AD: label=-1→이상 / scores 출력(decision function)을 rawScore로 반환
        /// FD: label=클래스인덱스 / probabilities[pred]를 rawScore로 반환
        /// </summary>
        private bool TrySklOnnxScore(OnnxSklModel skl, string csvPath,
            out bool isAnomaly, out int predClass, out float[] probabilities, out double rawScore, out string info)
            => TrySklOnnxScore(skl, csvPath, skl?.YColumn, out isAnomaly, out predClass, out probabilities, out rawScore, out info);

        private bool TrySklOnnxScore(OnnxSklModel skl, string csvPath, string yColumn,
            out bool isAnomaly, out int predClass, out float[] probabilities, out double rawScore, out string info)
        {
            isAnomaly = false; predClass = -1; probabilities = null; rawScore = 0.0; info = "";
            if (skl?.OnnxSession == null || skl.Features == null || skl.Features.Length == 0) return false;
            if (string.IsNullOrWhiteSpace(yColumn)) return false;

            double[] vec = BuildFeatureVectorFromCsv(csvPath, yColumn, skl.Features);
            if (vec == null || vec.Length != skl.Features.Length) return false;

            return TrySklOnnxScoreFromVec(skl, vec, out isAnomaly, out predClass, out probabilities, out rawScore, out info);
        }

        /// <summary>
        /// CSV 헤더에서 토크 컬럼을 찾습니다.
        /// "torque" 컬럼이 없으면 "Ax{axis}_Trq(%)" 또는 "AXIS{axis}_FBKTRQ" 패턴을 탐색합니다.
        /// </summary>
        private static string ResolveTorqueColumn(string[] headers, int axis)
        {
            if (headers == null) return null;
            // 정확히 "torque" 컬럼이 있으면 그대로
            string exact = headers.FirstOrDefault(h => string.Equals(h?.Trim(), "torque", StringComparison.OrdinalIgnoreCase));
            if (exact != null) return exact;
            // AjinCsvLogger: "Ax0_Trq(%)"
            string ajin = headers.FirstOrDefault(h =>
                h != null && System.Text.RegularExpressions.Regex.IsMatch(h.Trim(),
                    $@"^Ax{axis}_Trq", System.Text.RegularExpressions.RegexOptions.IgnoreCase));
            if (ajin != null) return ajin;
            // WmxTorqueLogger: "AXIS0_FBKTRQ"
            string wmx = headers.FirstOrDefault(h =>
                h != null && System.Text.RegularExpressions.Regex.IsMatch(h.Trim(),
                    $@"^AXIS{axis}_FBKTRQ", System.Text.RegularExpressions.RegexOptions.IgnoreCase));
            if (wmx != null) return wmx;
            // 축 무관하게 첫 번째 매칭
            return headers.FirstOrDefault(h =>
                h != null && System.Text.RegularExpressions.Regex.IsMatch(h.Trim(),
                    @"Trq|torque|FBKTRQ", System.Text.RegularExpressions.RegexOptions.IgnoreCase));
        }

        private bool TrySklOnnxScoreFromVec(OnnxSklModel skl, double[] vec,
            out bool isAnomaly, out int predClass, out float[] probabilities, out double rawScore, out string info)
        {
            isAnomaly = false; predClass = -1; probabilities = null; rawScore = 0.0; info = "";
            if (skl == null || vec == null || vec.Length == 0) return false;

            // ★ knn AD: 학습 벡터가 있으면 C# kNN 거리로 rawScore 계산
            if (skl.Session == "AD" && skl.ModelType == "knn" &&
                skl.TrainVectors != null && skl.TrainVectors.Length > 0)
            {
                rawScore = SignalFeatures.ScoreKnn(vec, skl.TrainVectors, skl.K,
                                                   skl.Standardize, skl.Mean, skl.Std);
                double thr = skl.Threshold > 0 ? skl.Threshold : 1.0;
                isAnomaly = rawScore >= thr;
                info = $"KNN AD  score={rawScore:F4}  thr={thr:F4}  =>  {(isAnomaly ? "이상" : "정상")}";
                return true;
            }

            // 2) float 텐서 구성 (knn 외 모든 모델)
            var inputData = new DenseTensor<float>(new[] { 1, vec.Length });
            for (int i = 0; i < vec.Length; i++) inputData[0, i] = (float)vec[i];

            // ONNX 모델 입력 이름 자동 탐색
            string inputName = "float_input";
            try
            {
                var meta = skl.OnnxSession.InputMetadata;
                if (meta.Count > 0) inputName = meta.Keys.First();
            }
            catch { }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputData)
            };

            // 3) 추론
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            try { results = skl.OnnxSession.Run(inputs); }
            catch (Exception ex) { info = "ONNX Run 오류: " + ex.Message; return false; }

            using (results)
            {
                // "label" 출력 파싱
                var labelVal = results.FirstOrDefault(r => r.Name == "label");
                if (labelVal != null)
                {
                    try
                    {
                        // sklearn ONNX: label은 int64 또는 string
                        if (labelVal.ElementType == TensorElementType.Int64)
                        {
                            var lt = labelVal.AsTensor<long>();
                            long lbl = lt[0];
                            if (skl.Session == "AD")
                                isAnomaly = lbl == -1L;
                            else
                            {
                                predClass = (int)lbl;
                                isAnomaly = predClass != 0;
                            }
                        }
                    }
                    catch { }
                }

                // "probabilities" 또는 "output_probability" 출력 파싱 (FD 분류기)
                var probVal = results.FirstOrDefault(r => r.Name == "probabilities" || r.Name == "output_probability");
                if (probVal != null)
                {
                    try
                    {
                        var pt = probVal.AsTensor<float>();
                        probabilities = new float[pt.Length];
                        for (int i = 0; i < pt.Length; i++) probabilities[i] = pt[i];
                        // FD rawScore: 예측 클래스의 확률
                        if (predClass >= 0 && predClass < probabilities.Length)
                            rawScore = probabilities[predClass];
                    }
                    catch { }
                }

                // "scores" 출력 파싱 (AD 이상탐지: decision_function 값)
                // sklearn outlier detectors: 음수일수록 이상, 양수일수록 정상
                // skl2onnx는 scores를 shape (1,1)로 출력
                var scoresVal = results.FirstOrDefault(r => r.Name == "scores");
                if (scoresVal != null && skl.Session == "AD")
                {
                    try
                    {
                        var st = scoresVal.AsTensor<float>();
                        float s = st[0];  // decision function 값 (음수=이상, 양수=정상)
                        rawScore = -s;    // 대시보드 관례: 값이 클수록 이상 → 부호 반전
                    }
                    catch { }
                }
            }

            string labelStr = skl.Session == "AD"
                ? (isAnomaly ? "이상" : "정상")
                : (predClass >= 0 && skl.ClassNames != null && predClass < skl.ClassNames.Length ? skl.ClassNames[predClass] : predClass.ToString());

            info = $"{skl.ModelType?.ToUpperInvariant()} {skl.Session}  score={rawScore:F4}  =>  {labelStr}";
            return true;
        }

        // ── 전역 KNN 모델 로드 ───────────────────────────────────────────────
        private void LoadGlobalKnnModel()
        {
            using (var ofd = new OpenFileDialog { Filter = "PHM Model (*.json)|*.json|All files (*.*)|*.*", Title = "전역 KNN 모델 선택" })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;
                PersistedKnnModel model; string err;
                if (!TryLoadModelFromPath(ofd.FileName, out model, out err))
                { MessageBox.Show("전역 KNN 로드 실패: " + err); return; }
                _globalKnnModel = model;
                _globalKnnModelPath = ofd.FileName;
                MessageBox.Show("전역 KNN 모델 로드 완료\n" + Path.GetFileName(ofd.FileName)
                    + "\nYColumn=" + (model.YColumn ?? "(없음)"));
            }
        }

        // ── 전역 AE(ONNX) 모델 로드 ─────────────────────────────────────────
        private void LoadGlobalOnnxAeModel()
        {
            using (var ofd = new OpenFileDialog { Filter = "ONNX 모델 (*.onnx)|*.onnx|All files (*.*)|*.*", Title = "전역 AE ONNX 모델 선택" })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;
                try
                {
                    var session = new InferenceSession(ofd.FileName);
                    if (_globalOnnxAe?.Session != null) try { _globalOnnxAe.Session.Dispose(); } catch { }
                    _globalOnnxAe = new OnnxAxisModel
                    {
                        AxisId = -1,
                        ModelPath = ofd.FileName,
                        Kind = "AE-GLOBAL",
                        YColumn = "x",   // 기본값; 파일명 규칙 적용 시 변경 가능
                        C = 3,
                        InputName = "input",
                        ReconOutputName = "recon",
                        IsAutoencoder = true,
                        StandardizePerSample = true,
                        Threshold = DefaultThreshold,
                        Session = session
                    };
                    // 파일명에서 YColumn, C, Threshold 파싱 (선택 규칙)
                    string name = Path.GetFileNameWithoutExtension(ofd.FileName);
                    var yMatch = Regex.Match(name, @"y=(?<y>[A-Za-z0-9_]+)", RegexOptions.IgnoreCase);
                    if (yMatch.Success) _globalOnnxAe.YColumn = yMatch.Groups["y"].Value;
                    var cMatch = Regex.Match(name, @"c=(?<c>\d+)", RegexOptions.IgnoreCase);
                    if (cMatch.Success && int.TryParse(cMatch.Groups["c"].Value, out int tmpC) && tmpC > 0)
                        _globalOnnxAe.C = tmpC;
                    var thrMatch = Regex.Match(name, @"thr=(?<t>[-+]?\d*\.?\d+)", RegexOptions.IgnoreCase);
                    if (thrMatch.Success && double.TryParse(thrMatch.Groups["t"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double thr) && thr > 0)
                        _globalOnnxAe.Threshold = thr;

                    MessageBox.Show("전역 AE 모델 로드 완료\n" + Path.GetFileName(ofd.FileName)
                        + "\nYColumn=" + _globalOnnxAe.YColumn + "  C=" + _globalOnnxAe.C
                        + "  Thr=" + _globalOnnxAe.Threshold);
                }
                catch (Exception ex) { MessageBox.Show("전역 AE 로드 실패: " + ex.Message); }
            }
        }

        private void LoadAxisModelsFromFolder()
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Title = "AE/CLS ONNX 모델 폴더 선택";
                ofd.ValidateNames = false;
                ofd.CheckFileExists = false;
                ofd.CheckPathExists = true;
                ofd.FileName = "폴더 선택";
                string start = Path.Combine(DefaultLogsPath, "Models");
                try { Directory.CreateDirectory(start); } catch { }
                ofd.InitialDirectory = start;

                if (ofd.ShowDialog() != DialogResult.OK) return;
                string folder = Path.GetDirectoryName(ofd.FileName);
                if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder)) return;

                int loaded = 0, failed = 0;

                foreach (string file in Directory.EnumerateFiles(folder, "*.onnx", SearchOption.TopDirectoryOnly))
                {
                    string name = Path.GetFileNameWithoutExtension(file);

                    // 0) 모델 역할: _AE / _CLS (없으면 AE로 가정)
                    bool isAE = Regex.IsMatch(name, @"(^|_)(ae)($|_)", RegexOptions.IgnoreCase);
                    bool isCLS = Regex.IsMatch(name, @"(^|_)(cls|clf|class|classifier)($|_)", RegexOptions.IgnoreCase);
                    if (!isAE && !isCLS) isAE = true;

                    // 1) 축 ID
                    var axisM = Regex.Match(name, @"axis(?<id>\d+)", RegexOptions.IgnoreCase);
                    if (!axisM.Success || !int.TryParse(axisM.Groups["id"].Value, out int axis))
                    { failed++; continue; }

                    // 2) 모드(Acc/Trq)
                    var modM = Regex.Match(name, @"_(?<mod>acc|trq)($|_)", RegexOptions.IgnoreCase);
                    string mod = modM.Success ? modM.Groups["mod"].Value.ToLowerInvariant() : "";

                    // 기본값
                    string yCol = "FBTRQ" + axis;
                    int C = 1;
                    if (mod == "acc") { yCol = $"x{axis}"; C = 3; }
                    else if (mod == "trq") { yCol = "FBTRQ" + axis; C = 1; }

                    // 3) 선택 파라미터 오버라이드: y=, c=
                    var yMatch = Regex.Match(name, @"y=(?<y>[A-Za-z0-9_]+)", RegexOptions.IgnoreCase);
                    if (yMatch.Success) yCol = yMatch.Groups["y"].Value;

                    var cMatch = Regex.Match(name, @"c=(?<c>\d+)", RegexOptions.IgnoreCase);
                    if (cMatch.Success && int.TryParse(cMatch.Groups["c"].Value, out int tmpC) && tmpC > 0) C = tmpC;

                    // 4) AE 임계값: thr= 우선, 없으면 tr= (정규식의 \b 제거)
                    double thr = DefaultThreshold;
                    if (isAE)
                    {
                        var thrMatch = Regex.Match(name, @"thr=(?<t>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", RegexOptions.IgnoreCase);
                        var trMatch = Regex.Match(name, @"tr=(?<t>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", RegexOptions.IgnoreCase);
                        if (thrMatch.Success &&
                            double.TryParse(thrMatch.Groups["t"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double vThr) && vThr > 0)
                            thr = vThr;
                        else if (trMatch.Success &&
                            double.TryParse(trMatch.Groups["t"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double vTr) && vTr > 0)
                            thr = vTr;
                    }

                    // 5) Kind 추정
                    string kind = isAE ? "AE-CNN1D" : "CLS-CNN1D";
                    if (Regex.IsMatch(name, @"cnn1d", RegexOptions.IgnoreCase)) kind = isAE ? "AE-CNN1D" : "CNN1D";
                    else if (Regex.IsMatch(name, @"lstm", RegexOptions.IgnoreCase)) kind = isAE ? "AE-LSTM" : "LSTM";

                    try
                    {
                        var session = new InferenceSession(file);

                        if (isAE)
                        {
                            if (_axisOnnx.TryGetValue(axis, out OnnxAxisModel old) && old?.Session != null)
                            { try { old.Session.Dispose(); } catch { } }

                            var om = new OnnxAxisModel
                            {
                                AxisId = axis,
                                ModelPath = file,
                                Kind = kind,
                                YColumn = yCol,
                                C = C,
                                InputName = "input",
                                OutputName = "logits",   // AE에서는 사용 안 함 (스키마 일치 용)
                                ReconOutputName = "recon",
                                IsAutoencoder = true,
                                StandardizePerSample = true,
                                Threshold = thr,
                                Session = session
                            };

                            _axisOnnx[axis] = om;
                            loaded++;
                        }
                        else
                        {
                            if (_axisOnnxCls.TryGetValue(axis, out OnnxAxisModel oldCls) && oldCls?.Session != null)
                            { try { oldCls.Session.Dispose(); } catch { } }

                            var om = new OnnxAxisModel
                            {
                                AxisId = axis,
                                ModelPath = file,
                                Kind = kind,
                                YColumn = yCol,
                                C = C,
                                InputName = "input",
                                OutputName = "logits",   // 분류 기본 출력명
                                ReconOutputName = "recon", // (CLS에선 미사용)
                                IsAutoencoder = false,
                                StandardizePerSample = true,
                                Session = session
                            };

                            _axisOnnxCls[axis] = om;

                            int k = GetNumClassesFromOnnx(session, om.OutputName, (_clsLabels?.Length ?? 4));
                            SeedAxisGauge(axis, k);
                            loaded++;
                        }
                    }
                    catch
                    {
                        failed++;
                    }
                }

                _axisModels.Clear();
                RefreshModelPathList();

                MessageBox.Show("ONNX 일괄 로드 완료: " + loaded + "개 성공, " + failed + "개 실패\n경로: " + folder);
            }
        }

        private bool TryLoadModelFromPath(string path, out PersistedKnnModel model, out string error)
        {
            model = null; error = null;
            try
            {
                string json = File.ReadAllText(path);
                JsonSerializerOptions opt = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                PersistedKnnModel m = JsonSerializer.Deserialize<PersistedKnnModel>(json, opt);

                bool typeOk = m != null && (string.Equals(m.ModelType, "KNN", StringComparison.OrdinalIgnoreCase) ||
                                            string.Equals(m.ModelType, "KNN_AD", StringComparison.OrdinalIgnoreCase));
                if (!typeOk || m.Features == null || m.Train == null)
                { error = "지원하지 않는 모델 형식(KNN/KNN_AD JSON만)."; return false; }

                int D = m.Features.Length;
                if (m.Standardize && ((m.Mean == null || m.Mean.Length != D) || (m.Std == null || m.Std.Length != D)))
                { error = "Mean/Std 길이가 Features와 다릅니다."; return false; }
                for (int i = 0; i < m.Train.Length; i++)
                {
                    double[] v = m.Train[i];
                    if (v == null || v.Length != D) { error = "Train 벡터 차원이 Features와 다릅니다."; return false; }
                }
                if (string.IsNullOrWhiteSpace(m.YColumn))
                { error = "모델 JSON에 YColumn이 없습니다."; return false; }

                model = m; return true;
            }
            catch (Exception ex) { error = ex.Message; return false; }
        }

        private void RefreshModelPathList()
        {
            if (gridModelPaths == null) return;
            gridModelPaths.SuspendLayout();
            try
            {
                gridModelPaths.Rows.Clear();

                foreach (var kv in _axisOnnx.OrderBy(k => k.Key))
                {
                    var om = kv.Value;
                    string file = string.IsNullOrEmpty(om.ModelPath) ? "" : Path.GetFileName(om.ModelPath);
                    int r = gridModelPaths.Rows.Add(kv.Key, $"{file}  (ONNX-AE)");
                    gridModelPaths.Rows[r].Cells["Path"].ToolTipText = om.ModelPath;
                }

                foreach (var kv in _axisOnnxCls.OrderBy(k => k.Key))
                {
                    var om = kv.Value;
                    string file = string.IsNullOrEmpty(om.ModelPath) ? "" : Path.GetFileName(om.ModelPath);
                    int r = gridModelPaths.Rows.Add(kv.Key, $"{file}  (ONNX-CLS)");
                    gridModelPaths.Rows[r].Cells["Path"].ToolTipText = om.ModelPath;
                }

                foreach (var kv in _axisSklModels.OrderBy(k => k.Key))
                {
                    var sm = kv.Value;
                    string file = string.IsNullOrEmpty(sm.ModelPath) ? "" : Path.GetFileName(sm.ModelPath);
                    string tag = sm.Session == "AD" ? "SKL-AD" : "SKL-FD";
                    int r = gridModelPaths.Rows.Add("전체", $"{file}  ({tag}/{sm.ModelType?.ToUpperInvariant()})");
                    gridModelPaths.Rows[r].Cells["Path"].ToolTipText = sm.ModelPath;
                }

                foreach (var kv in _axisModels.OrderBy(k => k.Key))
                {
                    var am = kv.Value;
                    string file = string.IsNullOrEmpty(am.ModelPath) ? "" : Path.GetFileName(am.ModelPath);
                    int r = gridModelPaths.Rows.Add("전체", $"{file}  (KNN/JSON)");
                    gridModelPaths.Rows[r].Cells["Path"].ToolTipText = am.ModelPath;
                }

                if (gridModelPaths.Rows.Count == 0)
                    gridModelPaths.Rows.Add("-", "(모델이 없습니다)");
            }
            finally { gridModelPaths.ResumeLayout(); }
        }
        #endregion

        #region 폴더/Watcher
        private void SelectFolder()
        {
            using (FolderBrowserDialog fbd = new FolderBrowserDialog())
            {
                if (!string.IsNullOrEmpty(_watchFolder) && Directory.Exists(_watchFolder)) fbd.SelectedPath = _watchFolder;
                if (fbd.ShowDialog() == DialogResult.OK)
                    SetWatchFolder(fbd.SelectedPath);
            }
        }

        private void SetWatchFolder(string path)
        {
            if (string.IsNullOrEmpty(path) || !Directory.Exists(path)) return;
            _watchFolder = path;
            lblFolder.Text = "폴더: " + _watchFolder;
            lock (_sync)
            {
                _processing.Clear();
                foreach (KeyValuePair<string, CancellationTokenSource> kv in _debouncers) { try { kv.Value.Cancel(); } catch { } kv.Value.Dispose(); }
                _debouncers.Clear();
                _lastProcessedLen.Clear();
            }
            AddRecentFolder(path);
        }

        private void ShowFolderQuickMenu()
        {
            var cms = new ContextMenuStrip();

            // ── 고정 폴더 ─────────────────────────────────────────────
            string signalsRoot = Path.Combine(DefaultLogsPath, "Signals");
            AddFolderMenuItem(cms, "📂 Signals (기본)", signalsRoot);
            AddFolderMenuItem(cms, "📂 PHM_Logs", DefaultLogsPath);

            // ── Signals 하위 Axis* 폴더 ──────────────────────────────
            if (Directory.Exists(signalsRoot))
            {
                var axisDirs = Directory.GetDirectories(signalsRoot)
                    .Where(d => System.Text.RegularExpressions.Regex.IsMatch(
                        Path.GetFileName(d), @"[Aa]xis\d+"))
                    .OrderBy(d => d).ToArray();
                if (axisDirs.Length > 0)
                {
                    cms.Items.Add(new ToolStripSeparator());
                    foreach (var d in axisDirs)
                        AddFolderMenuItem(cms, "  📂 " + Path.GetFileName(d), d);
                }
            }

            // ── 최근 폴더 ─────────────────────────────────────────────
            var recent = LoadRecentFolders();
            if (recent.Count > 0)
            {
                cms.Items.Add(new ToolStripSeparator());
                cms.Items.Add(new ToolStripMenuItem("최근 폴더") { Enabled = false });
                foreach (var r in recent)
                    AddFolderMenuItem(cms, "  🕐 " + r, r);
                cms.Items.Add(new ToolStripSeparator());
                var clearItem = new ToolStripMenuItem("🗑 최근 기록 지우기");
                clearItem.Click += (s2, e2) => { try { File.Delete(MruFile); } catch { } };
                cms.Items.Add(clearItem);
            }

            cms.Show(btnQuickFolder, new System.Drawing.Point(0, btnQuickFolder.Height));
        }

        private void AddFolderMenuItem(ContextMenuStrip cms, string label, string path)
        {
            var item = new ToolStripMenuItem(label) { Enabled = Directory.Exists(path) };
            item.Click += (s, e) => SetWatchFolder(path);
            cms.Items.Add(item);
        }

        private List<string> LoadRecentFolders()
        {
            try
            {
                if (!File.Exists(MruFile)) return new List<string>();
                return File.ReadAllLines(MruFile)
                    .Where(l => !string.IsNullOrWhiteSpace(l) && Directory.Exists(l))
                    .Distinct()
                    .Take(MruMaxCount)
                    .ToList();
            }
            catch { return new List<string>(); }
        }

        private void AddRecentFolder(string path)
        {
            try
            {
                var list = LoadRecentFolders();
                list.Remove(path);
                list.Insert(0, path);
                try { Directory.CreateDirectory(DefaultLogsPath); } catch { }
                File.WriteAllLines(MruFile, list.Take(MruMaxCount));
            }
            catch { }
        }

        private void StartWatch()
        {
            bool hasKnn = _axisModels != null && _axisModels.Count > 0;
            bool hasAe = _axisOnnx != null && _axisOnnx.Values.Any(om => om?.Session != null && om.IsAutoencoder);
            bool hasCls = _axisOnnxCls != null && _axisOnnxCls.Values.Any(om => om?.Session != null && !om.IsAutoencoder);
            bool hasSkl = _axisSklModels != null && _axisSklModels.Values.Any(sm => sm?.OnnxSession != null);
            if (!hasKnn && !hasAe && !hasCls && !hasSkl)
            {
                MessageBox.Show("먼저 모델을 추가하세요. (SKL ONNX / AE ONNX / 분류 ONNX / KNN JSON)");
                return;
            }

            if (_isDbMode) { StartDbWatch(); return; }

            // 2) 폴더 체크
            if (string.IsNullOrEmpty(_watchFolder) || !Directory.Exists(_watchFolder))
            {
                MessageBox.Show("CSV 폴더를 먼저 선택하세요.");
                return;
            }

            // 3) YColumn 검증 (존재하는 모델들만)
            //    - KNN(JSON): Model.YColumn
            //    - ONNX: OnnxAxisModel.YColumn
            foreach (var am in _axisModels.Values)
            {
                if (am == null || am.Model == null || string.IsNullOrWhiteSpace(am.Model.YColumn))
                {
                    MessageBox.Show($"축 {am?.AxisId} KNN 모델에 YColumn 정보가 없습니다.");
                    return;
                }
            }
            foreach (var om in _axisOnnx.Values)
            {
                if (om == null || om.Session == null) continue;
                if (string.IsNullOrWhiteSpace(om.YColumn))
                {
                    MessageBox.Show($"축 {om?.AxisId} ONNX 모델에 YColumn 정보가 없습니다.");
                    return;
                }
            }

            // 4) 기존 Watcher 정리
            StopWatch();

            // 5) Watcher 설정
            _watcher = new FileSystemWatcher(_watchFolder, "*.csv");
            _watcher.IncludeSubdirectories = WatchSubdirectories;
            _watcher.InternalBufferSize = 64 * 1024;
            _watcher.NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.Size | NotifyFilters.CreationTime;
            _watcher.Created += OnFileCreatedOrChanged;
            _watcher.Changed += OnFileCreatedOrChanged;
            _watcher.Renamed += OnFileRenamed;
            _watcher.Error   += OnWatcherError;

            // 6) 기존 파일 길이를 베이스라인으로 기록 — 진단 시작 이후 새로 생기는 파일만 처리
            var option = WatchSubdirectories ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            lock (_sync)
            {
                _processing.Clear();
                foreach (var kv in _debouncers) { try { kv.Value.Cancel(); } catch { } try { kv.Value.Dispose(); } catch { } }
                _debouncers.Clear();
                _lastProcessedLen.Clear();
                foreach (string f in Directory.EnumerateFiles(_watchFolder, "*.csv", option))
                {
                    try { _lastProcessedLen[f] = new FileInfo(f).Length; } catch { }
                }
            }

            // 7) 이벤트 시작
            _watcher.EnableRaisingEvents = true;

            foreach (var kv in _axisOnnxCls)
            {
                var om = kv.Value;
                if (om?.Session == null) continue;
                int k = GetNumClassesFromOnnx(om.Session, om.OutputName ?? "logits", (_clsLabels?.Length ?? 4));
                SeedAxisGauge(kv.Key, k);
            }

            btnStart.Enabled = false;
            btnStop.Enabled = true;
            lblStatus.Text = "상태: 수집 중..." + (WatchSubdirectories ? " (하위 폴더 포함)" : "");
        }

        private void StopWatch()
        {
            if (_watcher != null)
            {
                try
                {
                    _watcher.EnableRaisingEvents = false;
                    _watcher.Created -= OnFileCreatedOrChanged;
                    _watcher.Changed -= OnFileCreatedOrChanged;
                    _watcher.Renamed -= OnFileRenamed;
                    _watcher.Error   -= OnWatcherError;
                    _watcher.Dispose();
                }
                catch { }
                finally { _watcher = null; }
            }
            lock (_sync)
            {
                foreach (KeyValuePair<string, CancellationTokenSource> kv in _debouncers) { try { kv.Value.Cancel(); } catch { } kv.Value.Dispose(); }
                _debouncers.Clear();
                _processing.Clear();
            }

            // DB 폴링도 중지
            StopDbWatch();

            btnStart.Enabled = true; btnStop.Enabled = false; lblStatus.Text = "상태: 중지";
        }

        // ── DB 모드 전환 ──────────────────────────────────────────────────────
        private void SwitchSourceMode(bool dbMode)
        {
            _isDbMode = dbMode;
            if (pnlCsvSource != null) pnlCsvSource.Visible = !dbMode;
            if (pnlDbSource != null)  pnlDbSource.Visible  =  dbMode;
        }

        // ── InfluxDB 설정 파일 자동 탐색 ─────────────────────────────────────
        private static string FindInfluxConfigPath()
        {
            var candidates = new[]
            {
                Path.Combine(ResolveDataRoot(), "Tests", "influx_config.json"),
                @"D:\Dev\hvs\WorkingSource\DAQ_Test\infra\influx_config.json",
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "influx_config.json"),
            };
            foreach (var p in candidates)
                if (File.Exists(p)) return p;
            return candidates[2]; // fallback: exe 폴더
        }
        private static string ResolveDataRoot()
        {
            foreach (var r in new[] { @"E:\Data\PHM_Logs", @"C:\Data\PHM_Logs", @"C:\PHM_Logs" })
                if (Directory.Exists(r)) return r;
            return @"C:\Data\PHM_Logs";
        }

        // ── DB 장치/레이블 목록 갱신 ─────────────────────────────────────────
        private async void RefreshDbDevices()
        {
            btnDbRefresh.Enabled = false;
            try
            {
                EnsureInfluxSource();
                var devices = await _influxSource.GetDevicesAsync();
                var labels  = await _influxSource.GetLabelsAsync();
                BeginInvoke(new Action(() =>
                {
                    string prevDev = cmbDbDevice.Text;
                    string prevLbl = cmbDbLabel.Text;
                    cmbDbDevice.Items.Clear();
                    cmbDbLabel.Items.Clear();
                    cmbDbDevice.Items.Add("");
                    foreach (var d in devices) cmbDbDevice.Items.Add(d);
                    cmbDbLabel.Items.Add("");
                    foreach (var l in labels)  cmbDbLabel.Items.Add(l);
                    cmbDbDevice.Text = prevDev;
                    cmbDbLabel.Text  = prevLbl;
                    AppendEventLog($"[DB] 장치 {devices.Count}개, 레이블 {labels.Count}개 로드됨");
                }));
            }
            catch (Exception ex)
            {
                BeginInvoke(new Action(() => AppendEventLog($"[DB] 목록 갱신 오류: {ex.Message}")));
            }
            finally { BeginInvoke(new Action(() => btnDbRefresh.Enabled = true)); }
        }

        private void EnsureInfluxSource()
        {
            // ServerSettings.Current 우선 사용, 없으면 파일에서 로드
            var cfg = Services.ServerSettings.Current.ToInfluxConfig();
            if (string.IsNullOrEmpty(cfg.Url) || cfg.Url == "http://localhost:8086")
            {
                string cfgPath = FindInfluxConfigPath();
                cfg = InfluxConfig.LoadOrDefault(cfgPath);
            }
            if (_influxSource == null)
                _influxSource = new InfluxDbDataSource(cfg);
        }

        // ── DB 진단 시작/중지 ────────────────────────────────────────────────
        private void StartDbWatch()
        {
            StopDbWatch();
            EnsureInfluxSource();

            string device = cmbDbDevice.Text?.Trim() ?? "";
            string label  = cmbDbLabel.Text?.Trim()  ?? "";
            var    from   = dtpDbFrom.Value.ToUniversalTime();
            var    to     = dtpDbTo.Value.ToUniversalTime();

            if (from >= to)
            {
                AppendEventLog("[DB] 오류: 시작 시각이 종료 시각보다 뒤입니다.");
                return;
            }

            _influxPollCts = new CancellationTokenSource();
            var token = _influxPollCts.Token;

            btnStart.Enabled = false;
            btnStop.Enabled  = true;
            lblStatus.Text   = $"상태: DB 진단 중...";
            AppendEventLog($"[DB] 진단 시작 — 장치='{device}' 레이블='{label}' 기간={dtpDbFrom.Value:yyyy-MM-dd HH:mm:ss} ~ {dtpDbTo.Value:yyyy-MM-dd HH:mm:ss}");

            Task.Run(() => RunDbDiagnosisAsync(device, label, from, to, token), token)
                .ContinueWith(t =>
                {
                    if (t.IsFaulted)
                        BeginInvoke(new Action(() => AppendEventLog($"[DB] 진단 오류: {t.Exception?.GetBaseException().Message}")));
                    BeginInvoke(new Action(() =>
                    {
                        btnStart.Enabled = true;
                        btnStop.Enabled  = false;
                        lblStatus.Text   = "상태: 대기";
                    }));
                });
        }

        private void StopDbWatch()
        {
            if (_influxPollCts != null)
            {
                try { _influxPollCts.Cancel(); } catch { }
                try { _influxPollCts.Dispose(); } catch { }
                _influxPollCts = null;
            }
        }

        // ── DB 단발성 진단 ────────────────────────────────────────────────────
        private static readonly HashSet<string> TorqueYColumns =
            new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "torque", "fbtrq", "trq" };

        /// <summary>로드된 모델 중 하나라도 토크 채널을 사용하면 true</summary>
        private bool HasTorqueModels()
        {
            foreach (var kv in _axisSklModels)
                if (TorqueYColumns.Contains(kv.Value?.YColumn ?? "")) return true;
            foreach (var kv in _axisModels)
                if (TorqueYColumns.Contains(kv.Value?.Model?.YColumn ?? "")) return true;
            return false;
        }

        /// <summary>
        /// 지정 기간의 InfluxDB 데이터를 조회하여 진단합니다. (단발성, 폴링 없음)
        /// 전체 기간을 하나의 세그먼트로 병합하여 진단 1회 수행합니다.
        /// </summary>
        private async Task RunDbDiagnosisAsync(string device, string label, DateTime from, DateTime to, CancellationToken ct)
        {
            string devArg = string.IsNullOrEmpty(device) ? null : device;
            string lblArg = string.IsNullOrEmpty(label)  ? null : label;

            // 전체 기간을 하나의 세그먼트로 취급 — 기간 길이를 segmentSeconds 로 설정
            double winSecs = (to - from).TotalSeconds + 1.0;
            int totalSegs = 0;

            // ── (A) accel 세그먼트 ────────────────────────────────────────────
            var accelSegs = await _influxSource.QuerySegmentsAsync(
                devArg, lblArg, from, to, segmentSeconds: winSecs, ct: ct);

            if (accelSegs.Count > 0)
            {
                BeginInvoke(new Action(() => ProcessInfluxSegment(MergeSegments(accelSegs))));
                totalSegs += accelSegs.Count;
            }

            // ── (B) torque 세그먼트 (토크 모델이 있을 때만) ───────────────────
            if (HasTorqueModels())
            {
                var torqueSegs = await _influxSource.QueryTorqueSegmentsAsync(
                    devArg, lblArg, from, to, segmentSeconds: winSecs, ct: ct);

                if (torqueSegs.Count > 0)
                {
                    BeginInvoke(new Action(() => ProcessInfluxSegment(MergeSegments(torqueSegs))));
                    totalSegs += torqueSegs.Count;
                }
            }

            if (totalSegs == 0)
                BeginInvoke(new Action(() => AppendEventLog("[DB] 해당 기간에 데이터가 없습니다.")));
        }

        /// <summary>DB 장치/레이블 전체 기간을 dtpDbFrom/dtpDbTo 에 자동 채웁니다.</summary>
        private async Task FillDbFullRangeAsync()
        {
            EnsureInfluxSource();
            string device = cmbDbDevice.Text?.Trim() ?? "";
            string label  = cmbDbLabel.Text?.Trim()  ?? "";
            try
            {
                var (first, last) = await _influxSource.GetTimeRangeAsync(
                    string.IsNullOrEmpty(device) ? null : device,
                    string.IsNullOrEmpty(label)  ? null : label);
                dtpDbFrom.Value = first.ToLocalTime();
                dtpDbTo.Value   = last.ToLocalTime();
                AppendEventLog($"[DB] 전체 기간: {dtpDbFrom.Value:yyyy-MM-dd HH:mm:ss} ~ {dtpDbTo.Value:yyyy-MM-dd HH:mm:ss}");
            }
            catch (Exception ex)
            {
                AppendEventLog($"[DB] 기간 조회 오류: {ex.Message}");
            }
        }

        /// <summary>
        /// 동일 폴 창의 세그먼트를 시간 순으로 이어 붙여 하나로 반환합니다.
        /// 이동 1 회분 데이터가 여러 세그먼트로 분할된 경우에도 진단을 1 회만 수행합니다.
        /// </summary>
        private static SignalSegment MergeSegments(List<SignalSegment> segs)
        {
            if (segs.Count == 1) return segs[0];

            // 시작 시각 순 정렬
            segs.Sort((a, b) => a.StartTime.CompareTo(b.StartTime));
            var first = segs[0];

            bool hasX      = segs.Any(s => s.X      != null && s.X.Length      > 0);
            bool hasY      = segs.Any(s => s.Y      != null && s.Y.Length      > 0);
            bool hasZ      = segs.Any(s => s.Z      != null && s.Z.Length      > 0);
            bool hasTorque = segs.Any(s => s.Torque != null && s.Torque.Length > 0);

            int total = segs.Sum(s => s.SampleCount);
            var time   = new double[total];
            var xArr   = hasX      ? new double[total] : null;
            var yArr   = hasY      ? new double[total] : null;
            var zArr   = hasZ      ? new double[total] : null;
            var trqArr = hasTorque ? new double[total] : null;

            int offset = 0;
            foreach (var seg in segs)
            {
                double tBase = (seg.StartTime - first.StartTime).TotalSeconds;
                int n = seg.SampleCount;
                for (int i = 0; i < n; i++)
                {
                    time[offset + i] = tBase + (seg.Time != null && i < seg.Time.Length ? seg.Time[i] : i * 0.001);
                    if (hasX      && seg.X      != null && i < seg.X.Length)      xArr  [offset + i] = seg.X     [i];
                    if (hasY      && seg.Y      != null && i < seg.Y.Length)      yArr  [offset + i] = seg.Y     [i];
                    if (hasZ      && seg.Z      != null && i < seg.Z.Length)      zArr  [offset + i] = seg.Z     [i];
                    if (hasTorque && seg.Torque != null && i < seg.Torque.Length) trqArr[offset + i] = seg.Torque[i];
                }
                offset += n;
            }

            return new SignalSegment
            {
                Name      = first.Name,
                Label     = first.Label,
                Device    = first.Device,
                StartTime = first.StartTime,
                Time      = time,
                X         = xArr,
                Y         = yArr,
                Z         = zArr,
                Torque    = trqArr,
            };
        }

        // ── InfluxDB 세그먼트 처리 (ProcessCsvSafe의 DB 버전) ────────────────
        private void ProcessInfluxSegment(SignalSegment seg)
        {
            if (seg == null) return;

            // ── 이상치 필터 ─────────────────────────────────────────────────
            if (!SegmentValidator.IsValid(seg, out var rejectReason))
            {
                AppendEventLog($"[SKIP] segment {seg.Name} — outlier rejected ({rejectReason})");
                return;
            }

            // 세그먼트 이름/장치에서 axis 파싱 (예: "Axis0", "seg_0000" 등)
            var axesByName = AxesFromDevice(seg.Device ?? "") ;
            if (axesByName.Count == 0) axesByName = AxesFromDevice(seg.Name ?? "");

            // estSr: 타임스탬프로부터 계산
            double estSr = 1000.0;
            if (seg.Time != null && seg.Time.Length >= 2)
            {
                double dur = seg.Time[seg.Time.Length - 1] - seg.Time[0];
                if (dur > 0) estSr = (seg.Time.Length - 1) / dur;
            }

            // ---------- (B) KNN — 전체 축 공용 모델 ----------
            foreach (KeyValuePair<int, AxisModel> kv in _axisModels)
            {
                int axis = axesByName.Count > 0 ? axesByName.First() : 0;
                var am = kv.Value;
                if (am?.Model == null) continue;

                var m = am.Model;
                if (string.IsNullOrWhiteSpace(m.YColumn)) continue;

                double[] arr = seg.GetChannel(m.YColumn);
                if (arr == null || arr.Length < 4) continue;

                double[] sample = SignalFeatures.BuildFeatureVectorFromSeries(arr.ToList(), m.Features, estSr);
                if (sample == null) continue;

                double score = SignalFeatures.ScoreKnn(sample, m.Train, m.K, m.Standardize, m.Mean, m.Std);
                double thr   = m.Threshold > 0 ? m.Threshold : DefaultThreshold;
                bool isAnom  = score >= thr;
                AlarmLevel level = isAnom ? (score >= thr * 10.0 ? AlarmLevel.Danger : AlarmLevel.Warning) : AlarmLevel.Normal;

                var captAxis = axis; var captScore = score; var captThr = thr;
                var captLevel = level; var captSeg = seg; var captYCol = m.YColumn;
                BeginInvoke(new Action(() =>
                {
                    RenderSegmentChart(captSeg, captYCol, captAxis);
                    UpdateKpiAndLog(captAxis, captScore, captThr, captLevel,
                        $"[KNN] axis {captAxis}  score={captScore:F2}  thr={captThr:F2}  => {AlarmText(captLevel)}  ({captSeg.Name})",
                        captSeg.StartTime);
                }));
            }

            // ---------- (C) sklearn ONNX — 전체 축 공용 모델 ----------
            foreach (KeyValuePair<int, OnnxSklModel> kv in _axisSklModels.OrderBy(k => k.Key))
            {
                int axis = axesByName.Count > 0 ? axesByName.First() : 0;
                OnnxSklModel skl = kv.Value;
                if (skl == null || skl.OnnxSession == null) continue;
                if (string.IsNullOrWhiteSpace(skl.YColumn)) continue;

                double[] arr = seg.GetChannel(skl.YColumn);
                if (arr == null || arr.Length < 4) continue;

                double[] vec = SignalFeatures.BuildFeatureVectorFromSeries(arr.ToList(), skl.Features, estSr);
                if (vec == null || vec.Length != skl.Features.Length) continue;

                bool isAnom; int predClass; float[] probs; double rawScore; string sklInfo;
                if (!TrySklOnnxScoreFromVec(skl, vec, out isAnom, out predClass, out probs, out rawScore, out sklInfo))
                    continue;

                bool isKnnAd = skl.Session == "AD" && skl.ModelType == "knn"
                               && skl.TrainVectors != null && skl.TrainVectors.Length > 0;
                double thr = isKnnAd ? skl.Threshold
                           : skl.ScoreThreshold > 0 ? skl.ScoreThreshold : 0.0;
                bool useScoreThreshold = isKnnAd || skl.ScoreThreshold > 0;

                AlarmLevel level;
                if (skl.Session == "AD")
                    level = useScoreThreshold
                          ? (rawScore >= thr * 1.5 ? AlarmLevel.Danger : rawScore >= thr ? AlarmLevel.Warning : AlarmLevel.Normal)
                          : (isAnom ? AlarmLevel.Warning : AlarmLevel.Normal);
                else
                    level = isAnom ? AlarmLevel.Warning : AlarmLevel.Normal;

                var captAxis = axis; var captSkl = skl; var captRaw = rawScore;
                var captThr2 = useScoreThreshold ? thr : skl.Threshold;
                var captLevel = level; var captInfo = sklInfo; var captProbs = probs;
                var captPred = predClass; var captSeg = seg;
                BeginInvoke(new Action(() =>
                {
                    RenderSegmentChart(captSeg, captSkl.YColumn, captAxis);
                    UpdateKpiAndLog(captAxis, captRaw, captThr2, captLevel,
                        $"[SKL-{captSkl.Session}] axis {captAxis}  {captInfo}  => {AlarmText(captLevel)}  ({captSeg.Name})",
                        captSeg.StartTime);
                    if (captProbs != null && captProbs.Length > 0)
                        UpdateAxisClassGauge(captAxis, captProbs, captPred >= 0 ? captPred : (isAnom ? 1 : 0));
                }));
            }

            // ---------- (D) 전역 모델 폴백 (DB 모드) ----------
            if (_globalKnnModel != null || _globalOnnxAe != null)
            {
                var coveredAxes = new HashSet<int>(
                    _axisModels.Keys.Concat(_axisSklModels.Keys));
                var candidateAxes = axesByName.Count > 0
                    ? new HashSet<int>(axesByName)
                    : new HashSet<int>(coveredAxes);
                if (candidateAxes.Count == 0) candidateAxes.Add(0);

                foreach (int axis in candidateAxes)
                {
                    if (coveredAxes.Contains(axis)) continue;

                    if (_globalKnnModel != null)
                    {
                        var gm = _globalKnnModel;
                        double[] arr = string.IsNullOrWhiteSpace(gm.YColumn)
                            ? null
                            : seg.GetChannel(gm.YColumn);
                        if (arr != null && arr.Length >= 4)
                        {
                            double[] sample = SignalFeatures.BuildFeatureVectorFromSeries(arr.ToList(), gm.Features, estSr);
                            if (sample != null)
                            {
                                double score = SignalFeatures.ScoreKnn(sample, gm.Train, gm.K, gm.Standardize, gm.Mean, gm.Std);
                                double thr = gm.Threshold > 0 ? gm.Threshold : DefaultThreshold;
                                AlarmLevel level = score >= thr
                                    ? (score >= thr * 10.0 ? AlarmLevel.Danger : AlarmLevel.Warning)
                                    : AlarmLevel.Normal;
                                var captAxis = axis; var captScore = score; var captThr = thr; var captSeg = seg;
                                BeginInvoke(new Action(() =>
                                    UpdateKpiAndLog(captAxis, captScore, captThr, level,
                                        $"[G-KNN] axis {captAxis}  score={captScore:F2}  thr={captThr:F2}  => {AlarmText(level)}  ({captSeg.Name})",
                                        captSeg.StartTime)));
                            }
                        }
                    }
                }
            }

            // ── 샘플 차트 렌더링 (DB 모드 — 인메모리 배열 사용) ──────────────
            if (seg.X != null && seg.X.Length > 0)
            {
                var xsSec = new List<double>(seg.X.Length);
                var yMain = new List<double>(seg.X.Length);
                var chXList = new List<double>(seg.X.Length);
                var chYList = new List<double>(seg.X.Length);
                var chZList = new List<double>(seg.X.Length);

                double t0 = seg.Time != null && seg.Time.Length > 0 ? seg.Time[0] : 0.0;
                for (int i = 0; i < seg.X.Length; i++)
                {
                    double t = seg.Time != null && i < seg.Time.Length
                        ? seg.Time[i] - t0
                        : i * (estSr > 0 ? 1.0 / estSr : 0.001);
                    double vx = seg.X[i];
                    double vy = seg.Y != null && i < seg.Y.Length ? seg.Y[i] : 0.0;
                    double vz = seg.Z != null && i < seg.Z.Length ? seg.Z[i] : 0.0;
                    xsSec.Add(t);
                    yMain.Add(Math.Sqrt(vx * vx + vy * vy + vz * vz));
                    chXList.Add(vx);
                    chYList.Add(vy);
                    chZList.Add(vz);
                }

                // 추론한 모든 축에 차트 렌더링
                var chartAxes = new HashSet<int>(axesByName.Count > 0
                    ? axesByName
                    : _axisModels.Keys.Concat(_axisSklModels.Keys));
                if (chartAxes.Count == 0) chartAxes.Add(0); // 기본 axis 0

                var captXs  = xsSec;
                var captYm  = yMain;
                var captChX = chXList;
                var captChY = chYList;
                var captChZ = chZList;
                foreach (int chartAxis in chartAxes)
                {
                    var captAx = chartAxis;
                    BeginInvoke(new Action(() =>
                        RenderSampleChartFromArrays(captAx, "Accel |a|", captXs, captYm, captChX, captChY, captChZ, true)));
                }
            }
        }

        // ── Device 이름에서 axis 번호 파싱 ──────────────────────────────────
        private static HashSet<int> AxesFromDevice(string deviceOrName)
        {
            var axes = new HashSet<int>();
            if (string.IsNullOrEmpty(deviceOrName)) return axes;
            var m = Regex.Matches(deviceOrName, @"Axis(?<id>\d+)", RegexOptions.IgnoreCase);
            foreach (Match mm in m)
                if (int.TryParse(mm.Groups["id"].Value, out int ax)) axes.Add(ax);
            return axes;
        }

        // ── KPI 업데이트 + 이벤트 로그 공통 헬퍼 ──────────────────────────
        private void UpdateKpiAndLog(int axis, double score, double thr, AlarmLevel level, string logMsg, DateTime timestamp)
        {
            if (level == AlarmLevel.Danger)  Interlocked.Increment(ref cntDanger);
            else if (level == AlarmLevel.Warning) Interlocked.Increment(ref cntWarning);
            Interlocked.Increment(ref cycles);
            cardDanger.ValueText  = cntDanger  + " 건";
            cardWarning.ValueText = cntWarning + " 건";
            cardCycles.ValueText  = cycles     + " 회";

            AppendEventLog($"[{DateTime.Now:HH:mm:ss}] {logMsg}");

            if (level != AlarmLevel.Normal)
            {
                ShowToast(level, axis, score);
                rows.Add(new EventRow
                {
                    TimeLine     = timestamp.ToLocalTime().ToString("yyyy.MM.dd HH:mm:ss"),
                    Sensor       = $"Ax{axis}",
                    AnomalyScore = Math.Round(score, 4),
                    Threshold    = Math.Round(thr, 4),
                    Alarm        = level == AlarmLevel.Danger ? "위험" : "경고"
                });
                if (grid.Rows.Count > 0)
                    try { grid.FirstDisplayedScrollingRowIndex = grid.Rows.Count - 1; } catch { }
            }

            scoreSeries.Enqueue(Tuple.Create(axis, DateTime.Now, score));
            while (scoreSeries.Count > 3000) { Tuple<int, DateTime, double> dump; scoreSeries.TryDequeue(out dump); }
        }

        private static string AlarmText(AlarmLevel l)
            => l == AlarmLevel.Danger ? "DANGER" : l == AlarmLevel.Warning ? "WARN" : "OK";

        private void OnFileRenamed(object sender, RenamedEventArgs e)
        {
            var dir = Path.GetDirectoryName(e.FullPath) ?? _watchFolder ?? "";
            var name = Path.GetFileName(e.FullPath);
            OnFileCreatedOrChanged(sender, new FileSystemEventArgs(WatcherChangeTypes.Changed, dir, name));
        }

        private void OnFileCreatedOrChanged(object sender, FileSystemEventArgs e)
        {
            // CSV만 처리 (임시 파일/기타 확장자 무시)
            if (!e.FullPath.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
                return;

            try
            {
                CancellationTokenSource cts;

                lock (_sync)
                {
                    // 같은 경로에 대한 기존 디바운서 취소/정리
                    if (_debouncers.TryGetValue(e.FullPath, out var old))
                    {
                        try { old.Cancel(); } catch { /* ignore */ }
                        try { old.Dispose(); } catch { /* ignore */ }
                    }

                    cts = new CancellationTokenSource();
                    _debouncers[e.FullPath] = cts;
                }

                // 더 민첩하게: 75ms 디바운스
                _ = Task.Delay(75, cts.Token).ContinueWith(t =>
                {
                    if (t.IsCanceled) return;

                    // 파일이 실제 존재하는지 재확인
                    if (!File.Exists(e.FullPath)) return;

                    // 안정화/증분 체크는 ProcessCsvSafe 내부에서 수행
                    ProcessCsvSafe(e.FullPath);

                    // 디바운서 정리
                    lock (_sync)
                    {
                        if (_debouncers.TryGetValue(e.FullPath, out var removed))
                        {
                            _debouncers.Remove(e.FullPath);
                            try { removed?.Dispose(); } catch { /* ignore */ }
                        }
                    }
                }, CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex);
            }
        }

        private void OnWatcherError(object sender, ErrorEventArgs e)
        {
            BeginInvoke(new Action(() =>
                AppendEventLog($"[FSW-ERROR] 감시 오류: {e.GetException()?.Message} — 재시작 중...")));

            // 오류 발생 시 Watcher 재시작
            try
            {
                if (_watcher != null)
                {
                    _watcher.EnableRaisingEvents = false;
                    _watcher.EnableRaisingEvents = true;
                }
            }
            catch (Exception ex)
            {
                BeginInvoke(new Action(() =>
                    AppendEventLog($"[FSW-ERROR] 재시작 실패: {ex.Message}")));
            }
        }
        #endregion

        #region 파일 처리/스코어링
        private void ProcessCsvSafe(string path)
        {
            if (!TryMarkProcessing(path)) return;
            try
            {
                WaitUntilStable(path, 3000, 80, 2);
                long curLen;
                try { curLen = new FileInfo(path).Length; } catch { return; }

                long lastLen = 0;
                lock (_sync) { _lastProcessedLen.TryGetValue(path, out lastLen); }
                if (curLen <= lastLen)
                    return;

                string[] headers = Retry<string[]>(() => SignalFeatures.GetCsvHeaders(path), 5, 100);
                if (headers == null || headers.Length == 0) return;

                var axesByName = AxesFromFilename(path);
                HashSet<string> headerSet = new HashSet<string>(headers.Select(h => h == null ? null : h.Trim()).Where(h => !string.IsNullOrEmpty(h)), StringComparer.OrdinalIgnoreCase);
                List<int> movedAxes = DetermineMovedAxes(path, headers, MotionEps);

                bool anyAxisProcessed = false;

                // ---------- (A) AE 우선 ----------
                foreach (var kv in _axisOnnx.OrderBy(k => k.Key))
                {
                    int axis = kv.Key;
                    var ae = kv.Value;

                    if (axesByName.Count > 0 && !axesByName.Contains(axis)) continue;
                    if (ae == null || ae.Session == null || !ae.IsAutoencoder) continue;
                    if (movedAxes.Count > 0 && !movedAxes.Contains(axis)) continue;
                    if (string.IsNullOrWhiteSpace(ae.YColumn) || !HasYColumns(headers, ae, axis)) continue;

                    double scoreAe; string infoAe;
                    if (!TryOnnxAeScoreOnce(axis, path, out scoreAe, out infoAe)) continue;

                    double thrAe = (ae.Threshold > 0) ? ae.Threshold : DefaultThreshold;
                    bool isAnom = scoreAe >= thrAe;
                    AlarmLevel level = isAnom ? (scoreAe >= thrAe * 10.0 ? AlarmLevel.Danger : AlarmLevel.Warning) : AlarmLevel.Normal;
                    anyAxisProcessed = true;

                    BeginInvoke(new Action(() =>
                    {
                        RenderSampleChartSafe(path, ae.YColumn, axis);

                        // KPI/로그
                        if (level == AlarmLevel.Danger) Interlocked.Increment(ref cntDanger);
                        else if (level == AlarmLevel.Warning) Interlocked.Increment(ref cntWarning);
                        Interlocked.Increment(ref cycles);
                        cardDanger.ValueText = cntDanger + " 건";
                        cardWarning.ValueText = cntWarning + " 건";
                        cardCycles.ValueText = cycles + " 회";

                        var alarmText = (level == AlarmLevel.Danger) ? "DANGER" : (level == AlarmLevel.Warning) ? "WARN" : "OK";
                        AppendEventLog($"[AE] axis {axis}  mae={scoreAe:0.0000} thr={thrAe:0.0000} => {alarmText} ({Path.GetFileName(path)})");

                        if (level != AlarmLevel.Normal)
                        {
                            ShowToast(level, axis, scoreAe);
                            rows.Add(new EventRow
                            {
                                TimeLine     = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"),
                                Sensor       = $"가속도 Ax{axis}",
                                AnomalyScore = Math.Round(scoreAe, 4),
                                Threshold    = Math.Round(thrAe, 4),
                                Alarm        = (level == AlarmLevel.Danger) ? "위험" : "경고"
                            });
                            if (grid.Rows.Count > 0)
                                try { grid.FirstDisplayedScrollingRowIndex = grid.Rows.Count - 1; } catch { }
                        }

                        // 라인 차트에 AE 점수 반영
                        scoreSeries.Enqueue(Tuple.Create(axis, DateTime.Now, scoreAe));
                        while (scoreSeries.Count > 3000) { Tuple<int, DateTime, double> dump; scoreSeries.TryDequeue(out dump); }
                        lblStatus.Text = $"상태: 처리완료 {DateTime.Now:HH:mm:ss} (AE axis {axis}, {Path.GetFileName(path)})";
                    }));

                    if (!isAnom && _axisOnnxCls.ContainsKey(axis) && _axisGauges.TryGetValue(axis, out var g))
                    {
                        // 마지막 값 유지 대신, 0으로 리셋하거나 PredIndex = -1 처리
                        int k = g.Labels?.Length ?? (_clsLabels?.Length ?? 4);
                        UpdateAxisClassGauge(axis, Enumerable.Repeat(0f, k).ToArray(), -1);
                    }

                    // ---------- (B) 비정상일 때만 분류 실행 ----------
                    if (isAnom && _axisOnnxCls.TryGetValue(axis, out var cls) && cls?.Session != null)
                    {
                        // 기존 TryOnnxInferOnce 재사용 (분류)
                        if (TryOnnxInferOnce(axis, path, out int pred, out float[] probs, out string infoCls))
                        {
                            float p = (probs != null && probs.Length > 0 && pred >= 0 && pred < probs.Length) ? probs[pred] : 0f;
                            BeginInvoke(new Action(() =>
                            {
                                AppendEventLog($"[CLS] axis {axis}  pred={pred} p={p:0.000} ({infoCls}) file={Path.GetFileName(path)}");
                                UpdateAxisClassGauge(axis, probs, pred); // 게이지는 여기서만 갱신
                            }));
                        }
                    }
                }

                // ---------- (A2) CLS 단독 — AE 없이 분류만 등록된 축 처리 ----------
                foreach (var kv in _axisOnnxCls.OrderBy(k => k.Key))
                {
                    int axis = kv.Key;
                    var cls = kv.Value;
                    if (cls == null || cls.Session == null || cls.IsAutoencoder) continue;
                    // AE가 이미 커버한 축은 (B)에서 처리됐으므로 skip
                    if (_axisOnnx.ContainsKey(axis) && _axisOnnx[axis]?.Session != null) continue;
                    if (axesByName.Count > 0 && !axesByName.Contains(axis)) continue;
                    if (movedAxes.Count > 0 && !movedAxes.Contains(axis)) continue;
                    if (string.IsNullOrWhiteSpace(cls.YColumn) || !HasYColumns(headers, cls, axis)) continue;

                    int predCls; float[] probsCls; string infoCls;
                    if (!TryOnnxInferOnce(axis, path, out predCls, out probsCls, out infoCls)) continue;

                    anyAxisProcessed = true;
                    var capAxis = axis; var capPred = predCls; var capProbs = probsCls;
                    var capInfo = infoCls; var capYCol = cls.YColumn;
                    BeginInvoke(new Action(() =>
                    {
                        RenderSampleChartSafe(path, capYCol, capAxis);
                        Interlocked.Increment(ref cycles);
                        cardCycles.ValueText = cycles + " 회";

                        // 게이지에서 레이블 조회
                        ProbGaugeControl gauge;
                        string[] gaugeLabels = _axisGauges.TryGetValue(capAxis, out gauge) ? gauge.Labels : null;
                        string predLabel = (gaugeLabels != null && capPred >= 0 && capPred < gaugeLabels.Length)
                            ? gaugeLabels[capPred] : capPred.ToString();
                        float p = (capProbs != null && capProbs.Length > 0 && capPred >= 0 && capPred < capProbs.Length)
                            ? capProbs[capPred] : 0f;

                        AppendEventLog($"[CLS] axis {capAxis}  pred={predLabel} p={p:0.000}  ({capInfo})  ({Path.GetFileName(path)})");
                        UpdateAxisClassGauge(capAxis, capProbs, capPred);
                        lblStatus.Text = $"상태: 처리완료 {DateTime.Now:HH:mm:ss} (CLS axis {capAxis}, {Path.GetFileName(path)})";
                    }));
                }

                foreach (KeyValuePair<int, AxisModel> kv in _axisModels.OrderBy(k => k.Key))
                {
                    int axis = kv.Key;
                    AxisModel axisModel = kv.Value;
                    PersistedKnnModel m = axisModel == null ? null : axisModel.Model;
                    if (m == null) continue;
                    if (movedAxes.Count > 0 && !movedAxes.Contains(axis)) continue; // 움직인 축만 처리
                    if (string.IsNullOrWhiteSpace(m.YColumn) || !headerSet.Contains(m.YColumn)) continue;

                    double[] sample = Retry<double[]>(() => BuildFeatureVectorFromCsv(path, m.YColumn, m.Features), 5, 100);
                    if (sample == null) continue;

                    double score = SignalFeatures.ScoreKnn(sample, m.Train, m.K, m.Standardize, m.Mean, m.Std);
                    double thr = (m.Threshold > 0) ? m.Threshold : DefaultThreshold;
                    bool isAnom = score >= thr;
                    AlarmLevel level = isAnom ? (score >= thr * 10.0 ? AlarmLevel.Danger : AlarmLevel.Warning) : AlarmLevel.Normal;

                    anyAxisProcessed = true;
                    BeginInvoke(new Action(delegate ()
                    {
                        RenderSampleChartSafe(path, m.YColumn, axis);

                        // KPI
                        if (level == AlarmLevel.Danger) Interlocked.Increment(ref cntDanger);
                        else if (level == AlarmLevel.Warning) Interlocked.Increment(ref cntWarning);
                        Interlocked.Increment(ref cycles);
                        cardDanger.ValueText = cntDanger + " 건";
                        cardWarning.ValueText = cntWarning + " 건";
                        cardCycles.ValueText = cycles + " 회";

                        // 실시간 로그 (정상도 포함해서 모든 결과를 위쪽 로그 영역에 남김)
                        var alarmText = (level == AlarmLevel.Danger) ? "DANGER"
                                      : (level == AlarmLevel.Warning) ? "WARN"
                                      : "OK";
                        var msg = $"[{DateTime.Now:HH:mm:ss}] axis {axis}  score={score:F2}  thr={thr:F2}  => {alarmText}  ({Path.GetFileName(path)})";
                        AppendEventLog(msg);

                        if (level == AlarmLevel.Danger || level == AlarmLevel.Warning)
                            ShowToast(level, axis, score);
                        // 그리드(DataGridView)에는 경고/위험만 기록
                        if (level != AlarmLevel.Normal)
                        {
                            rows.Add(new EventRow
                            {
                                TimeLine     = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"),
                                Sensor       = $"분류 Ax{axis}",
                                AnomalyScore = Math.Round(score, 1),
                                Threshold    = Math.Round(thr, 1),
                                Alarm        = (level == AlarmLevel.Danger) ? "위험" : "경고"
                            });

                            if (grid.Rows.Count > 0)
                            {
                                try { grid.FirstDisplayedScrollingRowIndex = grid.Rows.Count - 1; } catch { }
                            }
                        }

                        // 라인차트 큐
                        scoreSeries.Enqueue(Tuple.Create(axis, DateTime.Now, score));
                        while (scoreSeries.Count > 3000)
                        {
                            Tuple<int, DateTime, double> dump;
                            scoreSeries.TryDequeue(out dump);
                        }

                        // --- ONNX 분류 테스트 (있으면 실행) ---
                        if (TryOnnxInferOnce(axis, path, out int pred, out float[] probs, out string info))
                        {
                            float p = (probs != null && probs.Length > 0) ? probs[pred] : 0f;
                            AppendEventLog($"[ONNX] axis {axis}  pred={pred}  prob={p:0.000}  ({info})  file={Path.GetFileName(path)}");

                            UpdateAxisClassGauge(axis, probs, pred);
                        }

                        lblStatus.Text = "상태: 처리완료 " + DateTime.Now.ToString("HH:mm:ss") + " (axis " + axis + ", " + Path.GetFileName(path) + ")";
                    }));
                }

                // ---------- (C) sklearn ONNX (AIForm 모델) — 전체 축 공용 모델 ----------
                foreach (KeyValuePair<int, OnnxSklModel> kv in _axisSklModels.OrderBy(k => k.Key))
                {
                    int axis = axesByName.Count > 0 ? axesByName.First() : 0;
                    OnnxSklModel skl = kv.Value;
                    if (skl == null || skl.OnnxSession == null) continue;
                    if (string.IsNullOrWhiteSpace(skl.YColumn)) continue;

                    // 토크 모델은 토크 CSV 컬럼 이름으로 재매핑 (예: "torque" → "Ax0_Trq(%)")
                    string effectiveYColumn = TorqueYColumns.Contains(skl.YColumn)
                        ? ResolveTorqueColumn(headers, axesByName.Count > 0 ? axesByName.First() : 0)
                        : skl.YColumn;
                    if (string.IsNullOrEmpty(effectiveYColumn) || !headerSet.Contains(effectiveYColumn)) continue;

                    bool isAnom; int predClass; float[] probs; double rawScore; string sklInfo;
                    if (!TrySklOnnxScore(skl, path, effectiveYColumn, out isAnom, out predClass, out probs, out rawScore, out sklInfo))
                        continue;

                    // 임계값 결정:
                    // knn AD: C# kNN 거리 기준 → Threshold 직접 사용
                    // 그 외 AD: score_threshold(decision function 기반) 있으면 사용, 없으면 label만
                    bool isKnnAd = skl.Session == "AD" && skl.ModelType == "knn"
                                   && skl.TrainVectors != null && skl.TrainVectors.Length > 0;
                    double thr = isKnnAd                      ? skl.Threshold
                               : skl.ScoreThreshold > 0      ? skl.ScoreThreshold
                                                              : 0.0;
                    bool useScoreThreshold = isKnnAd || skl.ScoreThreshold > 0;

                    AlarmLevel level;
                    if (skl.Session == "AD")
                    {
                        if (useScoreThreshold)
                            level = rawScore >= thr * 1.5 ? AlarmLevel.Danger
                                  : rawScore >= thr       ? AlarmLevel.Warning
                                  : AlarmLevel.Normal;
                        else
                            level = isAnom ? AlarmLevel.Warning : AlarmLevel.Normal;
                    }
                    else
                        level = isAnom ? AlarmLevel.Warning : AlarmLevel.Normal;

                    anyAxisProcessed = true;

                    BeginInvoke(new Action(() =>
                    {
                        RenderSampleChartSafe(path, effectiveYColumn, axis);

                        if (level == AlarmLevel.Danger) Interlocked.Increment(ref cntDanger);
                        else if (level == AlarmLevel.Warning) Interlocked.Increment(ref cntWarning);
                        Interlocked.Increment(ref cycles);
                        cardDanger.ValueText = cntDanger + " 건";
                        cardWarning.ValueText = cntWarning + " 건";
                        cardCycles.ValueText = cycles + " 회";

                        var alarmText = level == AlarmLevel.Danger ? "DANGER" : level == AlarmLevel.Warning ? "WARN" : "OK";
                        AppendEventLog($"[SKL-{skl.Session}] axis {axis}  {sklInfo}  => {alarmText}  ({Path.GetFileName(path)})");

                        if (probs != null && probs.Length > 0)
                            UpdateAxisClassGauge(axis, probs, predClass >= 0 ? predClass : (isAnom ? 1 : 0));

                        if (level != AlarmLevel.Normal)
                        {
                            ShowToast(level, axis, rawScore);
                            rows.Add(new EventRow
                            {
                                TimeLine     = DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"),
                                Sensor       = $"SKL Ax{axis}",
                                AnomalyScore = Math.Round(rawScore, 4),
                                Threshold    = Math.Round(useScoreThreshold ? thr : skl.Threshold, 4),
                                Alarm        = level == AlarmLevel.Danger ? "위험" : "경고"
                            });
                            if (grid.Rows.Count > 0)
                                try { grid.FirstDisplayedScrollingRowIndex = grid.Rows.Count - 1; } catch { }
                        }

                        scoreSeries.Enqueue(Tuple.Create(axis, DateTime.Now, rawScore));
                        while (scoreSeries.Count > 3000) { Tuple<int, DateTime, double> dump; scoreSeries.TryDequeue(out dump); }
                        lblStatus.Text = $"상태: 처리완료 {DateTime.Now:HH:mm:ss} (SKL axis {axis}, {Path.GetFileName(path)})";
                    }));
                }

                // ---------- (D) 전역 모델 폴백 — 축별 모델이 없는 축에 적용 ----------
                if (_globalKnnModel != null || _globalOnnxAe != null)
                {
                    // 이미 처리된 축 수집
                    var coveredAxes = new HashSet<int>(
                        _axisModels.Keys.Concat(_axisOnnx.Keys).Concat(_axisSklModels.Keys));

                    // 이 파일의 후보 축 (파일명 기반 + 커버된 축)
                    var candidateAxes = axesByName.Count > 0
                        ? new HashSet<int>(axesByName)
                        : new HashSet<int>(coveredAxes);
                    if (candidateAxes.Count == 0) candidateAxes.Add(0);

                    foreach (int axis in candidateAxes)
                    {
                        if (coveredAxes.Contains(axis)) continue; // 이미 처리됨

                        // 전역 KNN
                        if (_globalKnnModel != null)
                        {
                            var gm = _globalKnnModel;
                            if (!string.IsNullOrWhiteSpace(gm.YColumn) && headerSet.Contains(gm.YColumn))
                            {
                                double[] sample = Retry<double[]>(() => BuildFeatureVectorFromCsv(path, gm.YColumn, gm.Features), 5, 100);
                                if (sample != null)
                                {
                                    double score = SignalFeatures.ScoreKnn(sample, gm.Train, gm.K, gm.Standardize, gm.Mean, gm.Std);
                                    double thr = gm.Threshold > 0 ? gm.Threshold : DefaultThreshold;
                                    bool isAnom = score >= thr;
                                    AlarmLevel level = isAnom ? (score >= thr * 10.0 ? AlarmLevel.Danger : AlarmLevel.Warning) : AlarmLevel.Normal;
                                    anyAxisProcessed = true;
                                    var captAxis = axis; var captScore = score; var captThr = thr; var captLevel = level;
                                    BeginInvoke(new Action(() =>
                                    {
                                        RenderSampleChartSafe(path, gm.YColumn, captAxis);
                                        if (captLevel == AlarmLevel.Danger) Interlocked.Increment(ref cntDanger);
                                        else if (captLevel == AlarmLevel.Warning) Interlocked.Increment(ref cntWarning);
                                        Interlocked.Increment(ref cycles);
                                        cardDanger.ValueText = cntDanger + " 건"; cardWarning.ValueText = cntWarning + " 건"; cardCycles.ValueText = cycles + " 회";
                                        var alarmText = captLevel == AlarmLevel.Danger ? "DANGER" : captLevel == AlarmLevel.Warning ? "WARN" : "OK";
                                        AppendEventLog($"[G-KNN] axis {captAxis}  score={captScore:F2}  thr={captThr:F2}  => {alarmText}  ({Path.GetFileName(path)})");
                                        if (captLevel != AlarmLevel.Normal) ShowToast(captLevel, captAxis, captScore);
                                        scoreSeries.Enqueue(Tuple.Create(captAxis, DateTime.Now, captScore));
                                        while (scoreSeries.Count > 3000) { Tuple<int, DateTime, double> dump; scoreSeries.TryDequeue(out dump); }
                                        lblStatus.Text = $"상태: 처리완료 {DateTime.Now:HH:mm:ss} (G-KNN axis {captAxis}, {Path.GetFileName(path)})";
                                    }));
                                }
                            }
                        }

                        // 전역 AE(ONNX)
                        if (_globalOnnxAe?.Session != null)
                        {
                            var gae = _globalOnnxAe;
                            if (!string.IsNullOrWhiteSpace(gae.YColumn) && HasYColumns(headers, gae, axis))
                            {
                                double scoreAe; string infoAe;
                                if (TryOnnxAeScoreOnce(axis, path, gae, out scoreAe, out infoAe))
                                {
                                    double thrAe = gae.Threshold > 0 ? gae.Threshold : DefaultThreshold;
                                    bool isAnom = scoreAe >= thrAe;
                                    AlarmLevel level = isAnom ? (scoreAe >= thrAe * 10.0 ? AlarmLevel.Danger : AlarmLevel.Warning) : AlarmLevel.Normal;
                                    anyAxisProcessed = true;
                                    var captAxis = axis; var captScore = scoreAe; var captThr = thrAe; var captLevel = level;
                                    BeginInvoke(new Action(() =>
                                    {
                                        RenderSampleChartSafe(path, gae.YColumn, captAxis);
                                        if (captLevel == AlarmLevel.Danger) Interlocked.Increment(ref cntDanger);
                                        else if (captLevel == AlarmLevel.Warning) Interlocked.Increment(ref cntWarning);
                                        Interlocked.Increment(ref cycles);
                                        cardDanger.ValueText = cntDanger + " 건"; cardWarning.ValueText = cntWarning + " 건"; cardCycles.ValueText = cycles + " 회";
                                        var alarmText = captLevel == AlarmLevel.Danger ? "DANGER" : captLevel == AlarmLevel.Warning ? "WARN" : "OK";
                                        AppendEventLog($"[G-AE] axis {captAxis}  mae={captScore:F4}  thr={captThr:F4}  => {alarmText}  ({Path.GetFileName(path)})");
                                        if (captLevel != AlarmLevel.Normal) ShowToast(captLevel, captAxis, captScore);
                                        scoreSeries.Enqueue(Tuple.Create(captAxis, DateTime.Now, captScore));
                                        while (scoreSeries.Count > 3000) { Tuple<int, DateTime, double> dump; scoreSeries.TryDequeue(out dump); }
                                        lblStatus.Text = $"상태: 처리완료 {DateTime.Now:HH:mm:ss} (G-AE axis {captAxis}, {Path.GetFileName(path)})";
                                    }));
                                }
                            }
                        }
                    }
                }

                // 처리 시도 여부와 무관하게 현재 길이 저장 → 동일 파일 재처리 방지
                lock (_sync) { _lastProcessedLen[path] = curLen; }

                if (!anyAxisProcessed)
                {
                    // 모델 매칭 실패 로그 (디버그용)
                    BeginInvoke(new Action(() =>
                        AppendEventLog($"[FSW] 매칭 모델 없음: {Path.GetFileName(path)} (헤더/Y컬럼 불일치?)")));
                }
            }
            catch (Exception ex) { System.Diagnostics.Debug.WriteLine(ex); }
            finally { UnmarkProcessing(path); }
        }

        private static bool WaitUntilStable(string path, int timeoutMs, int sampleMs, int stableSamples)
        {
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            long? lastLen = null; int stableCount = 0;
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                long len;
                try { FileInfo fi = new FileInfo(path); fi.Refresh(); len = fi.Length; }
                catch { Thread.Sleep(sampleMs); continue; }
                if (lastLen.HasValue && len == lastLen.Value) { stableCount++; if (stableCount >= stableSamples) return true; }
                else { stableCount = 0; lastLen = len; }
                Thread.Sleep(sampleMs);
            }
            return false;
        }

        private List<int> DetermineMovedAxes(string filePath, string[] headers, double motionEps)
        {
            List<AxisCol> axisCols = new List<AxisCol>();
            for (int i = 0; i < headers.Length; i++)
            {
                for (int r = 0; r < AxisPosRegexes.Length; r++)
                {
                    Match m = AxisPosRegexes[r].Match(headers[i]);
                    int axisIdParsed;
                    if (m.Success && int.TryParse(m.Groups["id"].Value, out axisIdParsed))
                    { axisCols.Add(new AxisCol { AxisId = axisIdParsed, ColIndex = i }); break; }
                }
            }
            if (axisCols.Count == 0) return new List<int>();

            HashSet<int> moved = new HashSet<int>();
            try
            {
                using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (StreamReader sr = new StreamReader(fs))
                {
                    sr.ReadLine(); // header
                    Dictionary<int, double> mins = new Dictionary<int, double>();
                    Dictionary<int, double> maxs = new Dictionary<int, double>();
                    for (int i = 0; i < axisCols.Count; i++)
                    { if (!mins.ContainsKey(axisCols[i].AxisId)) mins[axisCols[i].AxisId] = double.PositiveInfinity; if (!maxs.ContainsKey(axisCols[i].AxisId)) maxs[axisCols[i].AxisId] = double.NegativeInfinity; }

                    string line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        string[] cells = SplitCsvLine(line);
                        for (int i = 0; i < axisCols.Count; i++)
                        {
                            int AxisId = axisCols[i].AxisId; int ColIndex = axisCols[i].ColIndex;
                            if (ColIndex >= cells.Length) continue;
                            double v;
                            if (double.TryParse(cells[ColIndex], NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                            { if (v < mins[AxisId]) mins[AxisId] = v; if (v > maxs[AxisId]) maxs[AxisId] = v; }
                        }
                    }
                    foreach (int axisId in new List<int>(mins.Keys))
                    {
                        if (double.IsInfinity(mins[axisId]) || double.IsInfinity(maxs[axisId])) continue;
                        if (maxs[axisId] - mins[axisId] > motionEps) moved.Add(axisId);
                    }
                }
            }
            catch (Exception ex) { System.Diagnostics.Debug.WriteLine(ex); }
            List<int> result = new List<int>(moved);
            result.Sort();
            return result;
        }

        private void FlushLineChart()
        {
            // ─ 로컬 진단 스코어 (chartLine이 있을 때만) ──────────────────────
            if (chartLine != null && !chartLine.IsDisposed && !scoreSeries.IsEmpty)
            {
                if (_lineFirstFlush)
                {
                    var dmy = chartLine.Series.FindByName(SkeletonSeriesName);
                    if (dmy != null) chartLine.Series.Remove(dmy);
                    var area = chartLine.ChartAreas["a"];
                    area.AxisX.Minimum = area.AxisX.Maximum = double.NaN;
                    area.AxisY.Minimum = area.AxisY.Maximum = double.NaN;
                    _lineFirstFlush = false;
                }
                Tuple<int, DateTime, double> item;
                while (scoreSeries.TryDequeue(out item))
                {
                    var s = EnsureAxisSeries(item.Item1);
                    s.Points.AddXY(item.Item2.ToOADate(), item.Item3);
                }
                foreach (Series s in chartLine.Series)
                {
                    if (s.Name == SkeletonSeriesName) continue;
                    while (s.Points.Count > ChartKeepPoints) s.Points.RemoveAt(0);
                }
                chartLine.ChartAreas["a"].RecalculateAxesScale();
            }

            // ─ 서버 추론 스코어 → _chartAccel / _chartTorque ─────────────────
            if (_chartAccel == null || _chartTorque == null) return;
            if (_liveScoreQueue.IsEmpty) return;

            // 첫 실데이터 수신 시 스켈레톤 시리즈 제거 (X축 고정 방지)
            if (_liveChartFirstData)
            {
                foreach (var ch in new[] { _chartAccel, _chartTorque })
                {
                    var sk = ch.Series.FindByName("_sk_");
                    if (sk != null) ch.Series.Remove(sk);
                }
                _liveChartFirstData = false;
            }

            var latestTime = DateTime.MinValue;
            Tuple<string, DateTime, double> liveItem;
            while (_liveScoreQueue.TryDequeue(out liveItem))
            {
                bool isAccel = liveItem.Item1.StartsWith("accel", StringComparison.OrdinalIgnoreCase);
                Chart target = isAccel ? _chartAccel : _chartTorque;
                var ls = EnsureLiveSeries(target, liveItem.Item1, isAccel);
                ls.Points.AddXY(liveItem.Item2.ToOADate(), liveItem.Item3);
                if (liveItem.Item2 > latestTime) latestTime = liveItem.Item2;
            }

            if (latestTime == DateTime.MinValue) latestTime = DateTime.Now;
            double xMax = latestTime.AddSeconds(10).ToOADate();

            // X 최솟값: 실제 데이터 첫 포인트와 10분 롤링 창 중 더 최근 값 사용
            // → 막 시작했을 때는 데이터 시작점에 맞게 좁히고, 10분 이상 쌓이면 롤링 창으로 전환
            double xMinRolling = latestTime.AddMinutes(-10).ToOADate();
            double xMinData    = double.MaxValue;
            foreach (Chart _ch in new[] { _chartAccel, _chartTorque })
            {
                if (_ch == null || _ch.IsDisposed) continue;
                foreach (Series _s in _ch.Series)
                    if (!_s.Name.StartsWith("_") && _s.Points.Count > 0
                        && _s.Points[0].XValue < xMinData)
                        xMinData = _s.Points[0].XValue;
            }
            const double PadOA = 10.0 / 86400.0;  // 10초 여백 (OADate 단위)
            double xMin = (xMinData < double.MaxValue && xMinData > xMinRolling)
                ? xMinData - PadOA   // 데이터 시작 10초 전 (짧은 구간)
                : xMinRolling;       // 10분 롤링 창 (긴 구간)

            foreach (Chart ch in new[] { _chartAccel, _chartTorque })
            {
                if (ch == null || ch.IsDisposed || ch.ChartAreas.Count == 0) continue;
                foreach (Series s in ch.Series)
                    while (s.Points.Count > ChartKeepPoints) s.Points.RemoveAt(0);

                var area = ch.ChartAreas[0];
                area.AxisX.Minimum = xMin;
                area.AxisX.Maximum = xMax;
                area.AxisY.Minimum = 0;
                area.AxisY.Maximum = double.NaN;
            }

            // ── MA 이평선 업데이트 ────────────────────────────────────────────
            Tuple<string, DateTime, double> maItem;
            while (_liveMaQueue.TryDequeue(out maItem))
            {
                bool isAccelMa = maItem.Item1.StartsWith("accel", StringComparison.OrdinalIgnoreCase);
                Chart tgtMa = isAccelMa ? _chartAccel : _chartTorque;
                if (tgtMa == null || tgtMa.IsDisposed) continue;
                var mas = EnsureLiveMaSeries(tgtMa, maItem.Item1, isAccelMa);
                mas.Points.AddXY(maItem.Item2.ToOADate(), maItem.Item3);
            }
            foreach (Chart ch in new[] { _chartAccel, _chartTorque })
            {
                if (ch == null || ch.IsDisposed) continue;
                foreach (Series s in ch.Series)
                    if (s.Name.StartsWith("ma-"))
                        while (s.Points.Count > ChartKeepPoints) s.Points.RemoveAt(0);
            }
        }

        private Series EnsureAxisSeries(int axis)
        {
            string name = "Axis " + axis;
            var s = chartLine.Series.FindByName(name);
            if (s != null) return s;

            s = new Series(name)
            {
                ChartType = SeriesChartType.FastLine,
                XValueType = ChartValueType.DateTime,
                BorderWidth = 2,
                LegendText = name
            };
            chartLine.Series.Add(s);
            return s;
        }

        /// <summary>서버 추론 전용 시리즈를 chart 에서 찾거나 생성합니다.</summary>
        /// <param name="chart">대상 차트 (_chartAccel 또는 _chartTorque)</param>
        /// <param name="key">예: "accel_ax0", "torque_ax2"</param>
        /// <param name="isAccel">가속도 여부 (색상 팔레트 선택)</param>
        private Series EnsureLiveSeries(Chart chart, string key, bool isAccel)
        {
            string seriesName = "srv-" + key;
            var s = chart.Series.FindByName(seriesName);
            if (s != null) return s;

            int axIdx = -1;  // -1 = 전역(global) 모델
            int ux = key.LastIndexOf("_ax", StringComparison.OrdinalIgnoreCase);
            if (ux >= 0) int.TryParse(key.Substring(ux + 3), out axIdx);
            bool isGlobal = axIdx < 0;
            if (isGlobal) axIdx = 0;  // 색상 인덱스용

            Color[] accelColors  = { Color.FromArgb(0, 112, 204),  Color.FromArgb(0, 170, 230),  Color.FromArgb(70, 130, 210), Color.FromArgb(100, 160, 240) };
            Color[] torqueColors = { Color.FromArgb(210, 55, 20),  Color.FromArgb(240, 100, 40), Color.FromArgb(255, 150, 0),  Color.FromArgb(200, 75, 55)  };
            Color c = isAccel ? accelColors[axIdx % accelColors.Length]
                               : torqueColors[axIdx % torqueColors.Length];

            string legendText = isGlobal
                ? (isAccel ? "가속도 (전역)" : "토크 (전역)")
                : (isAccel ? $"Ax{axIdx} 가속" : $"Ax{axIdx} 토크");

            s = new Series(seriesName)
            {
                ChartType   = SeriesChartType.FastLine,
                XValueType  = ChartValueType.DateTime,
                BorderWidth = 2,
                LegendText  = legendText,
                Color       = c
            };
            // 스켈레톤 제거 (첫 실제 시리즈 추가 시)
            var sk = chart.Series.FindByName("_sk_");
            if (sk != null) chart.Series.Remove(sk);
            chart.Series.Add(s);
            return s;
        }

        /// <summary>MA(이동평균)선 시리즈를 chart 에서 찾거나 생성합니다.</summary>
        private Series EnsureLiveMaSeries(Chart chart, string key, bool isAccel)
        {
            string seriesName = "ma-" + key;
            var s = chart.Series.FindByName(seriesName);
            if (s != null) return s;

            int axIdx = 0;
            int ux = key.LastIndexOf("_ax", StringComparison.OrdinalIgnoreCase);
            if (ux >= 0) int.TryParse(key.Substring(ux + 3), out axIdx);

            // raw선보다 밝고 굵게 — 추세선임을 시각적으로 구분
            Color[] accelMa  = { Color.FromArgb(0, 170, 255),  Color.FromArgb(80, 210, 255), Color.FromArgb(120, 180, 255), Color.FromArgb(160, 200, 255) };
            Color[] torqueMa = { Color.FromArgb(255, 100, 50), Color.FromArgb(255, 150, 70), Color.FromArgb(255, 190, 30),  Color.FromArgb(240, 120, 100) };
            Color c = isAccel ? accelMa[axIdx % accelMa.Length]
                               : torqueMa[axIdx % torqueMa.Length];

            s = new Series(seriesName)
            {
                ChartType   = SeriesChartType.FastLine,
                XValueType  = ChartValueType.DateTime,
                BorderWidth = 3,
                LegendText  = isAccel ? "가속도 이평" : "토크 이평",
                Color       = c,
            };
            chart.Series.Add(s);
            return s;
        }

        /// <summary>
        /// 센서+축 조합 칩이 _statusFlow 에 없으면 동적으로 추가합니다.
        /// UI 스레드에서만 호출해야 합니다.
        /// </summary>
        private void EnsureLiveChip(string key, string displayName)
        {
            if (_statusChips.ContainsKey(key) || _statusFlow == null) return;

            var chip = new Panel
            {
                Width = 128, Height = 42,
                Margin = new Padding(0, 4, 8, 4),
                BackColor = Color.FromArgb(243, 244, 246),
                BorderStyle = BorderStyle.FixedSingle
            };
            var lblName = new Label
            {
                Text = displayName,
                Font = new Font("Segoe UI", 7.5f, FontStyle.Bold),
                Left = 5, Top = 4, Width = 118, Height = 15,
                ForeColor = Color.FromArgb(55, 55, 65)
            };
            var lblState = new Label
            {
                Text = "—",
                Font = new Font("Segoe UI", 8f),
                Left = 5, Top = 21, Width = 70, Height = 16,
                ForeColor = Color.Gray
            };
            var lblScore = new Label
            {
                Text = "",
                Font = new Font("Segoe UI", 7.5f),
                Left = 74, Top = 23, Width = 50, Height = 14,
                ForeColor = Color.FromArgb(80, 80, 90),
                TextAlign = ContentAlignment.MiddleRight
            };
            // 우클릭 컨텍스트 메뉴: 임계값 설정
            var cms = new ContextMenuStrip();
            var miSet = new ToolStripMenuItem("임계값 설정...");
            string capturedKey = key;
            miSet.Click += (s, e) => ShowThresholdDialog(capturedKey);
            cms.Items.Add(miSet);
            chip.ContextMenuStrip = cms;
            foreach (Control c in new Control[] { lblName, lblState, lblScore })
                c.ContextMenuStrip = cms;

            chip.Controls.AddRange(new Control[] { lblName, lblState, lblScore });

            // 체크박스 상태에 따라 초기 Visible 결정
            // key = "accel_ax0" / "torque_ax1" / "combined_ax0" — 첫 번째 '_' 앞이 sensorType
            string chipSensor = key.Contains("_") ? key.Substring(0, key.IndexOf('_')) : key;
            chip.Visible = _IsSensorEnabled(chipSensor);

            _statusFlow.Controls.Add(chip);

            _statusChips[key]      = chip;
            _liveStatusLabels[key] = lblState;
            _liveScoreLabels[key]  = lblScore;
        }

        /// <summary>
        /// 축별 임계값을 설정하는 다이얼로그를 표시합니다.
        /// </summary>
        private void ShowThresholdDialog(string key)
        {
            double current = _axisThresholds.TryGetValue(key, out double v) ? v : 1.0;

            using (var dlg = new Form
            {
                Text            = $"임계값 설정 — {key}",
                Size            = new Size(300, 145),
                FormBorderStyle = FormBorderStyle.FixedDialog,
                StartPosition   = FormStartPosition.CenterParent,
                MaximizeBox     = false, MinimizeBox = false
            })
            {
                var lbl = new Label
                {
                    Text     = $"[{key}] 이상 판정 임계값:",
                    Left = 12, Top = 14, Width = 270, Height = 18,
                    Font = new Font("Segoe UI", 9f)
                };
                var nud = new NumericUpDown
                {
                    Left = 12, Top = 36, Width = 120, Height = 26,
                    Minimum = 0.001M, Maximum = 9999M,
                    DecimalPlaces = 3, Increment = 0.1M,
                    Value = (decimal)Math.Max(0.001, Math.Min(9999, current)),
                    Font = new Font("Segoe UI", 10f)
                };
                var lblNote = new Label
                {
                    Text      = "차트에는 score ÷ threshold 정규화 값이 표시됩니다.",
                    Left = 12, Top = 68, Width = 270, Height = 16,
                    Font = new Font("Segoe UI", 7.5f), ForeColor = Color.Gray
                };
                var btnOk = new Button
                {
                    Text = "확인", Left = 140, Top = 88, Width = 70, Height = 26,
                    DialogResult = DialogResult.OK
                };
                var btnCancel = new Button
                {
                    Text = "취소", Left = 218, Top = 88, Width = 60, Height = 26,
                    DialogResult = DialogResult.Cancel
                };
                dlg.AcceptButton = btnOk;
                dlg.CancelButton = btnCancel;
                dlg.Controls.AddRange(new Control[] { lbl, nud, lblNote, btnOk, btnCancel });

                if (dlg.ShowDialog(this) == DialogResult.OK)
                {
                    _axisThresholds[key] = (double)nud.Value;
                    SaveAxisThresholds();
                    // 툴팁 즉시 갱신
                    if (_statusChips.TryGetValue(key, out Panel chip))
                        _chipToolTip.SetToolTip(chip, $"임계값: {nud.Value:F3}  (우클릭 → 변경)");
                    AppEvents.RaiseLog($"[임계값] {key} = {nud.Value:F3} 설정됨");
                }
            }
        }

        /// <summary>
        /// 축별 임계값을 파일에서 읽어옵니다 (앱 시작 시 호출).
        /// 파일 형식: key=value (한 줄에 하나)
        /// </summary>
        private void LoadAxisThresholds()
        {
            try
            {
                if (!File.Exists(ThresholdsFile)) return;
                foreach (string line in File.ReadAllLines(ThresholdsFile))
                {
                    int eq = line.IndexOf('=');
                    if (eq <= 0) continue;
                    string k = line.Substring(0, eq).Trim();
                    if (double.TryParse(line.Substring(eq + 1).Trim(),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out double val) && val > 0)
                        _axisThresholds[k] = val;
                }
            }
            catch { /* 무시 — 파일 없으면 서버 전달값으로 자동 초기화 */ }
        }

        /// <summary>
        /// 현재 축별 임계값을 파일에 저장합니다 (폼 닫힐 때 + 설정 변경 시 호출).
        /// </summary>
        private void SaveAxisThresholds()
        {
            try
            {
                Directory.CreateDirectory(DefaultLogsPath);
                var lines = _axisThresholds
                    .OrderBy(kv => kv.Key)
                    .Select(kv => $"{kv.Key}={kv.Value.ToString("F4", System.Globalization.CultureInfo.InvariantCulture)}");
                File.WriteAllLines(ThresholdsFile, lines);
            }
            catch { }
        }

        /// <summary>
        /// AppEvents.InferenceResultReceived 핸들러 — 서버 추론 결과를 UI에 반영합니다.
        /// </summary>
        private void OnLiveInferenceResult(string sensorType, InferenceResult result)
        {
            if (result == null) return;
            if (!_IsSensorEnabled(sensorType)) return;

            // 서버 오류 응답: 칩에 오류 상태 표시 후 즉시 반환
            if (result.IsError)
            {
                // key 복원: axis 정보가 없으므로 sensorType만 사용
                string errKey = sensorType;
                if (!IsHandleCreated || IsDisposed) return;
                BeginInvoke(new Action(() =>
                {
                    if (_statusChips.TryGetValue(errKey, out Panel chip) && chip != null)
                        chip.BackColor = Color.FromArgb(254, 240, 200); // 연노랑 = 오류
                    if (_liveStatusLabels.TryGetValue(errKey, out Label lblErr) && lblErr != null)
                    {
                        lblErr.Text     = "⚠ 오류";
                        lblErr.ForeColor = Color.FromArgb(160, 80, 0);
                    }
                    if (_liveScoreLabels.TryGetValue(errKey, out Label lblScoreErr) && lblScoreErr != null)
                        lblScoreErr.Text = "---";
                }));
                return;
            }

            bool isAccel = string.Equals(sensorType, "accel", StringComparison.OrdinalIgnoreCase);
            // 키: "accel_ax0", "torque_ax2" 등 (axis 없으면 "accel", "torque")
            string key = result.Axis.HasValue
                         ? $"{sensorType}_ax{result.Axis.Value}"
                         : sensorType;

            // 축별 임계값 초기화: 파일에 저장된 값 없으면 서버 전달값으로 초기화
            if (result.Threshold > 0)
                _axisThresholds.TryAdd(key, result.Threshold);

            // 정규화 스코어를 차트 큐에 추가 (score / threshold → 임계선 항상 1.0)
            double axThr      = _axisThresholds.TryGetValue(key, out double t) ? t : 1.0;
            double normScore  = axThr > 0 ? (double)result.AnomalyScore / axThr : (double)result.AnomalyScore;

            // EMA 평활화 적용 (차트 노이즈 감소 — 이상 판정은 rawScore 기반으로 별도 수행)
            // ── 단기 EMA (차트 raw선) ─────────────────────────────────────────
            double prevEma    = _chartEma.GetOrAdd(key, normScore);
            double smoothed   = ChartEmaAlpha * normScore + (1.0 - ChartEmaAlpha) * prevEma;
            _chartEma[key]    = smoothed;

            _liveScoreQueue.Enqueue(Tuple.Create(key, DateTime.Now, smoothed));
            while (_liveScoreQueue.Count > 6000)
            {
                Tuple<string, DateTime, double> _discard;
                _liveScoreQueue.TryDequeue(out _discard);
            }

            // ── 장기 EMA (MA 이평선) ──────────────────────────────────────────
            double prevMa  = _chartMaEma.GetOrAdd(key, normScore);
            double maScore = ChartMaAlpha * normScore + (1.0 - ChartMaAlpha) * prevMa;
            _chartMaEma[key] = maScore;

            _liveMaQueue.Enqueue(Tuple.Create(key, DateTime.Now, maScore));
            while (_liveMaQueue.Count > 6000)
            {
                Tuple<string, DateTime, double> _discardMa;
                _liveMaQueue.TryDequeue(out _discardMa);
            }

            // UI 컨트롤 업데이트는 UI 스레드에서
            if (!IsHandleCreated || IsDisposed) return;
            BeginInvoke(new Action(() =>
            {
                string axLabel     = result.Axis.HasValue ? $" Ax{result.Axis.Value}" : "";
                string sensorLabel = isAccel ? "가속도"
                                   : string.Equals(sensorType, "combined", StringComparison.OrdinalIgnoreCase) ? "결합"
                                   : "토크";
                string displayName = sensorLabel + axLabel;

                // 칩이 없으면 상태 바에 동적 추가
                EnsureLiveChip(key, displayName);

                _liveStatusLabels.TryGetValue(key, out Label lblStatus);
                _liveScoreLabels.TryGetValue(key, out Label lblScore);
                _statusChips.TryGetValue(key, out Panel chip);
                if (lblStatus == null) return;

                // 클라이언트 임계값 기준 이상 판정 (per-axis threshold 반영)
                double clientThr = _axisThresholds.TryGetValue(key, out double ct) ? ct : 1.0;
                double rawScore  = (double)result.AnomalyScore;
                bool   threshAnomaly = rawScore >= clientThr;

                // ── 연속 카운터: N회 연속 초과 시 확정 (단발 노이즈 제거) ──────────
                int prevConsec;
                _consecutiveAnomalyCount.TryGetValue(key, out prevConsec);
                int newConsec = threshAnomaly ? prevConsec + 1 : 0;
                _consecutiveAnomalyCount[key] = newConsec;
                bool confirmedThreshAnomaly = newConsec >= AnomalyConfirmCount;

                // ── EMA 베이스라인 대비 급증 감지 (외력 등 순간 이상 — 즉시 반응) ──
                var baseline = _scoreBaseline.GetOrAdd(key, (rawScore, 0));
                double ema   = baseline.ema;
                int    cnt   = baseline.count;
                bool   spikeAnomaly = cnt >= SpikeWarmup && rawScore > ema * SpikeFactor;
                // EMA 업데이트: 항상 반영 (스파이크 구간만 제외하면 베이스라인이 낮게 고착됨)
                double newEma = ema * (1 - SpikeEmaAlpha) + rawScore * SpikeEmaAlpha;
                _scoreBaseline[key] = (newEma, cnt + 1);

                // 최종 이상 판정: 연속 N회 초과(확정) 또는 급증 스파이크(즉시)
                bool   anomaly    = confirmedThreshAnomaly || spikeAnomaly;
                string spikeTag   = (spikeAnomaly && !threshAnomaly) ? " ↑급증" : "";
                string stateText  = anomaly ? "⚠ 이상" : "✓ 정상";
                Color  stateClr   = anomaly ? Color.FromArgb(180, 25, 25) : Color.FromArgb(18, 120, 55);

                lblStatus.Text      = stateText;
                lblStatus.ForeColor = stateClr;

                // 점수 표시: RawMae+RawThreshold가 있으면 uncapped 정규화 점수 사용
                // (서버가 anomaly_score를 1.0으로 cap하는 경우 raw 값으로 보정)
                float displayScore = (result.RawMae.HasValue && result.RawThreshold.HasValue
                                      && result.RawThreshold.Value > 0)
                    ? result.RawMae.Value / result.RawThreshold.Value
                    : result.AnomalyScore;
                if (lblScore != null) lblScore.Text = $"{displayScore:F3}";

                // 칩 툴팁: 현재 임계값 표시
                _chipToolTip.SetToolTip(chip, $"임계값: {clientThr:F3}  (우클릭 → 변경)");

                // 칩 배경 색상 (이상=연빨강, 정상=연초록, 처음=연회색)
                if (chip != null)
                    chip.BackColor = anomaly
                        ? Color.FromArgb(254, 226, 226)
                        : Color.FromArgb(220, 252, 231);

                // 실시간 분류 현황 매트릭스 갱신 (정상/경고/위험 모두 반영)
                // 가속도(전역 단일 모델, axis 없음) → _pnlAccelAeBar 패널
                // 토크(per-axis) → DGV 매트릭스 행
                if (isAccel && !result.Axis.HasValue)
                    UpdateAccelAeDisplay(normScore);
                else if (!isAccel && result.Axis.HasValue)
                    UpdateClassMatrix(result.Axis.Value, normScore);

                // 사용 모델 파일명 라벨 갱신
                if (!string.IsNullOrEmpty(result.ModelFile))
                    UpdateModelInfoLabel(isAccel, result.ModelFile);

                bool hasCls = !string.IsNullOrEmpty(result.ClassName) &&
                              !string.Equals(result.ClassName, "normal", StringComparison.OrdinalIgnoreCase) &&
                              !string.Equals(result.ClassName, "anomaly", StringComparison.OrdinalIgnoreCase);
                string cls = hasCls ? $" ({result.ClassName})" : "";

                // /predict fallback 경로: ClassName이 있으면 CLS 매트릭스도 갱신
                // (/predict/combined 성공 시에는 OnLiveClsInferenceResult에서 처리되므로 중복 업데이트 방지)
                if (hasCls && result.Axis.HasValue)
                {
                    object confVal = result.Confidence.HasValue
                        ? (object)result.Confidence.Value
                        : (object)"-";
                    UpdateClassMatrixCls(result.Axis.Value, sensorType, result.ClassName, confVal);
                }

                // 이상 감지 시 KPI / 이벤트 로그 갱신
                // ── 이상/정상 이벤트 로그 (상태 전환 + 30초 쿨다운으로 홍수 방지) ──
                bool wasAnomaly  = _prevAnomalyState.TryGetValue(key, out bool _pw) && _pw;
                bool cooldownOk  = !_lastAnomalyLogTime.TryGetValue(key, out DateTime _lt)
                                   || (DateTime.Now - _lt) >= AnomalyLogCooldown;

                if (anomaly)
                {
                    bool isDanger  = normScore >= DangerMultiplier;
                    string spikeInfo = spikeAnomaly && !threshAnomaly
                        ? $"  ema={ema:F3}→{rawScore:F3}(×{(ema>0?rawScore/ema:0):F1})" : "";
                    string levelTag  = isDanger ? "🔴 위험" : "🟡 경고";

                    // 상태 전환(정상→이상) 또는 쿨다운 경과 시만 기록 — AE(이상탐지) 전 센서 대상
                    bool shouldLog = !wasAnomaly || cooldownOk;
                    if (shouldLog)
                    {
                        // ── KPI 카드 업데이트 ─────────────────────────────────
                        if (isDanger) Interlocked.Increment(ref cntDanger);
                        else          Interlocked.Increment(ref cntWarning);
                        cardDanger.ValueText  = cntDanger  + " 건";
                        cardWarning.ValueText = cntWarning + " 건";
                        if (isDanger) ShowToast(AlarmLevel.Danger, displayName, displayScore);

                        AppendEventLog(
                            $"[{DateTime.Now:HH:mm:ss}] {levelTag} {displayName} 이상{spikeTag}  " +
                            $"score={displayScore:F3}  thr={clientThr:F3}{cls}{spikeInfo}");
                        _lastAnomalyLogTime[key] = DateTime.Now;

                        rows.Add(new EventRow
                        {
                            TimeLine     = DateTime.Now.ToString("HH:mm:ss"),
                            Sensor       = displayName,
                            AnomalyScore = Math.Round(displayScore, 4),
                            Threshold    = Math.Round(clientThr,    4),
                            Alarm        = levelTag + " " + displayName + " 이상" + spikeTag + cls
                        });
                        if (rows.Count > 500) rows.RemoveAt(0);
                    }
                }
                else if (wasAnomaly)
                {
                    // 이상→정상 복귀 시 1회 기록
                    AppendEventLog(
                        $"[{DateTime.Now:HH:mm:ss}] ✅ 복귀 {displayName}  score={displayScore:F3}");
                }
                _prevAnomalyState[key] = anomaly;
            }));
        }

        /// <summary>
        /// AppEvents.ClsInferenceResultReceived 핸들러 — CLS 결함진단 결과를 분류 매트릭스에 반영합니다.
        /// ClsAvailable=false이면 "모델없음"으로 표시, true이면 결함명/신뢰도 표시.
        /// </summary>
        private void OnLiveClsInferenceResult(string sensorType, CombinedInferenceResult combined)
        {
            if (combined == null || combined.IsError || !combined.Axis.HasValue) return;
            if (!_IsSensorEnabled(sensorType)) return;

            bool isAccel = string.Equals(sensorType, "accel", StringComparison.OrdinalIgnoreCase);

            if (!IsHandleCreated || IsDisposed) return;
            BeginInvoke(new Action(() =>
            {
                string clsName;
                object confVal;
                if (!combined.ClsAvailable)
                {
                    clsName = "모델없음";
                    confVal = "-";
                }
                else
                {
                    clsName = combined.ClsClassName ?? "-";
                    confVal = combined.ClsConfidence.HasValue
                              ? (object)(combined.ClsConfidence.Value)
                              : (object)"-";
                }
                UpdateClassMatrixCls(combined.Axis.Value, sensorType, clsName, confVal);

                // 사용 모델 파일명 라벨 갱신 (AE + CLS)
                if (!string.IsNullOrEmpty(combined.AeModelFile) || !string.IsNullOrEmpty(combined.ClsModelFile))
                    UpdateModelInfoLabel(isAccel, combined.AeModelFile, combined.ClsModelFile);

                // 결함 감지 시 KPI 카운트 + 이벤트 로그 기록
                if (combined.ClsAvailable && combined.ClsIsFault == true)
                {
                    string axLabel   = $" Ax{combined.Axis.Value}";
                    string sensor    = isAccel ? "가속도"
                                     : string.Equals(sensorType, "combined", StringComparison.OrdinalIgnoreCase) ? "결합"
                                     : "토크";
                    string conf      = combined.ClsConfidence.HasValue
                                       ? $"  신뢰도={combined.ClsConfidence.Value:P0}" : "";

                    // 신뢰도 0.7 이상 → 위험, 미만 → 경고
                    bool isCritical = combined.ClsConfidence.HasValue
                                      ? combined.ClsConfidence.Value >= 0.7f
                                      : true; // 신뢰도 없으면 위험으로 분류
                    if (isCritical) { cntDanger++;  cardDanger.ValueText  = cntDanger  + " 건"; }
                    else            { cntWarning++; cardWarning.ValueText = cntWarning + " 건"; }

                    AppendEventLog(
                        $"[{DateTime.Now:HH:mm:ss}] 🔶 결함진단 {sensor}{axLabel} = {clsName}{conf}");
                }
            }));
        }

        /// <summary>
        /// 실시간 분류 현황 매트릭스 CLS 열 갱신 (UI 스레드에서만 호출)
        /// torque/combined per-axis 결과만 처리; accel은 전역 단일 모델이므로 여기 오지 않음
        /// </summary>
        private void UpdateClassMatrixCls(int axis, string sensorType, string className, object confValue)
        {
            if (_classMatrixDgv == null || axis < 0) return;

            // accel CLS는 per-axis DGV 대상 아님 (전역 단일 모델) — 무시
            if (string.Equals(sensorType, "accel", StringComparison.OrdinalIgnoreCase)) return;

            // 행 확보 (AE 결과보다 먼저 올 수도 있으므로 여기서도 생성)
            if (!_classMatrixAxisRow.TryGetValue(axis, out int rowIdx))
            {
                rowIdx = _classMatrixDgv.Rows.Add();
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_AXIS].Value     = axis;
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TS].Value       = 0.0;
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TSTATE].Value   = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TCLS].Value     = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TCLSCONF].Value = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_CCLS].Value     = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_CCLSCONF].Value = "-";
                _classMatrixAxisRow[axis] = rowIdx;
            }

            int clsCol, confCol;
            switch (sensorType?.ToLowerInvariant())
            {
                case "combined":
                    clsCol  = CMG_COL_CCLS;  confCol = CMG_COL_CCLSCONF; break;
                default: // "torque"
                    clsCol  = CMG_COL_TCLS;  confCol = CMG_COL_TCLSCONF; break;
            }

            _classMatrixDgv.Rows[rowIdx].Cells[clsCol].Value  = className ?? "-";
            _classMatrixDgv.Rows[rowIdx].Cells[confCol].Value = confValue ?? (object)"-";
            _classMatrixDgv.InvalidateRow(rowIdx);
        }

        /// <summary>
        // ── 센서 표시 필터 헬퍼 ─────────────────────────────────────────────────

        /// <summary>체크박스 상태 기반으로 해당 sensorType의 표시 여부를 반환합니다.</summary>
        private bool _IsSensorEnabled(string sensorType)
        {
            switch (sensorType?.ToLowerInvariant())
            {
                case "accel":    return _chkShowAccel    == null || _chkShowAccel.Checked;
                case "torque":   return _chkShowTorque   == null || _chkShowTorque.Checked;
                case "combined": return _chkShowCombined == null || _chkShowCombined.Checked;
                default:         return true;
            }
        }

        /// <summary>
        /// 체크박스 상태에 따라 _dualChartPanel 의 열 너비와 차트 래퍼 패널 Visible 을 조정합니다.
        /// 가속도만 표시: 가속도 100% / 토크 열 0px
        /// 토크만 표시:   가속도 열 0px / 토크 100%
        /// 둘 다 표시:    각 50%
        /// 둘 다 숨김:    두 래퍼 모두 Visible=false
        /// </summary>
        private void _UpdateChartLayout()
        {
            if (_dualChartPanel == null) return;
            bool showA = _chkShowAccel?.Checked  ?? true;
            bool showT = _chkShowTorque?.Checked ?? false;

            if (_accelChartWrap  != null) _accelChartWrap.Visible  = showA;
            if (_torqueChartWrap != null) _torqueChartWrap.Visible = showT;

            _dualChartPanel.SuspendLayout();
            _dualChartPanel.ColumnStyles.Clear();
            if (showA && showT)
            {
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            }
            else if (showA)
            {
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 0));
            }
            else if (showT)
            {
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 0));
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            }
            else
            {
                // 둘 다 꺼진 경우 — 래퍼가 Visible=false이므로 비율은 무관
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
                _dualChartPanel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            }
            _dualChartPanel.ResumeLayout(true);
        }

        /// <summary>
        /// 체크박스 CheckedChanged 시 호출 — 해당 sensorType의 칩과 컨트롤을 표시/숨깁니다.
        /// 가속도: _pnlAccelAeBar 패널 (전역 단일 모델)
        /// 토크/결합: DGV 열 그룹
        /// </summary>
        private void ApplySensorVisibility(string sensorType, bool visible)
        {
            // 1) 상태 바 칩 show/hide
            foreach (var kv in _statusChips)
            {
                string chipSensor = kv.Key.Contains("_")
                    ? kv.Key.Substring(0, kv.Key.IndexOf('_'))
                    : kv.Key;
                if (string.Equals(chipSensor, sensorType, StringComparison.OrdinalIgnoreCase))
                    kv.Value.Visible = visible;
            }

            // 2) 센서별 컨트롤 show/hide
            switch (sensorType?.ToLowerInvariant())
            {
                case "accel":
                    // 가속도는 전역 단일 모델 → _pnlAccelAeBar 패널
                    if (_pnlAccelAeBar != null) _pnlAccelAeBar.Visible = visible;
                    break;
                case "torque":
                    if (_classMatrixDgv != null)
                    {
                        _classMatrixDgv.Columns[CMG_COL_TS].Visible       = visible;
                        _classMatrixDgv.Columns[CMG_COL_TSTATE].Visible   = visible;
                        _classMatrixDgv.Columns[CMG_COL_TCLS].Visible     = visible;
                        _classMatrixDgv.Columns[CMG_COL_TCLSCONF].Visible = visible;
                    }
                    break;
                case "combined":
                    if (_classMatrixDgv != null)
                    {
                        _classMatrixDgv.Columns[CMG_COL_CCLS].Visible     = visible;
                        _classMatrixDgv.Columns[CMG_COL_CCLSCONF].Visible = visible;
                    }
                    break;
            }
        }

        // ─────────────────────────────────────────────────────────────────────

        /// <summary>Teaching Sequence 한 회차 완료 → 설비 사용률(cycles) 카드 갱신</summary>
        private void OnLoopCompleted(int count)
        {
            cycles = count;
            if (!IsHandleCreated || IsDisposed) return;
            BeginInvoke(new Action(() =>
            {
                cardCycles.ValueText = cycles + " 회";
            }));
        }

        // ─────────────────────────────────────────────────────────────────────
        // 실시간 분류 현황 매트릭스
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// 실시간 분류 현황 패널을 생성합니다. (행=축, 열=가속도/토크 점수+상태)
        /// </summary>
        private Panel BuildClassMatrixPanel()
        {
            var wrap = new Panel { Dock = DockStyle.Fill, Padding = new Padding(0, 0, 0, 4) };

            var lbl = new Label
            {
                Text = "실시간 분류 현황  ( AE: 토크 이상탐지 점수/판정 │ CLS: 토크/결합 결함명/신뢰도 )",
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                Dock = DockStyle.Top, Height = 22,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(4, 0, 0, 0)
            };

            _classMatrixDgv = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AllowUserToResizeRows = false,
                RowHeadersVisible = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                MultiSelect = false,
                AutoGenerateColumns = false,
                EnableHeadersVisualStyles = false,
                ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing,
                ColumnHeadersHeight = 22,
                BackgroundColor = SystemColors.Window,
                BorderStyle = BorderStyle.None,
                GridColor = Color.FromArgb(220, 220, 225),
                DefaultCellStyle = new DataGridViewCellStyle
                {
                    Font = new Font("Segoe UI", 8.5f),
                    Alignment = DataGridViewContentAlignment.MiddleCenter
                },
                ColumnHeadersDefaultCellStyle = new DataGridViewCellStyle
                {
                    Font = new Font("Segoe UI", 8f, FontStyle.Bold),
                    BackColor = Color.FromArgb(50, 50, 65),
                    ForeColor = Color.White,
                    Alignment = DataGridViewContentAlignment.MiddleCenter
                }
            };
            _classMatrixDgv.RowTemplate.Height = 24;

            // col 0: 축
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "Axis",    HeaderText = "축",       Width = 34, ReadOnly = true });
            // col 1-2: AE 토크 (per-axis)
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "TScore",  HeaderText = "AE 토크",  Width = 80, ReadOnly = true,
                  DefaultCellStyle = new DataGridViewCellStyle { Format = "0.000", Alignment = DataGridViewContentAlignment.MiddleCenter } });
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "TState",  HeaderText = "판정",     Width = 68, ReadOnly = true });
            // col 3-4: CLS 토크 결함 (per-axis)
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "TCLS",    HeaderText = "토크 결함",  Width = 90, ReadOnly = true });
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "TCLSC",   HeaderText = "신뢰도",   Width = 52, ReadOnly = true,
                  DefaultCellStyle = new DataGridViewCellStyle { Format = "0%", Alignment = DataGridViewContentAlignment.MiddleCenter } });
            // col 5-6: CLS 결합 결함 (per-axis)
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "CCLS",    HeaderText = "결합 결함",  Width = 90, ReadOnly = true });
            _classMatrixDgv.Columns.Add(new DataGridViewTextBoxColumn
                { Name = "CCLSC",   HeaderText = "신뢰도",   Width = 52, ReadOnly = true,
                  DefaultCellStyle = new DataGridViewCellStyle { Format = "0%", Alignment = DataGridViewContentAlignment.MiddleCenter } });

            // 마지막 열은 남은 공간 채우기
            _classMatrixDgv.Columns[CMG_COL_CCLSCONF].AutoSizeMode = DataGridViewAutoSizeColumnMode.Fill;

            _classMatrixDgv.CellFormatting += ClassMatrixDgv_CellFormatting;

            wrap.Controls.Add(_classMatrixDgv);
            wrap.Controls.Add(lbl);   // Top — 최상단 헤더
            return wrap;
        }

        /// <summary>
        /// 실시간 분류 현황 매트릭스 셀 색상 (점수 기반 3단계)
        /// </summary>
        private void ClassMatrixDgv_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            if (e.RowIndex < 0) return;

            // ── AE 토크 이상탐지 열 색상 (점수 기반 3단계) ─────────────────
            if (e.ColumnIndex == CMG_COL_TS || e.ColumnIndex == CMG_COL_TSTATE)
            {
                double score = 0;
                var cell = _classMatrixDgv.Rows[e.RowIndex].Cells[CMG_COL_TS];
                if (cell.Value != null)
                    double.TryParse(cell.Value.ToString(), System.Globalization.NumberStyles.Any,
                        System.Globalization.CultureInfo.InvariantCulture, out score);

                Color bg, fg;
                if (score <= 0)                            { bg = Color.FromArgb(245,245,248); fg = Color.Gray; }
                else if (score >= DangerMultiplier)        { bg = Color.FromArgb(255,220,220); fg = Color.FromArgb(160,20,20); }
                else if (score >= WarnMultiplier)          { bg = Color.FromArgb(255,248,210); fg = Color.FromArgb(150,100,0); }
                else                                       { bg = Color.FromArgb(220,248,228); fg = Color.FromArgb(20,110,50); }

                e.CellStyle.BackColor          = bg;
                e.CellStyle.ForeColor          = fg;
                e.CellStyle.SelectionBackColor = bg;
                e.CellStyle.SelectionForeColor = fg;
                return;
            }

            // ── CLS 결함진단 열 색상 (결함명 기반) ──────────────────────────
            if (e.ColumnIndex == CMG_COL_TCLS  || e.ColumnIndex == CMG_COL_TCLSCONF ||
                e.ColumnIndex == CMG_COL_CCLS  || e.ColumnIndex == CMG_COL_CCLSCONF)
            {
                int nameCol = (e.ColumnIndex == CMG_COL_TCLS || e.ColumnIndex == CMG_COL_TCLSCONF)
                              ? CMG_COL_TCLS
                              : CMG_COL_CCLS;

                string cls = _classMatrixDgv.Rows[e.RowIndex].Cells[nameCol].Value?.ToString() ?? "";

                Color bg, fg;
                if (string.IsNullOrEmpty(cls) || cls == "-" || cls == "모델없음")
                    { bg = Color.FromArgb(245,245,248); fg = Color.Gray; }
                else if (string.Equals(cls, "normal", StringComparison.OrdinalIgnoreCase))
                    { bg = Color.FromArgb(220,248,228); fg = Color.FromArgb(20,110,50); }
                else
                    { bg = Color.FromArgb(255,235,200); fg = Color.FromArgb(140,70,0); }  // 결함 = 주황

                e.CellStyle.BackColor          = bg;
                e.CellStyle.ForeColor          = fg;
                e.CellStyle.SelectionBackColor = bg;
                e.CellStyle.SelectionForeColor = fg;
            }
        }

        /// <summary>
        /// 실시간 분류 현황 매트릭스 갱신 — 토크 per-axis (UI 스레드에서만 호출)
        /// 가속도 전역 결과는 UpdateAccelAeDisplay() 를 사용하세요.
        /// </summary>
        private void UpdateClassMatrix(int axis, double normScore)
        {
            if (_classMatrixDgv == null || axis < 0) return;

            // 행 확보
            if (!_classMatrixAxisRow.TryGetValue(axis, out int rowIdx))
            {
                rowIdx = _classMatrixDgv.Rows.Add();
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_AXIS].Value     = axis;
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TS].Value       = 0.0;
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TSTATE].Value   = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TCLS].Value     = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TCLSCONF].Value = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_CCLS].Value     = "-";
                _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_CCLSCONF].Value = "-";
                _classMatrixAxisRow[axis] = rowIdx;
            }

            // 상태 텍스트
            string stateText;
            if (normScore <= 0)                     stateText = "-";
            else if (normScore >= DangerMultiplier)  stateText = "🔴 위험";
            else if (normScore >= WarnMultiplier)    stateText = "🟡 경고";
            else                                     stateText = "✓ 정상";

            _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TS].Value     = normScore;
            _classMatrixDgv.Rows[rowIdx].Cells[CMG_COL_TSTATE].Value = stateText;

            // 색 갱신을 위해 해당 행 무효화
            _classMatrixDgv.InvalidateRow(rowIdx);
        }

        /// <summary>
        /// 가속도 전역 AE 결과를 _pnlAccelAeBar 패널에 반영합니다 (UI 스레드에서만 호출).
        /// </summary>
        /// <summary>수신된 모델 파일명을 차트 하단 모델 정보 라벨에 반영합니다.</summary>
        private void UpdateModelInfoLabel(bool isAccel, string modelFile, string clsModelFile = null)
        {
            if (string.IsNullOrWhiteSpace(modelFile)) return;

            var seen  = isAccel ? _seenAccelModels  : _seenTorqueModels;
            var lbl   = isAccel ? _lblAccelModelInfo : _lblTorqueModelInfo;
            if (lbl == null || lbl.IsDisposed) return;

            bool changed = false;
            string fn = System.IO.Path.GetFileName(modelFile);
            if (!string.IsNullOrEmpty(fn) && seen.Add(fn)) changed = true;

            if (!string.IsNullOrEmpty(clsModelFile))
            {
                string cfn = System.IO.Path.GetFileName(clsModelFile);
                if (!string.IsNullOrEmpty(cfn) && seen.Add("cls:" + cfn)) changed = true;
            }

            if (!changed) return;

            // AE 모델: 파일명 목록, CLS 모델: "cls:" prefix 제거 후 표시
            var aeFiles  = new System.Collections.Generic.List<string>();
            var clsFiles = new System.Collections.Generic.List<string>();
            foreach (var s in seen)
            {
                if (s.StartsWith("cls:")) clsFiles.Add(s.Substring(4));
                else aeFiles.Add(s);
            }

            var parts = new System.Collections.Generic.List<string>();
            if (aeFiles.Count  > 0) parts.Add("AE: "  + string.Join(", ", aeFiles));
            if (clsFiles.Count > 0) parts.Add("CLS: " + string.Join(", ", clsFiles));
            lbl.Text = "모델  " + string.Join("  |  ", parts);
        }

        private void UpdateAccelAeDisplay(double normScore)
        {
            _accelAeNormScore = normScore;

            string stateText;
            Color  stateColor;
            if (normScore <= 0)
            {
                stateText  = "—";
                stateColor = Color.Gray;
            }
            else if (normScore >= DangerMultiplier)
            {
                stateText  = "🔴 위험";
                stateColor = Color.FromArgb(160, 20, 20);
            }
            else if (normScore >= WarnMultiplier)
            {
                stateText  = "🟡 경고";
                stateColor = Color.FromArgb(150, 100, 0);
            }
            else
            {
                stateText  = "✓ 정상";
                stateColor = Color.FromArgb(20, 110, 50);
            }

            if (_lblAccelAeScore != null) _lblAccelAeScore.Text      = normScore > 0 ? $"{normScore:F3}" : "—";
            if (_lblAccelAeState != null)
            {
                _lblAccelAeState.Text      = stateText;
                _lblAccelAeState.ForeColor = stateColor;
            }

            // 패널 배경색 갱신
            if (_pnlAccelAeBar != null)
            {
                _pnlAccelAeBar.BackColor = normScore <= 0
                    ? Color.FromArgb(245, 245, 248)
                    : normScore >= DangerMultiplier ? Color.FromArgb(255, 220, 220)
                    : normScore >= WarnMultiplier   ? Color.FromArgb(255, 248, 210)
                    : Color.FromArgb(220, 248, 228);
            }
        }

        private static bool HasYColumns(string[] headers, OnnxAxisModel om, int axis)
        {
            if (headers == null || om == null) return false;
            if (om.C > 1)
            {
                // ACC: 축까지 고려하여 x/y/z 중 아무 형태라도 존재하면 true
                int ix, iy, iz;
                return TryResolveAccelColumns(headers, om.YColumn, axis, out ix, out iy, out iz);
            }
            // 단일 채널(토크 등)
            return Array.FindIndex(headers, h => string.Equals(h?.Trim(), om.YColumn, StringComparison.OrdinalIgnoreCase)) >= 0;
        }
        private bool TryMarkProcessing(string path) { lock (_sync) return _processing.Add(path); }
        private void UnmarkProcessing(string path) { lock (_sync) _processing.Remove(path); }

        private T Retry<T>(Func<T> action, int maxAttempts, int initialDelayMs)
        {
            int delay = initialDelayMs;
            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                try { return action(); }
                catch (Exception ex) when ((ex is IOException || ex is UnauthorizedAccessException) && attempt < maxAttempts)
                { Thread.Sleep(delay); delay = Math.Min(delay * 2, 2000); }
            }
            return action();
        }

        private static string[] SplitCsvLine(string line)
        {
            return line.Split(new[] { ',', ';', '\t' }, StringSplitOptions.None);
        }

        private double[] BuildFeatureVectorFromCsv(string filePath, string yColumn, string[] featureKeysInOrder)
        {
            if (featureKeysInOrder == null || featureKeysInOrder.Length == 0) return null;
            double sr = AppState.GetForColumn(yColumn);

            return SignalFeatures.BuildFeatureVectorFromCsv(
                filePath,
                yColumn,
                featureKeysInOrder);
        }
        #endregion

        #region 유틸 컨트롤
        private class KpiCard : UserControl
        {
            public string Title { get { return _title.Text; } set { _title.Text = value; } }
            public string ValueText { get { return _value.Text; } set { _value.Text = value; } }
            public string DeltaText { get { return _delta.Text; } set { _delta.Text = value; } }
            public string Footnote { get { return _foot.Text; } set { _foot.Text = value; } }

            private readonly Label _title = new Label { Font = new Font("Segoe UI", 10, FontStyle.Bold) };
            private readonly Label _value = new Label { Font = new Font("Segoe UI", 20, FontStyle.Bold) };
            private readonly Label _delta = new Label { Font = new Font("Segoe UI", 9) };
            private readonly Label _foot = new Label { Font = new Font("Segoe UI", 8), ForeColor = Color.Gray };

            public KpiCard()
            {
                this.DoubleBuffered = true;
                this.Padding = new Padding(10, 8, 10, 8);   // 카드 내부 여백 (좌우10, 상하8)
                this.BackColor = Color.White;

                // ---- 내부 레이아웃: 행을 퍼센트 비율로 꽉 채움 ----
                var stack = new TableLayoutPanel
                {
                    Dock = DockStyle.Fill,
                    ColumnCount = 1,
                    RowCount = 4,
                    Margin = Padding.Empty,
                    Padding = Padding.Empty
                };

                // 비율 예시: 제목 20% / 값 40% / 증감 15% / 푸터 25%
                // (원하면 숫자만 바꿔 미세 조정)
                stack.RowStyles.Add(new RowStyle(SizeType.Percent, 20f));
                stack.RowStyles.Add(new RowStyle(SizeType.Percent, 40f));
                stack.RowStyles.Add(new RowStyle(SizeType.Percent, 15f));
                stack.RowStyles.Add(new RowStyle(SizeType.Percent, 25f));
                stack.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));

                // 라벨들이 셀을 가득 채우고, 서로 겹치지 않도록 설정
                _title.AutoSize = false;
                _title.Dock = DockStyle.Fill;
                _title.Margin = new Padding(0, 0, 0, 2);
                _title.TextAlign = ContentAlignment.MiddleLeft;

                _value.AutoSize = false;
                _value.Dock = DockStyle.Fill;
                _value.Margin = new Padding(0, 2, 0, 2);
                _value.TextAlign = ContentAlignment.MiddleLeft;

                _delta.AutoSize = false;
                _delta.Dock = DockStyle.Fill;
                _delta.Margin = new Padding(0, 2, 0, 2);
                _delta.TextAlign = ContentAlignment.MiddleLeft;

                _foot.AutoSize = false;
                _foot.Dock = DockStyle.Fill;
                _foot.Margin = new Padding(0, 2, 0, 0);
                _foot.TextAlign = ContentAlignment.MiddleLeft;

                // (멀티라인 방지/말줄임표가 필요하면)
                _title.AutoEllipsis = _value.AutoEllipsis = _delta.AutoEllipsis = _foot.AutoEllipsis = true;

                stack.Controls.Add(_title, 0, 0);
                stack.Controls.Add(_value, 0, 1);
                stack.Controls.Add(_delta, 0, 2);
                stack.Controls.Add(_foot, 0, 3);

                this.Controls.Add(stack);

                // 외부에서 Dock = Fill 로 쓰면 카드가 셀 크기에 맞게 자동 확장됨
                // (Size는 기본값만 두고 고정하지 않는 편이 좋습니다)
            }

            protected override void OnPaint(PaintEventArgs e)
            {
                base.OnPaint(e);
                Graphics g = e.Graphics; g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                Rectangle shadowRect = this.ClientRectangle; shadowRect.Inflate(-3, -4); shadowRect.Offset(2, 2);
                using (System.Drawing.Drawing2D.GraphicsPath shadowPath = Rounded(shadowRect, 14))
                using (SolidBrush shadowBrush = new SolidBrush(Color.FromArgb(35, 0, 0, 0)))
                { g.FillPath(shadowBrush, shadowPath); }
                Rectangle rect = this.ClientRectangle; rect.Inflate(-4, -6);
                using (System.Drawing.Drawing2D.GraphicsPath path = Rounded(rect, 16))
                using (SolidBrush bg = new SolidBrush(this.BackColor))
                using (Pen pen = new Pen(Color.FromArgb(210, 210, 210), 1))
                { g.FillPath(bg, path); g.DrawPath(pen, path); }
            }

            private static System.Drawing.Drawing2D.GraphicsPath Rounded(Rectangle r, int radius)
            {
                System.Drawing.Drawing2D.GraphicsPath gp = new System.Drawing.Drawing2D.GraphicsPath(); int d = radius * 2;
                gp.AddArc(r.X, r.Y, d, d, 180, 90);
                gp.AddArc(r.Right - d, r.Y, d, d, 270, 90);
                gp.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
                gp.AddArc(r.X, r.Bottom - d, d, d, 90, 90); gp.CloseFigure(); return gp;
            }
        }

        private class InputBox : Form
        {
            public string InputText { get { return _tb.Text; } }
            private TextBox _tb;
            public InputBox(string title, string message)
            {
                this.Text = title; this.Width = 360; this.Height = 160; this.StartPosition = FormStartPosition.CenterParent;
                Label lbl = new Label { Text = message, Dock = DockStyle.Top, Height = 32, TextAlign = ContentAlignment.MiddleLeft, Padding = new Padding(10, 8, 10, 4) };
                _tb = new TextBox { Dock = DockStyle.Top, Margin = new Padding(10) };
                FlowLayoutPanel flow = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 40, FlowDirection = FlowDirection.RightToLeft };
                Button ok = new Button { Text = "확인", DialogResult = DialogResult.OK, Width = 80 };
                Button cancel = new Button { Text = "취소", DialogResult = DialogResult.Cancel, Width = 80 };
                flow.Controls.Add(ok); flow.Controls.Add(cancel);
                this.Controls.Add(flow); this.Controls.Add(_tb); this.Controls.Add(lbl);
                this.AcceptButton = ok; this.CancelButton = cancel;
            }
        }
        #endregion
    }
}