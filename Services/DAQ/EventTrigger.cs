using System;
using System.Collections.Generic;
using System.Linq;

namespace PHM_Project_DockPanel.Services.DAQ
{
    // =========================================================================
    //  이벤트 트리거 설정
    // =========================================================================
    public sealed class TriggerConfig
    {
        // RMS 게이트 (느슨하게 설정, 주요 게이트는 Peak)
        public double OnsetThreshold    { get; set; } = 0.005; // 고정 임계값 (적응형 OFF 시)
        public double OffsetThreshold   { get; set; } = 0.005;
        public int    MinEventFrames    { get; set; } = 5;
        public int    MaxEventFrames    { get; set; } = 5000;
        public int    PreFrames         { get; set; } = 3;
        public int    PostFrames        { get; set; } = 5;

        // Layer 1: 피크 (로봇 진동의 주요 감지 기준)
        public double PeakThreshold     { get; set; } = 0.025; // 기존 0.05 → 0.025 g

        // Layer 1: STA/LTA
        public int    StaFrames         { get; set; } = 3;
        public int    LtaFrames         { get; set; } = 50;
        public double StaLtaRatio       { get; set; } = 2.0;   // 기존 3.0 → 2.0

        // Layer 2: 적응형 노이즈 floor
        public bool   UseAdaptiveThreshold   { get; set; } = false; // 기본 OFF (고정 임계값 사용)
        public double OnsetNoiseMultiplier   { get; set; } = 2.0;   // 기존 4.0 → 2.0
        public double OffsetNoiseMultiplier  { get; set; } = 1.5;
        public double NoiseFloorGamma        { get; set; } = 0.98;

        // Layer 2: Onset 게이트 조합
        public bool   RequirePeakForOnset    { get; set; } = true;
        public bool   RequireStaLtaForOnset  { get; set; } = false; // 기본 OFF (로봇 진동에 부적합)

        // Layer 3: HoldOff
        public int    HoldOffFrames     { get; set; } = 3;
    }

    // =========================================================================
    //  감지된 이벤트 데이터
    // =========================================================================
    public sealed class VibrationEvent
    {
        public DateTime StartUtc    { get; }
        public DateTime EndUtc      { get; set; }
        public List<double[,]> Frames { get; } = new List<double[,]>();
        public int    ChannelCount  { get; }
        public int    SampleRate    { get; }
        public string Device        { get; }
        public string EventId       { get; } = Guid.NewGuid().ToString("N").Substring(0, 12);
        public string Label         { get; set; } = "unknown";
        public string EventType     { get; set; } = "sustained";

        public double OnsetStaLta      { get; set; }
        public double OnsetPeak        { get; set; }
        public double OnsetNoiseFloor  { get; set; }

        public VibrationEvent(string device, int channelCount, int sampleRate, DateTime startUtc)
        {
            Device = device; ChannelCount = channelCount;
            SampleRate = sampleRate; StartUtc = startUtc;
        }

        public int    TotalSamples    => Frames.Sum(f => f.GetLength(1));
        public double DurationSeconds => TotalSamples / (double)SampleRate;

        public double[] ComputeOverallRms()
        {
            int ch = ChannelCount;
            var sum = new double[ch];
            int n = 0;
            foreach (var frame in Frames)
            {
                int fn = frame.GetLength(1);
                for (int c = 0; c < ch; c++)
                    for (int i = 0; i < fn; i++) { double v = frame[c, i]; sum[c] += v * v; }
                n += fn;
            }
            if (n > 0)
                for (int c = 0; c < ch; c++) sum[c] = Math.Sqrt(sum[c] / n);
            return sum;
        }

        public double[] ComputePeak()
        {
            int ch = ChannelCount;
            var peak = new double[ch];
            foreach (var frame in Frames)
            {
                int fn = frame.GetLength(1);
                for (int c = 0; c < ch; c++)
                    for (int i = 0; i < fn; i++)
                    {
                        double v = Math.Abs(frame[c, i]);
                        if (v > peak[c]) peak[c] = v;
                    }
            }
            return peak;
        }
    }

    // =========================================================================
    //  EventTrigger — 3-Layer 이벤트 감지
    //
    //  Layer 1: RMS · 순간 피크 · STA/LTA 비율
    //  Layer 2: 적응형 노이즈 floor 기반 Onset 게이트 (AND 조합)
    //  Layer 3: HoldOff + Impulse/Sustained 분류
    // =========================================================================
    public sealed class EventTrigger
    {
        public TriggerConfig Config { get; set; }
        public event Action<VibrationEvent> EventDetected;

        private enum State { Idle, Active, PostBuffer }
        private State _state = State.Idle;

        private VibrationEvent _current;
        private int _postCount;
        private int _holdOffCount;

        private readonly Queue<double[,]> _preBuffer = new Queue<double[,]>();
        private readonly Queue<double>    _staBuffer = new Queue<double>();
        private readonly Queue<double>    _ltaBuffer = new Queue<double>();

        private double _noiseFloor = -1.0;

        // ── 진단용 공개 속성 ─────────────────────────────────────────────────
        public double CurrentNoiseFloor   => _noiseFloor < 0 ? 0 : _noiseFloor;
        public double CurrentStaLta       { get; private set; }
        public double CurrentPeak         { get; private set; }
        public double CurrentRms          { get; private set; }
        public bool   IsActive            => _state != State.Idle;
        public string CurrentStateName    => _state.ToString();   // "Idle" / "Active" / "PostBuffer"

        public double EffectiveOnsetThreshold =>
            Config.UseAdaptiveThreshold && _noiseFloor > 0
                ? _noiseFloor * Config.OnsetNoiseMultiplier
                : Config.OnsetThreshold;

        public double EffectiveOffsetThreshold =>
            Config.UseAdaptiveThreshold && _noiseFloor > 0
                ? _noiseFloor * Config.OffsetNoiseMultiplier
                : Config.OffsetThreshold;

        public EventTrigger(TriggerConfig cfg) => Config = cfg;

        // ── 메인 진입점 ───────────────────────────────────────────────────────
        public void Feed(
            string device, int channelCount, int sampleRate,
            double[] rmsVector, double[,] rawFrame,
            DateTime frameTime, string label = "unknown")
        {
            double maxRms      = rmsVector.Max();
            double frameEnergy = maxRms * maxRms;
            double framePeak   = ComputeFramePeak(rawFrame);

            _staBuffer.Enqueue(frameEnergy);
            _ltaBuffer.Enqueue(frameEnergy);
            while (_staBuffer.Count > Config.StaFrames) _staBuffer.Dequeue();
            while (_ltaBuffer.Count > Config.LtaFrames) _ltaBuffer.Dequeue();

            double sta    = _staBuffer.Average();
            double lta    = _ltaBuffer.Count >= 2 ? _ltaBuffer.Average() : sta;
            double staLta = lta > 1e-12 ? sta / lta : 1.0;

            CurrentRms    = maxRms;
            CurrentPeak   = framePeak;
            CurrentStaLta = staLta;

            // Layer 2: 노이즈 floor (Idle 에서만 갱신)
            if (_state == State.Idle)
            {
                if (_noiseFloor < 0) _noiseFloor = maxRms;
                else _noiseFloor = Config.NoiseFloorGamma * _noiseFloor
                                   + (1.0 - Config.NoiseFloorGamma) * maxRms;
            }

            double onsetThr  = EffectiveOnsetThreshold;
            double offsetThr = EffectiveOffsetThreshold;

            bool rmsOk   = maxRms >= onsetThr;
            bool peakOk  = !Config.RequirePeakForOnset  || Config.PeakThreshold <= 0  || framePeak >= Config.PeakThreshold;
            bool staLtaOk= !Config.RequireStaLtaForOnset|| Config.StaLtaRatio   <= 0  || staLta    >= Config.StaLtaRatio;
            bool onsetGate = rmsOk && peakOk && staLtaOk;

            switch (_state)
            {
                case State.Idle:
                    if (onsetGate)
                    {
                        _current = new VibrationEvent(device, channelCount, sampleRate, frameTime)
                        {
                            Label = label,
                            OnsetStaLta     = staLta,
                            OnsetPeak       = framePeak,
                            OnsetNoiseFloor = _noiseFloor > 0 ? _noiseFloor : 0
                        };
                        foreach (var f in _preBuffer) _current.Frames.Add(f);
                        _preBuffer.Clear();
                        _current.Frames.Add(CopyFrame(rawFrame));
                        _holdOffCount = 0;
                        _state = State.Active;
                    }
                    else
                    {
                        _preBuffer.Enqueue(CopyFrame(rawFrame));
                        while (_preBuffer.Count > Config.PreFrames) _preBuffer.Dequeue();
                    }
                    break;

                case State.Active:
                    _current.Frames.Add(CopyFrame(rawFrame));
                    if (_current.Frames.Count >= Config.MaxEventFrames)
                    { FinalizeEvent(frameTime); break; }
                    if (maxRms < offsetThr)
                    { _postCount = 0; _holdOffCount = Config.HoldOffFrames; _state = State.PostBuffer; }
                    break;

                case State.PostBuffer:
                    _current.Frames.Add(CopyFrame(rawFrame));
                    _postCount++;
                    if (_holdOffCount > 0) _holdOffCount--;
                    if (_holdOffCount == 0 && onsetGate)
                    { _state = State.Active; _postCount = 0; break; }
                    if (_postCount >= Config.PostFrames)
                        FinalizeEvent(frameTime);
                    break;
            }
        }

        private void FinalizeEvent(DateTime endTime)
        {
            _current.EndUtc   = endTime;
            _current.EventType = _current.Frames.Count < Config.MinEventFrames ? "impulse" : "sustained";
            _state = State.Idle;
            _preBuffer.Clear();
            EventDetected?.Invoke(_current);
            _current = null;
        }

        private static double ComputeFramePeak(double[,] frame)
        {
            int ch = frame.GetLength(0), n = frame.GetLength(1);
            double peak = 0;
            for (int c = 0; c < ch; c++)
                for (int i = 0; i < n; i++) { double v = Math.Abs(frame[c, i]); if (v > peak) peak = v; }
            return peak;
        }

        private static double[,] CopyFrame(double[,] src)
        {
            int ch = src.GetLength(0), n = src.GetLength(1);
            var dst = new double[ch, n];
            Buffer.BlockCopy(src, 0, dst, 0, src.Length * sizeof(double));
            return dst;
        }

        public void ResetNoiseFloor(double value) => _noiseFloor = value > 0 ? value : -1.0;
    }
}
