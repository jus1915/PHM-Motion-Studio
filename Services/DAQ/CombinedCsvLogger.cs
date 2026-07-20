using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.DAQ
{
    /// <summary>
    /// 토크(1ms 폴링)와 가속도(DAQ 블록, 단일 센서)를 원래 해상도로 단일 CSV에 기록합니다.
    /// PHM 이상 탐지 학습(가속도 = 전체 구간, 토크 = 실제 동작 구간)에 필요한 축별 위치/속도/
    /// 모션 구분 정보까지 동일 행에 함께 기록합니다.
    ///
    /// ▶ 두 신호 해상도 동시 보존 방법:
    ///   1. 폴링 루프(1ms)가 링 버퍼에 (Stopwatch 틱, torque[], op[], pos[], cmdPos[], vel[], motionId[], seqId) 를 연속 저장
    ///   2. 가속도 블록 도착 시 각 샘플의 Stopwatch 추정 타임스탬프를 계산
    ///   3. 투 포인터로 링 버퍼를 순회 → 각 Accel 샘플에 가장 가까운(=시간적으로 가장 근접한 최신) 폴링 값을 매핑
    ///   → Accel 원해상도 + 나머지 신호 ≈1ms 해상도 정합 (보간 없음 — 항상 최근접 폴링 시점의 최신값 사용)
    ///
    /// CSV 컬럼 (열 순서):
    ///   time_s, x, y, z,
    ///   Ax{n}_Trq(%)...,                 ← 기존 컬럼(이름/의미 불변) — Ajin: AxmStatusReadTorque(%), Sim: 가상 토크(%)
    ///   [Op_Ax{n}...,                    ← 기존 컬럼(이름/의미 불변) — "Idle" | "Pos" (그 외 상태는 현재 코드가 구분하지 않음)
    ///    Ax{n}_ActPos,                   ← 실제 위치 피드백(mm) — Ajin: AxmStatusGetActPos / Sim: 내부 시뮬레이션 위치
    ///    Ax{n}_CmdPos,                   ← 명령(목표) 위치(mm) — Ajin: AxmStatusGetCmdPos / Sim: 마지막 MoveAbs/MoveRel 목표값
    ///    Ax{n}_ActVel,                   ← 실제 속도 피드백(mm/s) — Ajin: AxmStatusReadVel / Sim: 해석적 순시속도
    ///    Ax{n}_MotionID...]              ← 축별 모션 일련번호 (Idle=0, Pos 중엔 해당 모션의 ID 유지)
    ///   SequenceID                       ← 여러 축이 조합된 장비 시퀀스 1회 반복 식별자 (TeachingForm 회차 시작 시 +1, 없으면 0)
    ///
    /// [ ] 안의 5개 컬럼군은 제어기가 연결되어 있을 때(getAxisOp != null)만 기록됩니다 — 제어기 미연결이면
    ///     위치/속도/모션 정보 자체가 없으므로 컬럼을 생략합니다 (열 개수는 세션 내내 고정).
    /// 값을 확보할 수 없는 경우(주입된 콜백이 없거나 예외) ActPos/CmdPos/ActVel은 NaN, MotionID/SequenceID는
    /// 0으로 기록합니다 — 결측을 0으로 위장하지 않기 위함입니다 (MotionID/SequenceID의 0은 "미배정/Idle"이라는
    /// 명확한 의미를 가지므로 결측과 구분됩니다).
    ///
    /// ▶ 가속도는 물리적으로 단일 센서 → x, y, z 3열만 기록 (축별 구분 없음)
    /// ▶ 토크/Op/ActPos/CmdPos/ActVel/MotionID는 모터(축)별로 별도 열 유지
    /// ▶ 파일 회전: 세션 하나가 1시간 또는 500MB를 넘으면 새 파일(_partNNN)을 열어 이어서 기록합니다.
    ///   time_s는 세션 시작 기준 Stopwatch를 그대로 이어 쓰므로 회전 후에도 파일 간 연속성이 유지됩니다.
    /// </summary>
    public sealed class CombinedCsvLogger : IDisposable
    {
        [DllImport("winmm.dll")] private static extern uint timeBeginPeriod(uint uPeriod);
        [DllImport("winmm.dll")] private static extern uint timeEndPeriod(uint uPeriod);

        // ── 외부 주입 ──────────────────────────────────────────────
        private readonly Func<int, double> _getTorque;
        private readonly Func<int, string> _getAxisOp;
        private readonly Func<int, double> _getActPos;
        private readonly Func<int, double> _getCmdPos;
        private readonly Func<int, double> _getActVel;
        private readonly Func<int, int>    _getMotionId;
        private readonly Func<int>         _getSequenceId;
        private readonly int[]             _axes;
        private readonly Action<string>    _log;
        /// <summary>제어기 연결 시 true → Op_Ax/ActPos/CmdPos/ActVel/MotionID 컬럼 포함. 미연결 시 false → 생략.</summary>
        private readonly bool              _hasAxisColumns;

        // ── 링 버퍼 (토크 + Op + 위치/속도/모션 동시 저장) ────────────────
        // 크기 = 8192: 1kHz 기준 ≈8초, 빠른 샘플레이트에도 여유
        private const int RING = 8192;
        private readonly double[] _tBuf;    // [ringIdx * nAxes + axIdx] 토크
        private readonly string[] _opBuf;   // [ringIdx * nAxes + axIdx] Op 상태 (intern)
        private readonly double[] _posBuf;  // [ringIdx * nAxes + axIdx] ActPos(mm)
        private readonly double[] _cmdBuf;  // [ringIdx * nAxes + axIdx] CmdPos(mm)
        private readonly double[] _velBuf;  // [ringIdx * nAxes + axIdx] ActVel(mm/s)
        private readonly int[]    _midBuf;  // [ringIdx * nAxes + axIdx] MotionID
        private readonly int[]    _seqBuf;  // [ringIdx] SequenceID (축 무관, 링 슬롯당 1개)
        private readonly long[]   _tTick;   // 각 폴링의 Stopwatch 틱
        private long _tTotal = 0;
        private readonly object _tLock = new object();

        // ── Accel 샘플 카운터 (타임스탬프 계산용, 단일 센서) ──────
        private long _accelTotal = 0;

        // ── 파일 ──────────────────────────────────────────────────
        private StreamWriter _writer;
        private readonly object _writerLock = new object();
        private string _filePath;
        private string _dir;
        private string _baseNameRoot;

        // ── 파일 회전 정책 ────────────────────────────────────────
        // 하나의 CSV가 모션마다 새로 생기지 않고 한 세션 동안 연속 기록되므로, 장시간 연속 수집 시
        // 파일이 무한정 커지지 않도록 시간/용량 중 먼저 도달하는 기준으로 새 파일(_partNNN)을 엽니다.
        private const double RotateMaxHours = 1.0;
        private const long   RotateMaxBytes = 500L * 1024 * 1024; // 500MB
        private Stopwatch _fileOpenSw;   // 현재 파일이 열린 시점부터 경과 시간 (회전 판단용, 세션 전체 시간과 별개)
        private int _rotationIndex = 0;  // 0 = 원본 파일명, 1+ = _part001, _part002 ...

        // ── 공유 Stopwatch (세션 시작 기준, 회전과 무관하게 계속 흐름 → time_s 연속성 보장) ──
        private Stopwatch _sw;

        // ── 상태 ──────────────────────────────────────────────────
        private CancellationTokenSource _cts;
        private Task _pollTask;
        private bool _running;
        private bool _disposed;

        public bool   IsLogging  => _running;
        public string OutputPath => _filePath;

        /// <summary>
        /// 토크 폴링 루프가 각 샘플을 기록할 때 추가로 호출할 콜백.
        /// 파라미터: (axisIndex, torqueValue, sampleTimeUtc)
        /// InfluxDB 피드 등 외부 처리에 사용.
        /// </summary>
        public Action<int, double, DateTime> TorqueSampled { get; set; }

        public CombinedCsvLogger(
            Func<int, double> getTorque,
            Func<int, string> getAxisOp,
            int[]             axes,
            Action<string>    log = null,
            Func<int, double> getActPos = null,
            Func<int, double> getCmdPos = null,
            Func<int, double> getActVel = null,
            Func<int, int>    getMotionId = null,
            Func<int>         getSequenceId = null)
        {
            _getTorque      = getTorque ?? throw new ArgumentNullException(nameof(getTorque));
            _getAxisOp      = getAxisOp;
            _getActPos      = getActPos;
            _getCmdPos      = getCmdPos;
            _getActVel      = getActVel;
            _getMotionId    = getMotionId;
            _getSequenceId  = getSequenceId;
            _axes           = axes ?? new int[0];
            _log            = log ?? (_ => { });
            _hasAxisColumns = (getAxisOp != null);   // 제어기 연결 여부

            int n    = _axes.Length;
            _tBuf    = new double[RING * n];
            _opBuf   = new string[RING * n];
            _posBuf  = new double[RING * n];
            _cmdBuf  = new double[RING * n];
            _velBuf  = new double[RING * n];
            _midBuf  = new int[RING * n];
            _seqBuf  = new int[RING];
            _tTick   = new long[RING];

            // 초기값: "Pos" (intern으로 GC 부담 최소화)
            for (int i = 0; i < _opBuf.Length; i++)
                _opBuf[i] = string.Intern("Pos");
        }

        // ── 시작 ──────────────────────────────────────────────────
        public bool Start(string dir, string baseName)
        {
            if (_running) return false;
            try
            {
                Directory.CreateDirectory(dir);
                _dir          = dir;
                _baseNameRoot = baseName;
                _rotationIndex = 0;
                _filePath = BuildFilePath(_rotationIndex);

                _writer = new StreamWriter(
                    _filePath, false, Encoding.UTF8, bufferSize: 65536);
                _writer.WriteLine(BuildHeader());
                _fileOpenSw = Stopwatch.StartNew();

                _sw = Stopwatch.StartNew();

                _cts      = new CancellationTokenSource();
                _pollTask = Task.Run(() => TorqueOpPollLoop(_cts.Token), _cts.Token);

                _running = true;
                return true;
            }
            catch (Exception ex)
            {
                _log($"[CombinedLog] 시작 실패: {ex.Message}");
                _writer?.Dispose();
                _writer = null;
                return false;
            }
        }

        private string BuildFilePath(int rotationIndex)
        {
            string suffix = rotationIndex > 0 ? $"_part{rotationIndex:D3}" : "";
            return Path.Combine(_dir, _baseNameRoot + "_Combined" + suffix + ".csv");
        }

        // 헤더: 가속도(단일 센서 x/y/z) | 토크(축별) | [Op | ActPos | CmdPos | ActVel | MotionID](축별, 제어기 연결 시만) | SequenceID
        private string BuildHeader()
        {
            var hdr = new StringBuilder("time_s,x,y,z");
            foreach (int ax in _axes) hdr.Append($",Ax{ax}_Trq(%)");
            if (_hasAxisColumns)
            {
                foreach (int ax in _axes) hdr.Append($",Op_Ax{ax}");
                foreach (int ax in _axes) hdr.Append($",Ax{ax}_ActPos");
                foreach (int ax in _axes) hdr.Append($",Ax{ax}_CmdPos");
                foreach (int ax in _axes) hdr.Append($",Ax{ax}_ActVel");
                foreach (int ax in _axes) hdr.Append($",Ax{ax}_MotionID");
            }
            hdr.Append(",SequenceID");
            return hdr.ToString();
        }

        // 시간(1시간) 또는 용량(500MB) 중 먼저 도달하는 기준으로 새 파일을 엽니다.
        // 호출 전제: _writerLock을 이미 보유한 상태에서만 호출.
        private void RotateIfNeeded()
        {
            if (_writer == null || _fileOpenSw == null) return;

            bool timeExceeded = _fileOpenSw.Elapsed.TotalHours >= RotateMaxHours;
            long lenBytes = 0;
            try { lenBytes = _writer.BaseStream.Length; } catch { }
            bool sizeExceeded = lenBytes >= RotateMaxBytes;

            if (!timeExceeded && !sizeExceeded) return;

            try { _writer.Flush(); } catch { }
            try { _writer.Dispose(); } catch { }

            _rotationIndex++;
            string newPath = BuildFilePath(_rotationIndex);
            try
            {
                _writer = new StreamWriter(newPath, false, Encoding.UTF8, bufferSize: 65536);
                _writer.WriteLine(BuildHeader());
                _filePath = newPath;
                _fileOpenSw = Stopwatch.StartNew();
                _log($"[CombinedLog] 파일 회전({(timeExceeded ? "1시간" : "500MB")} 도달) → {newPath}");
            }
            catch (Exception ex)
            {
                _log($"[CombinedLog] 회전 후 파일 생성 실패: {ex.Message}");
                _writer = null;
            }
        }

        // ── 정지 ──────────────────────────────────────────────────
        public void Stop()
        {
            if (!_running) return;
            _running = false;

            try { _cts?.Cancel(); _pollTask?.Wait(2000); }
            catch { }
            finally { _cts?.Dispose(); _cts = null; _pollTask = null; }

            lock (_writerLock)
            {
                try { _writer?.Flush(); } catch { }
                try { _writer?.Dispose(); } catch { }
                _writer = null;
            }
        }

        // ── 가속도 블록 처리 (DAQ BlockReceived에서 호출) ──────────
        /// <summary>
        /// 단일 가속도 센서의 블록 N개 샘플 각각에 대해
        /// 시간적으로 가장 가까운 토크/Op/위치/속도/모션 값을 링 버퍼에서 찾아 한 행씩 기록합니다.
        /// </summary>
        public void ProcessAccelBlock(int modIdx, double[,] block, int n, double sampleRate)
        {
            if (!_running || _writer == null || n <= 0) return;
            if (sampleRate <= 0) sampleRate = 1000.0;

            // 블록 도착 시각 (공유 Stopwatch 기준)
            long blockArrivalTick = _sw.ElapsedTicks;
            double ticksPerSample = (double)Stopwatch.Frequency / sampleRate;

            // 링 버퍼 현재 크기 스냅샷
            long tTotal;
            lock (_tLock) { tTotal = _tTotal; }
            int tAvail = (int)Math.Min(tTotal, RING);
            if (tAvail == 0)
            {
                // 링 버퍼 미준비: 현재 상태로 채움
                WriteBlockNoRingData(block, n, sampleRate);
                return;
            }

            int nAxes = _axes.Length;
            long baseCount = _accelTotal;

            // ── 투 포인터 매핑: Accel 샘플(오름차순) ↔ 링 버퍼(오름차순) ──
            // 블록 첫 샘플 타임스탬프 = 도착 시각 - (n-1) * ticksPerSample
            // 링에서 해당 시각에 가장 가까운 폴링 위치부터 시작
            long riStart = Math.Max(0, tTotal - tAvail);
            long ri = riStart;

            lock (_writerLock)
            {
                if (_writer == null) return;
                try
                {
                    var sb = new StringBuilder(192);

                    for (int i = 0; i < n; i++)
                    {
                        // 이 샘플의 추정 타임스탬프 (Stopwatch 틱)
                        long sampleTick = blockArrivalTick - (long)((n - 1 - i) * ticksPerSample);

                        // ri 전진: 다음 항목이 sampleTick에 더 가까우면 전진
                        while (ri + 1 < tTotal)
                        {
                            int ci = (int)(ri       % RING);
                            int ni = (int)((ri + 1) % RING);
                            long dCurr = Abs64(_tTick[ci] - sampleTick);
                            long dNext = Abs64(_tTick[ni] - sampleTick);
                            if (dNext <= dCurr) ri++;
                            else break;
                        }

                        int rIdx  = (int)(ri % RING);
                        int rBase = rIdx * nAxes;

                        // time_s
                        double t_s = (baseCount + i) / sampleRate;
                        sb.Clear();
                        sb.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                        // 가속도: 단일 센서 x, y, z
                        sb.Append(",").Append(block[0, i].ToString("G6", CultureInfo.InvariantCulture));
                        sb.Append(",").Append(block[1, i].ToString("G6", CultureInfo.InvariantCulture));
                        sb.Append(",").Append(block[2, i].ToString("G6", CultureInfo.InvariantCulture));

                        // 토크: 축별 (링 버퍼 → 해당 시점의 값)
                        for (int ai = 0; ai < nAxes; ai++)
                            sb.Append(",").Append(_tBuf[rBase + ai].ToString("F4", CultureInfo.InvariantCulture));

                        if (_hasAxisColumns)
                        {
                            // Op: 축별 (링 버퍼 → 해당 시점의 상태)
                            for (int ai = 0; ai < nAxes; ai++)
                                sb.Append(",").Append(_opBuf[rBase + ai] ?? "Pos");
                            // ActPos(mm)
                            for (int ai = 0; ai < nAxes; ai++)
                                sb.Append(",").Append(_posBuf[rBase + ai].ToString("F4", CultureInfo.InvariantCulture));
                            // CmdPos(mm)
                            for (int ai = 0; ai < nAxes; ai++)
                                sb.Append(",").Append(_cmdBuf[rBase + ai].ToString("F4", CultureInfo.InvariantCulture));
                            // ActVel(mm/s)
                            for (int ai = 0; ai < nAxes; ai++)
                                sb.Append(",").Append(_velBuf[rBase + ai].ToString("F4", CultureInfo.InvariantCulture));
                            // MotionID
                            for (int ai = 0; ai < nAxes; ai++)
                                sb.Append(",").Append(_midBuf[rBase + ai].ToString(CultureInfo.InvariantCulture));
                        }

                        // SequenceID (축 무관, 링 슬롯 1개)
                        sb.Append(",").Append(_seqBuf[rIdx].ToString(CultureInfo.InvariantCulture));

                        _writer.WriteLine(sb.ToString());
                    }

                    _writer.Flush();   // 비정상 종료 대비 — DAQ 블록 주기(수백ms)마다 1회이므로 오버헤드 작음
                    RotateIfNeeded();
                }
                catch (Exception ex)
                {
                    // 예: CSV를 Excel 등이 잠근 상태에서 쓰기 실패 — 이 블록만 유실시키고 계속 진행
                    _log($"[CombinedLog] 쓰기 오류(이 블록 스킵): {ex.Message}");
                }
            }

            _accelTotal += n;
        }

        // 링 버퍼 미준비(수집 시작 직후 첫 블록) 시 현재 상태로 채운 행 기록
        private void WriteBlockNoRingData(double[,] block, int n, double sampleRate)
        {
            int nAxes = _axes.Length;
            long baseCount = _accelTotal;

            // 현재 상태 스냅샷 (링 버퍼 없으므로 현재값 사용 — 보간하지 않음)
            string[] opNow  = new string[nAxes];
            double[] posNow = new double[nAxes];
            double[] cmdNow = new double[nAxes];
            double[] velNow = new double[nAxes];
            int[]    midNow = new int[nAxes];
            for (int ai = 0; ai < nAxes; ai++)
            {
                opNow[ai]  = SafeGetOp(_getAxisOp, _axes[ai]);
                posNow[ai] = SafeGet(_getActPos, _axes[ai]);
                cmdNow[ai] = SafeGet(_getCmdPos, _axes[ai]);
                velNow[ai] = SafeGet(_getActVel, _axes[ai]);
                midNow[ai] = SafeGetInt(_getMotionId, _axes[ai]);
            }
            int seqNow = SafeGetInt(_getSequenceId);

            lock (_writerLock)
            {
                if (_writer == null) return;
                try
                {
                    var sb = new StringBuilder(192);

                    for (int i = 0; i < n; i++)
                    {
                        double t_s = (baseCount + i) / sampleRate;
                        sb.Clear();
                        sb.Append(t_s.ToString("F6", CultureInfo.InvariantCulture));

                        // 가속도: 단일 센서 x, y, z
                        sb.Append(",").Append(block[0, i].ToString("G6", CultureInfo.InvariantCulture));
                        sb.Append(",").Append(block[1, i].ToString("G6", CultureInfo.InvariantCulture));
                        sb.Append(",").Append(block[2, i].ToString("G6", CultureInfo.InvariantCulture));

                        // 토크: 0 채움 (링 버퍼 미준비 — 기존 동작 유지)
                        for (int ai = 0; ai < nAxes; ai++) sb.Append(",0");

                        if (_hasAxisColumns)
                        {
                            for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(opNow[ai]);
                            for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(posNow[ai].ToString("F4", CultureInfo.InvariantCulture));
                            for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(cmdNow[ai].ToString("F4", CultureInfo.InvariantCulture));
                            for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(velNow[ai].ToString("F4", CultureInfo.InvariantCulture));
                            for (int ai = 0; ai < nAxes; ai++) sb.Append(",").Append(midNow[ai].ToString(CultureInfo.InvariantCulture));
                        }

                        sb.Append(",").Append(seqNow.ToString(CultureInfo.InvariantCulture));

                        _writer.WriteLine(sb.ToString());
                    }

                    _writer.Flush();
                    RotateIfNeeded();
                }
                catch (Exception ex)
                {
                    _log($"[CombinedLog] 쓰기 오류(이 블록 스킵): {ex.Message}");
                }
            }
            _accelTotal += n;
        }

        // ── 토크 + Op + 위치/속도/모션 1ms 폴링 루프 ──────────────
        private void TorqueOpPollLoop(CancellationToken token)
        {
            int nAxes = _axes.Length;
            long ticksPerMs = Stopwatch.Frequency / 1000;
            long nextTick   = _sw.ElapsedTicks + ticksPerMs;

            Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
            timeBeginPeriod(1);
            try
            {
                while (!token.IsCancellationRequested)
                {
                    long tNow;
                    int  wi;
                    double[] torqueSnap = null;
                    lock (_tLock)
                    {
                        wi   = (int)(_tTotal % RING);
                        tNow = _sw.ElapsedTicks;
                        _tTick[wi] = tNow;
                        if (TorqueSampled != null) torqueSnap = new double[nAxes];
                        int wBase = wi * nAxes;
                        for (int ai = 0; ai < nAxes; ai++)
                        {
                            double v = SafeGet(_getTorque, _axes[ai]);
                            _tBuf[wBase + ai] = v;
                            if (torqueSnap != null) torqueSnap[ai] = v;

                            // Op 상태도 동일 링 슬롯에 저장 (intern으로 중복 할당 방지)
                            _opBuf[wBase + ai] = string.Intern(SafeGetOp(_getAxisOp, _axes[ai]));

                            // 위치/속도/모션 정보도 동일 링 슬롯에 저장 (Op와 동일한 정합 방식)
                            _posBuf[wBase + ai] = SafeGet(_getActPos, _axes[ai]);
                            _cmdBuf[wBase + ai] = SafeGet(_getCmdPos, _axes[ai]);
                            _velBuf[wBase + ai] = SafeGet(_getActVel, _axes[ai]);
                            _midBuf[wBase + ai] = SafeGetInt(_getMotionId, _axes[ai]);
                        }
                        _seqBuf[wi] = SafeGetInt(_getSequenceId);
                        _tTotal++;
                    }

                    // InfluxDB 등 외부 콜백 (lock 밖에서 호출)
                    if (torqueSnap != null)
                    {
                        var cb     = TorqueSampled;
                        var utcNow = DateTime.UtcNow;
                        if (cb != null)
                            for (int ai = 0; ai < nAxes; ai++)
                                try { cb(_axes[ai], torqueSnap[ai], utcNow); } catch { }
                    }

                    long rem;
                    while ((rem = nextTick - _sw.ElapsedTicks) > 0)
                    {
                        if (rem > ticksPerMs / 2) Thread.Sleep(1);
                    }
                    nextTick += ticksPerMs;
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex) { _log($"[CombinedLog] 폴링 오류: {ex.Message}"); }
            finally { timeEndPeriod(1); }
        }

        private static long Abs64(long v) => v < 0 ? -v : v;

        private static double SafeGet(Func<int, double> fn, int ax)
        {
            try { return fn(ax); }
            catch { return double.NaN; }
        }

        private static string SafeGetOp(Func<int, string> fn, int ax)
        {
            try { return fn?.Invoke(ax) ?? "Pos"; }
            catch { return "Pos"; }
        }

        /// <summary>MotionID 폴링. 콜백 없음/예외 시 0("모션 없음/미배정") — 결측을 위장하지 않는 명확한 값.</summary>
        private static int SafeGetInt(Func<int, int> fn, int ax)
        {
            try { return fn != null ? fn(ax) : 0; }
            catch { return 0; }
        }

        /// <summary>SequenceID 폴링. 콜백 없음/예외 시 0("시퀀스 컨텍스트 없음").</summary>
        private static int SafeGetInt(Func<int> fn)
        {
            try { return fn != null ? fn() : 0; }
            catch { return 0; }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Stop();
            _disposed = true;
        }
    }
}
