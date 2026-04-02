using NationalInstruments.DAQmx;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO.Compression;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

using DaqTask = NationalInstruments.DAQmx.Task;
using SysTask = System.Threading.Tasks.Task;

namespace PHM_Project_DockPanel.Services.DAQ
{
    public class DaqAccelHttpSender : IDisposable
    {
        // ===== 추가 필드 =====
        private string _currentSessionId;          // null이면 idle
        private string _currentRobotId;
        private int[] _currentAxes;
        private readonly object _sessionLock = new object();
        public bool IsRunning => _running;

        public event Action<string> FrameStatusUpdated;

        // ===== 공개 제어 메서드 =====
        public bool StartStreaming() => Start();
        public void StopStreaming() => Stop();

        private bool _isSessionActive = false;

        // ===== 세션 시작/종료 =====
        public void BeginSession(string sessionId, string robotId, int[] axes)
        {
            lock (_sessionLock)
            {
                if (_isSessionActive) return;             // 이미 세션 중이면 무시
                _isSessionActive = true;
                _currentSessionId = sessionId;
                _currentRobotId = robotId;
                _currentAxes = axes != null ? (int[])axes.Clone() : null;
            }
            //_log($"[SESSION] Begin {sessionId} ({robotId})");
        }

        public void EndSession()
        {
            bool wasActive;
            lock (_sessionLock)
            {
                wasActive = _isSessionActive;
                _isSessionActive = false;
                _currentSessionId = null;
                _currentRobotId = null;
                _currentAxes = null;
            }
            //if (wasActive) _log("[SESSION] End current session");
        }

        private (bool active, string sid, string rid, int[] axes) SnapshotSession()
        {
            lock (_sessionLock)
            {
                return (_isSessionActive, _currentSessionId, _currentRobotId,
                        _currentAxes != null ? (int[])_currentAxes.Clone() : null);
            }
        }

        // ===== 세션 필드 적용 유틸 =====
        private void FillSessionTags(Dictionary<string, object> payload)
        {
            string sid, rid; int[] axes;
            lock (_sessionLock)
            {
                sid = _currentSessionId;
                rid = _currentRobotId;
                axes = _currentAxes;
            }

            if (!string.IsNullOrEmpty(sid))
            {
                payload["session_id"] = sid;
                if (!string.IsNullOrEmpty(rid)) payload["robot_id"] = rid;
                if (axes != null) payload["axes"] = axes;
                payload["phase"] = "run";
            }
            else
            {
                payload["phase"] = "idle";
            }
        }

        // ===== 사용자/환경 설정 =====
        public string RealtimeDeviceId { get; set; } = "realtime";
        public string DeviceId { get; set; } = "realtime";
        public string[] Modules { get; set; } = new[] { "cDAQ1Mod1" };
        public string ChannelTriplet { get; set; } = "ai0:2";
        public double SampleRate { get; set; } = 1280;  // FS
        public int FrameSamples { get; set; } = 64;     // FRAME_N
        public double IepeMilliAmps { get; set; } = 4.0; // mA
        public double MinG { get; set; } = -25.0;
        public double MaxG { get; set; } = 25.0;

        public string ServerUrl { get; set; } = "http://10.100.17.221:8000/api/ingest";
        public TimeSpan HttpTimeout { get; set; } = TimeSpan.FromSeconds(2.0);
        public int NumWorkers { get; set; } = 6;
        public int QueueCapacity { get; set; } = 200;    // 드롭-올드 전략

        private readonly Action<string> _log;

        private class AxisSens { public double X; public double Y; public double Z; }
        private class AxisOffset { public double X; public double Y; public double Z; }
        private readonly Dictionary<string, AxisSens> _sensByModule = new Dictionary<string, AxisSens>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, AxisOffset> _offsetByModule = new Dictionary<string, AxisOffset>(StringComparer.OrdinalIgnoreCase);
        private const double DEFAULT_SENS_MVPG = 100.0;

        // ===== 런타임 상태 =====
        private DaqTask _aiTask;
        private AnalogMultiChannelReader _reader;
        private AsyncCallback _cb;
        private volatile bool _running;

        private long _frameSeq = 0;

        private BlockingCollection<Dictionary<string, object>> _q;
        private readonly System.Collections.Generic.List<SysTask> _workers = new System.Collections.Generic.List<SysTask>();

        // 콜백 동시성 추적
        private int _inCallback = 0;
        private readonly object _stopLock = new object();
        private bool _stopping = false;

        private readonly HttpClient _http;

        public DaqAccelHttpSender(Action<string> logAction = null)
        {
            _log = logAction ?? delegate { };
            _q = new BlockingCollection<Dictionary<string, object>>(QueueCapacity);

            if (ServicePointManager.DefaultConnectionLimit < 64)
                ServicePointManager.DefaultConnectionLimit = 64;

            var handler = new HttpClientHandler
            {
                AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate
            };
            _http = new HttpClient(handler);
            _http.Timeout = HttpTimeout;
            _http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }

        // ===== 민감도/오프셋 =====
        public void SetModuleSensitivity(string module, double x_mVpg, double y_mVpg, double z_mVpg)
        {
            if (string.IsNullOrWhiteSpace(module)) return;
            _sensByModule[module] = new AxisSens { X = x_mVpg, Y = y_mVpg, Z = z_mVpg };
        }
        public void SetModuleOffset(string module, double x_g, double y_g, double z_g)
        {
            if (string.IsNullOrWhiteSpace(module)) return;
            _offsetByModule[module] = new AxisOffset { X = x_g, Y = y_g, Z = z_g };
        }

        public void LoadSensitivityCsv(string path)
        {
            try
            {
                if (!File.Exists(path))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
                    File.WriteAllText(path,
                        @"module,x_mVpg,y_mVpg,z_mVpg
                        cDAQ1Mod1,1005,993,965
                        ");
                    _log("민감도 CSV가 없어 샘플을 생성했습니다: " + path);
                    return;
                }
                int count = 0;
                foreach (var line in File.ReadAllLines(path))
                {
                    var ln = line.Trim();
                    if (ln.Length == 0 || ln.StartsWith("#") ||
                        ln.StartsWith("module", StringComparison.OrdinalIgnoreCase)) continue;

                    var toks = ln.Split(',');
                    if (toks.Length < 4) continue;
                    string mod = toks[0].Trim();
                    double sx, sy, sz;
                    if (!double.TryParse(toks[1], out sx)) continue;
                    if (!double.TryParse(toks[2], out sy)) continue;
                    if (!double.TryParse(toks[3], out sz)) continue;

                    SetModuleSensitivity(mod, sx, sy, sz);
                    count++;
                }
                _log("민감도 CSV 로드 완료: " + path + " (모듈 " + count + "개)");
            }
            catch (Exception ex)
            {
                _log("[민감도 CSV 로드 오류] " + ex.Message + " → 기본값 사용");
            }
        }

        public void LoadOffsetCsv(string path)
        {
            try
            {
                if (!File.Exists(path))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
                    File.WriteAllText(path,
                        @"module,x_g,y_g,z_g
                        cDAQ1Mod1,0,0,0
                        ");
                    _log("오프셋 CSV가 없어 샘플을 생성했습니다: " + path);
                    return;
                }
                int count = 0;
                foreach (var line in File.ReadAllLines(path))
                {
                    var ln = line.Trim();
                    if (ln.Length == 0 || ln.StartsWith("#") ||
                        ln.StartsWith("module", StringComparison.OrdinalIgnoreCase)) continue;

                    var toks = ln.Split(',');
                    if (toks.Length < 4) continue;
                    string mod = toks[0].Trim();
                    double ox, oy, oz;
                    if (!double.TryParse(toks[1], out ox)) continue;
                    if (!double.TryParse(toks[2], out oy)) continue;
                    if (!double.TryParse(toks[3], out oz)) continue;

                    SetModuleOffset(mod, ox, oy, oz);
                    count++;
                }
                _log("오프셋 CSV 로드 완료: " + path + " (모듈 " + count + "개)");
            }
            catch (Exception ex)
            {
                _log("[오프셋 CSV 로드 오류] " + ex.Message + " → 0,0,0 사용");
            }
        }

        // ===== 수집 시작/종료 =====
        public bool Start()
        {
            if (_running) { _log("[주의] Start 요청 전에 이전 세션을 강제 종료합니다."); SafeStop(); }

            try
            {
                if (_q == null || _q.IsAddingCompleted)
                    _q = new BlockingCollection<Dictionary<string, object>>(QueueCapacity);

                if (Modules == null || Modules.Length == 0)
                    throw new InvalidOperationException("Modules가 비어 있습니다.");
                if (string.IsNullOrWhiteSpace(ChannelTriplet))
                    throw new InvalidOperationException("ChannelTriplet(ai0:2 등)이 비어 있습니다.");

                // ★ 새 세션 시작
                _frameSeq = 0;

                _aiTask = new DaqTask("cDAQ_AI_HTTP");

                // 모듈별 3축 채널 생성
                for (int i = 0; i < Modules.Length; i++)
                {
                    string mod = Modules[i];
                    AxisSens s = GetAxisSens(mod);
                    CreateAccel(mod + "/ai0", s.X);
                    CreateAccel(mod + "/ai1", s.Y);
                    CreateAccel(mod + "/ai2", s.Z);
                }

                // 버퍼: 샘플레이트 * 5초
                int inBuffer = (int)Math.Round(SampleRate * 5);
                _aiTask.Timing.ConfigureSampleClock(
                    "",
                    SampleRate,
                    SampleClockActiveEdge.Rising,
                    SampleQuantityMode.ContinuousSamples,
                    inBuffer
                );
                _aiTask.Stream.Timeout = 10000;

                _reader = new AnalogMultiChannelReader(_aiTask.Stream);
                _reader.SynchronizeCallbacks = false;
                _cb = new AsyncCallback(ReadCallback);

                // 워커 기동
                for (int i = 0; i < NumWorkers; i++)
                    _workers.Add(SysTask.Run(() => SenderWorker(_q, _http, ServerUrl, _log)));

                _aiTask.Start();
                _running = true;

                _reader.BeginReadMultiSample(Math.Max(FrameSamples, 1), _cb, null);

                _log(string.Format("[INFO] DAQ streaming start (Accel): fs={0}, frame={1}, modules=[{2}]",
                    SampleRate, FrameSamples, string.Join(",", Modules)));
                return true;
            }
            catch (Exception ex)
            {
                _log("[오류] " + ex.Message);
                SafeStop();
                return false;
            }
        }

        public void Stop() => SafeStop();

        private void SafeStop()
        {
            lock (_stopLock)
            {
                if (_stopping) return;
                _stopping = true;
            }

            try
            {
                bool wasRunning = _running;
                _running = false;

                _cb = null;
                _reader = null;

                try { if (_aiTask != null) _aiTask.Control(TaskAction.Abort); }
                catch { try { if (_aiTask != null) _aiTask.Stop(); } catch { } }
                finally { try { if (_aiTask != null) _aiTask.Dispose(); } catch { } _aiTask = null; }

                // 콜백 빠질 때까지 잠깐 대기
                var sw = System.Diagnostics.Stopwatch.StartNew();
                while (Interlocked.CompareExchange(ref _inCallback, 0, 0) != 0 && sw.ElapsedMilliseconds < 1000)
                    Thread.Sleep(5);

                try { _q?.CompleteAdding(); } catch { }

                if (wasRunning) _log("[STOP] 수집 종료");
            }
            finally
            {
                try { SysTask.WaitAll(_workers.ToArray(), 1000); } catch { }
                _workers.Clear();
                lock (_stopLock) { _stopping = false; }
            }
        }

        public void Dispose()
        {
            SafeStop();
            if (_http != null) _http.Dispose();
            if (_q != null) _q.Dispose();
        }

        private void CreateAccel(string phys, double sens_mVpg)
        {
            _aiTask.AIChannels.CreateAccelerometerChannel(
                phys,
                "",
                AITerminalConfiguration.Pseudodifferential,
                MinG,
                MaxG,
                sens_mVpg,
                AIAccelerometerSensitivityUnits.MillivoltsPerG,
                AIExcitationSource.Internal,
                IepeMilliAmps / 1000.0,
                AIAccelerationUnits.G
            );
        }

        private AxisSens GetAxisSens(string module)
        {
            AxisSens s;
            if (string.IsNullOrWhiteSpace(module)) return new AxisSens { X = DEFAULT_SENS_MVPG, Y = DEFAULT_SENS_MVPG, Z = DEFAULT_SENS_MVPG };
            if (_sensByModule.TryGetValue(module, out s)) return s;
            return new AxisSens { X = DEFAULT_SENS_MVPG, Y = DEFAULT_SENS_MVPG, Z = DEFAULT_SENS_MVPG };
        }

        private AxisOffset GetAxisOffset(string module)
        {
            AxisOffset o;
            if (string.IsNullOrWhiteSpace(module)) return new AxisOffset { X = 0, Y = 0, Z = 0 };
            if (_offsetByModule.TryGetValue(module, out o)) return o;
            return new AxisOffset { X = 0, Y = 0, Z = 0 };
        }

        private void ReadCallback(IAsyncResult ar)
        {
            Interlocked.Increment(ref _inCallback);
            try
            {
                if (!_running || _reader == null) return;

                double[,] block = _reader.EndReadMultiSample(ar);
                int n = block.GetLength(1);

                // n이 FrameSamples의 배수가 아닐 수 있음 → 완전 프레임만 처리
                int frames = n / FrameSamples;
                if (frames <= 0) goto _scheduleNext;

                for (int f = 0; f < frames; f++)
                {
                    if (!_running || _q == null || _q.IsAddingCompleted) break;

                    int start = f * FrameSamples;
                    int count = FrameSamples;

                    double[] gx = new double[count];
                    double[] gy = new double[count];
                    double[] gz = new double[count];

                    AxisOffset off = GetAxisOffset(Modules[0]);

                    for (int i = 0; i < count; i++)
                    {
                        gx[i] = block[0, start + i] - off.X;
                        gy[i] = block[1, start + i] - off.Y;
                        gz[i] = block[2, start + i] - off.Z;
                    }

                    // ★ 프레임 단위 증가
                    long seq = Interlocked.Increment(ref _frameSeq);

                    var (active, sid, rid, axes) = SnapshotSession();

                    // 4-1) 항상: realtime 채널로 1건
                    var realtimePayload = new Dictionary<string, object>
                    {
                        ["device_id"] = RealtimeDeviceId,            // "realtime"
                        ["fs"] = SampleRate,
                        ["t0_utc"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffK"),
                        ["seq"] = seq,
                        ["ax"] = gx,
                        ["ay"] = gy,
                        ["az"] = gz,
                        ["phase"] = active ? "run" : "idle"          // 참고 정보로만 사용 (폴더는 device_id로 분기)
                    };
                    TryEnqueue(realtimePayload);

                    // 4-2) 세션 활성일 때만: RBxx 채널로 1건 더
                    if (active && !string.IsNullOrEmpty(rid))
                    {
                        var sessionPayload = new Dictionary<string, object>
                        {
                            ["device_id"] = rid,                     // "RB01" 같은 로봇 식별자
                            ["fs"] = SampleRate,
                            ["t0_utc"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffK"),
                            ["seq"] = seq,                           // 같은 프레임 seq 재사용(채널은 다름)
                            ["ax"] = gx,
                            ["ay"] = gy,
                            ["az"] = gz,
                            ["session_id"] = sid,
                            ["axes"] = axes,
                            ["phase"] = "run"
                        };
                        TryEnqueue(sessionPayload);
                    }

                    // fps마다 라벨 갱신 (그대로)
                    int fps = Math.Max(1, (int)(SampleRate / FrameSamples));
                    if (seq % fps == 0)
                    {
                        string msg = $"[OK] seq={seq}, samples/chan={count}";
                        FrameStatusUpdated?.Invoke(msg);
                    }
                }

            _scheduleNext:
                if (_running && _reader != null && _cb != null)
                    _reader.BeginReadMultiSample(Math.Max(FrameSamples, 1), _cb, null);
            }
            catch (DaqException) { }
            catch (ObjectDisposedException) { }
            catch (Exception ex)
            {
                _log("[ReadCallback 오류] " + ex.Message);
            }
            finally
            {
                Interlocked.Decrement(ref _inCallback);
            }
        }

        private void TryEnqueue(Dictionary<string, object> payload)
        {
            if (_q == null || _q.IsAddingCompleted) return;
            try
            {
                if (!_q.TryAdd(payload))
                {
                    _q.TryTake(out _);
                    _q.TryAdd(payload);
                }
            }
            catch (InvalidOperationException)
            {
                // CompleteAdding 이후: 무시
            }
        }

        private static async SysTask SenderWorker(
            BlockingCollection<Dictionary<string, object>> q,
            HttpClient http,
            string serverUrl,
            Action<string> log)
        {
            Directory.CreateDirectory("spool");

            foreach (var payload in q.GetConsumingEnumerable())
            {
                try
                {
                    string json = JsonConvert.SerializeObject(payload);

                    using (var ms = new MemoryStream())
                    {
                        using (var gz = new GZipStream(ms, CompressionLevel.Fastest, true))
                        using (var sw = new StreamWriter(gz, new UTF8Encoding(false)))
                        {
                            sw.Write(json);
                        }
                        ms.Position = 0;

                        using (var req = new HttpRequestMessage(HttpMethod.Post, serverUrl))
                        {
                            req.Content = new StreamContent(ms);
                            req.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
                            req.Content.Headers.ContentEncoding.Add("gzip");

                            const int maxRetry = 3;
                            int attempt = 0;
                            bool ok = false;

                            while (attempt < maxRetry && !ok)
                            {
                                attempt++;
                                try
                                {
                                    using (var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead).ConfigureAwait(false))
                                    {
                                        if ((int)resp.StatusCode >= 200 && (int)resp.StatusCode < 300)
                                            ok = true;
                                    }
                                }
                                catch { }

                                if (!ok && attempt < maxRetry)
                                    await SysTask.Delay(TimeSpan.FromMilliseconds(200 * attempt));
                            }

                            if (!ok)
                            {
                                string t0 = payload.ContainsKey("t0_utc") && payload["t0_utc"] != null ? payload["t0_utc"].ToString() : "";
                                string seq = payload.ContainsKey("seq") && payload["seq"] != null ? payload["seq"].ToString() : "0";
                                string safe = SafeFilename(t0, seq);
                                string path = Path.Combine("spool", safe);
                                File.WriteAllText(path, json, new UTF8Encoding(false));
                                log("[SPOOL] " + path);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    try
                    {
                        string json = JsonConvert.SerializeObject(payload);
                        string t0 = payload.ContainsKey("t0_utc") && payload["t0_utc"] != null ? payload["t0_utc"].ToString() : "";
                        string seq = payload.ContainsKey("seq") && payload["seq"] != null ? payload["seq"].ToString() : "0";
                        string safe = SafeFilename(t0, seq);
                        string path = Path.Combine("spool", safe);
                        File.WriteAllText(path, json, new UTF8Encoding(false));
                        log("[SPOOL-EX] " + path + " (" + ex.GetType().Name + ")");
                    }
                    catch { }
                }
            }
        }

        private static string SafeFilename(string t0Utc, string seq)
        {
            string safe = (t0Utc ?? "").Replace(":", "-").Replace("+", "").Replace("T", "_");
            int dot = safe.IndexOf('.');
            if (dot >= 0) safe = safe.Substring(0, dot);
            return safe + "_" + seq + "_" + Guid.NewGuid().ToString("N") + ".json";
        }

        private static double Mean(double[] a)
        {
            if (a == null || a.Length == 0) return 0;
            double s = 0;
            for (int i = 0; i < a.Length; i++) s += a[i];
            return s / a.Length;
        }

        private static double Rms(double[] a)
        {
            if (a == null || a.Length == 0) return 0;
            double s = 0;
            for (int i = 0; i < a.Length; i++) s += a[i] * a[i];
            return Math.Sqrt(s / a.Length);
        }
    }
}
