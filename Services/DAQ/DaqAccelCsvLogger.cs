using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NationalInstruments.DAQmx;

using DaqTask = NationalInstruments.DAQmx.Task;
using SysTask = System.Threading.Tasks.Task;

namespace PHM_Project_DockPanel.Services.DAQ
{
    public class DaqAccelCsvLogger : IDisposable
    {
        private readonly Action<string> _logAction;

        public DaqAccelCsvLogger(Action<string> logAction = null)
        {
            _logAction = logAction;
        }

        // ===== 사용자 환경 =====
        private string[] _modules = new[] { "cDAQ2Mod2" };
        private string _aiRange = "ai0:2";      // 각 모듈 3채널(X/Y/Z)
        private double _rate = 0;         // 로깅 샘플레이트
        private static double AccelRate => 1.0 / AppState.GetPeriodForColumn("x");
        private int _readBlock = 2048;       // 채널당 블록 크기
        private double _minG = -5, _maxG = 5; // 9230: PseudoDiff

        public string[] Modules { get { return _modules; } set { _modules = value ?? new string[0]; } }
        public string AiRange { get { return _aiRange; } set { _aiRange = value ?? "ai0:2"; } }
        public double SampleRate { get { return _rate; } set { _rate = value > 0 ? value : 0; } }
        public int ReadBlock { get { return _readBlock; } set { _readBlock = Math.Max(1, value); } }
        public double MinVoltage { get { return _minG; } set { _minG = value; } }
        public double MaxVoltage { get { return _maxG; } set { _maxG = value; } }

        // ===== 민감도/오프셋 (mV/g, g) =====
        private class AxisSens { public double X; public double Y; public double Z; }
        private class AxisOffset { public double X; public double Y; public double Z; }

        private readonly Dictionary<string, AxisSens> _sensByModule =
            new Dictionary<string, AxisSens>(StringComparer.OrdinalIgnoreCase)
            {
                { "cDAQ3Mod1", new AxisSens{ X = 100.0, Y = 100.0, Z = 100.0 } },
                { "cDAQ3Mod2", new AxisSens{ X = 100.0, Y = 100.0, Z = 100.0 } },
                { "cDAQ3Mod3", new AxisSens{ X = 100.0, Y = 100.0, Z = 100.0 } },
            };

        private readonly Dictionary<string, AxisOffset> _offsetByModule =
            new Dictionary<string, AxisOffset>(StringComparer.OrdinalIgnoreCase)
            {
                { "cDAQ3Mod1", new AxisOffset{ X = 0.0, Y = 0.0, Z = 0.0 } },
                { "cDAQ3Mod2", new AxisOffset{ X = 0.0, Y = 0.0, Z = 0.0 } },
                { "cDAQ3Mod3", new AxisOffset{ X = 0.0, Y = 0.0, Z = 0.0 } },
            };

        private const double DEFAULT_SENS_MVPG = 100.0;

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
                    Directory.CreateDirectory(Path.GetDirectoryName(path));
                    File.WriteAllText(path,
                    @"module,x_mVpg,y_mVpg,z_mVpg
                    cDAQ3Mod1,100,100,100
                    cDAQ3Mod2,100,100,100
                    cDAQ3Mod3,100,100,100
                    ");
                    _logAction?.Invoke("민감도 CSV가 없어 샘플을 생성했습니다: " + path);
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
                _logAction?.Invoke("민감도 CSV 로드 완료: " + path + " (모듈 " + count + "개)");
            }
            catch (Exception ex)
            {
                _logAction?.Invoke("[민감도 CSV 로드 오류] " + ex.Message + " → 기본값 사용");
            }
        }
        public void LoadOffsetCsv(string path)
        {
            try
            {
                if (!File.Exists(path))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(path));
                    File.WriteAllText(path,
                    @"module,x_g,y_g,z_g
                    cDAQ3Mod1,0,0,0
                    cDAQ3Mod2,0,0,0
                    cDAQ3Mod3,0,0,0
                    ");
                    _logAction?.Invoke("오프셋 CSV가 없어 샘플을 생성했습니다: " + path);
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
                _logAction?.Invoke("오프셋 CSV 로드 완료: " + path + " (모듈 " + count + "개)");
            }
            catch (Exception ex)
            {
                _logAction?.Invoke("[오프셋 CSV 로드 오류] " + ex.Message + " → 0,0,0 사용");
            }
        }

        // ===== 로깅 내부 상태 =====
        private DaqTask _aiTask;
        private AnalogMultiChannelReader _reader;
        private AsyncCallback _cb;
        private bool _running;
        private long _samples;

        private string[] _csvPathByMod;
        private FileStream[] _fsByMod;
        private StreamWriter[] _swByMod;
        private string[] _lastCsvPathByMod; // survives SafeStop() — readable after Stop()

        public bool IsRunning { get { return _running; } }
        public string[] CsvPathByModule { get { return _csvPathByMod; } }
        public string[] LastCsvPaths { get { return _lastCsvPathByMod; } }

        public bool Start(int[] axisIndices, string filePath, string filename, uint durationMs = 5000)
        {
            // ★ 이미 실행 중이면 먼저 종료
            if (_running)
            {
                _logAction?.Invoke("[주의] Start 요청 전에 이전 세션을 강제 종료합니다.");
                SafeStop();
            }

            try
            {
                if (_modules == null || _modules.Length == 0)
                    throw new InvalidOperationException("Modules가 비어 있습니다.");
                if (string.IsNullOrWhiteSpace(_aiRange))
                    throw new InvalidOperationException("AiRange가 비어 있습니다.");

                _aiTask = new DaqTask("cDAQ_AI");

                foreach (var mod in _modules)
                {
                    CreateAccel($"{mod}/ai0");
                    CreateAccel($"{mod}/ai1");
                    CreateAccel($"{mod}/ai2");
                }

                double rate = (_rate > 0) ? _rate : AccelRate;

                _aiTask.Timing.ConfigureSampleClock(
                    "",
                    rate,
                    SampleClockActiveEdge.Rising,
                    SampleQuantityMode.ContinuousSamples,
                    (int)rate * 10  // 버퍼 크기도 동일 기준
                );
                _aiTask.Stream.Timeout = 10000;

                // CSV 준비(모듈별 폴더)
                var dir = string.IsNullOrWhiteSpace(filePath) ? @"C:\PHM_Logs\Tests" : filePath;
                Directory.CreateDirectory(dir);
                var baseName = string.IsNullOrWhiteSpace(filename)
                    ? DateTime.Now.ToString("yyyyMMdd_HHmmss")
                    : Path.GetFileNameWithoutExtension(filename);

                int modCount = _modules.Length;
                _csvPathByMod = new string[modCount];
                _lastCsvPathByMod = new string[modCount];
                _fsByMod = new FileStream[modCount];
                _swByMod = new StreamWriter[modCount];

                for (int m = 0; m < modCount; m++)
                {
                    var moduleDir = Path.Combine(dir, _modules[m]);
                    Directory.CreateDirectory(moduleDir);
                    var fileName = baseName + "_Accel.csv";

                    _csvPathByMod[m] = Path.Combine(moduleDir, fileName);
                    _lastCsvPathByMod[m] = _csvPathByMod[m];
                    _fsByMod[m] = new FileStream(
                        _csvPathByMod[m],
                        FileMode.Create,
                        FileAccess.Write,
                        FileShare.Read,
                        bufferSize: 64 * 1024,
                        options: FileOptions.SequentialScan);
                    var bs = new BufferedStream(_fsByMod[m], 64 * 1024);
                    // UTF-8, 4KB 내부 버퍼
                    _swByMod[m] = new StreamWriter(bs, new UTF8Encoding(false), 4096) { AutoFlush = false };
                    // 헤더
                    _swByMod[m].WriteLine("time_s,x,y,z");
                }

                _reader = new AnalogMultiChannelReader(_aiTask.Stream)
                {
                    // UI 스레드로 콜백 안 넘김
                    SynchronizeCallbacks = false
                };
                _cb = new AsyncCallback(ReadCallback);

                _aiTask.Start();
                _samples = 0;


                _running = true;
                _reader.BeginReadMultiSample(_readBlock, _cb, null);

                //_logAction?.Invoke("[DAQ] Start → " + string.Join(",", _modules.Select(m => m + "/" + _aiRange)) + " @ " + _rate + " Hz");
                for (int m = 0; m < _modules.Length; m++)
                {
                    var s = GetAxisSens(_modules[m]);
                    var o = GetAxisOffset(_modules[m]);
                    //_logAction?.Invoke($"→ {_modules[m]} : CSV={_csvPathByMod[m]} | Sens(mV/g) X={s.X},Y={s.Y},Z={s.Z} | Offset(g) X={o.X},Y={o.Y},Z={o.Z}");
                }
                return true;
            }
            catch (Exception ex)
            {
                _logAction?.Invoke("[오류] " + ex.Message);
                SafeStop();
                return false;
            }
        }

        private void CreateAccel(string phys)
        {
            // NI 9230은 IEPE 내장이 없는 하드웨어 AC 결합 전압 입력 모듈.
            // CreateAccelerometerChannel + Internal 익사이테이션을 사용하면
            // DAQmx 드라이버가 AC 결합을 DC 모드로 재설정해
            // IEPE 바이어스(~24 V) 전체가 측정값에 포함된다 (~24 V × 1000 / 100 mV/g = 240 g).
            // CreateVoltageChannel을 사용하면 하드웨어 AC 결합이 유지되어 DC 바이어스가 차단되고,
            // g 변환은 ReadCallback에서 수동으로 수행한다: g = V × 1000 / sensitivity_mVpg
            _aiTask.AIChannels.CreateVoltageChannel(
                phys,
                "",
                AITerminalConfiguration.Pseudodifferential,
                _minG,   // 전압 최솟값(V) — MainForm 기본값 −5V → 측정 범위 ±50 g
                _maxG,   // 전압 최댓값(V) — MainForm 기본값  +5V
                AIVoltageUnits.Volts
            );
        }

        public void Stop() { SafeStop(); }

        private void SafeStop()
        {
            try
            {
                // 1) 콜백 재진입 즉시 차단
                bool wasRunning = _running;
                _running = false;

                // 콜백이 null/flag 보고 바로 return 하도록
                _cb = null;
                _reader = null;

                // 2) 태스크 즉시 중단 (Abort가 Stop보다 빠름)
                try
                {
                    _aiTask?.Control(TaskAction.Abort);
                }
                catch
                {
                    try { _aiTask?.Stop(); } catch { }
                }

                // 3) 태스크 해제
                try { _aiTask?.Dispose(); } catch { }
                _aiTask = null;

                //if (wasRunning)
                //    _logAction?.Invoke("[STOP] 로깅 종료");
            }
            finally
            {
                // 4) 파일/스트림 닫기 (여기서 flush/close)
                if (_swByMod != null)
                {
                    for (int m = 0; m < _swByMod.Length; m++)
                    {
                        try { _swByMod[m]?.Flush(); } catch { }
                        try { _swByMod[m]?.Dispose(); } catch { }
                        _swByMod[m] = null;

                        try { _fsByMod[m]?.Dispose(); } catch { }
                        _fsByMod[m] = null;
                    }
                }

                _csvPathByMod = null;
            }
        }

        // ===== Read 콜백(로깅) =====
        private void ReadCallback(IAsyncResult ar)
        {
            try
            {
                if (!_running || _reader == null) return;

                double[,] block = _reader.EndReadMultiSample(ar); // [채널][샘플]
                int ch = block.GetLength(0);
                int n = block.GetLength(1);
                int modulesInBuffer = Math.Min(_modules.Length, ch / 3);

                double rate = (_rate > 0) ? _rate : AccelRate;

                for (int i = 0; i < n; i++)
                {
                    double t = (_samples + i) / rate;

                    for (int m = 0; m < modulesInBuffer; m++)
                    {
                        int baseIdx = m * 3;

                        // CreateVoltageChannel이므로 block[] 값은 Volts (AC 결합, DC 차단)
                        // g 변환: g = V × 1000 / sensitivity_mVpg
                        var sens = GetAxisSens(_modules[m]);
                        double gx = block[baseIdx + 0, i] * 1000.0 / sens.X;
                        double gy = block[baseIdx + 1, i] * 1000.0 / sens.Y;
                        double gz = block[baseIdx + 2, i] * 1000.0 / sens.Z;

                        // 오프셋 적용
                        var off = GetAxisOffset(_modules[m]);
                        gx -= off.X; gy -= off.Y; gz -= off.Z;

                        if (_swByMod[m] != null)
                            _swByMod[m].WriteLine(t.ToString("F6") + "," + gx.ToString("G6") + "," + gy.ToString("G6") + "," + gz.ToString("G6"));
                    }
                }

                _samples += n;

                if (_running && _reader != null && _cb != null)
                    _reader.BeginReadMultiSample(_readBlock, _cb, null);
            }
            catch (DaqException) { /* Stop 중일 수 있음 */ }
            catch (ObjectDisposedException) { }
        }

        // ===== 유틸 =====
        private AxisSens GetAxisSens(string module)
        {
            if (string.IsNullOrWhiteSpace(module))
                return new AxisSens { X = DEFAULT_SENS_MVPG, Y = DEFAULT_SENS_MVPG, Z = DEFAULT_SENS_MVPG };
            AxisSens s;
            if (_sensByModule.TryGetValue(module, out s)) return s;
            return new AxisSens { X = DEFAULT_SENS_MVPG, Y = DEFAULT_SENS_MVPG, Z = DEFAULT_SENS_MVPG };
        }

        private AxisOffset GetAxisOffset(string module)
        {
            if (string.IsNullOrWhiteSpace(module))
                return new AxisOffset { X = 0.0, Y = 0.0, Z = 0.0 };
            AxisOffset o;
            if (_offsetByModule.TryGetValue(module, out o)) return o;
            return new AxisOffset { X = 0.0, Y = 0.0, Z = 0.0 };
        }

        // ===== Dispose =====
        public void Dispose()
        {
            SafeStop();
        }
    }
}