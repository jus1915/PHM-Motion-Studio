using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PHM_Project_DockPanel.Services.Core;

namespace PHM_Project_DockPanel.Services
{
    public static class AppEvents
    {
        public static event Action<int, float> SimulatorInitializeRequested;
        public static event Action<float[]> SimulatorInitializeWithMaxPositionsRequested;

        public static event Action<int, float> SimulatorMaxPositionUpdateRequested;
        public static event Action<float[]> SimulatorPositionUpdateRequested;
        public static event Action RequestCloseSimulator;
        public static event Action RequestClearSimulator;
        public static event Action<string> LogRequested;
        public static event Action<string> ShowLogGraphRequested;

        public static event Action<bool> AccelRealtimeToggled;

        public static void RaiseAccelRealtimeToggled(bool enabled)
        {
            var h = AccelRealtimeToggled;
            if (h != null) h(enabled);
        }

        /// <summary>InfluxDB 저장 레이블 변경 (빈 문자열 = 레이블 없음)</summary>
        public static event Action<string> InfluxLabelChanged;
        public static void RaiseInfluxLabelChanged(string label)
            => InfluxLabelChanged?.Invoke(label ?? "");

        /// <summary>InfluxDB URL/Token 변경 (UI에서 편집 후 저장 시 발생)</summary>
        public static event Action<string, string> InfluxConfigChanged;
        public static void RaiseInfluxConfigChanged(string url, string token)
            => InfluxConfigChanged?.Invoke(url ?? "", token ?? "");

        /// <summary>서버 연결 설정 전체 변경 (ServerSettingsForm 저장 시 발생)</summary>
        public static event Action<ServerSettings> ServerSettingsChanged;
        public static void RaiseServerSettingsChanged(ServerSettings settings)
            => ServerSettingsChanged?.Invoke(settings);

        public enum LogDataKind { Torque, Accel }

        public static event Action<LogDataKind, string> ShowLogGraphRequestedEx;

        public static void RaiseShowLogGraph(LogDataKind kind, string path)
            => ShowLogGraphRequestedEx?.Invoke(kind, path);

        public static void RaiseSimulatorInitialize(int axisCount, float defaultMaxPos)
        {
            SimulatorInitializeRequested?.Invoke(axisCount, defaultMaxPos);
        }

        public static void RaiseSimulatorInitialize(float[] maxPositions)
        {
            SimulatorInitializeWithMaxPositionsRequested?.Invoke(maxPositions);
        }

        public static void RaiseSimulatorMaxPositionUpdate(int axisIndex, float maxPos)
        {
            SimulatorMaxPositionUpdateRequested?.Invoke(axisIndex, maxPos);
        }

        public static void RaiseSimulatorPositionUpdate(float[] positions)
            => SimulatorPositionUpdateRequested?.Invoke(positions);


        public static void RaiseShowLogGraph(string filePath)
        => ShowLogGraphRequested?.Invoke(filePath);



        public static void RaiseRequestCloseSimulator()
            => RequestCloseSimulator?.Invoke();

        public static void RaiseRequestClearSimulator()
            => RequestClearSimulator?.Invoke();

        public static void RaiseLog(string message) => LogRequested?.Invoke(message);

        // Passive Monitor ↔ PHM_Motion 리소스 조정
        public static event Action PassiveMonitorSuspendRequested;
        public static event Action PassiveMonitorResumeRequested;

        public static void RaisePassiveMonitorSuspend() => PassiveMonitorSuspendRequested?.Invoke();
        public static void RaisePassiveMonitorResume()  => PassiveMonitorResumeRequested?.Invoke();

        // ── 서버 추론 결과 ────────────────────────────────────────────────────
        /// <summary>
        /// ContinuousInferenceService 가 AE 이상탐지 결과를 수신했을 때 발생합니다.
        /// sensorType = "accel" | "torque"
        /// </summary>
        public static event Action<string, InferenceResult> InferenceResultReceived;
        public static void RaiseInferenceResult(string sensorType, InferenceResult result)
            => InferenceResultReceived?.Invoke(sensorType, result);

        /// <summary>
        /// ContinuousInferenceService 가 CLS 결함진단 결과를 수신했을 때 발생합니다.
        /// combined.ClsAvailable = false 이면 CLS 모델 없음(AE 전용 상태).
        /// combined.ClsAvailable = true  이면 CLS 결과(ClsClassName/ClsConfidence/ClsIsFault) 유효.
        /// sensorType = "accel" | "torque"
        /// </summary>
        public static event Action<string, CombinedInferenceResult> ClsInferenceResultReceived;
        public static void RaiseClsInferenceResult(string sensorType, CombinedInferenceResult result)
            => ClsInferenceResultReceived?.Invoke(sensorType, result);

        // ── 실시간 신호 모니터 (RealtimeSignalForm) ──────────────────────────
        /// <summary>
        /// 가속도 블록 수신 — 비UI 스레드에서 발생.
        /// module: 센서 모듈명  block: [3, N] (X/Y/Z, g단위)
        /// ts: 블록 마지막 샘플 UTC  sampleRate: Hz
        /// </summary>
        public static event Action<string, double[,], DateTime, double> AccelBlockReceived;
        public static void RaiseAccelBlock(string module, double[,] block, DateTime ts, double sampleRate)
            => AccelBlockReceived?.Invoke(module, block, ts, sampleRate);

        /// <summary>
        /// 토크 샘플 수신 — 비UI 스레드에서 발생 (1ms 주기).
        /// axis: 축 인덱스  value: 토크 [%]  ts: UTC
        /// </summary>
        public static event Action<int, double, DateTime> TorqueSampleReceived;
        public static void RaiseTorqueSample(int axis, double value, DateTime ts)
            => TorqueSampleReceived?.Invoke(axis, value, ts);

        // ── Teaching Sequence 루프 완료 ──────────────────────────────────────
        /// <summary>
        /// TeachingForm 에서 시퀀스 한 회차가 완료될 때마다 발생합니다.
        /// count = 현재까지 완료된 총 회차 수
        /// </summary>
        public static event Action<int> LoopCompleted;
        public static void RaiseLoopCompleted(int count)
            => LoopCompleted?.Invoke(count);

        // ── Teaching Sequence 회차 시작 (SequenceID 채번용) ──────────────────
        /// <summary>
        /// TeachingForm 에서 시퀀스 한 회차(전체 스텝 목록 1회 반복)가 시작될 때 발생합니다.
        /// PHM_Motion 이 이 이벤트를 구독해 CombinedCsvLogger 의 SequenceID 를 1씩 증가시킵니다.
        /// </summary>
        public static event Action SequenceStarted;
        public static void RaiseSequenceStarted()
            => SequenceStarted?.Invoke();
    }
}
