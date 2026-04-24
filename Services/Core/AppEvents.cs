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

        // ── Teaching Sequence 루프 완료 ──────────────────────────────────────
        /// <summary>
        /// TeachingForm 에서 시퀀스 한 회차가 완료될 때마다 발생합니다.
        /// count = 현재까지 완료된 총 회차 수
        /// </summary>
        public static event Action<int> LoopCompleted;
        public static void RaiseLoopCompleted(int count)
            => LoopCompleted?.Invoke(count);
    }
}
