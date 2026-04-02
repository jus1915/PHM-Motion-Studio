using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}
