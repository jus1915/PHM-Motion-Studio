using PHM_Project_DockPanel.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.DebugTools
{
    public static class AxisConfigDebugger
    {
        public static void LogChange(string source, AxisConfig[] configs)
        {
            string length = configs == null ? "null" : configs.Length.ToString();
            string hash = configs == null ? "null" : configs.GetHashCode().ToString();
            AppEvents.RaiseLog($"[디버그] {source} - 길이={length}, 해시={hash}");
        }
    }
}
