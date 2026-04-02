using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services.Logging
{
    public interface IMeasurementLogger
    {
        string Name { get; }
        bool IsRunning { get; }
        string OutputPath { get; }
        bool Start(LogStartContext ctx);
        void Stop();
    }

    public sealed class LogStartContext
    {
        public string Directory { get; set; }
        public string BaseFileName { get; set; }
        public int[] Axes { get; set; } = new int[0];
        public uint? DurationMs { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }

    public sealed class LoggerOutput
    {
        public string Name { get; private set; }
        public string Path { get; private set; }
        public LoggerOutput(string name, string path)
        {
            Name = name;
            Path = path;
        }
    }
}
