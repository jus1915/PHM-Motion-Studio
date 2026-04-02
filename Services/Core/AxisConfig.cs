using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services
{
    public class AxisConfig
    {
        public static int AxisCount { get; set; } = 0;
        public static bool IsConnected => AxisCount > 0;
        public double PositionMax { get; set; } = 900;  // Max Stroke
        public double PitchMmPerRev { get; set; } = 30;
        public double MaxVel { get; set; } = 1000;
        public double Acc { get; set; } = 5000;
        public double Dec { get; set; } = 5000;
    }
}
