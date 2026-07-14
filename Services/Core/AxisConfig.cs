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
        // 서보 모터/드라이브의 회전당 엔코더 펄스 수. 축마다 다른 드라이브(예: 이노반스 vs
        // 파나소닉)를 쓰면 이 값도 축별로 달라야 한다 — 0이면 AjinController의 기본값(8388608)을 사용.
        public int PulsePerRev { get; set; } = 0;
        public double MaxVel { get; set; } = 1000;
        public double Acc { get; set; } = 5000;
        public double Dec { get; set; } = 5000;
    }
}
