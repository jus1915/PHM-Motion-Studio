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
        // 드라이브 쪽 원인(모드/기어비 등)으로 명령한 거리와 실제 이동 거리가 일정 비율로
        // 어긋나는 축을 위한 임시 소프트웨어 보정 배율. 위치 피드백(실제 위치 표시)에는
        // 영향을 주지 않고, PHM_Motion이 컨트롤러에 보내는 "이동 명령" 목표값에만 곱해진다.
        // 예: 10mm 명령 시 실제로 10mm/2000만 움직이면 2000으로 설정. 1이면 보정 없음(기본).
        public double MoveCommandScale { get; set; } = 1.0;
        public double MaxVel { get; set; } = 1000;
        public double Acc { get; set; } = 5000;
        public double Dec { get; set; } = 5000;
    }
}
