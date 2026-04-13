using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PHM_Project_DockPanel.Windows;

namespace PHM_Project_DockPanel.Services
{
    public static class AppState
    {
        public static LogWriterForm LogWriter { get; set; }

        /// <summary>현재 선택된 데이터 레이블. 로컬 CSV 저장 경로에 반영됩니다.</summary>
        public static string CurrentLabel { get; set; } = "";

        public static LogGraphForm.LogKind LogGraphPreferredKind { get; set; } = LogGraphForm.LogKind.Torque;
        public static double TorqueCycleSeconds { get; set; } = 0.001; // 1 tick = 1ms (필요시 변경)
        public static List<string> LastAccelCsvs { get; set; } = new List<string>();
        public static string LastAccelSelectedPath { get; set; }

        // 기본값: 원하시는 값으로 바꾸세요.
        public static double Accel { get; private set; } = 12800.0;
        public static double Torque { get; private set; } = 1000.0; // TODO: 실제 토크 레이트로 변경

        /// <summary>런타임에서 값 변경(옵션).</summary>
        public static void Configure(double? accel = null, double? torque = null)
        {
            if (accel.HasValue && accel.Value > 0) Accel = accel.Value;
            if (torque.HasValue && torque.Value > 0) Torque = torque.Value;
        }

        /// <summary>
        /// YColumn 같은 채널명으로부터 적절한 샘플링레이트를 추정.
        /// </summary>
        public static double GetForColumn(string column)
        {
            if (string.IsNullOrEmpty(column)) return Accel;

            var s = column.ToLowerInvariant();

            // 토크 키워드
            if (s.Contains("torque") || s.Contains("torq") || s.Contains("trq") || s.Contains("tq"))
                return Torque;

            // 기본은 가속도(진동)로 간주
            return Accel;
        }

        public static double GetPeriodForColumn(string yColumn)
        {
            var sr = GetForColumn(yColumn);                 // Hz
            return 1.0 / Math.Max(sr, 1e-9);                // sec/sample
        }
    }
}
