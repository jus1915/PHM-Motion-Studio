using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PHM_Project_DockPanel.Services
{
    public class UnitConverter
    {
        public const double EncoderResolution = 8388608;

        public static double EncoderToMm(double encoderPos, double pitchMmPerRev) =>
            encoderPos / EncoderResolution * pitchMmPerRev;

        public static double MmToEncoder(double mmPos, double pitchMmPerRev) =>
            mmPos / pitchMmPerRev * EncoderResolution;
    }
}
