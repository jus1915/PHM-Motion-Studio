using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Services.WMX
{
    public class WmxTorqueLogger
    {
        private readonly Log _logger;
        private readonly uint _channel;
        private bool _isLogging;
        private readonly Action<string> _logAction;

        public WmxTorqueLogger(Log loggerInstance, uint channelIndex = 0, Action<string> logAction = null)
        {
            _logger = loggerInstance ?? throw new ArgumentNullException(nameof(loggerInstance));
            _channel = channelIndex;
            _isLogging = false;
            _logAction = logAction;
            _logAction?.Invoke($"FeedbackLogger 생성됨: Channel={_channel}");
        }

        public bool Start(int[] axisIndices, string filePath, string filename, uint durationMs = 5000)
        {
            if (!_logger.IsDeviceValid())
            {
                _logAction?.Invoke("디바이스가 유효하지 않습니다.");
                return false;
            }

            // 1. 이전 로그 상태 리셋
            _logger.ResetLog(_channel);

            // 2. 축 선택
            AxisSelection axisSel = new AxisSelection
            {
                AxisCount = axisIndices.Length,
                Axis = new int[axisIndices.Length]
            };
            for (int i = 0; i < axisIndices.Length; i++)
                axisSel.Axis[i] = (ushort)axisIndices[i];

            // 3. 로깅할 데이터 선택
            CoreMotionLogInput input = new CoreMotionLogInput();
            input.AxisSel = axisSel;
            input.AxisOptions.CommandPos = 1;
            input.AxisOptions.FeedbackPos = 1;
            input.AxisOptions.CommandVelocity = 1;
            input.AxisOptions.FeedbackVelocity = 1;
            input.AxisOptions.CommandTrq = 1;
            input.AxisOptions.FeedbackTrq = 1;

            int err = _logger.SetLog(_channel, input);
            if (err != ErrorCode.None)
            {
                _logAction?.Invoke($"SetLog 실패: {Log.ErrorToString(err)}");
                return false;
            }

            // 4. 로그 파일 경로 설정
            if (!Directory.Exists(filePath))
                Directory.CreateDirectory(filePath);

            var path = new LogFilePath { DirPath = filePath, FileName = filename };
            err = _logger.SetLogFilePath(_channel, path);
            if (err != ErrorCode.None)
            {
                _logAction?.Invoke($"SetLogFilePath 실패: {Log.ErrorToString(err)}");
                return false;
            }

            // 5. 옵션 설정
            var options = new LogChannelOptions
            {
                SamplingTimeMilliseconds = durationMs,
                SamplingPeriodInCycles = 1,
                Precision = 4,
                MaxLogFileSize = 0,
                MaxLogFileCount = 0,
                IsRotateFile = false,
                StopLoggingOnBufferOverflow = true,
                IsDelimInLastCol = true,
                Delimiter = ",",
                TriggerOnCondition = 0,
                TriggerOnEvent = 0,
                TriggerEventID = 0
            };
            err = _logger.SetLogOption(_channel, options);
            if (err != ErrorCode.None)
            {
                _logAction?.Invoke($"SetLogOption 실패: {Log.ErrorToString(err)}");
                return false;
            }

            // 6. Start
            err = _logger.StartLog(_channel);
            if (err != ErrorCode.None)
            {
                _logAction?.Invoke($"StartLog 실패: {Log.ErrorToString(err)}");
                return false;
            }

            _isLogging = true;
            return true;
        }

        public void Stop()
        {
            if (!_isLogging)
            {
                _logAction?.Invoke("현재 로깅 중이 아닙니다.");
                return;
            }

            int err = _logger.StopLog(_channel);
            if (err != ErrorCode.None)
            {
                _logAction?.Invoke($"StopLog 실패: {Log.ErrorToString(err)}");
            }
            else
            {
                Thread.Sleep(500); // 파일 닫힘 대기
                _isLogging = false;
            }
        }

        public bool IsLogging() => _isLogging;
    }
}
