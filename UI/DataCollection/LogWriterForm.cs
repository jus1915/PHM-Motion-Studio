using PHM_Project_DockPanel.Services;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace PHM_Project_DockPanel.Windows
{
    public class LogWriterForm : DockContent
    {
        private readonly TextBox _logBox;
        const int MaxLogChars = 200_000;
        public LogWriterForm()
        {
            Text = "Log";
            TabText = "Log";

            BackColor = Color.Black;
            AutoScroll = true;

            _logBox = new TextBox
            {
                Multiline = true,
                Dock = DockStyle.Fill,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                BackColor = Color.Black,
                ForeColor = Color.LightGreen,
                Font = new Font("Consolas", 9)
            };

            Controls.Add(_logBox);

            // 전역 로그 이벤트 구독
            AppEvents.LogRequested += AppendLog;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                AppEvents.LogRequested -= AppendLog;
            }
            base.Dispose(disposing);
        }

        /// <summary>
        /// 로그 추가 (스레드 안전)
        /// </summary>
        public void AppendLog(string message)
        {
            if (InvokeRequired) { BeginInvoke(new Action<string>(AppendLog), message); return; }

            string timeStamp = DateTime.Now.ToString("HH:mm:ss");
            _logBox.AppendText($"[{timeStamp}] {message}{Environment.NewLine}");
            _logBox.SelectionStart = _logBox.Text.Length;
            _logBox.ScrollToCaret();

            if (_logBox.TextLength > MaxLogChars)
            {
                // 앞쪽 절반 제거
                _logBox.Select(0, _logBox.TextLength / 2);
                _logBox.SelectedText = string.Empty;
            }
        }

        /// <summary>
        /// 로그 전체 지우기
        /// </summary>
        public void ClearLog()
        {
            if (_logBox.InvokeRequired)
            {
                _logBox.Invoke((Action)(ClearLog));
                return;
            }
            _logBox.Clear();
        }
    }
}