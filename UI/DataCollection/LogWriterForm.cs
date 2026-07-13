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

            // 핸들을 UI 스레드(생성자 호출 스레드)에서 미리 만들어 둔다.
            // 이 폼은 MainForm 생성자에서 만들어지지만, 저장된 layout.xml에 Log Writer
            // 패널이 없으면(예: 이전에 닫아둔 적이 있으면) Show()가 호출되지 않아 핸들이
            // 생기지 않는다. 핸들이 없는 Control은 InvokeRequired가 어느 스레드에서
            // 호출해도 항상 false를 반환하므로, AppendLog의 크로스스레드 체크가
            // 무력화되어 백그라운드 스레드가 _logBox를 직접 건드려 크래시가 난다.
            var _ = this.Handle;

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
            if (IsDisposed || Disposing) return;
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