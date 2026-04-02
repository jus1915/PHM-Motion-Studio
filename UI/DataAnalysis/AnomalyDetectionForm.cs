using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;

namespace PHM_Project_DockPanel.UI.DataAnalysis
{
    public partial class AnomalyDetectionForm : DockContent
    {
        private TabControl tabControl;

        public AnomalyDetectionForm()
        {
            InitializeComponent();
            Text = "이상 탐지";
        }

        private void InitializeComponent()
        {
            this.tabControl = new TabControl();
            var tabOffline = new TabPage("오프라인 분석");
            var tabRealtime = new TabPage("실시간 모니터링");

            tabOffline.Controls.Add(new Label() { Text = "오프라인 이상 탐지 UI", Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter });
            tabRealtime.Controls.Add(new Label() { Text = "실시간 이상 탐지 UI", Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter });

            tabControl.Dock = DockStyle.Fill;
            tabControl.TabPages.Add(tabOffline);
            tabControl.TabPages.Add(tabRealtime);

            this.Controls.Add(tabControl);
            this.ClientSize = new System.Drawing.Size(900, 600);
        }
    }
}
