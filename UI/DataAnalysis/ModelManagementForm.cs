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
    public partial class ModelManagementForm : DockContent
    {
        private TabControl tabControl;

        public ModelManagementForm()
        {
            InitializeComponent();
            Text = "AI 모델 관리";
        }

        private void InitializeComponent()
        {
            this.tabControl = new TabControl();
            var tabTrain = new TabPage("모델 학습");
            var tabValidate = new TabPage("모델 검증");
            var tabManage = new TabPage("모델 관리");

            tabTrain.Controls.Add(new Label() { Text = "모델 학습 UI", Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter });
            tabValidate.Controls.Add(new Label() { Text = "모델 검증 UI", Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter });
            tabManage.Controls.Add(new Label() { Text = "모델 불러오기/저장 UI", Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter });

            tabControl.Dock = DockStyle.Fill;
            tabControl.TabPages.Add(tabTrain);
            tabControl.TabPages.Add(tabValidate);
            tabControl.TabPages.Add(tabManage);

            this.Controls.Add(tabControl);
            this.ClientSize = new System.Drawing.Size(900, 600);
        }
    }
}
