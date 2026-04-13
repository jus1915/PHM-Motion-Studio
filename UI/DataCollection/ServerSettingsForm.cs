using System;
using System.Drawing;
using System.Windows.Forms;
using PHM_Project_DockPanel.Services;

namespace PHM_Project_DockPanel.UI.DataCollection
{
    // =========================================================================
    //  ServerSettingsForm — 서버 연결 설정 (InfluxDB / MLflow / Airflow)
    //  MainForm 메뉴 → 환경 설정 → 연결 설정
    // =========================================================================
    public sealed class ServerSettingsForm : Form
    {
        private readonly string _settingsPath;

        // InfluxDB
        private TextBox _txtInfluxUrl, _txtInfluxToken, _txtInfluxOrg, _txtInfluxBucket;
        // MLflow
        private TextBox _txtMlflowUrl;
        // Airflow
        private TextBox _txtAirflowUrl;
        // 일괄 변경
        private TextBox _txtBulkIp;

        public ServerSettingsForm(string settingsPath)
        {
            _settingsPath = settingsPath;
            BuildUI();
            LoadFromSettings();
        }

        // ── UI 구성 ────────────────────────────────────────────────────────────
        private void BuildUI()
        {
            Text            = "서버 연결 설정";
            Size            = new Size(500, 520);
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition   = FormStartPosition.CenterParent;
            MaximizeBox     = false;
            MinimizeBox     = false;

            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill, ColumnCount = 1,
                Padding = new Padding(12, 10, 12, 8),
                RowCount = 5,
            };
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 64));   // 일괄 변경
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 138));  // InfluxDB
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 62));   // MLflow
            root.RowStyles.Add(new RowStyle(SizeType.Absolute, 62));   // Airflow
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));   // 버튼

            // ── [0] IP 일괄 변경 ───────────────────────────────────────────────
            var gbBulk = new GroupBox { Text = "IP 일괄 변경", Dock = DockStyle.Fill, Padding = new Padding(8, 4, 8, 4) };
            var bulkRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false };
            bulkRow.Controls.Add(new Label { Text = "새 IP:", AutoSize = true, Margin = new Padding(0, 6, 4, 0) });
            _txtBulkIp = new TextBox { Width = 200, Height = 24 };
            var btnBulk = new Button { Text = "모든 URL에 적용", Height = 24, AutoSize = true, Margin = new Padding(6, 1, 0, 0) };
            btnBulk.Click += BtnBulk_Click;
            bulkRow.Controls.AddRange(new Control[] { _txtBulkIp, btnBulk });
            gbBulk.Controls.Add(bulkRow);
            root.Controls.Add(gbBulk, 0, 0);

            // ── [1] InfluxDB ──────────────────────────────────────────────────
            var gbInflux = new GroupBox { Text = "InfluxDB", Dock = DockStyle.Fill, Padding = new Padding(8, 4, 8, 4) };
            var tlInflux = MakeTl(4);
            int r = 0;
            tlInflux.Controls.Add(Lbl("URL:"),    0, r); _txtInfluxUrl    = Txt();                       tlInflux.Controls.Add(_txtInfluxUrl,    1, r++);
            tlInflux.Controls.Add(Lbl("Token:"),  0, r); _txtInfluxToken  = Txt(password: true);         tlInflux.Controls.Add(_txtInfluxToken,  1, r++);
            tlInflux.Controls.Add(Lbl("Org:"),    0, r); _txtInfluxOrg    = Txt();                       tlInflux.Controls.Add(_txtInfluxOrg,    1, r++);
            tlInflux.Controls.Add(Lbl("Bucket:"), 0, r); _txtInfluxBucket = Txt();                       tlInflux.Controls.Add(_txtInfluxBucket, 1, r++);
            gbInflux.Controls.Add(tlInflux);
            root.Controls.Add(gbInflux, 0, 1);

            // ── [2] MLflow ────────────────────────────────────────────────────
            var gbMlflow = new GroupBox { Text = "MLflow", Dock = DockStyle.Fill, Padding = new Padding(8, 4, 8, 4) };
            var tlMlflow = MakeTl(1);
            tlMlflow.Controls.Add(Lbl("URL:"), 0, 0); _txtMlflowUrl = Txt(); tlMlflow.Controls.Add(_txtMlflowUrl, 1, 0);
            gbMlflow.Controls.Add(tlMlflow);
            root.Controls.Add(gbMlflow, 0, 2);

            // ── [3] Airflow ───────────────────────────────────────────────────
            var gbAirflow = new GroupBox { Text = "Airflow", Dock = DockStyle.Fill, Padding = new Padding(8, 4, 8, 4) };
            var tlAirflow = MakeTl(1);
            tlAirflow.Controls.Add(Lbl("URL:"), 0, 0); _txtAirflowUrl = Txt(); tlAirflow.Controls.Add(_txtAirflowUrl, 1, 0);
            gbAirflow.Controls.Add(tlAirflow);
            root.Controls.Add(gbAirflow, 0, 3);

            // ── [4] 버튼 ──────────────────────────────────────────────────────
            var btnRow = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.RightToLeft, Padding = new Padding(0, 4, 0, 0) };
            var btnCancel = new Button { Text = "닫기",      Width = 80, Height = 28, DialogResult = DialogResult.Cancel };
            var btnSave   = new Button { Text = "💾 저장 & 적용", Width = 120, Height = 28,
                BackColor = Color.FromArgb(0, 120, 212), ForeColor = Color.White, FlatStyle = FlatStyle.Flat };
            btnSave.Click   += BtnSave_Click;
            btnRow.Controls.Add(btnCancel);
            btnRow.Controls.Add(btnSave);
            root.Controls.Add(btnRow, 0, 4);

            Controls.Add(root);
            AcceptButton = btnSave;
            CancelButton = btnCancel;
        }

        // ── 데이터 로드 ────────────────────────────────────────────────────────
        private void LoadFromSettings()
        {
            var s = ServerSettings.Current;
            _txtInfluxUrl.Text    = s.InfluxUrl    ?? "";
            _txtInfluxToken.Text  = s.InfluxToken  ?? "";
            _txtInfluxOrg.Text    = s.InfluxOrg    ?? "";
            _txtInfluxBucket.Text = s.InfluxBucket ?? "";
            _txtMlflowUrl.Text    = s.MlflowUrl    ?? "";
            _txtAirflowUrl.Text   = s.AirflowUrl   ?? "";
        }

        // ── 저장 & 적용 ────────────────────────────────────────────────────────
        private void BtnSave_Click(object sender, EventArgs e)
        {
            var s = ServerSettings.Current;
            s.InfluxUrl    = _txtInfluxUrl.Text.Trim();
            s.InfluxToken  = _txtInfluxToken.Text.Trim();
            s.InfluxOrg    = _txtInfluxOrg.Text.Trim();
            s.InfluxBucket = _txtInfluxBucket.Text.Trim();
            s.MlflowUrl    = _txtMlflowUrl.Text.Trim();
            s.AirflowUrl   = _txtAirflowUrl.Text.Trim();
            s.Save(_settingsPath);

            AppEvents.RaiseServerSettingsChanged(s);
            MessageBox.Show("저장되었습니다.\n변경사항이 즉시 적용됩니다.",
                "연결 설정", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        // ── IP 일괄 변경 ───────────────────────────────────────────────────────
        private void BtnBulk_Click(object sender, EventArgs e)
        {
            string newIp = _txtBulkIp.Text.Trim();
            if (string.IsNullOrEmpty(newIp))
            {
                MessageBox.Show("새 IP를 입력하세요.", "입력 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            _txtInfluxUrl.Text  = ServerSettings.ReplaceHost(_txtInfluxUrl.Text,  newIp);
            _txtMlflowUrl.Text  = ServerSettings.ReplaceHost(_txtMlflowUrl.Text,  newIp);
            _txtAirflowUrl.Text = ServerSettings.ReplaceHost(_txtAirflowUrl.Text, newIp);
        }

        // ── UI 헬퍼 ────────────────────────────────────────────────────────────
        private static TableLayoutPanel MakeTl(int rows)
        {
            var tl = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = rows };
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 70));
            tl.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            for (int i = 0; i < rows; i++)
                tl.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
            return tl;
        }

        private static Label Lbl(string text) =>
            new Label { Text = text, AutoSize = false, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft };

        private static TextBox Txt(bool password = false) =>
            new TextBox { Dock = DockStyle.Fill, UseSystemPasswordChar = password };
    }
}
