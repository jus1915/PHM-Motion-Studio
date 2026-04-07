using System.Globalization;
using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.DebugTools;
using PHM_Project_DockPanel.Services;
using PHM_Project_DockPanel.Windows;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using WeifenLuo.WinFormsUI.Docking;
using WMX3ApiCLR;

namespace PHM_Project_DockPanel.Windows
{
    public class AxisInfoForm : DockContent
    {
        // --- UI: 상단/좌/우 ---
        private Label lblStatus;
        private Button btnConnect, btnDisconnect;
        private Panel scrollPanel;

        private Panel parameterPanel;
        private Panel actionPanel;

        // --- 좌측: DataGridView ---
        private DataGridView _grid;

        // ▶ 수집 옵션: 가속도 / 토크 (기존 단일 체크박스 분리)
        private CheckBox _chkAccelCollect;     // 가속도 수집
        private CheckBox _chkTorqueCollect;    // 토크 수집
        private CheckBox _chkRealtime;
        private ComboBox _cmbLabel;
        private Label _lblLabelCaption;

        // ▷ 레거시 호환용(외부 코드가 LogCheckBox에 접근하던 경우 대응)
        private CheckBox _chkLogCombined = new CheckBox { Visible = false }; // 두 체크의 OR, UI에 미표시
        private bool _syncingLegacy = false;   // 이벤트 루프 방지
        private Label _lblRealtimeStatus;
        private Label _lblDaqStatus;

        // (선택) 외부에서 접근할 수 있도록 공개 프로퍼티
        public CheckBox AccelCheckBox => _chkAccelCollect;
        public CheckBox TorqueCheckBox => _chkTorqueCollect;
        [Obsolete("Use AccelCheckBox or TorqueCheckBox instead.")]
        public CheckBox LogCheckBox => _chkLogCombined;
        public CheckBox RealtimeSendCheckBox => _chkRealtime;

        // DGV 컬럼 인덱스
        private const int COL_SEL   = 0;
        private const int COL_AXIS  = 1;
        private const int COL_POS   = 2;
        private const int COL_SERVO = 3;
        private const int COL_ALARM = 4;
        private const int COL_OP    = 5;
        private const int COL_TGT   = 6;   // 축별 개별 Target(mm)

        private bool[]   _checkedArr;
        private float[]  _posArr;
        private bool[]   _servoArr;
        private bool[]   _alarmOkArr;
        private string[] _opArr;
        private double[] _targetArr;  // 축별 개별 target (Multi-Move용)

        // --- 우측 액션 ---
        private Button btnServoOnGlobal, btnServoOffGlobal, btnAbsMoveGlobal, btnRelMoveGlobal;
        private Button btnAlarmClear;
        private TextBox txtTargetGlobal;
        private Label lblCheckedAxes;   // 체크된 축 표시

        // --- 우측 파라미터 ---
        private TextBox txtMaxVel, txtAcc, txtDec, txtMaxStroke, txtPitch;

        // --- 상태/데이터 ---
        private int _selectedAxis = -1;           // 단일 선택(서보/파라미터용)
        private int _axisCount = 0;               // Connect 후 실제 축 수로 설정됨

        private PHM_Motion _motion;
        private AxisConfig[] _axisConfigs;
        private AxisMonitor _monitor;

        public AxisInfoForm(PHM_Motion pHM_motion, AxisConfig[] axisConfigs)
        {
            _motion = pHM_motion;
            _axisConfigs = axisConfigs;
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            Text = "Axis Information";
            Padding = new Padding(5);
            this.MinimumSize = new Size(960, 560);

            // === 상단 패널 ===
            var topPanel = new Panel { Dock = DockStyle.Top, Height = 60, Padding = new Padding(0) };

            // ===== (좌측) 상태 라벨들: 세로 정렬 =====
            var leftStatusPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Left,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                Margin = new Padding(0),
                Padding = new Padding(0)
            };

            lblStatus = new Label
            {
                Text = "Status: Disconnected",
                AutoSize = true,
                TextAlign = ContentAlignment.MiddleLeft,
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                ForeColor = Color.DarkRed,
                Margin = new Padding(0, 2, 0, 0)
            };

            _lblRealtimeStatus = new Label
            {
                Text = "실시간 전송: 대기 중",
                ForeColor = Color.Gray,
                AutoSize = true,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0, 2, 0, 0)
            };

            _lblDaqStatus = new Label
            {
                Text = "DAQ 상태: 대기 중",
                ForeColor = Color.DarkSlateGray,
                AutoSize = true,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0, 2, 0, 0)
            };

            leftStatusPanel.Controls.Add(lblStatus);
            leftStatusPanel.Controls.Add(_lblRealtimeStatus);
            leftStatusPanel.Controls.Add(_lblDaqStatus);

            // ===== (우측) 체크박스 + Connect/Disconnect 버튼: 가로 정렬 =====
            var rightControlPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Right,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false,
                Margin = new Padding(0),
                Padding = new Padding(0)
            };

            _chkAccelCollect = new CheckBox { Text = "가속도 수집", AutoSize = true, Margin = new Padding(5, 8, 5, 0) };
            _chkTorqueCollect = new CheckBox { Text = "토크 수집", AutoSize = true, Margin = new Padding(5, 8, 5, 0) };
            _chkRealtime = new CheckBox { Text = "실시간 데이터 전송", AutoSize = true, Margin = new Padding(5, 8, 5, 0) };

            _chkRealtime.CheckedChanged += (s, e) =>
            {
                bool enabled = _chkRealtime.Checked;
                UpdateRealtimeStatusLabel(enabled);
                _cmbLabel.Enabled = enabled;
                AppEvents.RaiseAccelRealtimeToggled(enabled); // 로그 출력 안 함
            };

            // 레이블 콤보박스
            _lblLabelCaption = new Label
            {
                Text = "레이블:",
                AutoSize = true,
                Margin = new Padding(10, 10, 2, 0)
            };

            _cmbLabel = new ComboBox
            {
                DropDownStyle = ComboBoxStyle.DropDown,
                Width = 120,
                Margin = new Padding(0, 6, 5, 0),
                Enabled = false
            };
            _cmbLabel.Items.AddRange(new object[]
            {
                "", "normal", "fault", "bearing_fault", "gear_fault", "imbalance", "looseness"
            });
            _cmbLabel.SelectedIndex = 0;
            _cmbLabel.TextChanged += (s, e) =>
                AppEvents.RaiseInfluxLabelChanged(_cmbLabel.Text.Trim());
            _cmbLabel.SelectedIndexChanged += (s, e) =>
                AppEvents.RaiseInfluxLabelChanged(_cmbLabel.Text.Trim());

            btnConnect = new Button { Text = "Connect", Width = 100, Margin = new Padding(8, 4, 0, 0) };
            btnConnect.Click += BtnConnect_Click;

            btnDisconnect = new Button { Text = "Disconnect", Width = 100, Enabled = false, Margin = new Padding(5, 4, 0, 0) };
            btnDisconnect.Click += BtnDisconnect_Click;

            rightControlPanel.Controls.Add(_chkAccelCollect);
            rightControlPanel.Controls.Add(_chkTorqueCollect);
            rightControlPanel.Controls.Add(_chkRealtime);
            rightControlPanel.Controls.Add(_lblLabelCaption);
            rightControlPanel.Controls.Add(_cmbLabel);
            rightControlPanel.Controls.Add(btnConnect);
            rightControlPanel.Controls.Add(btnDisconnect);

            // === 조립 ===
            topPanel.Controls.Add(leftStatusPanel);   // 좌측 세로 라벨
            topPanel.Controls.Add(rightControlPanel); // 우측 체크박스+버튼 가로 정렬

            // === 왼쪽 스크롤 영역 ===
            scrollPanel = new Panel { Dock = DockStyle.Fill, AutoScroll = true };
            scrollPanel.MinimumSize = new Size(400, 0);

            _grid = new DataGridView
            {
                Dock = DockStyle.Fill,
                ReadOnly = false,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false,
                AllowUserToResizeRows = false,
                AllowUserToResizeColumns = false,
                RowHeadersVisible = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                MultiSelect = false,
                AutoGenerateColumns = false,
                VirtualMode = true,
                EnableHeadersVisualStyles = false,
                ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing,
                ColumnHeadersHeight = 26,
                BackgroundColor = SystemColors.Window,
                EditMode = DataGridViewEditMode.EditOnKeystrokeOrF2
            };
            MakeDgvDoubleBuffered(_grid);

            // 컬럼 (체크박스 포함)
            var colSel = new DataGridViewCheckBoxColumn
            {
                Name = "Sel",
                HeaderText = "✔",
                Width = 36,
                ThreeState = false
            };
            _grid.Columns.Add(colSel);
            _grid.Columns.Add(new DataGridViewTextBoxColumn { Name = "Axis", HeaderText = "Axis Num", Width = 70 });
            _grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Pos",
                HeaderText = "Pos",
                Width = 90,
                DefaultCellStyle = new DataGridViewCellStyle { Alignment = DataGridViewContentAlignment.MiddleRight, Format = "N3" }
            });
            _grid.Columns.Add(new DataGridViewTextBoxColumn { Name = "Servo", HeaderText = "Servo", Width = 70 });
            _grid.Columns.Add(new DataGridViewTextBoxColumn { Name = "Alarm", HeaderText = "Alarm", Width = 70 });
            _grid.Columns.Add(new DataGridViewTextBoxColumn { Name = "Op", HeaderText = "Operation", Width = 90 });
            _grid.Columns.Add(new DataGridViewTextBoxColumn
            {
                Name = "Target",
                HeaderText = "Target(mm)",
                Width = 90,
                ReadOnly = false,
                DefaultCellStyle = new DataGridViewCellStyle
                {
                    Alignment = DataGridViewContentAlignment.MiddleRight,
                    Format = "0.###",
                    BackColor = Color.LightYellow
                }
            });

            _grid.RowTemplate.Height = 24;
            _grid.AlternatingRowsDefaultCellStyle.BackColor = Color.WhiteSmoke;

            _grid.CellValueNeeded += Grid_CellValueNeeded;
            _grid.CellValuePushed += Grid_CellValuePushed;
            _grid.CellFormatting += Grid_CellFormatting;
            _grid.CellBeginEdit += (s, e) =>
            {
                // Target 컬럼과 체크박스 컬럼만 편집 허용
                if (e.ColumnIndex != COL_TGT && e.ColumnIndex != COL_SEL)
                    e.Cancel = true;
            };
            _grid.CellClick += (s, e) =>
            {
                if (e.RowIndex >= 0)
                {
                    if (e.ColumnIndex == COL_SEL)
                    {
                        ToggleCheck(e.RowIndex);
                    }
                    else
                    {
                        SelectAxis(e.RowIndex);
                    }
                }
            };
            _grid.CurrentCellDirtyStateChanged += (s, e) =>
            {
                if (_grid.IsCurrentCellDirty)
                    _grid.CommitEdit(DataGridViewDataErrorContexts.Commit);
            };
            _grid.SelectionChanged += (s, e) =>
            {
                if (_grid.CurrentRow != null && _grid.CurrentRow.Index >= 0 && _grid.CurrentRow.Index < _axisCount)
                    SelectAxis(_grid.CurrentRow.Index);
            };

            scrollPanel.Controls.Add(_grid);

            // === 우측 컨테이너 ===
            var rightContainer = new Panel { Dock = DockStyle.Right, Width = 520, BackColor = Color.Gainsboro };
            rightContainer.MinimumSize = new Size(700, 0);

            // 우측 내부 2열 레이아웃 (Actions | Params)
            var rightLayout = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 2,
                RowCount = 1,
                BackColor = Color.Gainsboro,
                Padding = new Padding(0),
                Margin = new Padding(0)
            };
            rightLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 240f)); // Actions 폭 약간 확대
            rightLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f));
            rightLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100f));

            // === 공용 액션 패널 ===
            actionPanel = CreateActionPanel();
            actionPanel.Dock = DockStyle.Fill;
            actionPanel.Margin = new Padding(8);

            // === 파라미터 패널 ===
            parameterPanel = CreateParameterPanel();
            parameterPanel.Dock = DockStyle.Fill;
            parameterPanel.Margin = new Padding(8);

            rightLayout.Controls.Add(actionPanel, 0, 0);
            rightLayout.Controls.Add(parameterPanel, 1, 0);
            rightContainer.Controls.Add(rightLayout);

            // === 컨트롤 추가 순서 ===
            Controls.Add(scrollPanel);      // Fill
            Controls.Add(rightContainer);   // Right
            Controls.Add(topPanel);         // Top
        }

        public void UpdateDaqStatus(string msg)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action(() => UpdateDaqStatus(msg)));
                return;
            }

            if (_lblDaqStatus != null)
            {
                _lblDaqStatus.Text = msg;
                _lblDaqStatus.ForeColor = Color.SeaGreen;
            }
        }

        private void UpdateRealtimeStatusLabel(bool enabled)
        {
            if (_lblRealtimeStatus == null) return;

            if (enabled)
            {
                _lblRealtimeStatus.Text = "실시간 전송 중...";
                _lblRealtimeStatus.ForeColor = Color.SeaGreen;
            }
            else
            {
                _lblRealtimeStatus.Text = "대기 중";
                _lblRealtimeStatus.ForeColor = Color.Gray;
            }
        }


        // 레거시-신규 동기화
        private void SyncCombinedFromChildren()
        {
            if (_syncingLegacy) return;
            try
            {
                _syncingLegacy = true;
                _chkLogCombined.Checked = _chkAccelCollect.Checked || _chkTorqueCollect.Checked;
            }
            finally { _syncingLegacy = false; }
        }
        private void SyncChildrenFromCombined()
        {
            if (_syncingLegacy) return;
            try
            {
                _syncingLegacy = true;
                bool v = _chkLogCombined.Checked;
                _chkAccelCollect.Checked = v;
                _chkTorqueCollect.Checked = v;
            }
            finally { _syncingLegacy = false; }
        }

        private void BtnConnect_Click(object sender, EventArgs e)
        {
            try
            {
                _motion.Controller.Connect();

                // 실제 연결된 축 수 결정
                // - WMX3: AxesStatus에서 ServoOffline=false인 축만 카운트
                // - Ajin : GetAxisCount() 직접 사용 (AxesStatus 구조가 다름)
                var status = _motion.Controller.GetStatus();
                int detectedCount = status.AxesStatus?.Count(a => !a.ServoOffline) ?? 0;
                if (detectedCount <= 0 || detectedCount > 64)
                {
                    // Ajin: GetAxisCount() 직접 사용
                    var ajin = _motion.Controller.AsAjin;
                    if (ajin != null)
                        detectedCount = ajin.GetAxisCount();

                    // Simulation: 사용자가 선택한 축 수 직접 사용
                    var sim = _motion.Controller.AsSimulation;
                    if (sim != null)
                        detectedCount = sim.GetAxisCount();
                }
                _axisCount = Math.Max(1, detectedCount);
                AxisConfig.AxisCount = _axisCount;

                // 데이터 버퍼 준비
                _checkedArr  = new bool[_axisCount];
                _posArr      = new float[_axisCount];
                _servoArr    = new bool[_axisCount];
                _alarmOkArr  = new bool[_axisCount];
                _opArr       = Enumerable.Repeat("Idle", _axisCount).ToArray();
                _targetArr   = new double[_axisCount];  // 초기값 0
                _grid.RowCount = _axisCount;
                _grid.ClearSelection();
                UpdateCheckedAxesLabel();

                // 모니터 시작
                _monitor = new AxisMonitor(_motion.Controller, null, _axisCount, _axisConfigs);
                _monitor.PositionUpdated += AppEvents.RaiseSimulatorPositionUpdate;
                _monitor.PositionUpdated += OnPositionsUpdated;                 // float[]
                _monitor.StatusUpdated += OnStatusUpdated;                    // (axis,servo,ok,op)
                _monitor.Start();

                // 시뮬레이터 초기화
                var maxPositions = _axisConfigs
                    .Take(_axisCount)
                    .Select(cfg => (float)cfg.PositionMax)
                    .ToArray();
                AppEvents.RaiseSimulatorInitialize(maxPositions);

                // UI 상태 갱신
                lblStatus.Text = "Status: Connected";
                lblStatus.ForeColor = Color.DarkGreen;
                btnConnect.Enabled = false;
                btnDisconnect.Enabled = true;

                AppEvents.RaiseLog("Controller connected.");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[에러] 연결 실패: {ex.Message}");
                MessageBox.Show("컨트롤러 연결 중 오류가 발생했습니다.", "연결 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void BtnDisconnect_Click(object sender, EventArgs e)
        {
            try
            {
                _monitor?.Stop(); _monitor = null;
                _motion.Controller.Disconnect();
                AxisConfig.AxisCount = 0;

                _grid.RowCount = 0;
                _checkedArr = null; _posArr = null; _servoArr = null; _alarmOkArr = null; _opArr = null; _targetArr = null;
                _selectedAxis = -1;
                UpdateCheckedAxesLabel();
                SetActionButtonsEnabled(false);

                if (_chkRealtime.Checked) _chkRealtime.Checked = false;

                AppEvents.RaiseRequestClearSimulator();
                AppEvents.RaiseLog("Controller disconnected.");

                lblStatus.Text = "Status: Disconnected";
                lblStatus.ForeColor = Color.DarkRed;
                btnConnect.Enabled = true;
                btnDisconnect.Enabled = false;
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[에러] 연결 해제 실패: {ex.Message}");
                MessageBox.Show("컨트롤러 연결 해제 중 오류가 발생했습니다.", "해제 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Grid_CellValueNeeded(object sender, DataGridViewCellValueEventArgs e)
        {
            if (_axisCount == 0 || e.RowIndex < 0) return;

            int r = e.RowIndex;
            switch (e.ColumnIndex)
            {
                case COL_SEL:   e.Value = (_checkedArr != null && r < _checkedArr.Length) ? _checkedArr[r] : false; break;
                case COL_AXIS:  e.Value = r.ToString(); break;
                case COL_POS:   e.Value = (r < _posArr?.Length) ? _posArr[r] : 0f; break;
                case COL_SERVO: e.Value = (r < _servoArr?.Length && _servoArr[r]) ? "● ON" : "● OFF"; break;
                case COL_ALARM: e.Value = (r < _alarmOkArr?.Length && _alarmOkArr[r]) ? "● OK" : "● ALM"; break;
                case COL_OP:    e.Value = (r < _opArr?.Length) ? _opArr[r] : "Idle"; break;
                case COL_TGT:   e.Value = (r < _targetArr?.Length) ? _targetArr[r].ToString("0.###") : "0"; break;
            }
        }

        private void Grid_CellValuePushed(object sender, DataGridViewCellValueEventArgs e)
        {
            if (e.RowIndex < 0) return;

            if (e.ColumnIndex == COL_SEL && _checkedArr != null && e.RowIndex < _checkedArr.Length)
            {
                bool v = false;
                if (e.Value is bool b) v = b;
                _checkedArr[e.RowIndex] = v;
                UpdateCheckedAxesLabel();
            }
            else if (e.ColumnIndex == COL_TGT && _targetArr != null && e.RowIndex < _targetArr.Length)
            {
                if (double.TryParse(e.Value?.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out double val))
                    _targetArr[e.RowIndex] = val;
            }
        }

        private void Grid_CellFormatting(object sender, DataGridViewCellFormattingEventArgs e)
        {
            if (e.RowIndex < 0) return;
            var cell = _grid.Rows[e.RowIndex].Cells[e.ColumnIndex];

            if (e.ColumnIndex == COL_SERVO)
            {
                bool on = _servoArr != null && e.RowIndex < _servoArr.Length && _servoArr[e.RowIndex];
                cell.Style.ForeColor = on ? Color.SeaGreen : Color.Gray;
                cell.Style.Font = new Font(_grid.Font, FontStyle.Bold);
                e.FormattingApplied = true;
            }
            else if (e.ColumnIndex == COL_ALARM)
            {
                bool ok = _alarmOkArr != null && e.RowIndex < _alarmOkArr.Length && _alarmOkArr[e.RowIndex];
                cell.Style.ForeColor = ok ? Color.SeaGreen : Color.Firebrick;
                cell.Style.Font = new Font(_grid.Font, FontStyle.Bold);
                e.FormattingApplied = true;
            }
            else if (e.ColumnIndex == COL_OP)
            {
                cell.Style.ForeColor = SystemColors.ControlText;
                cell.Style.Font = new Font(_grid.Font, FontStyle.Regular);
                e.FormattingApplied = true;
            }
        }

        private static void MakeDgvDoubleBuffered(DataGridView dgv)
        {
            try
            {
                typeof(DataGridView).InvokeMember("DoubleBuffered",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.SetProperty,
                    null, dgv, new object[] { true });
            }
            catch { /* ignore */ }
        }

        private void OnPositionsUpdated(float[] positions)
        {
            if (positions == null) return;
            if (InvokeRequired) { BeginInvoke(new Action(() => OnPositionsUpdated(positions))); return; }

            int n = Math.Min(_axisCount, positions.Length);
            for (int i = 0; i < n; i++) _posArr[i] = positions[i];

            _grid.InvalidateColumn(COL_POS);
        }

        private void OnStatusUpdated(int axis, bool servoOn, bool alarmOk, string operation)
        {
            if (InvokeRequired) { BeginInvoke(new Action(() => OnStatusUpdated(axis, servoOn, alarmOk, operation))); return; }
            if (axis < 0 || axis >= _axisCount) return;

            _servoArr[axis] = servoOn;
            _alarmOkArr[axis] = alarmOk;
            _opArr[axis] = string.IsNullOrEmpty(operation) ? "Idle" : operation;

            _grid.InvalidateRow(axis);
        }

        // ======================= 선택/액션/모션 =======================
        private void SelectAxis(int axisIndex)
        {
            if (axisIndex < 0 || axisIndex >= _axisCount) return;

            _selectedAxis = axisIndex;
            SetActionButtonsEnabled(true);
            // 파라미터 패널 값 로드
            LoadAxisParametersToPanel(axisIndex);
        }

        private void ToggleCheck(int rowIndex)
        {
            if (_checkedArr == null || rowIndex < 0 || rowIndex >= _checkedArr.Length) return;
            _checkedArr[rowIndex] = !_checkedArr[rowIndex];
            _grid.InvalidateRow(rowIndex);
            UpdateCheckedAxesLabel();
        }

        private int[] CheckedAxes()
        {
            if (_checkedArr == null) return Array.Empty<int>();
            var list = new List<int>();
            for (int i = 0; i < _checkedArr.Length; i++) if (_checkedArr[i]) list.Add(i);
            return list.ToArray();
        }

        private void UpdateCheckedAxesLabel()
        {
            if (lblCheckedAxes == null) return;
            var arr = CheckedAxes();
            lblCheckedAxes.Text = (arr.Length > 0) ? $"Selected Axes: {string.Join(", ", arr)}" : "Selected Axes: -";
        }

        private Panel CreateParameterPanel()
        {
            var panel = new Panel
            {
                BackColor = Color.LightGray
            };

            txtMaxVel = CreateTextBox(panel, "MaxVel(mm/s)", 10, 10);
            txtAcc = CreateTextBox(panel, "Acc(mm/s²)", 10, 40);
            txtDec = CreateTextBox(panel, "Dec(mm/s²)", 10, 70);
            txtMaxStroke = CreateTextBox(panel, "Max Stroke", 200, 10);
            txtPitch = CreateTextBox(panel, "Pitch(mm/rev)", 200, 40);

            var btnApply = new Button { Text = "Apply to Axis", Left = 200, Top = 70, Width = 120 };
            btnApply.Click += (s, e) => ApplyParameters();
            panel.Controls.Add(btnApply);

            return panel;
        }

        private Panel CreateActionPanel()
        {
            const int CTRL_W = 180;
            const int BTN_H = 28;

            var root = new FlowLayoutPanel
            {
                BackColor = Color.Gainsboro,
                Padding = new Padding(8),
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink
            };

            lblCheckedAxes = new Label
            {
                Text = "Selected Axes: -",
                AutoSize = true,
                Font = new Font("Segoe UI", 10, FontStyle.Bold),
                Margin = new Padding(0, 0, 0, 8)
            };

            var lblTarget = new Label
            {
                Text = "Target (mm)",
                AutoSize = true,
                Margin = new Padding(0, 4, 0, 2)
            };

            // ===== NumericUpDown Target =====
            var numTarget = new NumericUpDown
            {
                Width = CTRL_W,
                DecimalPlaces = 3,
                Increment = 10M,
                Minimum = -99999.999M,
                Maximum = 99999.999M,
                TextAlign = HorizontalAlignment.Right,
                Margin = new Padding(0, 0, 0, 8)
            };
            // 내부 TextBox 참조(기존 코드 호환)
            txtTargetGlobal = numTarget.Controls
                .OfType<TextBox>()
                .FirstOrDefault() ?? new TextBox();

            // === 버튼 2x2 ===
            var grid = new TableLayoutPanel
            {
                ColumnCount = 2,
                RowCount = 3,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink
            };
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50f));
            grid.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            grid.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            grid.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            btnServoOnGlobal = new Button { Text = "Servo ON", Height = BTN_H, Dock = DockStyle.Fill, Margin = new Padding(0, 4, 6, 4) };
            btnServoOffGlobal = new Button { Text = "Servo OFF", Height = BTN_H, Dock = DockStyle.Fill, Margin = new Padding(6, 4, 0, 4) };
            btnAbsMoveGlobal = new Button { Text = "ABS Move", Height = BTN_H, Dock = DockStyle.Fill, Margin = new Padding(0, 6, 6, 0) };
            btnRelMoveGlobal = new Button { Text = "REL Move", Height = BTN_H, Dock = DockStyle.Fill, Margin = new Padding(6, 6, 0, 0) };

            // Set Zero: 체크된 축 모두의 현재 위치를 0으로 설정
            var btnSetZero = new Button
            {
                Text = "Set Zero",
                Height = BTN_H,
                Dock = DockStyle.Fill,
                Margin = new Padding(0, 6, 0, 0),
                BackColor = Color.MistyRose
            };
            btnSetZero.Click += (s, e) =>
            {
                var axes = CheckedAxes();
                if (axes.Length == 0 && _selectedAxis >= 0)
                    axes = new[] { _selectedAxis };

                if (axes.Length == 0)
                {
                    MessageBox.Show("축을 체크하거나 선택하세요.", "Set Zero", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                var ajin = _motion?.Controller?.AsAjin;
                if (ajin == null)
                {
                    MessageBox.Show("Set Zero는 Ajin 제어기에서만 지원됩니다.", "Set Zero", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                foreach (int ax in axes)
                {
                    ajin.SetZeroPos(ax);
                    AppEvents.RaiseLog($"[Set Zero] Axis {ax} 위치 초기화");
                }
            };

            btnServoOnGlobal.Click += (s, e) => { if (_selectedAxis >= 0) _motion.Controller.SetServo(_selectedAxis, true); };
            btnServoOffGlobal.Click += (s, e) => { if (_selectedAxis >= 0) _motion.Controller.SetServo(_selectedAxis, false); };
            btnAbsMoveGlobal.Click += async (s, e) => await MultiMove(true);
            btnRelMoveGlobal.Click += async (s, e) => await MultiMove(false);

            SetActionButtonsEnabled(false);

            grid.RowCount = 3;
            grid.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            grid.Controls.Add(btnServoOnGlobal, 0, 0);
            grid.Controls.Add(btnServoOffGlobal, 1, 0);
            grid.Controls.Add(btnAbsMoveGlobal, 0, 1);
            grid.Controls.Add(btnRelMoveGlobal, 1, 1);

            // Set Zero는 2행 전체(colspan=2)
            grid.SetColumnSpan(btnSetZero, 2);
            grid.Controls.Add(btnSetZero, 0, 2);

            // Alarm Clear
            btnAlarmClear = new Button
            {
                Text = "Alarm Clear",
                Height = BTN_H,
                Dock = DockStyle.Fill,
                Margin = new Padding(0, 6, 0, 0),
                BackColor = Color.LightSalmon
            };
            btnAlarmClear.Click += async (s, e) =>
            {
                var axes = CheckedAxes();
                if (axes.Length == 0 && _selectedAxis >= 0)
                    axes = new[] { _selectedAxis };

                if (axes.Length == 0)
                {
                    MessageBox.Show("축을 체크하거나 선택하세요.", "Alarm Clear",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                var ajin = _motion?.Controller?.AsAjin;
                if (ajin == null)
                {
                    MessageBox.Show("Alarm Clear는 Ajin 제어기에서만 지원됩니다.",
                        "Alarm Clear", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }

                btnAlarmClear.Enabled = false;
                try
                {
                    await Task.Run(() =>
                    {
                        foreach (int ax in axes)
                            ajin.ClearAlarm(ax);
                    });
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"알람 클리어 실패: {ex.Message}", "오류",
                        MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    btnAlarmClear.Enabled = true;
                }
            };
            grid.RowCount = 4;
            grid.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            grid.SetColumnSpan(btnAlarmClear, 2);
            grid.Controls.Add(btnAlarmClear, 0, 3);

            root.Controls.Add(lblCheckedAxes);
            root.Controls.Add(lblTarget);
            root.Controls.Add(numTarget);
            root.Controls.Add(grid);

            return root;
        }

        private async Task MultiMove(bool isAbs)
        {
            var axes = CheckedAxes();
            if (axes.Length == 0)
            {
                MessageBox.Show("체크된 축이 없습니다.", "안내", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            double[] targets;

            if (axes.Length == 1)
            {
                // 단일 축: 공용 NumericUpDown 사용
                if (!double.TryParse(txtTargetGlobal.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out double singleVal))
                {
                    MessageBox.Show("Target 값이 유효하지 않습니다.", "입력 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    return;
                }
                targets = new[] { singleVal };
            }
            else
            {
                // 다축: 그리드 각 행의 Target(mm) 컬럼 값 사용
                targets = axes.Select(ax =>
                    (_targetArr != null && ax < _targetArr.Length) ? _targetArr[ax] : 0.0
                ).ToArray();
            }

            SetActionButtonsEnabled(false);
            try
            {
                await _motion.RunMotionWithLogging(axes, isAbs, targets);
            }
            finally
            {
                SetActionButtonsEnabled(true);
            }
        }

        private void SetActionButtonsEnabled(bool enabled)
        {
            btnServoOnGlobal.Enabled = enabled && (_selectedAxis >= 0);
            btnServoOffGlobal.Enabled = enabled && (_selectedAxis >= 0);
            btnAbsMoveGlobal.Enabled = enabled;
            btnRelMoveGlobal.Enabled = enabled;
            if (btnAlarmClear != null) btnAlarmClear.Enabled = enabled;
        }

        private TextBox CreateTextBox(Panel panel, string label, int left, int top)
        {
            panel.Controls.Add(new Label { Text = label, Left = left, Top = top, Width = 110 });
            var txt = new TextBox { Left = left + 120, Top = top, Width = 60 };
            panel.Controls.Add(txt);
            return txt;
        }

        private void LoadAxisParametersToPanel(int axisIndex)
        {
            if (_axisConfigs == null || axisIndex < 0 || axisIndex >= _axisConfigs.Length) return;

            var cfg = _axisConfigs[axisIndex];
            txtMaxVel.Text = cfg.MaxVel.ToString("0.###");
            txtAcc.Text = cfg.Acc.ToString("0.###");
            txtDec.Text = cfg.Dec.ToString("0.###");
            txtMaxStroke.Text = cfg.PositionMax.ToString("0.###");
            txtPitch.Text = cfg.PitchMmPerRev.ToString("0.###");
        }

        private void ApplyParameters()
        {
            if (_selectedAxis < 0 || _axisConfigs == null) return;

            var cfg = _axisConfigs[_selectedAxis];

            cfg.MaxVel = double.TryParse(txtMaxVel.Text, out var v1) ? v1 : cfg.MaxVel;
            cfg.Acc = double.TryParse(txtAcc.Text, out var v2) ? v2 : cfg.Acc;
            cfg.Dec = double.TryParse(txtDec.Text, out var v3) ? v3 : cfg.Dec;
            cfg.PositionMax = double.TryParse(txtMaxStroke.Text, out var v4) ? v4 : cfg.PositionMax;
            cfg.PitchMmPerRev = double.TryParse(txtPitch.Text, out var v5) ? v5 : cfg.PitchMmPerRev;

            _motion.SetAxisConfigs(_axisConfigs);
            AppEvents.RaiseSimulatorMaxPositionUpdate(_selectedAxis, (float)cfg.PositionMax);
            LoadAxisParametersToPanel(_selectedAxis);

            MessageBox.Show($"Parameters applied to Axis {_selectedAxis}");
        }
    }
}