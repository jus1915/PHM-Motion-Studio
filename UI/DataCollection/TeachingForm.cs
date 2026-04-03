using PHM_Project_DockPanel.Services;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using WeifenLuo.WinFormsUI.Docking;
using WMX3ApiCLR;
using System.Drawing;

namespace PHM_Project_DockPanel.Windows
{
    public class TeachingForm : DockContent
    {
        private PHM_Motion _motion;

        private StatusStrip _statusStrip;
        private ToolStripStatusLabel _iterStatusLabel;
        private NumericUpDown _repeatUpDown;
        private CheckBox _loopCheck;
        private Button _runButton, _stopButton, _saveButton, _loadButton;
        private DataGridView _teachingGrid;

        private CancellationTokenSource _teachCts;

        // --- 컬럼 인덱스/이름(고정) ---
        private const int COL_RUN = 0;
        private const int COL_MODE = 1;     // "Single" | "Multi"
        private const int COL_AXIS = 2;     // Single에서 사용
        private const int COL_SINGLE_TGT = 3; // Single Target
        private const int COL_VEL_OVERRIDE = 4; // 속도 배율 (%)
        // 다축 타겟은 이후에 동적 생성 (이들 다음에 들어감)
        // 마지막에 Wait(ms) 컬럼을 둔다

        private const string MODE_SINGLE = "Single";
        private const string MODE_MULTI = "Multi";
        private const string MULTI_COL_PREFIX = "T"; // T0, T1, ...
        private const string DefaultTeachingDir = @"C:\Data\PHM_Logs\Teaching";

        public TeachingForm(PHM_Motion pHM_motion)
        {
            _motion = pHM_motion;
            Text = "Teaching";
            TabText = "Teaching";
            InitializeTeachingPanel();

            // 폼이 보일 때마다 실제 축 수로 컬럼 갱신 (Connect/Disconnect 이후 반영)
            this.VisibleChanged += (s, e) =>
            {
                if (this.Visible) EnsureDynamicTargetColumns();
            };
        }

        private void InitializeTeachingPanel()
        {
            // DPI에서 클리핑 방지
            this.AutoScaleMode = AutoScaleMode.Dpi;

            BackColor = Color.LightSteelBlue;
            Padding = new Padding(6);

            // ── 메인: 상(툴바 AutoSize) / 중(그리드) / 하(컨트롤 AutoSize) / 상태바 ──
            var main = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 4
            };
            main.RowStyles.Add(new RowStyle(SizeType.AutoSize));         // 상단 툴바
            main.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));    // 그리드
            main.RowStyles.Add(new RowStyle(SizeType.AutoSize));         // 하단 컨트롤
            main.RowStyles.Add(new RowStyle(SizeType.Absolute, 24F));    // 상태바

            // ── 상단 툴바 ──
            var topBar = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false,
                Padding = new Padding(10, 8, 10, 8),
                Margin = new Padding(0, 0, 0, 6),
                BackColor = Color.AliceBlue
            };
            var btnAdd = new Button { Text = "+", Width = 40, Margin = new Padding(5) };
            var btnRemove = new Button { Text = "-", Width = 40, Margin = new Padding(5) };
            _saveButton = new Button { Text = "Save", Width = 80, Margin = new Padding(5) };
            _loadButton = new Button { Text = "Load", Width = 80, Margin = new Padding(5) };

            btnAdd.Click += (s, e) =>
            {
                EnsureDynamicTargetColumns();
                var newRow = _teachingGrid.Rows[_teachingGrid.Rows.Add(false, MODE_SINGLE, 0, 0.0)];
                newRow.Cells[COL_VEL_OVERRIDE].Value = 100;
                var waitCol = FindWaitColumnIndex();
                if (waitCol >= 0)
                    _teachingGrid.Rows[_teachingGrid.Rows.Count - 1].Cells[waitCol].Value = 500;
                ApplyMultiColsVisibilityByMode(); // ← 추가
            };

            btnRemove.Click += (s, e) =>
            {
                if (_teachingGrid.SelectedRows.Count > 0)
                {
                    foreach (DataGridViewRow r in _teachingGrid.SelectedRows)
                        if (!r.IsNewRow) _teachingGrid.Rows.Remove(r);
                }
                else MessageBox.Show("삭제할 행을 선택하세요.");

                ApplyMultiColsVisibilityByMode(); // ← 추가
            };
            _saveButton.Click += (s, e) => SaveTeachingSequence();
            _loadButton.Click += (s, e) => LoadTeachingSequence();

            topBar.Controls.AddRange(new Control[] { btnAdd, btnRemove, _saveButton, _loadButton });

            // ── 그리드 ──
            _teachingGrid = new DataGridView
            {
                Dock = DockStyle.Fill,
                AllowUserToAddRows = false,
                RowHeadersVisible = false,
                AutoGenerateColumns = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                BackgroundColor = Color.White
            };

            if (_teachingGrid.Columns.Count == 0)
            {
                // [Run]
                _teachingGrid.Columns.Add(new DataGridViewCheckBoxColumn { HeaderText = "Run", Width = 50 });

                // [Mode]
                var modeCol = new DataGridViewComboBoxColumn
                {
                    HeaderText = "Mode",
                    Width = 80,
                    FlatStyle = FlatStyle.Flat,
                    DisplayStyle = DataGridViewComboBoxDisplayStyle.DropDownButton
                };
                modeCol.Items.AddRange(new[] { MODE_SINGLE, MODE_MULTI });
                _teachingGrid.Columns.Add(modeCol);

                // [Axis] (Single에서 사용)
                _teachingGrid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Axis", Width = 50 });

                // [Target(mm)] (Single에서 사용)
                _teachingGrid.Columns.Add(new DataGridViewTextBoxColumn { HeaderText = "Target(mm)", Width = 100 });

                // [VelOverride(%)] 행별 속도 배율 (10~100%, 기본 100)
                _teachingGrid.Columns.Add(new DataGridViewTextBoxColumn
                {
                    Name = "VelOverride",
                    HeaderText = "Vel(%)",
                    Width = 60,
                    DefaultCellStyle = new DataGridViewCellStyle
                    {
                        Alignment = DataGridViewContentAlignment.MiddleCenter,
                        BackColor = Color.LightYellow
                    }
                });

                // 동적 다축 타겟 컬럼 생성
                EnsureDynamicTargetColumns();

                // [Wait(ms)] (항상 마지막)
                EnsureWaitColumn();

                // 초기 예시 2행
                for (int i = 0; i < 2; i++)
                {
                    var r = new object[_teachingGrid.Columns.Count];
                    r[COL_RUN] = false;
                    r[COL_MODE] = MODE_SINGLE;
                    r[COL_AXIS] = 0;
                    r[COL_SINGLE_TGT] = 0.0;
                    r[COL_VEL_OVERRIDE] = 100;
                    var waitCol = FindWaitColumnIndex();
                    if (waitCol >= 0) r[waitCol] = 500;
                    _teachingGrid.Rows.Add(r);
                }

                // 모드 변경 시 시각적 도움(선택): Single/Multi에 따라 셀 배경 톤 변경
                _teachingGrid.CellFormatting += (s, e) =>
                {
                    if (e.RowIndex < 0) return;
                    var mode = Convert.ToString(_teachingGrid.Rows[e.RowIndex].Cells[COL_MODE].Value) ?? MODE_SINGLE;
                    bool isMulti = mode == MODE_MULTI;

                    if (e.ColumnIndex == COL_AXIS || e.ColumnIndex == COL_SINGLE_TGT)
                    {
                        _teachingGrid.Rows[e.RowIndex].Cells[e.ColumnIndex].Style.BackColor =
                            isMulti ? Color.WhiteSmoke : Color.White;
                    }
                    else if (IsMultiTargetColumn(e.ColumnIndex))
                    {
                        _teachingGrid.Rows[e.RowIndex].Cells[e.ColumnIndex].Style.BackColor =
                            isMulti ? Color.White : Color.WhiteSmoke;
                    }
                };
            }
            // 모드 콤보 바로 반영되도록
            _teachingGrid.CurrentCellDirtyStateChanged += (s, e) =>
            {
                if (_teachingGrid.IsCurrentCellDirty &&
                    _teachingGrid.CurrentCell.ColumnIndex == COL_MODE)
                    _teachingGrid.CommitEdit(DataGridViewDataErrorContexts.Commit);
            };

            // 모드 값 바뀔 때마다 다축 컬럼 재구성 + 보이기/숨기기 갱신
            _teachingGrid.CellValueChanged += (s, e) =>
            {
                if (e.RowIndex >= 0 && e.ColumnIndex == COL_MODE)
                {
                    EnsureDynamicTargetColumns(); // 축 수가 바뀌었을 수 있으므로 재구성
                    ApplyMultiColsVisibilityByMode();
                }
            };

            // 행 추가/삭제 시도 후에도 갱신
            _teachingGrid.RowsAdded += (s, e) => ApplyMultiColsVisibilityByMode();
            _teachingGrid.RowsRemoved += (s, e) => ApplyMultiColsVisibilityByMode();

            // 초기 2행 만든 다음 1회 적용
            ApplyMultiColsVisibilityByMode();
            // ── 하단 컨트롤: Loop/Repeat + Run/Stop ──
            var bottomBar = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                AutoSizeMode = AutoSizeMode.GrowAndShrink,
                FlowDirection = FlowDirection.LeftToRight,
                WrapContents = false,
                Padding = new Padding(10, 6, 10, 6),
                Margin = new Padding(0)
            };

            // Loop/Repeat
            _loopCheck = new CheckBox { Text = "Loop", Width = 60, Margin = new Padding(5) };
            var lblRepeat = new Label { Text = "Repeat", AutoSize = true, Margin = new Padding(8, 8, 0, 0) };
            _repeatUpDown = new NumericUpDown { Minimum = 1, Maximum = 100000, Value = 1, Width = 80, Margin = new Padding(5) };
            _loopCheck.CheckedChanged += (s, e) => _repeatUpDown.Enabled = !_loopCheck.Checked;
            _repeatUpDown.Enabled = !_loopCheck.Checked;

            // Run/Stop
            _runButton = new Button { Text = "Run", Width = 90, Margin = new Padding(12, 6, 6, 6) };
            _stopButton = new Button { Text = "Stop", Width = 90, Margin = new Padding(6) };
            _runButton.Click += async (s, e) => await RunTeachingSequence();
            _stopButton.Click += (s, e) => _teachCts?.Cancel();

            // (선택) 시각적 간격용 더미
            var spacer = new Label { AutoSize = false, Width = 24, Height = 1, Margin = new Padding(0) };

            bottomBar.Controls.AddRange(new Control[] { _loopCheck, lblRepeat, _repeatUpDown, spacer, _runButton, _stopButton });

            // ── 상태바 ──
            _statusStrip = new StatusStrip { SizingGrip = false };
            _iterStatusLabel = new ToolStripStatusLabel("반복 0/0");
            _statusStrip.Items.Add(_iterStatusLabel);

            // ── 조립 ──
            main.Controls.Add(topBar, 0, 0);
            main.Controls.Add(_teachingGrid, 0, 1);
            main.Controls.Add(bottomBar, 0, 2);
            main.Controls.Add(_statusStrip, 0, 3);

            Controls.Add(main);
        }

        /// <summary>연결 후 실제 축 수. 연결 전이면 AxisConfigs 길이로 폴백.</summary>
        private int GetCurrentAxisCount()
        {
            if (AxisConfig.AxisCount > 0) return AxisConfig.AxisCount;
            return _motion?.AxisConfigs?.Length ?? 0;
        }

        // ===== 동적 다축 타겟 컬럼 보장 =====
        private void EnsureDynamicTargetColumns()
        {
            int axisCount = GetCurrentAxisCount();
            if (axisCount <= 0) { EnsureWaitColumn(); return; }

            // Wait 컬럼은 절대 제거하지 않음
            EnsureWaitColumn();
            int waitColIdx = FindWaitColumnIndex();

            // 이미 존재하는 다축 타겟 컬럼 이름 모음
            var existing = new HashSet<string>(
                _teachingGrid.Columns
                    .Cast<DataGridViewColumn>()
                    .Select(c => c.Name)
                    .Where(n => n != null && n.StartsWith(MULTI_COL_PREFIX, StringComparison.OrdinalIgnoreCase))
            );

            // 필요한 만큼 생성 (T0..T{axisCount-1})
            for (int ax = 0; ax < axisCount; ax++)
            {
                string name = MULTI_COL_PREFIX + ax.ToString();
                if (!existing.Contains(name))
                {
                    var col = new DataGridViewTextBoxColumn
                    {
                        Name = name,
                        HeaderText = $"A{ax}(mm)",
                        Width = 90
                    };
                    _teachingGrid.Columns.Add(col);
                }
            }

            // 축 개수가 줄어들었다면 남는 다축 컬럼 제거 (Wait 영향 없음)
            var toRemove = _teachingGrid.Columns
                .Cast<DataGridViewColumn>()
                .Where(c => c.Name != null &&
                            c.Name.StartsWith(MULTI_COL_PREFIX, StringComparison.OrdinalIgnoreCase) &&
                            GetAxisIndexFromMultiColName(c.Name) >= axisCount)
                .Select(c => c.Name)
                .ToList();
            foreach (var n in toRemove) _teachingGrid.Columns.Remove(n);

            // 다축 컬럼들을 Wait 앞에 A0→A1→…→A{n-1} 순으로 배치
            // ※ 역순(마지막→첫번째)으로 DisplayIndex를 같은 위치에 넣으면
            //   T{n-1}부터 T0 순으로 앞쪽 자리를 차지해 결과적으로 T0,T1,…,T{n-1} 순이 됩니다.
            waitColIdx = FindWaitColumnIndex();
            if (waitColIdx >= 0)
            {
                var multiColsSorted = _teachingGrid.Columns
                    .Cast<DataGridViewColumn>()
                    .Where(c => c.Name != null &&
                                c.Name.StartsWith(MULTI_COL_PREFIX, StringComparison.OrdinalIgnoreCase))
                    .OrderBy(c => GetAxisIndexFromMultiColName(c.Name))
                    .ToList();

                int baseDisplay = _teachingGrid.Columns[waitColIdx].DisplayIndex;
                for (int i = multiColsSorted.Count - 1; i >= 0; i--)
                    multiColsSorted[i].DisplayIndex = baseDisplay;
            }
        }


        private bool IsMultiTargetColumn(int colIndex)
        {
            if (colIndex < 0 || colIndex >= _teachingGrid.Columns.Count) return false;
            var c = _teachingGrid.Columns[colIndex];
            return c?.Name != null && c.Name.StartsWith(MULTI_COL_PREFIX, StringComparison.OrdinalIgnoreCase);
        }

        private int GetAxisIndexFromMultiColName(string name)
        {
            if (string.IsNullOrEmpty(name)) return -1;
            if (!name.StartsWith(MULTI_COL_PREFIX, StringComparison.OrdinalIgnoreCase)) return -1;
            if (int.TryParse(name.Substring(MULTI_COL_PREFIX.Length), out int ax)) return ax;
            return -1;
        }

        private int FindWaitColumnIndex()
        {
            for (int i = 0; i < _teachingGrid.Columns.Count; i++)
            {
                if (_teachingGrid.Columns[i].HeaderText.Equals("Wait(ms)", StringComparison.OrdinalIgnoreCase))
                    return i;
            }
            return -1;
        }

        private void EnsureWaitColumn()
        {
            if (FindWaitColumnIndex() >= 0) return;

            var col = new DataGridViewTextBoxColumn
            {
                Name = "Wait",                       // 이름을 지정해두면 찾기도 편해요
                HeaderText = "Wait(ms)",
                Width = 100,
                ValueType = typeof(int),
                DefaultCellStyle = { Alignment = DataGridViewContentAlignment.MiddleRight, Format = "N0" } // 천단위 표시
            };
            _teachingGrid.Columns.Add(col);
        }

        private async Task RunTeachingSequence()
        {
            if (_teachCts != null) return;

            _teachingGrid.EndEdit(DataGridViewDataErrorContexts.Commit);
            this.Validate();

            _teachCts = new CancellationTokenSource();
            var token = _teachCts.Token;

            bool abort = false;
            SetUiRunning(true);

            try
            {
                EnsureDynamicTargetColumns();

                if (!TryReadStepsFromGrid(out var plan)) { abort = true; return; }

                var steps = plan.Steps.Where(s => s.Enabled).ToList();
                if (steps.Count == 0)
                {
                    MessageBox.Show("선택된 티칭 행이 없습니다.", "실행 불가",
                                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    abort = true; return;
                }

                int total = plan.Loop ? int.MaxValue : Math.Max(1, plan.RepeatCount);
                int iterDone = 0;

                // 초기 상태 (0/전체)
                UpdateIterationStatus(iterDone, plan.Loop ? -1 : total);

                while (!token.IsCancellationRequested && !abort && iterDone < total)
                {
                    // 이번 회차 시작 표시
                    UpdateIterationStatus(iterDone + 1, plan.Loop ? -1 : total);

                    foreach (var step in steps)
                    {
                        // 속도 배율 적용: AxisConfigs의 MaxVel을 임시로 스케일
                        double ratio = (step.VelOverride > 0 && step.VelOverride < 100)
                            ? step.VelOverride / 100.0
                            : 1.0;
                        double[] origVels = null;
                        if (ratio < 1.0 && _motion.AxisConfigs != null)
                        {
                            origVels = _motion.AxisConfigs.Select(c => c?.MaxVel ?? 0).ToArray();
                            foreach (var cfg in _motion.AxisConfigs)
                                if (cfg != null) cfg.MaxVel *= ratio;
                        }

                        bool success;
                        try
                        {
                            if (step.IsMulti)
                            {
                                var axesList = new List<int>();
                                var valsList = new List<double>();
                                int axisCount = GetCurrentAxisCount();

                                for (int ax = 0; ax < axisCount; ax++)
                                {
                                    double? v = (step.Targets != null && ax < step.Targets.Length) ? step.Targets[ax] : null;
                                    if (v.HasValue) { axesList.Add(ax); valsList.Add(v.Value); }
                                }

                                if (axesList.Count == 0)
                                {
                                    AppEvents.RaiseLog("[경고] 다축 행에 유효 타겟이 없습니다. 스킵.");
                                    continue;
                                }

                                success = await _motion.RunMotionWithLogging(
                                    axesList.ToArray(), true, valsList.ToArray(), null, plan.Loop);
                            }
                            else
                            {
                                if (step.Axis < 0 || step.Axis >= GetCurrentAxisCount())
                                {
                                    MessageBox.Show($"Axis {step.Axis} 인덱스가 잘못되었습니다.", "입력 오류",
                                                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                                    abort = true; break;
                                }

                                success = await _motion.RunMotionWithLogging(
                                    new[] { step.Axis }, true, step.Target, null, plan.Loop);
                            }
                        }
                        finally
                        {
                            // 속도 원복
                            if (origVels != null && _motion.AxisConfigs != null)
                            {
                                for (int i = 0; i < Math.Min(origVels.Length, _motion.AxisConfigs.Length); i++)
                                    if (_motion.AxisConfigs[i] != null) _motion.AxisConfigs[i].MaxVel = origVels[i];
                            }
                        }

                        if (!success || token.IsCancellationRequested) { abort = true; break; }

                        if (step.WaitMs > 0)
                        {
                            try { await Task.Delay(step.WaitMs, token); }
                            catch (TaskCanceledException) { abort = true; break; }
                        }
                    }

                    if (abort) break;
                    iterDone++;

                    if (plan.Loop)   // 무한 반복이면 done 카운트만 증가(표시는 n/∞)
                        UpdateIterationStatus(iterDone, -1);
                }
            }
            catch (OperationCanceledException)
            {
                AppEvents.RaiseLog("[중단] 사용자 요청에 의해 Teaching 루프가 취소되었습니다.");
            }
            catch (Exception ex)
            {
                AppEvents.RaiseLog($"[예외] Teaching 실행 중 오류 발생: {ex.Message}");
            }
            finally
            {
                SetUiRunning(false);
                _teachCts?.Dispose();
                _teachCts = null;
            }
        }

        private bool TryReadStepsFromGrid(out TeachingPlan plan)
        {
            EnsureDynamicTargetColumns();

            plan = new TeachingPlan { Loop = _loopCheck.Checked, RepeatCount = (int)_repeatUpDown.Value };

            int axisCount = GetCurrentAxisCount();

            foreach (DataGridViewRow row in _teachingGrid.Rows)
            {
                if (row.IsNewRow) continue;

                // Run
                bool enabled = row.Cells[COL_RUN].Value is bool b && b;

                // Mode
                string mode = Convert.ToString(row.Cells[COL_MODE].Value) ?? MODE_SINGLE;
                bool isMulti = (mode == MODE_MULTI);

                // Wait
                int waitMs;
                {
                    int waitCol = FindWaitColumnIndex();
                    if (waitCol < 0 || !TryParseWaitMs(row.Cells[waitCol].Value, out waitMs))
                    {
                        MessageBox.Show("Wait(ms) 입력이 올바르지 않습니다. (예: 500, 1,000, 500.0, 500ms)",
                                        "입력 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return false;
                    }
                }

                // VelOverride (%)
                int velOverride = 100;
                {
                    var velVal = row.Cells[COL_VEL_OVERRIDE].Value;
                    if (velVal != null && int.TryParse(velVal.ToString(), out int vo))
                        velOverride = Math.Max(1, Math.Min(100, vo));
                }

                if (isMulti)
                {
                    // 다축: 축별 타겟 읽기 (빈 칸은 null)
                    double?[] targets = new double?[axisCount];
                    for (int ax = 0; ax < axisCount; ax++)
                    {
                        int col = FindMultiTargetColumn(ax);
                        if (col < 0) continue;
                        string s = row.Cells[col].Value?.ToString();
                        if (string.IsNullOrWhiteSpace(s)) { targets[ax] = null; continue; }
                        if (double.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out double v))
                            targets[ax] = v;
                        else
                        {
                            MessageBox.Show($"A{ax}(mm) 입력이 올바르지 않습니다.", "입력 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                            return false;
                        }
                    }

                    plan.Steps.Add(new TeachingStep
                    {
                        Enabled = enabled,
                        IsMulti = true,
                        Targets = targets,
                        VelOverride = velOverride,
                        WaitMs = waitMs
                    });
                }
                else
                {
                    // 단축: Axis, Target
                    if (!int.TryParse(row.Cells[COL_AXIS].Value?.ToString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int axis))
                    {
                        MessageBox.Show("Axis 입력이 올바르지 않습니다.", "입력 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return false;
                    }
                    if (!double.TryParse(row.Cells[COL_SINGLE_TGT].Value?.ToString(), NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out double target))
                    {
                        MessageBox.Show("Target(mm) 입력이 올바르지 않습니다.", "입력 오류", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return false;
                    }

                    plan.Steps.Add(new TeachingStep
                    {
                        Enabled = enabled,
                        IsMulti = false,
                        Axis = axis,
                        Target = target,
                        VelOverride = velOverride,
                        WaitMs = waitMs
                    });
                }
            }
            return true;
        }

        private static bool TryParseWaitMs(object raw, out int waitMs)
        {
            waitMs = 0;
            if (raw == null) return true; // 비어있으면 0으로

            // 숫자 타입 직접 처리 (패턴 변수 이름 충돌 방지)
            switch (raw)
            {
                case int i: waitMs = i; return true;
                case long l: waitMs = (int)l; return true;
                case double d64: waitMs = (int)Math.Round(d64); return true;
                case float f32: waitMs = (int)Math.Round(f32); return true;
                case decimal m: waitMs = (int)Math.Round(m); return true;
            }

            // 문자열 처리
            var s = raw.ToString()?.Trim();
            if (string.IsNullOrEmpty(s)) return true; // 빈 문자열은 0으로

            // "500ms" 허용
            if (s.EndsWith("ms", StringComparison.OrdinalIgnoreCase))
                s = s.Substring(0, s.Length - 2).Trim();

            // 정수(천단위 구분 허용) - 로컬/Invariant 모두 시도
            if (int.TryParse(s, NumberStyles.Integer | NumberStyles.AllowThousands, CultureInfo.CurrentCulture, out var i1))
            { waitMs = i1; return true; }
            if (int.TryParse(s, NumberStyles.Integer | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out i1))
            { waitMs = i1; return true; }

            // 실수 → 반올림 후 int
            if (double.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.CurrentCulture, out var d1))
            { waitMs = (int)Math.Round(d1); return true; }
            if (double.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out d1))
            { waitMs = (int)Math.Round(d1); return true; }

            return false;
        }

        // ── Single 모드만 있을 땐 다축 컬럼 전체 숨기기 ──
        private void SetMultiColsVisible(bool visible)
        {
            int axisCount = GetCurrentAxisCount();
            // 현재 생성되어 있는 다축 컬럼 전체를 대상으로 한다
            for (int ax = 0; ax < axisCount; ax++)
            {
                int idx = FindMultiTargetColumn(ax);
                if (idx >= 0) _teachingGrid.Columns[idx].Visible = visible;
            }
        }

        private void ApplyMultiColsVisibilityByMode()
        {
            // 모든 행이 Single이면 숨김, 하나라도 Multi가 있으면 표시
            bool anyMulti = _teachingGrid.Rows
                .Cast<DataGridViewRow>()
                .Where(r => !r.IsNewRow)
                .Any(r => string.Equals(
                    Convert.ToString(r.Cells[COL_MODE].Value) ?? MODE_SINGLE,
                    MODE_MULTI, StringComparison.OrdinalIgnoreCase));

            SetMultiColsVisible(anyMulti);
        }

        private int FindMultiTargetColumn(int axisIndex)
        {
            string name = MULTI_COL_PREFIX + axisIndex.ToString();
            for (int i = 0; i < _teachingGrid.Columns.Count; i++)
            {
                if (string.Equals(_teachingGrid.Columns[i].Name, name, StringComparison.OrdinalIgnoreCase))
                    return i;
            }
            return -1;
        }

        private void WritePlanToGrid(TeachingPlan plan)
        {
            _loopCheck.Checked = plan?.Loop ?? false;
            _repeatUpDown.Value = (plan != null && plan.RepeatCount >= 1) ? plan.RepeatCount : 1;
            _repeatUpDown.Enabled = !_loopCheck.Checked;

            _teachingGrid.Rows.Clear();

            EnsureDynamicTargetColumns();

            if (plan?.Steps == null || plan.Steps.Count == 0) return;

            int axisCount = GetCurrentAxisCount();

            foreach (var s in plan.Steps)
            {
                var row = new DataGridViewRow();
                row.CreateCells(_teachingGrid);

                row.Cells[COL_RUN].Value = s.Enabled;
                row.Cells[COL_MODE].Value = s.IsMulti ? MODE_MULTI : MODE_SINGLE;

                if (s.IsMulti)
                {
                    // 단축 필드는 회색칸 취급이지만 값은 안써도 됨
                    if (s.Targets != null)
                    {
                        for (int ax = 0; ax < axisCount; ax++)
                        {
                            int col = FindMultiTargetColumn(ax);
                            if (col >= 0)
                                row.Cells[col].Value = s.Targets.Length > ax && s.Targets[ax].HasValue ? s.Targets[ax].Value : (double?)null;
                        }
                    }
                }
                else
                {
                    row.Cells[COL_AXIS].Value = s.Axis;
                    row.Cells[COL_SINGLE_TGT].Value = s.Target;
                }

                int waitCol = FindWaitColumnIndex();
                if (waitCol >= 0) row.Cells[waitCol].Value = s.WaitMs;

                row.Cells[COL_VEL_OVERRIDE].Value = s.VelOverride > 0 ? s.VelOverride : 100;

                _teachingGrid.Rows.Add(row);
            }
            ApplyMultiColsVisibilityByMode();
        }

        private void SaveTeachingSequence()
        {
            if (!TryReadStepsFromGrid(out var plan)) return;

            using (var sfd = new SaveFileDialog())
            {
                sfd.Title = "Save Teaching Plan";
                sfd.Filter = "Teaching Plan (*.teach.json)|*.teach.json|JSON (*.json)|*.json";
                sfd.DefaultExt = "teach.json"; sfd.AddExtension = true;
                if (sfd.ShowDialog(this) != DialogResult.OK) return;
                var options = new JsonSerializerOptions { WriteIndented = true };

                try
                {
                    string json = JsonSerializer.Serialize(plan, options);
                    File.WriteAllText(sfd.FileName, json, Encoding.UTF8);
                    AppEvents.RaiseLog($"[저장] Teaching 계획 저장: {sfd.FileName}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"저장 중 오류: {ex.Message}", "저장 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void LoadTeachingSequence()
        {
            using (var ofd = new OpenFileDialog())
            {
                try { Directory.CreateDirectory(DefaultTeachingDir); } catch { /* ignore */ }
                ofd.InitialDirectory = DefaultTeachingDir;
                ofd.Title = "Load Teaching Plan";
                ofd.Filter = "Teaching Plan (*.teach.json)|*.teach.json|JSON (*.json)|*.json";

                if (ofd.ShowDialog(this) != DialogResult.OK) return;
                try
                {
                    string json = File.ReadAllText(ofd.FileName, Encoding.UTF8);
                    var plan = JsonSerializer.Deserialize<TeachingPlan>(json);
                    if (plan == null)
                    {
                        MessageBox.Show("파일 형식이 올바르지 않습니다.", "불러오기 실패", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                    WritePlanToGrid(plan);
                    AppEvents.RaiseLog($"[불러오기] Teaching 계획 로드: {ofd.FileName}");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"불러오기 중 오류: {ex.Message}", "불러오기 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void SetUiRunning(bool running)
        {
            _runButton.Enabled = !running;
            _stopButton.Enabled = running;
            _saveButton.Enabled = !running;
            _loadButton.Enabled = !running;

            _loopCheck.Enabled = !running;
            _repeatUpDown.Enabled = !running && !_loopCheck.Checked;

            if (!running)
                UpdateIterationStatus(0, _loopCheck.Checked ? -1 : (int)_repeatUpDown.Value);
        }

        private void UpdateIterationStatus(int current, int total)
        {
            string totalStr = (total < 0 || total == int.MaxValue) ? "∞" : total.ToString(CultureInfo.InvariantCulture);
            string curStr = Math.Max(0, current).ToString(CultureInfo.InvariantCulture);

            if (InvokeRequired)
            {
                BeginInvoke(new Action(() => _iterStatusLabel.Text = $"반복 {curStr}/{totalStr}"));
            }
            else
            {
                _iterStatusLabel.Text = $"반복 {curStr}/{totalStr}";
            }
        }
    }

    public class TeachingStep
    {
        public bool Enabled { get; set; }

        // Single 모드(기존 호환)
        public int Axis { get; set; }
        public double Target { get; set; }

        // Multi 모드 확장
        public bool IsMulti { get; set; } = false;
        public double?[] Targets { get; set; }  // 축별 절대 타겟(mm). null은 무시

        /// <summary>이 행의 속도 배율 (1~100%). 0이면 100%로 처리.</summary>
        public int VelOverride { get; set; } = 100;

        public int WaitMs { get; set; }
    }

    public class TeachingPlan
    {
        public bool Loop { get; set; }
        public int RepeatCount { get; set; } = 1;
        public List<TeachingStep> Steps { get; set; } = new List<TeachingStep>();
    }
}