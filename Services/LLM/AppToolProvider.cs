using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using PHM_Project_DockPanel.Controller;
using PHM_Project_DockPanel.Services.Core;

namespace PHM_Project_DockPanel.Services.LLM
{
    /// <summary>
    /// LLM 도구 실행에 필요한 앱 컨텍스트.
    /// MainForm에서 생성하여 LlmChatPanel에 주입합니다.
    /// </summary>
    public class LlmAppContext
    {
        public ControllerManager Controller { get; set; }
        public PHM_Motion Motion { get; set; }
        public AxisConfig[] AxisConfigs { get; set; }

        /// <summary>사용자 확인 다이얼로그를 표시하고 Yes/No 결과를 반환합니다.</summary>
        public Func<string, bool> ConfirmAction { get; set; }

        /// <summary>축 설정 변경 후 컨트롤러에 적용합니다.</summary>
        public Action<AxisConfig[]> ApplyAxisConfigs { get; set; }
    }

    /// <summary>
    /// Claude Tool Use 정의 + 실제 실행 로직.
    /// </summary>
    public static class AppToolProvider
    {
        // =====================================================================
        // 도구 스키마 정의
        // =====================================================================
        public static JArray GetToolDefinitions()
        {
            return JArray.Parse(@"[
  {
    ""name"": ""get_system_status"",
    ""description"": ""컨트롤러 연결 상태, 제어기 종류(WMX3/Ajin/시뮬레이션), 총 축 수, 현재 레이블을 반환합니다."",
    ""input_schema"": { ""type"": ""object"", ""properties"": {} }
  },
  {
    ""name"": ""get_axes_status"",
    ""description"": ""각 축의 현재 위치(mm), 서보 ON/OFF, 알람 상태, 동작 상태(Idle/Moving 등)를 반환합니다."",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""axes"": {
          ""type"": ""array"",
          ""items"": { ""type"": ""integer"" },
          ""description"": ""조회할 축 번호 목록. 생략 시 전체 축.""
        }
      }
    }
  },
  {
    ""name"": ""get_axes_config"",
    ""description"": ""축의 파라미터(최대 속도 mm/s, 가감속 mm/s², 최대 스트로크 mm)를 반환합니다."",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""axis"": { ""type"": ""integer"", ""description"": ""축 번호. 생략 시 전체 축."" }
      }
    }
  },
  {
    ""name"": ""set_servo"",
    ""description"": ""서보를 ON 또는 OFF합니다. axes를 생략하면 전체 축에 적용됩니다. [사용자 확인 필요]"",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""on"": { ""type"": ""boolean"", ""description"": ""true=서보 ON, false=서보 OFF"" },
        ""axes"": {
          ""type"": ""array"",
          ""items"": { ""type"": ""integer"" },
          ""description"": ""대상 축 번호 목록. 생략 시 전체 축.""
        }
      },
      ""required"": [""on""]
    }
  },
  {
    ""name"": ""move_axes"",
    ""description"": ""축을 지정한 위치(mm)로 이동합니다. is_absolute=true면 절대 좌표, false면 현재 위치 기준 상대 이동입니다. [사용자 확인 필요]"",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""axes"": { ""type"": ""array"", ""items"": { ""type"": ""integer"" }, ""description"": ""이동할 축 번호 목록"" },
        ""targets"": { ""type"": ""array"", ""items"": { ""type"": ""number"" }, ""description"": ""mm 단위 목표 위치 (axes와 순서 일치)"" },
        ""is_absolute"": { ""type"": ""boolean"", ""description"": ""절대 이동 여부, 기본값 true"" }
      },
      ""required"": [""axes"", ""targets""]
    }
  },
  {
    ""name"": ""set_axis_param"",
    ""description"": ""축 파라미터(최대 속도/가감속/최대 스트로크)를 변경합니다. 변경하려는 항목만 지정하세요. [사용자 확인 필요]"",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""axis"": { ""type"": ""integer"", ""description"": ""대상 축 번호"" },
        ""max_vel"": { ""type"": ""number"", ""description"": ""최대 속도 (mm/s)"" },
        ""acc"": { ""type"": ""number"", ""description"": ""가속도 (mm/s²)"" },
        ""dec"": { ""type"": ""number"", ""description"": ""감속도 (mm/s²)"" },
        ""max_stroke"": { ""type"": ""number"", ""description"": ""최대 스트로크 (mm)"" }
      },
      ""required"": [""axis""]
    }
  },
  {
    ""name"": ""set_label"",
    ""description"": ""데이터 수집 레이블을 변경합니다. 허용값: normal, fault, bearing_fault, gear_fault, imbalance, looseness"",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""label"": { ""type"": ""string"" }
      },
      ""required"": [""label""]
    }
  },
  {
    ""name"": ""save_teaching_sequence"",
    ""description"": ""반복 동작 시퀀스를 생성하여 Teaching 파일(JSON)로 저장합니다. TeachingForm에서 바로 불러올 수 있습니다."",
    ""input_schema"": {
      ""type"": ""object"",
      ""properties"": {
        ""filename"": { ""type"": ""string"", ""description"": ""저장할 파일명 (.json 포함 또는 생략 가능)"" },
        ""steps"": {
          ""type"": ""array"",
          ""description"": ""시퀀스 스텝 목록"",
          ""items"": {
            ""type"": ""object"",
            ""properties"": {
              ""mode"": { ""type"": ""string"", ""enum"": [""Single"", ""Multi""], ""description"": ""Single: 단일 축 이동, Multi: 다축 동시 이동"" },
              ""axis"": { ""type"": ""integer"", ""description"": ""[Single 전용] 이동할 축 번호"" },
              ""target"": { ""type"": ""number"", ""description"": ""[Single 전용] 목표 위치 (mm)"" },
              ""targets"": { ""type"": ""array"", ""items"": { ""type"": ""number"" }, ""description"": ""[Multi 전용] 각 축 목표 위치 (mm), 축 0부터 순서대로"" },
              ""velocity_override"": { ""type"": ""number"", ""description"": ""속도 비율 (%), 기본값 100"" },
              ""wait_ms"": { ""type"": ""integer"", ""description"": ""이동 완료 후 대기 시간 (ms), 기본값 0"" }
            },
            ""required"": [""mode""]
          }
        },
        ""repeat_count"": { ""type"": ""integer"", ""description"": ""전체 시퀀스 반복 횟수, 기본값 1"" }
      },
      ""required"": [""filename"", ""steps""]
    }
  }
]");
        }

        // =====================================================================
        // 시스템 프롬프트 생성 (매 요청마다 현재 상태로 갱신)
        // =====================================================================
        public static string BuildSystemPrompt(LlmAppContext ctx)
        {
            var sb = new StringBuilder();
            sb.AppendLine("당신은 PHM Motion Studio의 AI 관제 보조원입니다.");
            sb.AppendLine("산업용 모션 제어 설비의 상태를 모니터링하고, 사용자 요청에 따라 제어 작업을 수행합니다.");
            sb.AppendLine();

            // 현재 설비 상태 스냅샷
            sb.AppendLine("## 현재 설비 상태");
            try
            {
                sb.AppendLine($"- 제어기: {GetControllerType(ctx.Controller)}");
                sb.AppendLine($"- 연결: {(ctx.Controller.IsConnected ? "연결됨" : "미연결")}");
                sb.AppendLine($"- 총 축 수: {ctx.AxisConfigs.Length}개 (번호 0~{ctx.AxisConfigs.Length - 1})");
                sb.AppendLine($"- 현재 레이블: {(string.IsNullOrEmpty(AppState.CurrentLabel) ? "(없음)" : AppState.CurrentLabel)}");

                if (ctx.Controller.IsConnected)
                {
                    var status = ctx.Controller.GetStatus();
                    int count = status?.AxesStatus?.Length ?? 0;
                    for (int i = 0; i < count && i < ctx.AxisConfigs.Length; i++)
                    {
                        var ax = status.AxesStatus[i];
                        sb.AppendLine($"- 축{i}: 위치={ax.ActualPos:F1}mm  서보={( ax.ServoOn ? "ON" : "OFF")}  알람={( ax.AmpAlarm ? "알람" : "정상")}  상태={ax.OpState}");
                    }
                }
            }
            catch
            {
                sb.AppendLine("- 상태 조회 실패");
            }

            sb.AppendLine();
            sb.AppendLine("## 행동 규칙");
            sb.AppendLine("- 반드시 한국어로 답하세요. 기술 용어는 영어 병기 가능합니다.");
            sb.AppendLine("- 서보 ON/OFF, 축 이동, 파라미터 변경 등 하드웨어에 영향을 주는 작업은 도구 호출 전에 의도를 먼저 설명하고, set_servo/move_axes/set_axis_param 도구를 통해서만 실행하세요.");
            sb.AppendLine("- 축 번호는 0부터 시작합니다.");
            sb.AppendLine("- 모르거나 불확실한 정보는 추측하지 말고 사용자에게 질문하세요.");
            sb.AppendLine("- 안전에 관련된 사항(과속, 초과 스트로크 등)은 반드시 사용자에게 경고하세요.");

            return sb.ToString();
        }

        private static string GetControllerType(ControllerManager c)
        {
            if (c.IsWmx3) return "WMX3 (SoftServo)";
            if (c.IsAjin) return "Ajin AMP";
            if (c.IsSimulationMode) return "시뮬레이션";
            return "알 수 없음";
        }

        // =====================================================================
        // 도구 실행
        // =====================================================================
        public static async Task<string> ExecuteToolAsync(
            string toolName, JObject input, LlmAppContext ctx)
        {
            try
            {
                switch (toolName)
                {
                    case "get_system_status":
                        return GetSystemStatus(ctx);

                    case "get_axes_status":
                    {
                        int[] axes = null;
                        var axArr = input["axes"] as JArray;
                        if (axArr != null)
                            axes = axArr.Select(t => t.Value<int>()).ToArray();
                        return GetAxesStatus(ctx, axes);
                    }

                    case "get_axes_config":
                    {
                        int? axis = null;
                        if (input["axis"] != null && input["axis"].Type != JTokenType.Null)
                            axis = input["axis"].Value<int>();
                        return GetAxesConfig(ctx, axis);
                    }

                    case "set_servo":
                        return await SetServoAsync(input, ctx);

                    case "move_axes":
                        return await MoveAxesAsync(input, ctx);

                    case "set_axis_param":
                        return await SetAxisParamAsync(input, ctx);

                    case "set_label":
                    {
                        string label = input["label"]?.ToString() ?? "normal";
                        AppState.CurrentLabel = label;
                        AppEvents.RaiseInfluxLabelChanged(label);
                        AppEvents.RaiseLog($"[AI] 레이블 변경: {label}");
                        return $"레이블이 '{label}'(으)로 변경되었습니다.";
                    }

                    case "save_teaching_sequence":
                        return SaveTeachingSequence(input);

                    default:
                        return $"알 수 없는 도구입니다: {toolName}";
                }
            }
            catch (Exception ex)
            {
                return $"도구 실행 오류: {ex.Message}";
            }
        }

        // ------------------------------------------------------------------
        // 읽기 전용 도구
        // ------------------------------------------------------------------
        private static string GetSystemStatus(LlmAppContext ctx)
        {
            return new JObject
            {
                ["controller_type"] = GetControllerType(ctx.Controller),
                ["connected"] = ctx.Controller.IsConnected,
                ["simulation"] = ctx.Controller.IsSimulationMode,
                ["axis_count"] = ctx.AxisConfigs.Length,
                ["current_label"] = AppState.CurrentLabel ?? ""
            }.ToString(Formatting.Indented);
        }

        private static string GetAxesStatus(LlmAppContext ctx, int[] filterAxes)
        {
            if (!ctx.Controller.IsConnected)
                return "{ \"error\": \"컨트롤러가 연결되지 않았습니다.\" }";

            var status = ctx.Controller.GetStatus();
            if (status?.AxesStatus == null)
                return "{ \"error\": \"상태 조회에 실패했습니다.\" }";

            var arr = new JArray();
            int count = Math.Min(status.AxesStatus.Length, ctx.AxisConfigs.Length);
            for (int i = 0; i < count; i++)
            {
                if (filterAxes != null && !filterAxes.Contains(i)) continue;
                var ax = status.AxesStatus[i];
                arr.Add(new JObject
                {
                    ["axis"] = i,
                    ["position_mm"] = Math.Round(ax.ActualPos, 2),
                    ["servo_on"] = ax.ServoOn,
                    ["alarm_active"] = ax.AmpAlarm,
                    ["op_state"] = ax.OpState.ToString()
                });
            }
            return arr.ToString(Formatting.Indented);
        }

        private static string GetAxesConfig(LlmAppContext ctx, int? filterAxis)
        {
            var arr = new JArray();
            for (int i = 0; i < ctx.AxisConfigs.Length; i++)
            {
                if (filterAxis.HasValue && filterAxis.Value != i) continue;
                var cfg = ctx.AxisConfigs[i];
                arr.Add(new JObject
                {
                    ["axis"] = i,
                    ["max_vel_mm_s"] = cfg.MaxVel,
                    ["acc_mm_s2"] = cfg.Acc,
                    ["dec_mm_s2"] = cfg.Dec,
                    ["max_stroke_mm"] = cfg.PositionMax
                });
            }
            return arr.ToString(Formatting.Indented);
        }

        // ------------------------------------------------------------------
        // 확인 필요 도구
        // ------------------------------------------------------------------
        private static async Task<string> SetServoAsync(JObject input, LlmAppContext ctx)
        {
            bool on = input["on"]?.Value<bool>() ?? false;

            int[] axes;
            var axArr = input["axes"] as JArray;
            if (axArr != null && axArr.Count > 0)
                axes = axArr.Select(t => t.Value<int>()).ToArray();
            else
                axes = Enumerable.Range(0, ctx.AxisConfigs.Length).ToArray();

            string axisStr = string.Join(", ", axes.Select(a => $"축{a}"));
            string confirmMsg = $"{axisStr} 서보를 {(on ? "ON" : "OFF")} 하시겠습니까?";

            bool confirmed = ctx.ConfirmAction?.Invoke(confirmMsg) ?? false;
            if (!confirmed) return "사용자가 취소했습니다.";

            foreach (int ax in axes)
            {
                ctx.Controller.SetServo(ax, on);
                await Task.Delay(30);
            }

            string result = $"{axisStr} 서보 {(on ? "ON" : "OFF")} 완료.";
            AppEvents.RaiseLog($"[AI] {result}");
            return result;
        }

        private static async Task<string> MoveAxesAsync(JObject input, LlmAppContext ctx)
        {
            var axArr = (input["axes"] as JArray) ?? new JArray();
            var tgtArr = (input["targets"] as JArray) ?? new JArray();
            bool isAbs = input["is_absolute"]?.Value<bool>() ?? true;

            if (axArr.Count == 0) return "오류: 이동할 축(axes)이 지정되지 않았습니다.";
            if (axArr.Count != tgtArr.Count)
                return $"오류: axes({axArr.Count}개)와 targets({tgtArr.Count}개) 배열 크기가 다릅니다.";

            int[] axes = axArr.Select(t => t.Value<int>()).ToArray();
            double[] targets = tgtArr.Select(t => t.Value<double>()).ToArray();

            // 유효 범위 검사
            var warnings = new List<string>();
            for (int i = 0; i < axes.Length; i++)
            {
                int ax = axes[i];
                if (ax < 0 || ax >= ctx.AxisConfigs.Length)
                {
                    return $"오류: 축 번호 {ax}가 유효하지 않습니다. (범위: 0~{ctx.AxisConfigs.Length - 1})";
                }
                double maxPos = ctx.AxisConfigs[ax].PositionMax;
                if (isAbs && (targets[i] < 0 || targets[i] > maxPos))
                    warnings.Add($"축{ax} 목표 {targets[i]:F1}mm가 범위(0~{maxPos:F1}mm)를 초과합니다.");
            }

            string mode = isAbs ? "절대" : "상대";
            string detail = string.Join("\n", axes.Select((a, i) => $"  축{a}: {targets[i]:F1} mm"));
            string warnStr = warnings.Count > 0 ? "\n\n⚠️ 경고:\n" + string.Join("\n", warnings) : "";
            string confirmMsg = $"다음 {mode} 이동을 실행하시겠습니까?\n{detail}{warnStr}";

            bool confirmed = ctx.ConfirmAction?.Invoke(confirmMsg) ?? false;
            if (!confirmed) return "사용자가 취소했습니다.";

            AppEvents.RaiseLog($"[AI] {mode} 이동 시작: 축 [{string.Join(",", axes)}] → [{string.Join(",", targets.Select(t => t.ToString("F1")))}]");
            bool success = await ctx.Motion.RunMotionWithLogging(axes, isAbs, targets);
            return success
                ? $"{mode} 이동 완료. 이동 축: {string.Join(", ", axes.Select(a => $"축{a}"))}"
                : "이동에 실패했거나 중단되었습니다.";
        }

        private static async Task<string> SetAxisParamAsync(JObject input, LlmAppContext ctx)
        {
            int axis = input["axis"]?.Value<int>() ?? 0;
            if (axis < 0 || axis >= ctx.AxisConfigs.Length)
                return $"오류: 축 번호 {axis}가 유효하지 않습니다.";

            var cfg = ctx.AxisConfigs[axis];
            var changes = new List<string>();

            double newMaxVel = cfg.MaxVel, newAcc = cfg.Acc, newDec = cfg.Dec, newMaxStroke = cfg.PositionMax;

            if (input["max_vel"] != null && input["max_vel"].Type != JTokenType.Null)
            {
                newMaxVel = input["max_vel"].Value<double>();
                changes.Add($"최대 속도: {cfg.MaxVel:F0} → {newMaxVel:F0} mm/s");
            }
            if (input["acc"] != null && input["acc"].Type != JTokenType.Null)
            {
                newAcc = input["acc"].Value<double>();
                changes.Add($"가속도: {cfg.Acc:F0} → {newAcc:F0} mm/s²");
            }
            if (input["dec"] != null && input["dec"].Type != JTokenType.Null)
            {
                newDec = input["dec"].Value<double>();
                changes.Add($"감속도: {cfg.Dec:F0} → {newDec:F0} mm/s²");
            }
            if (input["max_stroke"] != null && input["max_stroke"].Type != JTokenType.Null)
            {
                newMaxStroke = input["max_stroke"].Value<double>();
                changes.Add($"최대 스트로크: {cfg.PositionMax:F0} → {newMaxStroke:F0} mm");
            }

            if (changes.Count == 0) return "변경할 파라미터가 지정되지 않았습니다.";

            string confirmMsg = $"축{axis} 파라미터 변경:\n" + string.Join("\n", changes.Select(c => $"  {c}"));
            bool confirmed = ctx.ConfirmAction?.Invoke(confirmMsg) ?? false;
            if (!confirmed) return "사용자가 취소했습니다.";

            cfg.MaxVel = newMaxVel;
            cfg.Acc = newAcc;
            cfg.Dec = newDec;
            cfg.PositionMax = newMaxStroke;

            ctx.ApplyAxisConfigs?.Invoke(ctx.AxisConfigs);
            AppEvents.RaiseLog($"[AI] 축{axis} 파라미터 변경: {string.Join(", ", changes)}");

            await Task.CompletedTask;
            return $"축{axis} 파라미터 변경 완료:\n" + string.Join("\n", changes.Select(c => $"  {c}"));
        }

        private static string SaveTeachingSequence(JObject input)
        {
            string filename = input["filename"]?.ToString() ?? "ai_sequence";
            if (!filename.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                filename += ".json";

            var steps = input["steps"] as JArray ?? new JArray();
            int repeat = input["repeat_count"]?.Value<int>() ?? 1;
            if (repeat < 1) repeat = 1;

            string dir = @"C:\Data\PHM_Logs\Teaching";
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, filename);

            // TeachingForm JSON 포맷으로 변환
            var rows = new JArray();
            foreach (JObject step in steps)
            {
                string mode = step["mode"]?.ToString() ?? "Single";
                var row = new JObject
                {
                    ["Run"] = true,
                    ["Mode"] = mode,
                    ["Axis"] = step["axis"]?.Value<int>() ?? 0,
                    ["SingleTarget"] = step["target"]?.Value<double>() ?? 0.0,
                    ["VelocityOverride"] = step["velocity_override"]?.Value<double>() ?? 100.0,
                    ["WaitMs"] = step["wait_ms"]?.Value<int>() ?? 0
                };

                var multiTargets = step["targets"] as JArray;
                if (multiTargets != null)
                {
                    for (int i = 0; i < multiTargets.Count; i++)
                        row[$"T{i}"] = multiTargets[i].Value<double>();
                }

                rows.Add(row);
            }

            var doc = new JObject { ["RepeatCount"] = repeat, ["Rows"] = rows };
            File.WriteAllText(path, doc.ToString(Formatting.Indented), Encoding.UTF8);
            AppEvents.RaiseLog($"[AI] 시퀀스 저장: {path}");

            return $"시퀀스 '{filename}' 저장 완료.\n경로: {path}\n스텝 수: {rows.Count}개, 반복 횟수: {repeat}회";
        }
    }
}
