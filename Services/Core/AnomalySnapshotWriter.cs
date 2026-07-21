using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace PHM_Project_DockPanel.Services.Core
{
    /// <summary>
    /// 이상 탐지 시점의 원신호 윈도우를 PNG로 저장합니다.
    /// ContinuousInferenceService의 백그라운드 폴링 루프(Task.Run, UI 스레드 아님)에서
    /// 직접 호출되므로, System.Windows.Forms.Chart 같은 WinForms 컨트롤(스레드 친화적이지
    /// 않음)을 쓰지 않고 System.Drawing(GDI+) 만으로 Bitmap에 직접 그립니다 — GDI+는
    /// Win32 메시지 루프/STA에 묶여 있지 않아 어느 스레드에서 호출해도 안전합니다.
    /// </summary>
    public static class AnomalySnapshotWriter
    {
        private static readonly Color[] Palette =
        {
            Color.RoyalBlue, Color.OrangeRed, Color.SeaGreen, Color.Purple,
            Color.Goldenrod, Color.Teal, Color.Crimson, Color.SlateGray,
        };

        /// <summary>
        /// window(윈도우 크기 × 채널 수로 평탄화된 배열)를 채널별 라인 차트로 그리고,
        /// 점수/임계값/결함분류/채널별 통계를 함께 표시한 PNG를 outputDir에 저장합니다.
        /// 실패해도 예외를 던지지 않고 log로만 알립니다(호출부의 추론 루프를 절대 막지 않기 위함).
        /// </summary>
        public static void Save(
            string outputDir,
            float[] window,
            int nChannels,
            string[] channelNames,
            string sensorType,
            int? axis,
            string modelType,
            float score,
            float threshold,
            string clsClassName,
            float? clsConfidence,
            Action<string> log)
        {
            try
            {
                if (window == null || nChannels <= 0 || window.Length < nChannels)
                {
                    log?.Invoke("[이상 탐지] 스냅샷 저장 건너뜀 — 윈도우 데이터 없음");
                    return;
                }

                Directory.CreateDirectory(outputDir);

                string axisTag   = axis.HasValue ? $"_ax{axis.Value}" : "";
                string classTag  = string.IsNullOrWhiteSpace(clsClassName) ? "" : $"_{SanitizeForFileName(clsClassName)}";
                string fileName  = $"{DateTime.Now:yyyyMMdd_HHmmss_fff}_{sensorType}{axisTag}{classTag}.png";
                string path      = Path.Combine(outputDir, fileName);

                using (var bmp = Render(window, nChannels, channelNames, sensorType, axis,
                    modelType, score, threshold, clsClassName, clsConfidence))
                {
                    bmp.Save(path, ImageFormat.Png);
                }

                log?.Invoke($"[이상 탐지] 스냅샷 저장 → {path}");
            }
            catch (Exception ex)
            {
                log?.Invoke($"[이상 탐지] 스냅샷 저장 실패: {ex.Message}");
            }
        }

        private static string SanitizeForFileName(string s)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
                s = s.Replace(c, '_');
            return s;
        }

        private static Bitmap Render(
            float[] window, int nCh, string[] channelNames,
            string sensorType, int? axis, string modelType,
            float score, float threshold, string clsClassName, float? clsConfidence)
        {
            const int width = 900;
            const int height = 640;
            const int chartLeft = 60, chartRight = width - 20;
            const int chartTop = 70, chartHeight = 340;
            int chartBottom = chartTop + chartHeight;

            int windowSize = window.Length / nCh;

            var bmp = new Bitmap(width, height);
            using (var g = Graphics.FromImage(bmp))
            using (var titleFont  = new Font("맑은 고딕", 13f, FontStyle.Bold))
            using (var scoreFont  = new Font("맑은 고딕", 10f))
            using (var hdrFont    = new Font("맑은 고딕", 9.5f, FontStyle.Bold))
            using (var monoFont   = new Font("Consolas", 9f))
            using (var clsFont    = new Font("맑은 고딕", 10.5f, FontStyle.Bold))
            using (var legendFont = new Font("맑은 고딕", 8.5f))
            {
                g.SmoothingMode  = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
                g.Clear(Color.White);

                // ── 제목 / 메타 정보 ─────────────────────────────────────────
                string axisDesc = axis.HasValue ? $"Ax{axis.Value}" : "전역";
                string title = $"이상 탐지 스냅샷 — {sensorType} {axisDesc}  [{modelType}]";
                g.DrawString(title, titleFont, Brushes.Black, 20, 14);
                g.DrawString(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"), scoreFont, Brushes.Gray, 20, 42);

                bool isAnomaly = score >= threshold;
                string scoreLine = $"점수: {score:F4}   임계값: {threshold:F4}" + (isAnomaly ? "   ⚠ 이상" : "");
                g.DrawString(scoreLine, scoreFont, isAnomaly ? Brushes.Red : Brushes.Black, 320, 42);

                // ── 차트 테두리 ──────────────────────────────────────────────
                g.DrawRectangle(Pens.Gray, chartLeft, chartTop, chartRight - chartLeft, chartHeight);
                g.DrawString("샘플 (윈도우 내 인덱스, 채널별 자체 스케일 정규화)", legendFont, Brushes.Gray,
                    chartLeft, chartBottom + 4);

                // ── 채널별 라인 + 통계 계산 ──────────────────────────────────
                var stats = new List<(string Name, double Rms, double PeakAbs, Color Color)>();

                for (int c = 0; c < nCh; c++)
                {
                    var vals = new float[windowSize];
                    for (int t = 0; t < windowSize; t++)
                        vals[t] = window[t * nCh + c];

                    double sumSq = 0.0;
                    double peak  = 0.0;
                    float minV = vals.Length > 0 ? vals[0] : 0f;
                    float maxV = vals.Length > 0 ? vals[0] : 1f;
                    foreach (var v in vals)
                    {
                        sumSq += (double)v * v;
                        double av = Math.Abs(v);
                        if (av > peak) peak = av;
                        if (v < minV) minV = v;
                        if (v > maxV) maxV = v;
                    }
                    double rms = windowSize > 0 ? Math.Sqrt(sumSq / windowSize) : 0.0;

                    string name = (channelNames != null && c < channelNames.Length) ? channelNames[c] : $"ch{c}";
                    Color color = Palette[c % Palette.Length];
                    stats.Add((name, rms, peak, color));

                    float range = Math.Max(maxV - minV, 1e-6f);
                    if (windowSize > 1)
                    {
                        var points = new PointF[windowSize];
                        for (int t = 0; t < windowSize; t++)
                        {
                            float nx = chartLeft + (chartRight - chartLeft) * (t / (float)(windowSize - 1));
                            float ny = chartBottom - (vals[t] - minV) / range * chartHeight;
                            points[t] = new PointF(nx, ny);
                        }
                        using (var pen = new Pen(color, 1.3f))
                            g.DrawLines(pen, points);
                    }
                }

                // ── 범례 (차트 오른쪽 위, 채널명) — 신호선과 겹쳐도 읽히도록 흰 배경 판 ──
                int legendX = chartRight - 150;
                int legendBoxHeight = Math.Max(1, nCh) * 15 + 6;
                g.FillRectangle(new SolidBrush(Color.FromArgb(230, Color.White)),
                    legendX - 4, chartTop + 2, 150, legendBoxHeight);
                g.DrawRectangle(Pens.LightGray, legendX - 4, chartTop + 2, 150, legendBoxHeight);

                int legendY = chartTop + 6;
                foreach (var s in stats)
                {
                    g.FillRectangle(new SolidBrush(s.Color), legendX, legendY + 2, 10, 10);
                    g.DrawString(s.Name, legendFont, Brushes.Black, legendX + 14, legendY);
                    legendY += 15;
                }

                // ── 채널별 통계 표 (RMS 기준 내림차순 — 가장 두드러진 채널 먼저) ──
                int statY = chartBottom + 26;
                g.DrawString("채널별 통계 (진폭이 가장 큰 채널이 원인일 가능성이 높습니다):", hdrFont, Brushes.Black, 20, statY);
                statY += 20;
                foreach (var s in stats.OrderByDescending(x => x.Rms))
                {
                    g.FillRectangle(new SolidBrush(s.Color), 24, statY + 2, 8, 8);
                    g.DrawString($"{s.Name,-18} RMS={s.Rms,10:F4}   Peak={s.PeakAbs,10:F4}",
                        monoFont, Brushes.Black, 40, statY);
                    statY += 16;
                }

                // ── 결함 분류(CLS) — 있을 때만 ────────────────────────────────
                statY += 10;
                if (!string.IsNullOrWhiteSpace(clsClassName))
                {
                    string clsLine = $"결함 분류(CLS): {clsClassName}" +
                        (clsConfidence.HasValue ? $"  (신뢰도 {clsConfidence.Value:P1})" : "");
                    g.DrawString(clsLine, clsFont, Brushes.DarkRed, 20, statY);
                }
                else
                {
                    g.DrawString("결함 분류(CLS): 미실행/비활성화 — 위 채널별 통계만 참고하세요.",
                        scoreFont, Brushes.Gray, 20, statY);
                }
            }

            return bmp;
        }
    }
}
