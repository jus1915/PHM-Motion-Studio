using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System;

public class ProbGaugeControl : UserControl
{
    public string[] Labels { get; set; }
    public float[] Probs { get; set; }
    public int PredIndex { get; set; } = -1;
    public bool SortDesc { get; set; } = false;

      // ★ 추가: 컴팩트 조절용
    public int BarHeight { get; set; } = 14;
    public int BarGap { get; set; } = 4;
    public int LeftLabelWidth { get; set; } = 96;
    public int RightValueWidth { get; set; } = 48;
    public Padding InnerPadding { get; set; } = new Padding(6, 4, 6, 4);

    public ProbGaugeControl()
    {
        DoubleBuffered = true; BackColor = Color.White; ResizeRedraw = true;
        MinimumSize = new Size(260, 120);
        Font = new Font("Segoe UI", 9f);
    }

    public void SetData(string[] labels, float[] probs, int pred)
    {
        Labels = labels ?? Array.Empty<string>();
        Probs = probs ?? Array.Empty<float>();
        PredIndex = (pred >= 0 && pred < Probs.Length) ? pred : -1;
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var g = e.Graphics;
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        g.Clear(BackColor);

        if (Labels == null || Probs == null) return;
        int k = Math.Min(Labels.Length, Probs.Length);
        if (k == 0) return;

        Rectangle rect = ClientRectangle;
        rect = new Rectangle(
            rect.X + InnerPadding.Left,
            rect.Y + InnerPadding.Top,
            rect.Width - InnerPadding.Horizontal,
            rect.Height - InnerPadding.Vertical);

        // 전체 높이에 맞춰 실제 바높이/간격을 조정
        int totalHeight = k * BarHeight + (k - 1) * BarGap;
        float scaleY = (totalHeight > rect.Height) ? (float)rect.Height / totalHeight : 1f;
        int barH = (int)(BarHeight * scaleY);
        int gap = (int)(BarGap * scaleY);

        using (var font = new Font(Font.FontFamily, Math.Max(8f, Font.Size * scaleY), Font.Style))
        using (var textBrush = new SolidBrush(ForeColor))
        using (var valBrush = new SolidBrush(Color.Black))
        {
            for (int i = 0; i < k; i++)
            {
                int y = rect.Y + i * (barH + gap);

                // 확률
                float p = Probs[i];
                if (p < 0f) p = 0f; if (p > 1f) p = 1f;

                // 영역
                Rectangle barArea = new Rectangle(
                    rect.X + LeftLabelWidth,
                    y,
                    rect.Width - LeftLabelWidth - RightValueWidth,
                    barH);

                // 배경
                using (var bg = new SolidBrush(Color.Gainsboro))
                    g.FillRectangle(bg, barArea);

                // 채워진 부분
                int fillW = (int)(barArea.Width * p);
                Rectangle fill = new Rectangle(barArea.X, barArea.Y, fillW, barArea.Height);
                Color barColor = (i == PredIndex) ? Color.OrangeRed : Color.SteelBlue;
                using (var fg = new SolidBrush(barColor))
                    g.FillRectangle(fg, fill);

                // 라벨
                Rectangle lblRect = new Rectangle(rect.X, y, LeftLabelWidth - 4, barH);
                var sfL = new StringFormat { Alignment = StringAlignment.Near, LineAlignment = StringAlignment.Center };
                g.DrawString(Labels[i], font, textBrush, lblRect, sfL);

                // 값 (퍼센트)
                Rectangle valRect = new Rectangle(barArea.Right + 2, y, RightValueWidth - 2, barH);
                var sfR = new StringFormat { Alignment = StringAlignment.Far, LineAlignment = StringAlignment.Center };
                g.DrawString($"{p * 100:0.0}%", font, valBrush, valRect, sfR);
            }
        }
    }
}