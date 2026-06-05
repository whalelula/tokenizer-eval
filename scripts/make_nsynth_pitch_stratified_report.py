from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("outputs/nsynth_valid_manifest_pitch_stratified.csv")
OUTPUT_HTML = Path("outputs/nsynth_valid_manifest_pitch_stratified_report.html")


def main() -> None:
    frame = pd.read_csv(INPUT_CSV)
    pitch_counts = (
        frame.groupby("pitch", dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values("pitch")
        .rename(columns={"pitch": "Pitch", "sample_count": "样本量"})
    )
    family_counts = (
        frame.groupby("instrument_family_str", dropna=False)
        .size()
        .rename("sample_count")
        .reset_index()
        .sort_values("instrument_family_str")
        .rename(columns={"instrument_family_str": "Instrument Family", "sample_count": "样本量"})
    )
    pitch_family_counts = pd.crosstab(
        frame["pitch"],
        frame["instrument_family_str"],
        dropna=False,
    ).sort_index()
    pitch_family_counts.index.name = "Pitch"
    chart_html = _pitch_family_charts(pitch_family_counts)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NSynth Pitch 分层采样报告</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6b7a;
      --line: #d8dee8;
      --head: #eef3f8;
      --page: #f7f9fc;
      --paper: #ffffff;
      --accent: #216e7a;
    }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 24px 44px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 18px 0 4px;
    }}
    .summary-item {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .summary-label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .summary-value {{
      margin-top: 4px;
      color: var(--accent);
      font-size: 24px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      border-right: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      background: var(--head);
      color: #334155;
      font-weight: 700;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      background: inherit;
    }}
    tbody tr:nth-child(even) {{
      background: #fbfcfe;
    }}
    tbody tr:last-child td {{
      border-bottom: 0;
    }}
    th:last-child, td:last-child {{
      border-right: 0;
    }}
    .chart-stack {{
      display: grid;
      gap: 14px;
    }}
    .chart-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px 10px;
    }}
    .chart-title {{
      margin: 0 0 8px;
      font-size: 15px;
      font-weight: 700;
      color: #334155;
    }}
    .chart {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .tick-label {{
      fill: var(--muted);
      font-size: 10px;
    }}
    .grid-line {{
      stroke: #e5eaf1;
      stroke-width: 1;
    }}
    .axis-line {{
      stroke: #8592a3;
      stroke-width: 1.2;
    }}
    .bar {{
      fill: var(--accent);
    }}
  </style>
</head>
<body>
<main>
  <h1>NSynth Pitch 分层采样报告</h1>
  <p class="meta">数据源：{escape(str(INPUT_CSV))}</p>

  <section class="summary" aria-label="采样摘要">
    <div class="summary-item">
      <div class="summary-label">总样本量</div>
      <div class="summary-value">{len(frame)}</div>
    </div>
    <div class="summary-item">
      <div class="summary-label">Pitch 数量</div>
      <div class="summary-value">{frame["pitch"].nunique()}</div>
    </div>
    <div class="summary-item">
      <div class="summary-label">Instrument Family 数量</div>
      <div class="summary-value">{frame["instrument_family_str"].nunique()}</div>
    </div>
  </section>

  <h2>图：每个 Instrument Family 的 Pitch 分布</h2>
  <p class="meta">10 张图共用同一横轴 pitch 范围和同一纵轴样本量范围。</p>
  <div class="chart-stack">
    {chart_html}
  </div>

  <h2>表 1：每个 Pitch 的样本量</h2>
  <div class="table-wrap">
    {_frame_to_html(pitch_counts)}
  </div>

  <h2>表 2：每个 Instrument Family 的样本量</h2>
  <div class="table-wrap">
    {_frame_to_html(family_counts)}
  </div>

  <h2>表 3：每个 Pitch × Instrument Family 的样本量</h2>
  <div class="table-wrap">
    {_frame_to_html(pitch_family_counts.reset_index())}
  </div>
</main>
</body>
</html>
"""
    OUTPUT_HTML.write_text(html, encoding="utf-8")


def _frame_to_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, escape=True)


def _pitch_family_charts(pitch_family_counts: pd.DataFrame) -> str:
    pitch_values = [int(value) for value in pitch_family_counts.index.tolist()]
    families = [str(column) for column in pitch_family_counts.columns]
    y_max = int(pitch_family_counts.to_numpy().max())
    y_max = max(y_max, 1)

    return "\n".join(
        _pitch_distribution_svg(
            family=family,
            pitch_values=pitch_values,
            counts=[int(value) for value in pitch_family_counts[family].tolist()],
            y_max=y_max,
        )
        for family in families
    )


def _pitch_distribution_svg(
    *,
    family: str,
    pitch_values: list[int],
    counts: list[int],
    y_max: int,
) -> str:
    width = 1160
    height = 230
    left = 52
    right = 18
    top = 18
    bottom = 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    pitch_min = min(pitch_values)
    pitch_max = max(pitch_values)
    pitch_span = max(pitch_max - pitch_min, 1)
    step = plot_width / len(pitch_values)
    bar_width = max(step * 0.72, 1.0)

    def x_for_pitch(pitch: int) -> float:
        return left + ((pitch - pitch_min) / pitch_span) * plot_width

    def y_for_count(count: int) -> float:
        return top + plot_height - (count / y_max) * plot_height

    y_ticks = _nice_ticks(y_max)
    x_ticks = [pitch_min, *range(_next_multiple(pitch_min, 10), pitch_max + 1, 10)]
    if pitch_max not in x_ticks:
        x_ticks.append(pitch_max)
    x_ticks = sorted(set(value for value in x_ticks if pitch_min <= value <= pitch_max))

    grid_parts = []
    for tick in y_ticks:
        y = y_for_count(tick)
        grid_parts.append(
            f'<line class="grid-line" x1="{left:.1f}" y1="{y:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{y:.1f}" />'
        )
        grid_parts.append(
            f'<text class="tick-label" x="{left - 9:.1f}" y="{y + 3:.1f}" '
            f'text-anchor="end">{tick}</text>'
        )
    for tick in x_ticks:
        x = x_for_pitch(tick)
        grid_parts.append(
            f'<text class="tick-label" x="{x:.1f}" y="{height - 18:.1f}" '
            f'text-anchor="middle">{tick}</text>'
        )

    bars = []
    for pitch, count in zip(pitch_values, counts):
        x = x_for_pitch(pitch) - bar_width / 2
        y = y_for_count(count)
        bar_height = top + plot_height - y
        bars.append(
            f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{bar_height:.2f}"><title>pitch {pitch}: {count}</title></rect>'
        )

    escaped_family = escape(family)
    return f"""<section class="chart-card">
  <h3 class="chart-title">{escaped_family}</h3>
  <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escaped_family} pitch distribution">
    <line class="axis-line" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" />
    <line class="axis-line" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" />
    {''.join(grid_parts)}
    {''.join(bars)}
    <text class="axis-label" x="{left + plot_width / 2:.1f}" y="{height - 4:.1f}" text-anchor="middle">Pitch</text>
    <text class="axis-label" transform="translate(13 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">样本量</text>
  </svg>
</section>"""


def _nice_ticks(y_max: int) -> list[int]:
    if y_max <= 5:
        step = 1
    elif y_max <= 12:
        step = 2
    elif y_max <= 25:
        step = 5
    else:
        step = 10
    ticks = list(range(0, y_max + 1, step))
    if ticks[-1] != y_max:
        ticks.append(y_max)
    return ticks


def _next_multiple(value: int, base: int) -> int:
    return ((value + base - 1) // base) * base


if __name__ == "__main__":
    main()
