#!/usr/bin/env python3
"""Render aligned, mean GPU-power traces from the Agent Probe raw samples."""
import argparse
import json
from pathlib import Path


COLORS = ["#31577b", "#628c68", "#b56b3f", "#8d5d93", "#a88732", "#9a4141"]


def load_trace(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [row["power_w"] for row in rows]


def mean_by_sample(traces):
    return [sum(trace[i] for trace in traces if i < len(trace)) /
            sum(i < len(trace) for trace in traces)
            for i in range(max(map(len, traces)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    width, height = 920, 480
    left, right, top, bottom = 75, 25, 28, 62
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 240, 570
    series = []
    for color, n in zip(COLORS, (1, 2, 4, 8, 16, 32)):
        series.append((n, color, mean_by_sample([load_trace(p) for p in sorted(args.power_dir.glob(f"n{n:02d}_rep*.jsonl"))])))
    max_samples = max(len(values) for _, _, values in series)
    def x(i): return left + (i / (max_samples - 1)) * plot_w
    def y(value): return top + (y_max - value) / (y_max - y_min) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">GPU power traces by concurrency</title>',
             '<desc id="desc">Mean aligned GPU power sampled every 10 milliseconds across four runs for each concurrent trace count.</desc>',
             f'<rect width="{width}" height="{height}" fill="white"/>']
    for power in range(250, 571, 50):
        yy = y(power)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#d9d9d9"/>',
                  f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-family="system-ui,sans-serif" font-size="13" fill="#555">{power}</text>']
    parts += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#555"/>',
              f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>']
    for second in (0, 0.5, 1.0, 1.5):
        xx = left + (second / ((max_samples - 1) * .01)) * plot_w
        if xx <= width-right:
            parts.append(f'<text x="{xx:.1f}" y="{height-bottom+24}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="13" fill="#555">{second:g}</text>')
    for n, color, values in series:
        points = ' '.join(f'{x(i):.1f},{y(value):.1f}' for i, value in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    parts += [f'<text x="{left+plot_w/2}" y="{height-12}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" fill="#333">Seconds since batch start</text>',
              f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" fill="#333">GPU power (W)</text>']
    for i, (n, color, _) in enumerate(series):
        lx = left + i * 117
        parts += [f'<line x1="{lx}" y1="16" x2="{lx+20}" y2="16" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{lx+27}" y="21" font-family="system-ui,sans-serif" font-size="13" fill="#333">N={n}</text>']
    parts.append('</svg>')
    args.output.write_text('\n'.join(parts))


if __name__ == "__main__":
    main()
