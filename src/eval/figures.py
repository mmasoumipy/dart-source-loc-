"""Paper figures from results/metrics/<name>/s<seed>/frames.csv (no image content).

Usage: python -m src.eval.figures --name loso_s --seed 0
Writes results/figures/<name>_s<seed>_{reliability,risk_coverage,conf_vs_length}.{pdf,png}
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.evaluate import ece, risk_coverage

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
# fixed categorical order (validated reference palette): blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def reliability(ax, conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    x = [conf[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 5]
    y = [correct[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 5]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="loso_s")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = ROOT / "results" / "metrics" / args.name / f"s{args.seed}" / "frames.csv"
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(src)
    f = f[~f.ambiguous]
    stem = f"{args.name}_s{args.seed}"

    defs = (("correct", "conf_platt", BLUE, "centre within 5 mm"),
            ("correct_line", "conf_platt_line", ORANGE, "within 2 mm of source line"))

    # Reliability diagram: raw confidence against both definitions of correct
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    c = f.conf.to_numpy()
    for target, _, color, name in defs:
        y = f[target].to_numpy(float)
        x, yy = reliability(ax, c, y)
        ax.plot(x, yy, color=color, marker="o", ms=4, label=f"{name}, ECE {ece(c, y):.2f}")
    ax.set(xlabel="confidence (top-1 per frame)", ylabel="fraction correct", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_reliability.{ext}")

    # Risk-coverage: reject low-confidence frames first
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for target, _, color, name in defs:
        loss = 1 - f[target].to_numpy(float)
        cov, risk, aurc = risk_coverage(c, loss)
        ax.plot(cov, risk, color=color, label=f"{name} (AURC {aurc:.2f})")
        ax.axhline(loss.mean(), color=color, lw=1, ls="--")
    ax.plot([], [], color=MUTED, lw=1, ls="--", label="no rejection")
    ax.set(xlabel="coverage (fraction of frames accepted)", ylabel="risk (1 − fraction correct)",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_risk_coverage.{ext}")

    # Confidence by labelled visible length: partly visible (<12 mm) vs fully in plane (>=12 mm)
    p = f[f.has_seed].dropna(subset=["gt_long_mm"])
    groups = [("partly visible\n(< 12 mm)", p[p.gt_long_mm < 12].conf), ("fully in plane\n(≥ 12 mm)", p[p.gt_long_mm >= 12].conf)]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    rng = np.random.default_rng(0)
    for i, (name, v) in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=BLUE, alpha=0.25, linewidths=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2)
        ax.plot([i, i], [q1, q3], color=INK, lw=1)
        ax.text(i + 0.3, med, f"median {med:.2f}\nn={len(v)}", va="center", fontsize=7, color=INK2)
    ax.set_xticks([0, 1], [g[0] for g in groups])
    ax.set(ylabel="top-1 confidence", ylim=(0, 1), xlim=(-0.5, 1.9))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_conf_vs_length.{ext}")
    print("figures ->", out)


if __name__ == "__main__":
    main()
