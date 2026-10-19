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
    correct = f.correct.to_numpy(float)
    stem = f"{args.name}_s{args.seed}"

    # Reliability diagram: raw vs Platt-calibrated
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for col, color, name in (("conf", BLUE, "raw"), ("conf_platt", ORANGE, "Platt (val segment)")):
        c = f[col].to_numpy()
        x, y = reliability(ax, c, correct)
        ax.plot(x, y, color=color, marker="o", ms=4, label=f"{name}, ECE {ece(c, correct):.3f}")
    ax.set(xlabel="confidence (top-1 per frame)", ylabel="fraction correct (centre ≤ gate)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_reliability.{ext}")

    # Risk-coverage: reject low-confidence frames first
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    loss = 1 - correct
    for score, color, name in ((f.conf.to_numpy(), BLUE, "model confidence"),
                               (correct, AQUA, "oracle")):
        cov, risk, aurc = risk_coverage(score, loss)
        ax.plot(cov, risk, color=color, label=f"{name} (AURC {aurc:.3f})")
    ax.axhline(loss.mean(), color=MUTED, lw=1, ls="--", label="no rejection")
    ax.set(xlabel="coverage (fraction of frames accepted)", ylabel="risk (1 − fraction correct)", xlim=(0, 1), ylim=(0, None))
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_risk_coverage.{ext}")

    # Confidence vs labelled visible length (orientation proxy)
    p = f[f.has_seed].dropna(subset=["gt_long_mm"])
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.scatter(p.gt_long_mm, p.conf, s=8, color=BLUE, alpha=0.35, linewidths=0)
    bins = np.arange(0, p.gt_long_mm.max() + 2, 2)
    mid = p.groupby(pd.cut(p.gt_long_mm, bins), observed=True).conf.median()
    ax.plot([i.mid for i in mid.index], mid.to_numpy(), color=INK, lw=2, label="median per 2 mm bin")
    ax.set(xlabel="labelled visible source length (mm)", ylabel="top-1 confidence", ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}_conf_vs_length.{ext}")
    print("figures ->", out)


if __name__ == "__main__":
    main()
