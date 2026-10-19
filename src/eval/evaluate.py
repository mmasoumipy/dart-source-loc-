"""Centre-based evaluation of LOSO predictions: visibility (RQ1), localization (RQ2), reliability (RQ3).

Usage: python -m src.eval.evaluate --name loso_s --seed 0 [--gate-mm 5]

Definitions
  top-1        the highest-confidence prediction in a frame (conf 0 if none); the system's answer for that frame.
  correct      the frame has a labelled seed and the top-1 centre is within gate_mm of a labelled centre.
  ambiguous    empty-label frame in a short gap between labelled frames (likely missed label), unless reviewed as
               'absent' in review/negatives_reviewed.csv; excluded from visibility and reliability metrics.
  calibrated   Platt scaling on logit(conf), fitted on the fold's validation segment and applied to its test segment.

Outputs: results/metrics/<name>/s<seed>/{frames.csv, detections.csv, folds.csv, pooled.csv}
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.data.io import load_boxes, load_manifest, obb_geometry
from src.train import fold_segments

ROOT = Path(__file__).resolve().parents[2]
MM_PER_PX = yaml.safe_load((ROOT / "configs" / "pixel_spacing.yaml").read_text())["mm_per_px"]
EPS = 1e-6


# ---------- ground truth ----------

def load_gt():
    m = load_manifest()
    rows = []
    for r in m[m.n_boxes > 0].itertuples():
        c, long_, _, _ = obb_geometry(load_boxes(r.label_path, r.width, r.height))
        for i in range(len(c)):
            rows.append(dict(segment=r.segment, frame=r.frame, gx=c[i, 0], gy=c[i, 1], long_mm=long_[i] * MM_PER_PX))
    return m, pd.DataFrame(rows)


def ambiguous_frames():
    runs = pd.read_csv(ROOT / "results" / "data_audit" / "empty_runs.csv")
    amb = {(r.segment, f) for r in runs[runs.suspicious].itertuples() for f in range(r.first_frame, r.last_frame + 1)}
    review = ROOT / "review" / "negatives_reviewed.csv"
    if review.exists():
        rv = pd.read_csv(review).fillna("")
        amb -= {(r.segment, r.frame) for r in rv.itertuples() if r.verdict.strip() == "absent"}
        amb |= {(r.segment, r.frame) for r in rv.itertuples() if r.verdict.strip() in ("missed_seed", "unsure")}
    return amb


# ---------- per-frame and per-detection tables ----------

def frame_table(pred, m, gt, segment, amb, gate_mm):
    frames = m[m.segment == segment][["segment", "frame", "n_boxes"]].copy()
    p = pred.dropna(subset=["conf"]).sort_values("conf", ascending=False).drop_duplicates("frame")
    frames = frames.merge(p[["frame", "conf", "cx", "cy"]], on="frame", how="left")
    frames["conf"] = frames.conf.fillna(0.0)
    g = gt[gt.segment == segment].groupby("frame")
    err, length = [], []
    for r in frames.itertuples():
        if r.n_boxes == 0 or r.frame not in g.groups or np.isnan(r.cx):
            err.append(np.nan)
            length.append(g.get_group(r.frame).long_mm.max() if r.frame in g.groups else np.nan)
            continue
        gg = g.get_group(r.frame)
        d = np.hypot(gg.gx - r.cx, gg.gy - r.cy) * MM_PER_PX
        err.append(d.min())
        length.append(gg.long_mm.iloc[int(np.argmin(d))])
    frames["err_mm"] = err
    frames["gt_long_mm"] = length
    frames["has_seed"] = frames.n_boxes > 0
    frames["correct"] = frames.has_seed & (frames.err_mm <= gate_mm)
    frames["ambiguous"] = [(s, f) in amb for s, f in zip(frames.segment, frames.frame)]
    return frames


def detection_table(pred, gt, segment, gate_mm):
    """Greedy matching per frame in descending confidence; a detection is TP if within gate of an unmatched seed."""
    dets = pred.dropna(subset=["conf"]).sort_values(["frame", "conf"], ascending=[True, False]).copy()
    g = gt[gt.segment == segment].groupby("frame")
    tp, err = [], []
    for frame, fd in dets.groupby("frame", sort=False):
        gg = g.get_group(frame) if frame in g.groups else None
        used = set()
        for r in fd.itertuples():
            best, bd = None, np.inf
            if gg is not None:
                for j, q in enumerate(gg.itertuples()):
                    d = np.hypot(q.gx - r.cx, q.gy - r.cy) * MM_PER_PX
                    if j not in used and d < bd:
                        best, bd = j, d
            ok = best is not None and bd <= gate_mm
            if ok:
                used.add(best)
            tp.append(ok)
            err.append(bd if ok else np.nan)
    dets["tp"] = tp
    dets["err_mm"] = err
    return dets


# ---------- metrics ----------

def centre_ap(dets, n_gt):
    d = dets.sort_values("conf", ascending=False)
    tp = d.tp.to_numpy().astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, EPS)
    mrec = np.concatenate([[0], recall, [recall[-1] if len(recall) else 0]])
    mpre = np.concatenate([[1], precision, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum(np.diff(mrec) * mpre[1:]))


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def risk_coverage(conf, loss):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    loss = np.asarray(loss, float)[order]
    n = np.arange(1, len(loss) + 1)
    risk = np.cumsum(loss) / n
    return n / len(loss), risk, float(risk.mean())


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt(val_frames):
    v = val_frames[~val_frames.ambiguous]
    if v.correct.nunique() < 2:
        return lambda c: np.asarray(c, float)
    lr = LogisticRegression(C=1e4).fit(logit(v.conf.to_numpy())[:, None], v.correct.to_numpy())
    return lambda c: lr.predict_proba(logit(np.asarray(c, float))[:, None])[:, 1]


def youden_threshold(frames):
    f = frames[~frames.ambiguous]
    if f.has_seed.nunique() < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(f.has_seed, f.conf)
    return float(thr[np.argmax(tpr - fpr)])


def summarize(test, dets, n_gt, thr, gate_mm):
    f = test[~test.ambiguous]
    pos = test[test.has_seed]
    out = dict(frames=len(test), frames_ambiguous=int(test.ambiguous.sum()), frames_with_seed=int(test.has_seed.sum()),
               seeds=n_gt, threshold=thr)
    # RQ1 visibility
    if f.has_seed.nunique() == 2:
        out.update(vis_auroc=roc_auc_score(f.has_seed, f.conf), vis_auprc=average_precision_score(f.has_seed, f.conf))
    out.update(vis_sensitivity=float((pos.conf >= thr).mean()) if len(pos) else np.nan,
               vis_specificity=float((f[~f.has_seed].conf < thr).mean()) if (~f.has_seed).any() else np.nan)
    # RQ2 localization (top-1 on frames with a seed)
    det = pos[pos.conf >= thr]
    out.update(centre_ap=centre_ap(dets, n_gt),
               loc_detected_frac=len(det) / max(len(pos), 1),
               loc_err_median_mm=det.err_mm.median(), loc_err_q25_mm=det.err_mm.quantile(.25),
               loc_err_q75_mm=det.err_mm.quantile(.75),
               **{f"loc_within_{t}mm": float((pos.err_mm <= t).mean()) for t in (2, 5, 10)})
    # RQ3 reliability
    out.update(ece_raw=ece(f.conf, f.correct), ece_platt=ece(f.conf_platt, f.correct),
               aurc=risk_coverage(f.conf, ~f.correct)[2], aurc_oracle=risk_coverage(f.correct, ~f.correct)[2])
    if len(det) > 2:
        out["spearman_conf_err"] = spearmanr(det.conf, det.err_mm).statistic
    pl = pos.dropna(subset=["gt_long_mm"])
    if len(pl) > 2:
        out["spearman_conf_length"] = spearmanr(pl.conf, pl.gt_long_mm).statistic
        for name, sel in (("short", pl.gt_long_mm < 12), ("long", pl.gt_long_mm >= 12)):
            out[f"detected_frac_{name}"] = float((pl[sel].conf >= thr).mean()) if sel.any() else np.nan
            out[f"conf_median_{name}"] = float(pl[sel].conf.median()) if sel.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="loso_s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-mm", type=float, default=5.0)
    args = ap.parse_args()

    pdir = ROOT / "results" / "predictions" / args.name / f"s{args.seed}"
    out = ROOT / "results" / "metrics" / args.name / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    m, gt = load_gt()
    amb = ambiguous_frames()

    all_frames, all_dets, rows = [], [], []
    for k in range(1, 7):
        if not (pdir / f"fold{k}_test.csv").exists():
            print(f"fold {k}: no predictions yet")
            continue
        _, val_seg, test_seg = fold_segments(k)
        val = frame_table(pd.read_csv(pdir / f"fold{k}_val.csv"), m, gt, val_seg, amb, args.gate_mm)
        tp = pd.read_csv(pdir / f"fold{k}_test.csv")
        test = frame_table(tp, m, gt, test_seg, amb, args.gate_mm)
        test["conf_platt"] = platt(val)(test.conf)
        test["fold"] = k
        dets = detection_table(tp, gt, test_seg, args.gate_mm)
        dets["fold"] = k
        thr = youden_threshold(val)
        n_gt = int((gt.segment == test_seg).sum())
        rows.append(dict(fold=k, test=test_seg, val=val_seg, **summarize(test, dets, n_gt, thr, args.gate_mm)))
        all_frames.append(test.assign(threshold=thr))
        all_dets.append(dets)

    if not rows:
        return
    folds = pd.DataFrame(rows)
    frames = pd.concat(all_frames)
    dets = pd.concat(all_dets)
    # pooled: every frame is a test frame exactly once; per-fold thresholds/calibration are already applied
    pooled = summarize(frames.assign(conf=frames.conf), dets, int(gt.segment.isin(folds.test).sum()),
                       float("nan"), args.gate_mm)
    pos = frames[frames.has_seed]
    pooled["vis_sensitivity"] = float((pos.conf >= pos.threshold).mean())
    neg = frames[~frames.has_seed & ~frames.ambiguous]
    pooled["vis_specificity"] = float((neg.conf < neg.threshold).mean())
    det = pos[pos.conf >= pos.threshold]
    pooled.update(loc_detected_frac=len(det) / len(pos), loc_err_median_mm=det.err_mm.median(),
                  loc_err_q25_mm=det.err_mm.quantile(.25), loc_err_q75_mm=det.err_mm.quantile(.75))
    pooled.pop("threshold")

    frames.round(4).to_csv(out / "frames.csv", index=False)
    dets.drop(columns=[c for c in dets.columns if c.startswith("p") and c[1:].isdigit()]).round(4).to_csv(
        out / "detections.csv", index=False)
    folds.round(4).to_csv(out / "folds.csv", index=False)
    pd.Series(pooled).round(4).to_csv(out / "pooled.csv", header=["value"])
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(folds.round(3).T)
        print("\nPOOLED\n", pd.Series(pooled).round(3))


if __name__ == "__main__":
    main()
