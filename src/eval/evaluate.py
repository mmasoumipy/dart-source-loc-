"""Centre-based evaluation of LOSO predictions: visibility (RQ1), localization (RQ2), reliability (RQ3).

Usage: python -m src.eval.evaluate --name loso_s --seed 0 [--gate-mm 5] [--calib pooled]

Definitions
  top-1        the highest-confidence prediction in a frame (conf 0 if none); the system's answer for that frame.
  correct      the frame has a labelled seed and the top-1 centre is within gate_mm of a labelled centre.
  correct_line the frame has a labelled seed and the top-1 centre is within LINE_MM of the labelled source axis
               (ignores the position along the source, which is ambiguous for long visible tracks).
  ambiguous    empty-label frame in a short gap between labelled frames (likely missed label), unless reviewed as
               'absent' in review/negatives_reviewed.csv; excluded from visibility and reliability metrics.
  calibrated   Platt scaling on logit(conf), fitted on the fold's validation segment and applied to its test segment.
               With --calib pooled, the threshold and Platt fit use the validation frames of every fold whose
               validation segment is not this fold's test segment (outputs go to s<seed>_pooledcal/).
  fp_per_frame detections above the threshold that match no labelled seed, per non-ambiguous frame.
  short/long   labelled visible length < 12 mm (partly visible) or >= 12 mm (fully in plane).
  background   phantom or tissue per segment, from configs/backgrounds.yaml.

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

from src.data.io import fold_segments, load_boxes, load_manifest, obb_geometry

ROOT = Path(__file__).resolve().parents[2]
MM_PER_PX = yaml.safe_load((ROOT / "configs" / "pixel_spacing.yaml").read_text())["mm_per_px"]
EPS = 1e-6
LINE_MM = 2.0
TARGETS = {"": "correct", "_line": "correct_line"}


# ---------- ground truth ----------

def load_gt():
    m = load_manifest()
    rows = []
    for r in m[m.n_boxes > 0].itertuples():
        c, long_, _, ang = obb_geometry(load_boxes(r.label_path, r.width, r.height))
        for i in range(len(c)):
            rows.append(dict(segment=r.segment, frame=r.frame, gx=c[i, 0], gy=c[i, 1], long_mm=long_[i] * MM_PER_PX,
                             ux=np.cos(np.radians(ang[i])), uy=np.sin(np.radians(ang[i]))))
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
    err, along, cross, length = [], [], [], []
    for r in frames.itertuples():
        if r.n_boxes == 0 or r.frame not in g.groups or np.isnan(r.cx):
            err.append(np.nan); along.append(np.nan); cross.append(np.nan)
            length.append(g.get_group(r.frame).long_mm.max() if r.frame in g.groups else np.nan)
            continue
        gg = g.get_group(r.frame)
        dx, dy = (r.cx - gg.gx).to_numpy(), (r.cy - gg.gy).to_numpy()
        i = int(np.argmin(np.hypot(dx, dy)))
        err.append(np.hypot(dx[i], dy[i]) * MM_PER_PX)
        # offset along the labelled source axis and perpendicular to it
        along.append(abs(dx[i] * gg.ux.iloc[i] + dy[i] * gg.uy.iloc[i]) * MM_PER_PX)
        cross.append(abs(-dx[i] * gg.uy.iloc[i] + dy[i] * gg.ux.iloc[i]) * MM_PER_PX)
        length.append(gg.long_mm.iloc[i])
    frames["err_mm"] = err
    frames["err_along_mm"] = along
    frames["err_cross_mm"] = cross
    frames["gt_long_mm"] = length
    frames["has_seed"] = frames.n_boxes > 0
    frames["correct"] = frames.has_seed & (frames.err_mm <= gate_mm)
    frames["correct_line"] = frames.has_seed & (frames.err_cross_mm <= LINE_MM)
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


def platt(val_frames, target="correct"):
    v = val_frames[~val_frames.ambiguous]
    if v[target].nunique() < 2:
        return lambda c: np.asarray(c, float)
    lr = LogisticRegression(C=1e4).fit(logit(v.conf.to_numpy())[:, None], v[target].to_numpy())
    return lambda c: lr.predict_proba(logit(np.asarray(c, float))[:, None])[:, 1]


def youden_threshold(frames):
    f = frames[~frames.ambiguous]
    if f.has_seed.nunique() < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(f.has_seed, f.conf)
    return float(thr[np.argmax(tpr - fpr)])


def summarize(test, dets, n_gt, gate_mm):
    test = test.assign(above=test.conf >= test.threshold)
    f = test[~test.ambiguous]
    pos = test[test.has_seed]
    out = dict(frames=len(test), frames_ambiguous=int(test.ambiguous.sum()), frames_with_seed=int(test.has_seed.sum()),
               seeds=n_gt)
    # RQ1 visibility
    if f.has_seed.nunique() == 2:
        out.update(vis_auroc=roc_auc_score(f.has_seed, f.conf), vis_auprc=average_precision_score(f.has_seed, f.conf))
    out.update(vis_sensitivity=float(pos.above.mean()) if len(pos) else np.nan,
               vis_specificity=float((~f[~f.has_seed].above).mean()) if (~f.has_seed).any() else np.nan)
    # RQ2 localization (top-1 on frames with a seed)
    det = pos[pos.above]
    d = dets.merge(test[["segment", "frame", "threshold", "ambiguous"]], on=["segment", "frame"])
    d = d[~d.ambiguous & (d.conf >= d.threshold)]
    out.update(fp_per_frame=float((~d.tp).sum() / max(len(f), 1)),
               fp_per_empty_frame=float((~d[d.frame.isin(f[~f.has_seed].frame)].tp).sum() / max((~f.has_seed).sum(), 1)))
    out.update(centre_ap=centre_ap(dets, n_gt),
               loc_detected_frac=len(det) / max(len(pos), 1),
               loc_err_median_mm=det.err_mm.median(), loc_err_q25_mm=det.err_mm.quantile(.25),
               loc_err_q75_mm=det.err_mm.quantile(.75),
               loc_along_median_mm=det.err_along_mm.median(), loc_cross_median_mm=det.err_cross_mm.median(),
               loc_cross_within_2mm=float((pos.err_cross_mm <= 2).mean()),
               **{f"loc_within_{t}mm": float((pos.err_mm <= t).mean()) for t in (2, 5, 10)})
    # RQ3 reliability
    for suffix, target in TARGETS.items():
        y = f[target]
        out.update({f"accuracy{suffix}": float(y.mean()),
                    f"ece_raw{suffix}": ece(f.conf, y), f"ece_platt{suffix}": ece(f[f"conf_platt{suffix}"], y),
                    f"aurc{suffix}": risk_coverage(f.conf, ~y)[2], f"aurc_oracle{suffix}": risk_coverage(y, ~y)[2]})
    if len(det) > 2:
        out["spearman_conf_err"] = spearmanr(det.conf, det.err_mm).statistic
    pl = pos.dropna(subset=["gt_long_mm"])
    if len(pl) > 2:
        out["spearman_conf_length"] = spearmanr(pl.conf, pl.gt_long_mm).statistic
        for name, sel in (("short", pl.gt_long_mm < 12), ("long", pl.gt_long_mm >= 12)):
            out[f"detected_frac_{name}"] = float(pl[sel].above.mean()) if sel.any() else np.nan
            out[f"conf_median_{name}"] = float(pl[sel].conf.median()) if sel.any() else np.nan
            out[f"within_5mm_{name}"] = float((pl[sel].err_mm <= 5).mean()) if sel.any() else np.nan
            out[f"cross_within_2mm_{name}"] = float((pl[sel].err_cross_mm <= 2).mean()) if sel.any() else np.nan
            out[f"err_median_{name}"] = float(pl[sel & pl.above].err_mm.median()) if (sel & pl.above).any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="loso_s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-mm", type=float, default=5.0)
    ap.add_argument("--calib", choices=["fold", "pooled"], default="fold",
                    help="fit threshold/Platt on the fold's own val segment, or on all other folds' val frames")
    args = ap.parse_args()

    pdir = ROOT / "results" / "predictions" / args.name / f"s{args.seed}"
    out = ROOT / "results" / "metrics" / args.name / (f"s{args.seed}" + ("_pooledcal" if args.calib == "pooled" else ""))
    out.mkdir(parents=True, exist_ok=True)
    m, gt = load_gt()
    amb = ambiguous_frames()

    folds_done = [k for k in range(1, 7) if (pdir / f"fold{k}_test.csv").exists()]
    vals = {k: frame_table(pd.read_csv(pdir / f"fold{k}_val.csv"), m, gt, fold_segments(k)[1], amb, args.gate_mm)
            for k in folds_done}
    all_frames, all_dets, rows = [], [], []
    for k in range(1, 7):
        if k not in folds_done:
            print(f"fold {k}: no predictions yet")
            continue
        _, val_seg, test_seg = fold_segments(k)
        val = vals[k]
        if args.calib == "pooled":
            val = pd.concat([v for j, v in vals.items() if fold_segments(j)[1] != test_seg], ignore_index=True)
        tp = pd.read_csv(pdir / f"fold{k}_test.csv")
        test = frame_table(tp, m, gt, test_seg, amb, args.gate_mm)
        for suffix, target in TARGETS.items():
            test[f"conf_platt{suffix}"] = platt(val, target)(test.conf)
        test["fold"] = k
        test["threshold"] = youden_threshold(val)
        dets = detection_table(tp, gt, test_seg, args.gate_mm)
        dets["fold"] = k
        n_gt = int((gt.segment == test_seg).sum())
        rows.append(dict(fold=k, test=test_seg, val=val_seg, threshold=test.threshold.iloc[0],
                         **summarize(test, dets, n_gt, args.gate_mm)))
        all_frames.append(test)
        all_dets.append(dets)

    if not rows:
        return
    folds = pd.DataFrame(rows)
    frames = pd.concat(all_frames, ignore_index=True)
    dets = pd.concat(all_dets)
    # pooled: every frame is a test frame exactly once; each fold's own threshold and calibration apply
    pooled = summarize(frames, dets, int(gt.segment.isin(folds.test).sum()), args.gate_mm)

    bg = yaml.safe_load((ROOT / "configs" / "backgrounds.yaml").read_text())["segments"]
    by_bg = {}
    for name in sorted(set(bg.values())):
        segs = [s for s, b in bg.items() if b == name and s in set(folds.test)]
        if segs:
            by_bg[name] = summarize(frames[frames.segment.isin(segs)], dets[dets.segment.isin(segs)],
                                    int(gt.segment.isin(segs).sum()), args.gate_mm)
    pd.DataFrame(by_bg).round(4).to_csv(out / "by_background.csv")

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
