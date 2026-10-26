"""Train one LOSO fold of YOLO26 (OBB or axis-aligned detect) and save raw predictions for the val and test segments.

Usage: python -m src.train --fold 3 [--seed 0] [--epochs N] [--model weights/yolo26s-obb.pt --batch 8 --name loso_s]
       python -m src.train --fold 3 --model weights/yolo26s.pt --data-dir data/prepared_aabb --name loso_aabb

Outputs
  runs/loso/s<seed>/fold<k>/            Ultralytics run (weights, curves; git-ignored)
  results/predictions/s<seed>/fold<k>_{val,test}.csv   one row per prediction (numbers only)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from src.data.io import fold_segments, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def image_lists(cfg, k):
    m = load_manifest()
    m["prepared"] = [str(ROOT / cfg["data_dir"] / s / "images" / Path(p).name) for s, p in zip(m.segment, m.image)]
    train_segs, val, test = fold_segments(k)
    train = m[m.segment.isin(train_segs)]
    if cfg["exclude_suspicious_empty_from_train"]:
        runs = pd.read_csv(ROOT / "results" / "data_audit" / "empty_runs.csv")
        bad = {(r.segment, f) for r in runs[runs.suspicious].itertuples()
               for f in range(r.first_frame, r.last_frame + 1)}
        train = train[[(s, f) not in bad for s, f in zip(train.segment, train.frame)]]
    return {"train": train, "val": m[m.segment == val], "test": m[m.segment == test]}


def write_data_yaml(lists, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"path": str(run_dir), "names": {0: "seed"}}
    for role, df in lists.items():
        (run_dir / f"{role}.txt").write_text("".join(p + "\n" for p in df.prepared))
        cfg[role] = f"{role}.txt"
    path = run_dir / "data.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def predict_to_csv(model, df, out_csv, imgsz, pcfg):
    rows = []
    paths = list(df.prepared)
    for start in range(0, len(paths), 32):
        for path, frame, seg, r in zip(paths[start:start + 32], df.frame.iloc[start:start + 32],
                                       df.segment.iloc[start:start + 32],
                                       model.predict(paths[start:start + 32], imgsz=imgsz, conf=pcfg["conf"],
                                                     max_det=pcfg["max_det"], verbose=False)):
            b = r.obb if r.obb is not None else r.boxes
            if b is None or len(b) == 0:
                rows.append(dict(segment=seg, frame=frame, det=-1))
                continue
            if r.obb is not None:
                xywhr = b.xywhr.cpu().numpy()
                corners = b.xyxyxyxy.cpu().numpy().reshape(-1, 8)
            else:  # axis-aligned: rot 0, corners clockwise from top-left
                xywh = b.xywh.cpu().numpy()
                xywhr = np.c_[xywh, np.zeros(len(xywh))]
                x0, y0, x1, y1 = b.xyxy.cpu().numpy().T
                corners = np.stack([x0, y0, x1, y0, x1, y1, x0, y1], 1)
            for i, conf in enumerate(b.conf.cpu().numpy()):
                rows.append(dict(segment=seg, frame=frame, det=i, conf=float(conf),
                                 cx=xywhr[i, 0], cy=xywhr[i, 1], w=xywhr[i, 2], h=xywhr[i, 3], rot=xywhr[i, 4],
                                 **{f"p{j}": corners[i, j] for j in range(8)}))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--name", default="loso")
    ap.add_argument("--model", help="override config model, e.g. weights/yolo26s-obb.pt")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--data-dir", help="override config data_dir, e.g. data/prepared_aabb")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override train settings, e.g. --set translate=0.4 scale=0.5")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    tcfg = dict(cfg["train"], seed=args.seed)
    if args.epochs:
        tcfg["epochs"] = args.epochs
    if args.batch:
        tcfg["batch"] = args.batch
    if args.model:
        cfg["model"] = args.model
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    for kv in args.set:
        key, value = kv.split("=", 1)
        tcfg[key] = yaml.safe_load(value)
    run_dir = ROOT / "runs" / args.name / f"s{args.seed}" / f"fold{args.fold}"
    lists = image_lists(cfg, args.fold)
    print({k: f"{sorted(v.segment.unique())} n={len(v)}" for k, v in lists.items()})
    data_yaml = write_data_yaml(lists, run_dir / "data")

    model = YOLO(str(ROOT / cfg["model"]))
    model.train(data=str(data_yaml), project=str(run_dir), name="train", exist_ok=True, device=0, **tcfg)

    best = YOLO(str(run_dir / "train" / "weights" / "best.pt"))
    out = ROOT / "results" / "predictions" / args.name / f"s{args.seed}"
    for role in ("val", "test"):
        predict_to_csv(best, lists[role], out / f"fold{args.fold}_{role}.csv", tcfg["imgsz"], cfg["predict"])
    print("done:", out)


if __name__ == "__main__":
    main()
