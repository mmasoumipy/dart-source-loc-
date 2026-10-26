"""Train one LOSO fold of Rotated Faster R-CNN (ResNet-50 + FPN, Detectron2 RRPN/RROIHeads) and save raw predictions.

Runs in .venv-rcnn (Detectron2 built with CUDA). Same folds, masked images, prediction CSV format and evaluation
(src.eval.evaluate) as the YOLO runs. The checkpoint kept is the one with the best centre AP on the validation segment.

Usage: .venv-rcnn/bin/python -m src.rcnn --fold 3 [--seed 0] [--iters 6000] [--name loso_rcnn]

Outputs
  runs/<name>/s<seed>/fold<k>/                         Detectron2 run (weights, log; git-ignored)
  results/predictions/<name>/s<seed>/fold<k>_{val,test}.csv   one row per prediction
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.structures import BoxMode

from src.data.io import fold_segments, load_boxes, load_manifest, obb_geometry
from src.eval.evaluate import centre_ap, detection_table, load_gt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "prepared"
SHORT_EDGE, MAX_SIZE = 576, 1024  # 1920x1080 -> 1024x576, the same scale as the YOLO runs (imgsz 1024)


# ---------- data ----------

def frames_for(segments):
    m = load_manifest()
    m = m[m.segment.isin(segments)]
    m["prepared"] = [str(DATA_DIR / s / "images" / Path(p).name) for s, p in zip(m.segment, m.image)]
    return m


def to_xywha(corners):
    """(n, 4, 2) corners in image pixels -> Detectron2 XYWHA_ABS: centre, long side, short side, CCW angle in degrees."""
    c, long_, short, ang = obb_geometry(corners)
    a = -ang  # image y points down, Detectron2 angles are counter-clockwise
    a = (a + 90) % 180 - 90
    return np.c_[c, long_, short, a]


def xywha_corners(b):
    """Inverse of to_xywha: (n, 5) XYWHA_ABS -> (n, 4, 2) corners in image pixels."""
    t = -np.radians(b[:, 4])
    u = np.stack([np.cos(t), np.sin(t)], 1) * b[:, 2:3] / 2
    v = np.stack([-np.sin(t), np.cos(t)], 1) * b[:, 3:4] / 2
    c = b[:, None, :2]
    return c + np.stack([-u - v, u - v, u + v, -u + v], 1)


def records(segments, exclude=frozenset()):
    out = []
    for r in frames_for(segments).itertuples():
        if (r.segment, r.frame) in exclude:
            continue
        boxes = to_xywha(load_boxes(r.label_path, r.width, r.height))
        out.append(dict(file_name=r.prepared, image_id=f"{r.segment}/{r.frame}", height=r.height, width=r.width,
                        segment=r.segment, frame=r.frame,
                        annotations=[dict(bbox=b.tolist(), bbox_mode=BoxMode.XYWHA_ABS, category_id=0) for b in boxes]))
    return out


def suspicious_empty():
    runs = pd.read_csv(ROOT / "results" / "data_audit" / "empty_runs.csv")
    return {(r.segment, f) for r in runs[runs.suspicious].itertuples() for f in range(r.first_frame, r.last_frame + 1)}


class RotatedMapper:
    """Load an image, apply resize/flip/brightness to it and its rotated boxes, return Instances with RotatedBoxes."""

    def __init__(self, train, fliplr):
        aug = [T.ResizeShortestEdge(SHORT_EDGE, MAX_SIZE)]
        if train:
            aug += [T.RandomFlip(prob=fliplr, horizontal=True, vertical=False), T.RandomBrightness(0.7, 1.3)]
        self.aug, self.train = T.AugmentationList(aug), train

    def __call__(self, d):
        d = copy.deepcopy(d)
        image = utils.read_image(d["file_name"], format="BGR")
        inp = T.AugInput(image)
        tf = self.aug(inp)
        image = inp.image
        d["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.train:
            annos = [utils.transform_instance_annotations(a, tf, image.shape[:2]) for a in d.pop("annotations")]
            d["instances"] = utils.annotations_to_instances_rotated(annos, image.shape[:2])
        return d


# ---------- model ----------

def build_cfg(args, run_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.1, 0.25, 0.5, 1.0]]  # h/w; sources are long and thin
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, -60, -30, 0, 30, 60]]
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1, 1, 1, 1, 1)
    cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5, 1)
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST = SHORT_EDGE, MAX_SIZE
    cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = (f"{args.name}_f{args.fold}_train",), ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # empty frames are the negatives
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = 0.0025 * args.batch / 2
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.STEPS = (int(args.iters * 0.67), int(args.iters * 0.89))
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 10 ** 9  # BestCentreAP saves the checkpoint instead
    cfg.SOLVER.AMP.ENABLED = False  # rotated ops have no half-precision kernels
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = str(run_dir)
    return cfg


class Trainer(DefaultTrainer):
    fliplr = 0.5

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RotatedMapper(True, cls.fliplr))


# ---------- prediction and model selection ----------

@torch.no_grad()
def predict(model, recs, batch=8):
    model.eval()
    mapper, rows = RotatedMapper(False, 0), []
    for start in range(0, len(recs), batch):
        chunk = recs[start:start + batch]
        outs = model([mapper(r) for r in chunk])
        for r, o in zip(chunk, outs):
            inst = o["instances"].to("cpu")
            if len(inst) == 0:
                rows.append(dict(segment=r["segment"], frame=r["frame"], det=-1))
                continue
            b = inst.pred_boxes.tensor.numpy()
            corners = xywha_corners(b).reshape(-1, 8)
            for i, conf in enumerate(inst.scores.numpy()):
                rows.append(dict(segment=r["segment"], frame=r["frame"], det=i, conf=float(conf),
                                 cx=b[i, 0], cy=b[i, 1], w=b[i, 2], h=b[i, 3], rot=-np.radians(b[i, 4]),
                                 **{f"p{j}": corners[i, j] for j in range(8)}))
    model.train()
    return pd.DataFrame(rows)


class BestCentreAP(HookBase):
    """Every `period` iterations, score the validation segment by centre AP and keep the best weights."""

    def __init__(self, recs, segment, period, gt):
        self.recs, self.segment, self.period, self.gt = recs, segment, period, gt
        self.n_gt = int((gt.segment == segment).sum())
        self.best = -1.0

    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.period and it != self.trainer.max_iter:
            return
        pred = predict(self.trainer.model, self.recs)
        ap = centre_ap(detection_table(pred, self.gt, self.segment, 5.0), self.n_gt)
        self.trainer.storage.put_scalar("val_centre_ap", ap, smoothing_hint=False)
        if ap > self.best:
            self.best = ap
            self.trainer.checkpointer.save("model_best", iteration=it, val_centre_ap=ap)
        print(f"iter {it}: val centre AP {ap:.3f} (best {self.best:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-period", type=int, default=500)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--name", default="loso_rcnn")
    args = ap.parse_args()

    train_segs, val_seg, test_seg = fold_segments(args.fold)
    excl = suspicious_empty() if yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())[
        "exclude_suspicious_empty_from_train"] else frozenset()
    train_recs = records(train_segs, excl)
    DatasetCatalog.register(f"{args.name}_f{args.fold}_train", lambda: train_recs)
    val_recs, test_recs = records([val_seg]), records([test_seg])
    print({"train": f"{train_segs} n={len(train_recs)}", "val": f"{val_seg} n={len(val_recs)}",
           "test": f"{test_seg} n={len(test_recs)}"})

    run_dir = ROOT / "runs" / args.name / f"s{args.seed}" / f"fold{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(args, run_dir)
    (run_dir / "config.yaml").write_text(cfg.dump())
    Trainer.fliplr = args.fliplr
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    _, gt = load_gt()
    trainer.register_hooks([BestCentreAP(val_recs, val_seg, args.eval_period, gt)])
    trainer.train()

    trainer.checkpointer.load(str(run_dir / "model_best.pth"))
    out = ROOT / "results" / "predictions" / args.name / f"s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for role, recs in (("val", val_recs), ("test", test_recs)):
        predict(trainer.model, recs).round(4).to_csv(out / f"fold{args.fold}_{role}.csv", index=False)
    print("done:", out)


if __name__ == "__main__":
    main()
