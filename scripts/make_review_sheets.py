"""Contact sheets of empty-label frames for manual review (LOCAL ONLY: patient images, never commit).

One sheet per segment in review/: every suspicious empty frame (short run between labelled frames) plus a
sample of the other empty frames, each shown next to the nearest labelled frame for context.
Fill in review/negatives_reviewed.csv: verdict = absent | missed_seed | unsure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.io import load_boxes, load_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review"
TILE = (480, 270)
OTHER_SAMPLE = 12


def tile(row, title):
    im = Image.open(row.image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    for c in load_boxes(row.label_path, row.width, row.height):
        d.line([tuple(p) for p in c] + [tuple(c[0])], fill=(0, 255, 0), width=4)
    im = im.resize(TILE)
    ImageDraw.Draw(im).rectangle((0, 0, TILE[0], 18), fill="black")
    ImageDraw.Draw(im).text((4, 3), title, fill="yellow")
    return im


def main():
    OUT.mkdir(exist_ok=True)
    m = load_manifest()
    runs = pd.read_csv(ROOT / "results" / "data_audit" / "empty_runs.csv")
    records = []
    for seg, g in m.groupby("segment"):
        g = g.set_index("frame")
        labelled = g.index[g.n_boxes > 0].to_numpy()
        r = runs[runs.segment == seg]
        frames = [f for x in r[r.suspicious].itertuples() for f in g.loc[x.first_frame:x.last_frame].index]
        others = [f for f in g.index[g.n_boxes == 0] if f not in frames]
        if others:
            frames += list(np.array(others)[np.linspace(0, len(others) - 1, min(OTHER_SAMPLE, len(others))).astype(int)])
        tiles = []
        for f in frames:
            near = labelled[np.argmin(np.abs(labelled - f))] if len(labelled) else f
            tiles += [tile(g.loc[f], f"{seg} frame {f} EMPTY"), tile(g.loc[near], f"nearest labelled {near}")]
            records.append(dict(segment=seg, frame=int(f), suspicious=f not in others, verdict=""))
        cols = 4
        sheet = Image.new("RGB", (cols * TILE[0], -(-len(tiles) // cols) * TILE[1]))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * TILE[0], (i // cols) * TILE[1]))
        sheet.save(OUT / f"{seg}_empty.jpg", quality=85)
        print(seg, len(frames), "frames")
    pd.DataFrame(records).to_csv(OUT / "negatives_reviewed.csv", index=False)


if __name__ == "__main__":
    main()
