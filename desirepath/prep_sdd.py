"""Generate walkable-surface masks for SDD videos (DesirePath-NF v2, P0).

Auto-segmentation (no hand annotation) of each scene's reference.jpg:
  paved      = bright non-green  (the 'provided' path network)
  grass      = smooth vegetation (walkable but not paved) -> desire-path surface
  obstacle   = textured vegetation (trees/planters) + large dark structures
Decisions (delegated): shadow -> underlying ground by neighbourhood majority;
high-texture vegetation -> obstacle.  paved vs non-paved is color-separable;
grass vs tree is separated by local texture (grass smooth, tree textured).

Raw SDD comes from `git clone https://github.com/flclain/StanfordDroneDataset`
(annotations.txt + reference.jpg per video, no auth, no video files needed).
Set SDD_RAW to that checkout.  Output: data/sdd/{scene}/{video}/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import (binary_closing, distance_transform_edt, label,
                           uniform_filter)

HERE = Path(__file__).resolve().parent.parent
SDD_RAW = Path(os.environ.get(
    "SDD_RAW",
    "/private/tmp/claude-501/-Users-kusakabe-Library-CloudStorage-GoogleDrive-"
    "kan86yeaoh-gmail-com-My-Drive-Work-SMU-documentation-----------3dreconstruction"
    "-rosbag-DesirePath-NF/1d99534e-f003-40c6-8e07-1d1cc96d82ff/scratchpad/sdd_anno"))
OUT = HERE / "data" / "sdd"


def _local_std(V, size):
    m = uniform_filter(V, size)
    m2 = uniform_filter(V * V, size)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def segment(img):
    """img HxWx3 uint8/float -> (paved, grass, obstacle) bool masks."""
    img = img.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    V = img.mean(-1)
    grn = 2 * G - R - B
    veg = grn > 18
    tex = _local_std(V, 15)
    tree = veg & (tex > 17)                 # textured vegetation = obstacle
    grass = veg & ~tree
    nonveg = ~veg
    dark = nonveg & (V < 50)                # shadow / dark structure
    gf = uniform_filter(grass.astype(np.float32), 61)
    grass = grass | (dark & (gf > 0.4))     # shadow over grass -> grass
    dark = dark & ~(gf > 0.4)
    paved0 = nonveg & ~dark
    pf = uniform_filter(paved0.astype(np.float32), 61)
    paved = paved0 | (dark & (pf > 0.35))   # shadow over pavement -> pavement
    building = dark & ~(pf > 0.35)
    obstacle = tree | building
    lab, n = label(obstacle)                # drop obstacle speckle
    if n:
        sizes = np.bincount(lab.ravel())
        obstacle = np.isin(lab, np.where(sizes > 1500)[0]) & (lab > 0)
    obstacle = binary_closing(obstacle, iterations=2)
    grass = grass & ~obstacle
    paved = ~grass & ~obstacle
    return paved, grass, obstacle


def _peds(fp):
    ids, frame, cx, cy = [], [], [], []
    for line in open(fp):
        p = line.split()
        if p[-1].strip('"') != "Pedestrian" or int(p[6]) == 1:
            continue
        ids.append(int(p[0])); frame.append(int(p[5]))
        cx.append((float(p[1]) + float(p[3])) / 2)
        cy.append((float(p[2]) + float(p[4])) / 2)
    return (np.array(ids), np.array(frame),
            np.array(cx, np.float32), np.array(cy, np.float32))


def make_video(scene, video):
    import matplotlib.image as mpimg
    src = SDD_RAW / scene / video
    img = mpimg.imread(src / "reference.jpg")
    H, W = img.shape[:2]
    paved, grass, obstacle = segment(img)
    walkable = paved | grass
    dist_obs = distance_transform_edt(~obstacle).astype(np.float32)
    dist_pav = distance_transform_edt(~paved).astype(np.float32)
    od = OUT / scene / video
    od.mkdir(parents=True, exist_ok=True)

    def save(a, name):
        mpimg.imsave(od / name, a.astype(np.uint8) * 255, cmap="gray", vmin=0, vmax=255)
    save(paved, "paved_mask.png"); save(walkable, "walkable_mask.png")
    save(obstacle, "obstacle_mask.png")
    np.save(od / "dist_to_obstacle.npy", dist_obs)
    np.save(od / "dist_to_paved.npy", dist_pav)
    ids, frame, cx, cy = _peds(src / "annotations.txt")
    np.savez_compressed(od / "peds.npz", ids=ids, frame=frame, cx=cx, cy=cy)
    xi = np.clip(cx.astype(int), 0, W - 1); yi = np.clip(cy.astype(int), 0, H - 1)
    on_walk = float(walkable[yi, xi].mean()) if len(cx) else 0.0
    meta = dict(scene=scene, video=video, H=H, W=W, px_per_m=None,
                n_ped_rows=int(len(cx)), ped_on_walkable=on_walk,
                frac=dict(paved=float(paved.mean()), grass=float(grass.mean()),
                          obstacle=float(obstacle.mean())))
    (od / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(scenes=("hyang", "little", "quad", "gates")):
    for scene in scenes:
        vids = sorted(p.name for p in (SDD_RAW / scene).glob("video*"))
        rows = [make_video(scene, v) for v in vids]
        nrow = sum(m["n_ped_rows"] for m in rows)
        ow = np.mean([m["ped_on_walkable"] for m in rows])
        print(f"{scene:8s} {len(vids):2d} videos  ped-rows {nrow:7d}  "
              f"mean ped-on-walkable {ow*100:.0f}%")


if __name__ == "__main__":
    main()
