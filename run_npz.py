#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
import numpy as np
from portable_combined_feature import precompute_wall_angles_pca, compute_combined_feature

def run_one(npz_path: Path, out_path: Path, n_angles=360*64):
    d = np.load(npz_path)
    refl = d['refl'].astype(np.float32)
    trans = d['trans'].astype(np.float32)
    x_ant = float(d['x_ant_px'])
    y_ant = float(d['y_ant_px'])
    freq = float(d['freq_MHz'])
    px   = float(d['pixel_size_m'])
    mask = (refl + trans) > 0
    angles = precompute_wall_angles_pca(mask.astype(np.uint8))
    rad = np.deg2rad(angles + 90.0)
    nx = np.cos(rad).astype(np.float32)
    ny = np.sin(rad).astype(np.float32)
    nx[angles < 0] = 0.0
    ny[angles < 0] = 0.0
    feat = compute_combined_feature(
        refl, trans, x_ant, y_ant, freq,
        nx=nx, ny=ny,
        n_angles=n_angles, max_refl=5, max_trans=15,
        radial_step=1.0, pixel_size=px, max_loss=160.0
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, feat)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--npz_dir', type=str, default='data/sample100_npz')
    p.add_argument('--out_dir', type=str, default='data/sample100_preds')
    p.add_argument('--angles', type=int, default=360*64)
    args=p.parse_args()
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    paths = sorted(list(npz_dir.glob('*.npz')))
    for i, pth in enumerate(paths, 1):
        out = out_dir / (pth.stem + '.npy')
        run_one(pth, out, n_angles=int(args.angles))
        print(f"[{i}/{len(paths)}] -> {out}")
