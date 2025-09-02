#!/usr/bin/env python3
import os, re, argparse, csv
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

# Map filename stem to paths
NAME_RE = re.compile(r'^B(\d+)_Ant(\d+)_f(\d+)_S(\d+)$')


def load_gt(gt_dir: Path, stem: str) -> np.ndarray:
    p = gt_dir / f"{stem}.png"
    img = iio.imread(p)
    if img.ndim == 3:
        gt = img[..., 0].astype(np.float32)
    else:
        gt = img.astype(np.float32)
    gt = np.nan_to_num(gt, nan=160.0, posinf=160.0, neginf=0.0)
    if gt.max() > 0:
        gt = np.minimum(gt, 160.0)
    return gt


def align_shapes(a: np.ndarray, b: np.ndarray):
    if a.shape == b.shape:
        return a, b
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def save_triplet(gt: np.ndarray, pred: np.ndarray, out_png: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(gt, vmin=0, vmax=160)
    axes[0].set_title('GT')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pred, vmin=0, vmax=160)
    axes[1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    diff = np.abs(gt - pred)
    im2 = axes[2].imshow(diff, cmap='gray')
    axes[2].set_title('Diff |GT-Pred|')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser('Evaluate NPZ predictions vs GT and save PNGs')
    ap.add_argument('--npz_dir', type=str, default='data/sample100_npz')
    ap.add_argument('--pred_dir', type=str, default='data/sample100_preds')
    ap.add_argument('--gt_dir', type=str, default='data/sample100/Outputs')
    ap.add_argument('--out_dir', type=str, default='data/sample100_viz')
    ap.add_argument('--csv', type=str, default='data/sample100_metrics.csv')
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)

    stems = sorted([p.stem for p in npz_dir.glob('*.npz')])
    rmses = []

    for i, stem in enumerate(stems, 1):
        pred_path = pred_dir / f'{stem}.npy'
        if not pred_path.exists():
            print(f'[skip] missing pred: {pred_path}')
            continue
        pred = np.load(pred_path).astype(np.float32)
        pred = np.nan_to_num(pred, nan=160.0, posinf=160.0, neginf=0.0)
        pred = np.minimum(pred, 160.0)

        gt = load_gt(gt_dir, stem)
        gt, pred = align_shapes(gt, pred)
        rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
        rmses.append((stem, rmse))

        out_png = out_dir / f'{stem}.png'
        save_triplet(gt, pred, out_png, f'{stem}  RMSE={rmse:.2f} dB')
        print(f'[{i}/{len(stems)}] RMSE={rmse:.3f} -> {out_png}')

    if rmses:
        mean_rmse = float(np.mean([r for _, r in rmses]))
    else:
        mean_rmse = float('nan')

    # write CSV
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['stem', 'rmse'])
        for stem, r in rmses:
            w.writerow([stem, f'{r:.6f}'])
        w.writerow(['mean_rmse', f'{mean_rmse:.6f}'])
    print(f'Done. Mean RMSE={mean_rmse:.3f}. CSV -> {args.csv}')

if __name__ == '__main__':
    main()
