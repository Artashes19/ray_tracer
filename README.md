### ICASSP 2025 Ray-Tracing Approximation (Focused Overview)

This repository provides a fast, portable 2D ray-tracing style approximator to predict indoor pathloss (radiomap) from building inputs. The tracer produces a dB map using:

- Free-space path loss (FSPL)
- Transmission losses when crossing walls (from the transmittance map)
- Specular reflections at reflective walls using precomputed wall normals

This README focuses only on the ray tracer: data conventions (reflectance/transmittance), how the radiomap is computed, how normals are estimated, and how to run and extend the tracer. Our next step is diffraction (GTD/UTD) integration (to be decided).


### Dataset layout and paths

Expected dataset root (default for portable script):

```
/auto/home/artashes/data/train/
  ├─ Inputs/
  │   ├─ Task_2_ICASSP/                   # input PNGs: HxWx3 (R=reflectance, G=transmittance, B=unused)
  ├─ Outputs/
  │   ├─ Task_2_ICASSP/                   # ground-truth pathloss PNGs (single channel)
  ├─ Positions/                           # Positions_B{b}_Ant{ant}_f{f}.csv (columns: X, Y, Azimuth)
  └─ Radiation_Patterns/                  # Ant{ant}_Pattern.csv (360-long gain pattern)
```

Notes:
- The portable script defaults to `/auto/home/artashes/data` (override via `--data-root`).


### Reflectance and Transmittance (data semantics)

- Input images are HxWx3 where:
  - R channel = reflectance (dB-like, ≥ 0 where walls exist; 0 means non-reflective air)
  - G channel = transmittance (dB-like, ≥ 0 where walls exist; 0 means air)
  - B channel is unused
- Binary wall mask is derived from `(reflectance + transmittance) > 0`.
- Intuition:
  - Transmittance accumulates when exiting a wall (air → wall → air) along a ray. The value represents penetration loss added to the running dB budget.
  - Reflectance indicates whether a wall can spawn a specular reflection branch (if reflectance > 0 and a valid wall normal exists at the hit point).


### Quickstart: portable approximator (standalone)

The portable module provides a self-contained way to compute and visualize the combined feature (FSPL + walls + reflections) without touching the training pipeline.

Dependencies for portable run:
- numpy, numba, imageio, matplotlib

Batch normals + demo run (saves a single visualization):

```bash
python portable_combined_feature.py \
  --viz-demo \
  --buildings 1 \
  --data-root /auto/home/artashes/data/train
```

What it does:
- Computes wall normals for selected buildings and saves to `parsed_buildings/B{b}_normals.npz`.
- Finds one sample with both input and GT, reads antenna position/frequency, computes the radiomap via the ray tracer, and saves a 3-panel PNG at `portable_demo_viz.png` with RMSE vs GT.

You can also compute normals or features for specific files:

```bash
# normals from a 3-channel building image
python portable_combined_feature.py \
  --building /path/to/building.png \
  --out-normals /path/to/BX_normals.npz

# normals from separate arrays
python portable_combined_feature.py \
  --refl /path/to/refl.npy \
  --trans /path/to/trans.npy \
  --out-normals /path/to/BX_normals.npz
```


### How the ray tracer computes the radiomap

Inputs per sample:
- Reflectance `refl[y,x]` (R channel), Transmittance `trans[y,x]` (G channel)
- Antenna location `(x_ant, y_ant)` and frequency `f_MHz`
- Wall normals `(nx, ny)` at wall pixels (precomputed)

Algorithm (2D grid, image coordinates):
- Cast `n_angles` rays from the antenna uniformly in `[0, 2π)`.
- March along direction `(dx, dy)` in fixed `radial_step` pixel increments, up to `max_dist`.
- Wall crossing detection with transmittance: when state changes from `wall>0 → air(=0)` at a pixel boundary, add the transmittance value to the accumulated budget (penetration loss at exit).
- FSPL accumulation: for each step, add free-space loss at the current path distance. If `radial_step==1.0`, a LUT indexed by integer pixel distance accelerates FSPL.
- Reflection spawning: if at a hit the reflectance value > 0 and a valid normal `(nx, ny)` exists, create a new branch with direction reflected about the normal (specular). Limit to `max_refl` reflections.
- Painting: write the minimum dB value into each traversed pixel (min-merge), increment a hit counter array for diagnostics.
- Untouched fill: any pixel with zero hits receives direct FSPL from the antenna.

Key parameters:
- `n_angles`: angular resolution (e.g., `360*64`)
- `max_trans`: max number of transmission events per ray branch
- `max_refl`: max number of reflections per ray branch
- `radial_step`: step size in pixels (`1.0` enables FSPL LUT)
- `pixel_size`: meters per pixel (default `0.25`)
- `max_loss`: dB cap for numerical stability

Normals estimation (precomputation):
- For each wall pixel (mask=`(refl+trans)>0`), estimate wall angle in `[0,180)` via multi-scale PCA with an oriented-strip voting heuristic to reduce corner mixing.
- Convert to normals by rotating angle + 90°. Invalid estimates yield `(nx, ny)=(0,0)`.

Variants and extras:
- `approx.py` includes optional backfilling methods (FSPL fill, diffuse solver, direct LOS fill) to replace untouched regions or smooth residuals.
- `beamtrace.py` implements a finite-width beam variant that paints a swath per step for smoother coverage.


### Radiomap prediction via tracer (practical notes)

- The output map is in dB and clipped to `[0,160]` with NaN/Inf sanitization.
- Min-dB merge encourages the lowest-loss path per pixel to dominate (physically plausible in many indoor scenarios).
- FSPL floor ensures monotonicity with range; untouched regions are set to direct FSPL to avoid holes.
- Angular resolution and budgets (`n_angles`, `max_refl`, `max_trans`) drive runtime vs fidelity; start with `360*64`, `max_refl=5`, `max_trans=15`, `radial_step=1.0`.


### Batch feature generation (optional)

`batch_featurize.py` computes the tracer radiomap for all samples in parallel and stores per-sample NPZs plus a metrics CSV for analysis.


### Generating wall normals

Two options:
1) Portable script (single building):
```bash
python portable_combined_feature.py \
  --buildings 10 \
  --data-root /auto/home/artashes/data/train
```
2) Parser utility (batch):
```bash
python parse_buildings.py --buildings 1 2 3 4 5 --overwrite
```

Normals are saved to `parsed_buildings/B{b}_normals.npz` with keys `nx`, `ny`, and `angles`.


### Implementation details and tips

- Threading: many scripts cap threading for Numba and BLAS to avoid oversubscription/crashes in notebooks/REPL:
  - `NUMBA_THREADING_LAYER=workqueue`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`, `TBB_NUM_THREADS=1`.
  - `approx.py` and `portable_combined_feature.py` both set these by default.

- Performance:
  - Use `radial_step=1.0` so FSPL LUT indexing (`int(global_r)`) is valid in Numba kernels.
  - Tuning `n_angles` and reflection budgets (`max_refl`, `max_trans`) has the largest impact on runtime/accuracy.
  - `Approx.predict` supports process/threaded backends and controls Numba threads per worker for stability.

- Min-dB merge: among competing ray branches, the smallest dB at a pixel wins.
- Untouched pixels: direct FSPL fill avoids blank regions; optional backfill methods exist in `approx.py`.


### Module map (ray tracer only)

- `portable_combined_feature.py`: standalone normals + combined-feature computation and a single-sample visualization demo.
- `approx.py`: core Numba implementations (combined/transmission-only), backfill methods, batch runner.
- `beamtrace.py`: beam-based ray painter alternative.
- `normal_parser.py`, `parse_buildings.py`: robust wall-angle estimation and batch normals export.
- `portable_combined_feature.py`: standalone normals + combined-feature computation and single-sample visualization.
- `approx.py`: core Numba implementations (combined/transmission-only), backfills, batch runner.
- `beamtrace.py`: finite-width beam variant.
- `normal_parser.py`, `parse_buildings.py`: wall-angle estimation and batch normals export.
- `batch_featurize.py`: parallel tracer runs with metrics.
- `helper.py`, `_types.py`: dataclasses and utilities for reading/plotting samples.
- `profile_approx.py`, `profile_runtime.py`, `reflect_minimal.py`: profiling and visualization/debug tools.


### Common pitfalls

- Missing normals: combined feature requires `parsed_buildings/B{b}_normals.npz`. Run `parse_buildings.py` or use the portable script in normals mode first.
- Data root mismatch: portable defaults to `/auto/home/artashes/data`, training defaults to `./data/train/` relative to repo; pass `--data-root` to portable, or adjust `main.py` if needed.
- Environment: ensure `numba`, `numpy`, `imageio`, `matplotlib`, `scipy`, `torch`, `torchvision`, `segmentation_models_pytorch`, `piq`, `pandas`, `tqdm` are installed in your conda env.


### Example: quick viz (single-sample radiomap)

```bash
# 1) Compute normals for Building 1
python portable_combined_feature.py --buildings 1 --data-root /auto/home/artashes/data/train

# 2) Run demo to compute combined feature for one sample and visualize
python portable_combined_feature.py --viz-demo --data-root /auto/home/artashes/data/train
```

This outputs `portable_demo_viz.png` with GT, tracer radiomap, and absolute diff, plus an RMSE number.


### Next step: diffraction (GTD/UTD) plan

Goal: Incorporate edge diffraction so NLOS pixels behind corners receive energy without requiring specular reflection paths.

Proposed milestones:
1) Edge detection: derive candidate edges from the binary wall mask (e.g., Canny + thinning or vectorization from normals; or infer edges at wall-air boundaries). Store edge segments and local face normals.
2) Wedge modeling: approximate edges as 2D wedges with interior angle; assign material-dependent parameters if available (simplify initially to canonical coefficients).
3) Ray–edge interaction: for each primary (and optionally reflected) ray, detect proximity to edges and evaluate a diffraction coefficient D(·) (GTD or UTD) to spawn a diffracted branch with an outgoing direction toward the receiver pixel set or along a sampled fan.
4) Coefficient implementation: start amplitude-only (magnitude of D), ignoring phase to match current dB pipeline. If UTD is chosen, add transition functions later to handle shadow boundaries smoothly.
5) Integration and budgets: introduce `max_diff` (diffraction budget) and maintain min-dB merge with FSPL + transmission + reflection.
6) Validation: unit tests on canonical scenes (single wedge, corridor corner), compare coverage vs transmission-only and reflection-only cases; measure RMSE improvements.

Initial simplifications:
- 2D wedge, frequency-invariant diffraction gain; use simple angular dependence for gain.
- Spawn a single dominant diffracted direction per edge–ray interaction toward grid pixels along visibility fans to limit combinatorics (or sample a small fixed fan).

Performance considerations:
- Pre-index edges spatially (grid or R-tree) to query nearby edges per step efficiently.
- Keep `radial_step==1.0` to preserve the FSPL LUT path.



### Contact points

- Portable approximator entry points: `compute_and_save_normals`, `compute_combined_feature` in `portable_combined_feature.py`.
- Training entry point: `main.py`.
- Batch standalone feature metrics: `batch_featurize.py`.


