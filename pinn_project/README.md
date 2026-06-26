# PINN Haptic Force Prediction — Liver Simulation

A PyTorch model that predicts haptic contact force for a SOFA liver-tissue
simulation, trained on real recorded touch data and deployed inside SOFA's
haptic force-feedback loop (`LCPForceFeedback`) via a TorchScript + LibTorch
C++ predictor.

## Directory structure

```
pinn_project/
├── collect_data/      C++ SOFA plugin (PINNDataCollector) used to record training data
│   ├── DataCollector.cpp/.h    records tool position/velocity/force + per-vertex
│   │                            deformation/stress/strain into training_data.csv
│   ├── liver_collection.scn    scene used to actually collect data (real device)
│   └── build/                  this plugin's own CMake build (separate from main SOFA build)
│
├── data/               all CSV/NPY data — see "Where results are stored" below
│
├── train/              Python training scripts (offline, not part of the C++ build)
│   ├── pinn_model.py            all model architecture classes (see below)
│   ├── train_pinn_seqattn_accel.py   current best/recommended model — trains
│   │                                  LagSequenceAttentionAccel, saves
│   │                                  tissue_pinn_seqattn_accel.pth
│   └── train_pinn_*.py          other architectures tried (baseline U-Net, Huber
│                                  loss variant, dual-attention, etc.) — kept for
│                                  comparison, not currently recommended
│
├── cpp/                Files actually loaded by the deployed C++ predictor
│   ├── PINNPredictor.cpp/.h     real-time TorchScript inference engine — builds the
│   │                            963-dim feature vector, maintains 5-lag rolling
│   │                            buffers, runs the model, decodes force + deformation
│   ├── pinn_model_traced.pt     exported TorchScript model (output of export_for_cpp.py)
│   ├── normalization_stats.csv  X/Y mean+std, read by PINNPredictor at init
│   └── liver_vertices.csv       rest-pose vertex positions, read by PINNPredictor
│
└── test/               Everything used to validate / deploy-test the model
    ├── export_for_cpp.py             run after training: traces the model to
    │                                  TorchScript and writes the 3 files in cpp/ above
    ├── extract_test_path.py          pulls one recorded touch episode out of
    │                                  training_data.csv as a fixed replay trajectory
    │                                  -> data/test_path.csv
    ├── liver_replay_groundtruth.scn  replays test_path.csv with usePINN=false
    │                                  (real FEM physics) -> data/replay_groundtruth.csv
    ├── liver_replay_pinn.scn         replays the SAME path with usePINN=true
    │                                  (PINN drives force) -> data/replay_pinn.csv
    ├── liver_replay_pinn_livedevice.scn   same as above but driven through the REAL
    │                                       haptic device's hardware thread instead of
    │                                       a scripted batch loop (needs a connected,
    │                                       calibrated device)
    ├── compare_replay.py             the main deployment-test metric: loads the two
    │                                  replay CSVs above and computes relative-L2 /
    │                                  MAE between PINN and ground truth
    ├── deployment_distance_plot.py   plots PINN vs ground truth force against
    │                                  cumulative distance travelled (full trajectory
    │                                  + a zoomed highest-activity window)
    └── predict_session.py            runs the Python-side (non-C++) model directly
                                       on a recorded session — used to separate "is the
                                       model wrong" from "is the C++ deployment wrong"
```

## Building

This lives inside the main SOFA monorepo and the C++ predictor (`PINNPredictor.cpp`)
is compiled directly into `Sofa.Component.Haptics` (see that target's `CMakeLists.txt`
— it lists the file explicitly and links LibTorch). There is no separate build step
for it; building SOFA builds it.

```bash
cd /home/yogyaahuja/sofa/build
make -j$(nproc) Sofa.Component.Haptics Geomagic
```

Rebuild both targets after editing anything under `Sofa/Component/Haptics/` or
`applications/plugins/Geomagic/` (e.g. `LCPForceFeedback.inl`, `GeomagicDriver.cpp`)
or `pinn_project/cpp/PINNPredictor.cpp`.

**LibTorch path is currently hardcoded** in `Sofa/Component/Haptics/CMakeLists.txt`:
```
list(APPEND CMAKE_PREFIX_PATH "/home/yogyaahuja/.local/lib/python3.12/site-packages/torch/share/cmake")
```
Update this if LibTorch moves or on a different machine.

### Common first-time build error: "No CMAKE_CUDA_COMPILER could be found"

If the installed LibTorch is a CUDA-enabled build, CMake will detect any NVIDIA GPU/CUDA
toolkit present and try to enable CUDA compilation for it — even though **the deployed
model only ever runs on CPU** (`PINNPredictor.h` hardcodes `device_(torch::kCPU)`
specifically to avoid CUDA allocator conflicts). If `nvcc` isn't on `PATH` or isn't where
CMake expects, configuration fails with this error. Fix by pointing CMake at the actual
`nvcc` binary before configuring:
```bash
export CUDACXX=/usr/local/cuda/bin/nvcc   # adjust path if your CUDA install differs —
                                           # find it with: find /usr/local/cuda* -name nvcc
cd /home/yogyaahuja/sofa/build
cmake ..
make -j$(nproc) Sofa.Component.Haptics Geomagic
```

### Building the data-collection plugin (separate, optional)

Only needed if you're re-collecting training data — `PINNDataCollector` is a
standalone plugin, not built as part of the main SOFA build:
```bash
cd /home/yogyaahuja/sofa/pinn_project/collect_data/build
cmake .. && make -j$(nproc)
```

## Running things

All scenes are run with the main SOFA binary:
```bash
/home/yogyaahuja/sofa/build/bin/runSofa -g batch -n <iterations> <scene.scn>
```
`-g batch` runs headless without a GUI; `-g qt` opens the interactive viewer if you
want to watch it. `-n` sets how many simulation steps to run.

### 1. Collect new training data (needs a real, calibrated haptic device)
```bash
cd /home/yogyaahuja/sofa/pinn_project/collect_data
/home/yogyaahuja/sofa/build/bin/runSofa -g qt liver_collection.scn
```
Move the device by hand to poke the liver — mix of gentle and hard pokes, varying
locations, some short and some long touches. Output **appends** to
`data/training_data.csv` across runs (it does not overwrite).

**Before collecting**, back up the existing file if it has data you don't want mixed
with the new batch:
```bash
cp data/training_data.csv data/training_data_backup_$(date +%Y%m%d).csv
```

**After collecting, validate before training on it** — this caught two real,
significant data-quality bugs in this project (a 3-point vs 6-point anchor mismatch
between collection and deployment scenes, and merged touch episodes from
release-then-repoke). Don't skip this:

1. **Scene physics must match deployment exactly.** `liver_collection.scn`'s
   `FixedConstraint` indices (and any other physics parameter — contact distances,
   solver tolerances, mass, `forceCoef`) must be identical to whatever scene you'll
   deployment-test against (`liver_replay_*.scn`). If they ever diverge, the model
   trains on a different physical system than it gets deployed on. Quick check: any
   vertex listed in `FixedConstraint` should show **exactly 0.0** deformation
   (`dx{i}`/`dy{i}`/`dz{i}`) in every row of the new data — if it shows real movement,
   that vertex wasn't actually fixed when this batch was recorded.
2. **No NaN/Inf, no displacement outliers** — sanity-check
   `training_data.csv` for `inf`/`nan` and implausibly large position jumps between
   consecutive rows within a session (anything beyond a few units in one row is
   probably an explosion, not a real poke).
3. **Check for merged sessions.** If you released contact and re-poked a different
   spot quickly, the session-boundary detector (`collectEvery`'s force-crossing-
   threshold logic) may not have registered a new session, merging two unrelated
   touches under one `session_id`. Symptom: a large tool-position jump *within* a
   single `session_id`, immediately following a row where force had dropped near
   zero. If found, re-split that `session_id` into separate IDs at the jump point
   before training — otherwise the model's lag-window history will bleed across
   unrelated touches.
4. **Check the force distribution** has both low and high values — a dataset of only
   gentle pokes won't teach the model to predict hard contact, and vice versa.

### 2. Train a model
```bash
cd /home/yogyaahuja/sofa/pinn_project/train
python3 train_pinn_seqattn_accel.py
```
Saves `tissue_pinn_seqattn_accel.pth` (and `..._best.pth`, the best-val-loss
checkpoint) into `train/`.

### 3. Export the trained model for C++
```bash
cd /home/yogyaahuja/sofa/pinn_project/test
python3 export_for_cpp.py
```
Writes `pinn_model_traced.pt`, `normalization_stats.csv`, `liver_vertices.csv` into
`cpp/`. No rebuild needed — the model is loaded at runtime, not compiled in — but
**you must re-run this export after every training run**, including ones that don't
change the architecture. `runSofa` picks up whatever is currently sitting in `cpp/`
with no warning if it's stale, so a forgotten export silently deployment-tests the
wrong (old) model.

### 4. Run the deployment test (scripted, no device needed)
```bash
cd /home/yogyaahuja/sofa/pinn_project/test
python3 extract_test_path.py --session-id <id>      # picks a recorded episode -> data/test_path.csv
rm -f ../data/replay_groundtruth.csv ../data/replay_pinn.csv
/home/yogyaahuja/sofa/build/bin/runSofa -g batch -n 600 liver_replay_groundtruth.scn
/home/yogyaahuja/sofa/build/bin/runSofa -g batch -n 600 liver_replay_pinn.scn
python3 compare_replay.py                 # prints the Rel-L2/MAE deployment-test numbers
python3 deployment_distance_plot.py       # writes the full + zoomed comparison plots
```
Pick `-n` generously — it must be large enough for the scripted replay to finish
advancing through every row of `test_path.csv` (check the `[REPLAY] FINISHED` log line).

**Picking a fair `--session-id`** matters more than it looks:
- Check `Contact steps: N/total` in `compare_replay.py`'s output. If it's near 0,
  ground truth never registered meaningful contact in *this replay* (even if the
  original recording did) — relative-L2 becomes meaningless (huge/garbage percentages)
  when dividing by near-zero force. Pick a different session.
- **Replaying a recorded path does not reproduce its original force exactly** —
  small differences in simulation history mean the same path can replay softer *or*
  harder than it was originally recorded, sometimes substantially. Don't assume a
  session picked for "moderate difficulty" by its recorded force will replay as
  moderate; check the actual `replay_groundtruth.csv` force values after running it.
- A good test session has: substantial contact (check 1 above), a build-up phase the
  model can use its lag history on (not an instant onset spike — those hit a known,
  separate weak point: zero-padded history at first contact), and ideally both rising
  and sustained-contact portions, since the model behaves differently in each (see
  "Current status" below).
- `deployment_distance_plot.py --zoom-frac <0-1>` auto-picks the highest-mean-force
  window of that size to zoom into — useful for visually inspecting exactly where
  predictions diverge.

### 5. Run the deployment test with the real device (live hardware timing)
```bash
/home/yogyaahuja/sofa/build/bin/runSofa -g batch -n 4500 liver_replay_pinn_livedevice.scn
```
Requires a connected, **calibrated** Geomagic/Touch device — `runSofa` will print
`device is not calibrated` and abort the replay if it isn't. The device doesn't need
to be held or moved; its position is overridden by the recorded path inside the
hardware callback (`stateCallback` in `GeomagicDriver.cpp`), but it does need to be
plugged in and calibrated for that callback to run at all. Output goes to
`data/replay_pinn_livedevice.csv`, logged at the hardware thread's native rate (many
rows per recorded position — see `replayRowIntervalSec` on `GeomagicDriver` to control
how long each row is held, default ~0.118s to match the original recording's pace).

## Where results are stored

| File | What it is |
|---|---|
| `data/training_data.csv` | All recorded training data (appends across collection runs) |
| `data/test_path.csv` | The one recorded episode currently selected for deployment testing |
| `data/replay_groundtruth.csv` | Ground-truth force log from the `usePINN=false` replay |
| `data/replay_pinn.csv` | PINN-predicted force log from the `usePINN=true` replay |
| `data/replay_pinn_livedevice.csv` | Same, but from the live-device replay |
| `data/liver_vertices.npy` | Rest-pose mesh vertex positions (Python-side copy) |
| `train/tissue_pinn_seqattn_accel.pth` | Current best trained model checkpoint |
| `test/force_comparison.png` | Python-side (non-C++) prediction vs ground truth |
| `test/replay_comparison.png` | `compare_replay.py`'s plot (PINN vs GT, time-based) |
| `test/deployment_distance_full.png` / `..._zoom.png` | Distance-based deployment plots |

## What calculates what (the parts that matter most)

- **Real contact force inside the deployment loop**: `LCPForceFeedback::computeRealForceForPINNFeedback()`
  in `LCPForceFeedback.inl` — reads `m_realForceCache`, a value kept fresh every
  simulation step regardless of `usePINN` (see `doComputeForce()` in the same file).
  This is the input fed to the model as "what was the real force a moment ago."
- **Force/deformation prediction**: `PINNPredictor::predictForce()` in
  `pinn_project/cpp/PINNPredictor.cpp` — builds the 963-feature vector, runs the
  TorchScript model, decodes force (inverse log1p) and the 534-value deformation field.
- **Real FEM ground truth being pushed into the predictor's history buffer**:
  `PINNPredictor::updateFEM()`, called from `LCPForceFeedback::handleEvent()` every
  3rd simulation step (`PINN_CALL_STEP_STRIDE`), matching training's `collectEvery=3`.
- **Replay trajectory playback**: `GeomagicDriver::updatePosition()` (scripted) and
  `stateCallback()` (real hardware thread) in `GeomagicDriver.cpp`.

## Current status / known limitations

- Deployment-test error (PINN vs ground truth, on a hard recorded episode, scripted
  replay): **~21-24% relative L2**, down from ~88% before today's fixes. See git log
  for the three confirmed C++ bugs fixed (self-referential force input, EMA/delta
  update-rate mismatch, and the closed feedback-loop cache contamination once
  `usePINN` is on).
- The model specifically underestimates force during **sustained, multi-second high
  contact** (it tracks well during the approach/build-up, less well once force has
  been held steady for a while) — likely because the 5-lag history window leans on
  recent velocity, which goes flat during a held press, and because sustained presses
  are underrepresented in current training data relative to brief pokes.
- Live-device replay (hardware thread timing) currently performs *worse* than
  scripted batch replay with the same code — not yet root-caused; use the scripted
  replay test as the primary metric until this is resolved.
