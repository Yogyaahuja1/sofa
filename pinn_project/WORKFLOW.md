# PINN Project Workflow

## Models in `train/`
| File | What it is |
|------|-----------|
| `liver_E1500_best.pth` | Liver model, fixed E=1500 Pa |
| `liver_Egen_v5.pth` | Liver model, generalises across E=500–5000 Pa |
| `liver_E1500_vw1p0s_v1.pth` | Liver model, variable-length 1-second history window |
| `membrane_v3.pth` | Membrane (flat surface) model |

---

## Deployment Tests (used by all liver workflows)

After exporting any liver model to C++, run these 3 tests to evaluate it.
The targets below were measured for E=1500 — other models will have different numbers.

### Test 1 — Replay: FEM ground truth vs PINN (E=1500 target: ~4.6%)

Runs the same fixed trajectory twice — once with FEM, once with PINN — and compares forces.

**Step 0** — Pick the fixed trajectory (only needed once per test path; skip if
`data/test_path.csv` from a previous run is still the one you want):
```bash
cd pinn_project/test
python3 extract_test_path.py --session-id <id>   # -> data/test_path.csv
```
See "Picking a fair `--session-id`" in [README.md](README.md) — it matters more
than it looks.

**Step 1** — Run FEM ground truth scene:
```
SOFA → open  pinn_project/test/liver_replay_groundtruth.scn
```
Outputs: `data/replay_groundtruth.csv`

(Not `collect_data/liver_auto_collect.scn` — that scene replays a different,
fixed `auto_traj.csv` for scripted training-data collection and does not write
`replay_groundtruth.csv`.)

**Step 2** — Run PINN replay scene:
```
SOFA → open  pinn_project/test/liver_replay_pinn.scn
```
Outputs: `data/replay_pinn.csv` and `/tmp/cpp_position_seq.csv`

**Step 3** — Compare:
```bash
python3 pinn_project/test/compare_replay.py
```
Saves: `data/replay_comparison.png`

---

### Test 2 — Corrected replay: corrected input vs true force (E=1500 target: ~7%)

Reconstructs model inputs from a real recording at the replay's actual positions.
Requires Test 1 Step 2 to have run first (needs `/tmp/cpp_position_seq.csv`).

```bash
python3 pinn_project/test/predict_corrected_replay.py --session-id 25
```
Saves: `pinn_project/test/corrected_replay_session25.png`

Use any session-id with good contact coverage from `data/training_data.csv`.

---

### Test 3 — Live device: real-time PINN vs real LCP force (E=1500 target: ~15.7%)

Runs the liver simulation live with the haptic device. PINN predicts force in real-time.

**Step 1** — Run live scene:
```
SOFA → open  pinn_project/collect_data/liver_live_pinn_test.scn
```
Move the haptic device around the liver, then close.
Outputs: `data/live_pinn_vs_real.csv`

**Step 2** — Compare:
```bash
python3 pinn_project/test/compare_live.py
```
Saves: `pinn_project/test/live_comparison.png`

---

## Workflow 1 — Liver E=1500 (fixed stiffness)

### Step 1 — Collect training data
1. Connect haptic device
2. Open scene in SOFA:
```
SOFA → open  pinn_project/collect_data/liver_collection.scn
```
3. Probe the liver for a few minutes, then close
4. Data appends to `data/training_data.csv`

### Step 2 — Train
```bash
cd pinn_project/train
python3 train_liver_pinn.py
```
Auto-saves as `liver_E1500_v1.pth`, `liver_E1500_v2.pth`, ... (increments automatically).
Training data: `data/training_data.csv` (auto-detected, no flag needed).

### Step 3 — Export to C++
`export_for_cpp.py` has no `--model` flag — edit its hardcoded `FULL_PATH` (near
the top of the script) to point at `pinn_project/train/liver_E1500_v1.pth` (or
whichever checkpoint you just trained), then:
```bash
python3 pinn_project/test/export_for_cpp.py
```
Updates `cpp/pinn_model_traced.pt` and `cpp/normalization_stats.csv`.

### Step 4 — Run deployment tests
See **Deployment Tests** section above. Run all 3 tests.

---

## Workflow 2 — Liver E-Generalised (E=500–5000 Pa)

Trains on data collected across 7 stiffness values. The model takes log(E/1000) as
an extra input so it learns to scale forces with stiffness.

### Step 1 — Training data
Pre-built CSV already exists: `data/training_data_E_gen_30k_per_E_v2.csv`
(30k rows per E value × 7 values = 210k rows, balanced across sessions).

To re-collect: run `liver_collection.scn` at each E value and merge CSVs.

### Step 2 — Train
```bash
cd pinn_project/train
python3 train_liver_pinn.py --use-youngs
```
Auto-saves as `liver_Egen_v1.pth`, `liver_Egen_v2.pth`, ...
The `--use-youngs` flag selects the E-gen CSV and adds log(E/1000) as a feature.

### Step 3 — Export to C++
Edit `export_for_cpp.py`'s hardcoded `FULL_PATH` to point at
`pinn_project/train/liver_Egen_v5.pth` (or whichever checkpoint you just
trained — there's no `--model` flag), then:
```bash
python3 pinn_project/test/export_for_cpp.py
```

### Step 4 — Run deployment tests
See **Deployment Tests** section above. Run all 3 tests.

---

## Workflow 3 — Membrane (flat surface)

### Step 1 — Collect training data
```
SOFA → open  pinn_project/collect_data/membrane_sirmesh_collect.scn
```
Probe the membrane surface, then close. Appends to `data/flat_surface_training_data.csv`.

### Step 2 — Train
```bash
cd pinn_project/train
python3 train_membrane_pinn.py
```
Auto-saves as `membrane_v1.pth`, `membrane_v2.pth`, ...

### Step 3 — Live test
```
SOFA → open  pinn_project/collect_data/membrane_sirmesh_live.scn
```
No export step needed — membrane uses a separate C++ predictor.

---

## Workflow 4 — Liver Variable-Window (1-second history)

Uses ALL FEM rows from the past 1 second as history (variable count, up to 200 rows).
The model attends over the full second of history and ignores padded positions via masking.

### Step 1 — Training data
Uses same data as Workflow 1: `data/training_data.csv`.
No additional collection needed.

### Step 2 — Train
```bash
cd pinn_project/train
python3 train_liver_pinn.py --var-window 1.0
```
Auto-saves as `liver_E1500_vw1p0s_v1.pth`, `liver_E1500_vw1p0s_v2.pth`, ...

To train on E-gen data with variable window:
```bash
python3 train_liver_pinn.py --var-window 1.0 --use-youngs
```
Auto-saves as `liver_Egen_vw1p0s_v1.pth`, ...

To test on a subset first (faster):
```bash
python3 train_liver_pinn.py --var-window 1.0 --max-rows 80000
```

> **Note:** Feature building takes ~30–60 mins (scans all history rows per training row).
> Training itself is fast — model forward pass is fully vectorised.

### Step 3 — Export to C++
Edit `export_for_cpp.py`'s hardcoded `FULL_PATH` to point at
`pinn_project/train/liver_E1500_vw1p0s_v1.pth` (or whichever checkpoint you
just trained — there's no `--model` flag), then:
```bash
python3 pinn_project/test/export_for_cpp.py
```

### Step 4 — Run deployment tests
See **Deployment Tests** section above. Run all 3 tests.
Compare results against Workflow 1 targets to see if 1-second history helps.

---

## Queue multiple training runs

To run training jobs back-to-back automatically and push results when done:
```bash
nohup bash pinn_project/train/run_queue.sh &
tail -f pinn_project/train/queue.log   # monitor progress
```

Edit `run_queue.sh` to change which jobs run and in what order.
