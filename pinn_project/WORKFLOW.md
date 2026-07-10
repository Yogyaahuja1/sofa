# PINN Project Workflow

## Models in `train/`
| File | What it is |
|------|-----------|
| `liver_E1500_best.pth` | Best liver model, fixed E=1500 Pa |
| `liver_Egen_v4.pth` | Liver model, generalises across E=500–5000 Pa |
| `membrane_v3.pth` | Best membrane model |

---

## 1. Liver E=1500 Model

### Train
```bash
cd pinn_project/train
python3 train_liver_pinn.py
```
Auto-saves as `liver_E1500_v1.pth`, `liver_E1500_v2.pth`, ... (increments automatically).

### Export to C++ (after training a new model)
```bash
cd pinn_project/test
python3 export_for_cpp.py --model ../train/liver_E1500_best.pth
```
Updates `cpp/pinn_model_traced.pt` and `cpp/normalization_stats.csv`.

---

## 2. Deployment Tests

### Setup — collect a fresh test path
1. Run `collect_data/liver_collection.scn` in SOFA with haptic device
2. Probe the liver for a few seconds, then close
3. Appends rows to `data/training_data.csv`
4. Extract that session's rows to test_path.csv (replace `<id>` with last session_id):
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/training_data.csv')
sid = df['session_id'].max()
df[df['session_id']==sid].to_csv('data/test_path.csv', index=False)
print(f'test_path.csv: {(df.session_id==sid).sum()} rows  session_id={sid}')
"
```

---

### Test 1 — Replay: FEM ground truth vs PINN (target: ~4.6%)

Runs the same fixed trajectory twice — once with FEM, once with PINN — and compares.

**Step 1** — Run FEM ground truth scene:
```
SOFA → open  pinn_project/test/liver_replay_groundtruth.scn
```
Outputs: `data/replay_groundtruth.csv`

> **Note:** `liver_replay_groundtruth.scn` was deleted during cleanup. If you need
> to re-run this, restore it from git or re-create it with `usePINN=false`.
> Yesterday's result already exists in `data/replay_groundtruth.csv`.

**Step 2** — Run PINN replay scene:
```
SOFA → open  pinn_project/test/liver_replay_pinn.scn
```
Outputs: `data/replay_pinn.csv` and `/tmp/cpp_position_seq.csv`

**Step 3** — Compare:
```bash
python3 pinn_project/test/compare_replay.py
```
Saves `data/replay_comparison.png`

---

### Test 2 — Corrected replay: predicted vs interpolated true force (target: ~7%)

Uses the actual positions the PINN replay visited and interpolates the true
force from a reference real recording. Requires Test 1 Step 2 to have run
first (needs `/tmp/cpp_position_seq.csv`).

**Step 1** — Run the PINN replay scene (same as Test 1 Step 2 above):
```
SOFA → open  pinn_project/test/liver_replay_pinn.scn
```
This also writes `/tmp/cpp_position_seq.csv` — the actual position at every predict call.

**Step 2** — Pick a reference session from training_data.csv (use a session
with good contact coverage). Then run:
```bash
python3 pinn_project/test/predict_corrected_replay.py --session-id <id>
```
Example: `--session-id 25`

Saves `pinn_project/test/corrected_replay_session<id>.png`

---

### Test 3 — Live device: real-time PINN vs real LCP force (target: ~15.7%)

Runs the liver simulation live with the haptic device. PINN predicts force
in real-time; LCPForceFeedback logs both PINN and real force.

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
Saves `pinn_project/test/live_comparison.png`

---

## 3. E-Generalised Liver Model (E=500–5000 Pa)

### Train
```bash
cd pinn_project/train
python3 train_liver_pinn.py --use-youngs
```
Auto-saves as `liver_Egen_v1.pth`, `liver_Egen_v2.pth`, ... (increments automatically).
The `--use-youngs` flag adds log(E/1000) as a global feature so the model
learns to scale forces with stiffness.

### Export
```bash
python3 pinn_project/test/export_for_cpp.py --model pinn_project/train/liver_Egen_v4.pth
```

---

## 4. Membrane Model

### Train
```bash
cd pinn_project/train
python3 train_membrane_pinn.py
```
Auto-saves as `membrane_v1.pth`, `membrane_v2.pth`, ... (increments automatically).

### Collect membrane data
```
SOFA → open  pinn_project/collect_data/membrane_sirmesh_collect.scn
```
Probe the membrane, close when done. Appends to `data/flat_surface_training_data.csv`.

### Live membrane test
```
SOFA → open  pinn_project/collect_data/membrane_sirmesh_live.scn
```
