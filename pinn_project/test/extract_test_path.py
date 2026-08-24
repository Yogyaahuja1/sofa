"""
Extract a single, representative touch episode from training_data.csv as a fixed
test trajectory for deployment comparison.

Both the PINN-off (ground truth) and PINN-on (predicted) replay runs read this same
file via GeomagicDriver's replayFile, guaranteeing row-for-row identical input —
the only fair way to compare PINN ON vs OFF.

Usage:
  python3 extract_test_path.py                  # auto-picks the longest episode
  python3 extract_test_path.py --session-id 35   # specific episode
"""
import os as _os
SOFA_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import pandas as pd

CSV_PATH = f'{SOFA_ROOT}/pinn_project/data/training_data.csv'
OUT_PATH = f'{SOFA_ROOT}/pinn_project/data/test_path.csv'
DEFORM_OUT_PATH = f'{SOFA_ROOT}/pinn_project/data/initial_deform.csv'
N_VERTICES = 181

parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, default=None)
parser.add_argument('--full', action='store_true',
                     help='Use the ENTIRE recorded file (all episodes, in original '
                          'recorded order) instead of a single touch episode.')
args = parser.parse_args()

df = pd.read_csv(CSV_PATH)

if args.full:
    episode = df
    print(f"Using FULL recording: {len(episode)} rows across "
          f"{df[df.session_id != 0]['session_id'].nunique()} touch episodes")
    # Full-path replay starts at row 0 (true rest) and carries state forward itself —
    # remove any stale initial-deform file from a previous single-session extraction,
    # or the scene would wrongly apply it here too.
    if os.path.exists(DEFORM_OUT_PATH):
        os.remove(DEFORM_OUT_PATH)
else:
    if args.session_id is not None:
        sid = args.session_id
    else:
        sess_counts = df.groupby('session_id').size()
        sess_counts = sess_counts[sess_counts.index != 0]
        sid = sess_counts.idxmax()
    episode = df[df['session_id'] == sid].sort_values('step')
    orig_index = episode.index  # keep ORIGINAL df row numbers before any reset, so we
                                 # can look up the row just before the episode starts
    episode = episode.reset_index(drop=True)
    # session_id resets each time liver_collection.scn restarts, so the same small ID can
    # collide across MANY separate collection launches once their data is combined — not
    # just two. Split on every step-gap into contiguous blocks and keep the LARGEST one
    # (the real episode), not just the first (which can be a tiny stray fragment).
    step_gaps = episode['step'].diff().abs()
    gap_idx = list(step_gaps[step_gaps > 50].index)
    block_start = 0
    if len(gap_idx) > 0:
        bounds = [0] + gap_idx + [len(episode)]
        blocks = [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
        best = max(blocks, key=lambda b: b[1]-b[0])
        print(f"WARNING: session_id={sid} has {len(episode)} rows split into {len(blocks)} "
              f"colliding fragments across separate collection runs (sizes: "
              f"{[e-s for s,e in blocks]}). Keeping the largest: rows {best[0]}-{best[1]-1}.")
        episode = episode.iloc[best[0]:best[1]].reset_index(drop=True)
        block_start = best[0]
    print(f"Using session_id={sid}: {len(episode)} rows")

    # Export the liver's REAL residual deformation from the row immediately before
    # this touch actually began in the original recording — without this, replaying
    # the episode in isolation starts the liver from a clean rest pose, which is wrong
    # for any touch that wasn't the first one in its collection run. Confirmed via
    # measurement: every deployment-test session except one had 11-18 units of real
    # residual deformation at this point, vs ~0.8 for the one session that replayed well.
    first_orig_idx = orig_index[block_start]
    if first_orig_idx == 0:
        print("  (this episode starts at row 0 of the recording — already at rest, "
              "no initial-deform file needed)")
    else:
        prev_row = df.iloc[first_orig_idx - 1]
        vals = []
        for v in range(N_VERTICES):
            vals += [prev_row[f'dx{v}'], prev_row[f'dy{v}'], prev_row[f'dz{v}']]
        with open(DEFORM_OUT_PATH, 'w') as f:
            f.write(','.join(f'{x:.10f}' for x in vals) + '\n')
        print(f"  Saved real initial deformation: {DEFORM_OUT_PATH}")

out = episode[['tool_x', 'tool_y', 'tool_z', 'tool_vx', 'tool_vy', 'tool_vz', 'dt_since_last']]
out.to_csv(OUT_PATH, index=False)
print(f"Saved test trajectory: {OUT_PATH} ({len(out)} rows)")
