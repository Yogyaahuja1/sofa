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

import argparse
import pandas as pd

CSV_PATH = '/home/yogyaahuja/sofa/pinn_project/data/training_data.csv'
OUT_PATH = '/home/yogyaahuja/sofa/pinn_project/data/test_path.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--session-id', type=int, default=None)
args = parser.parse_args()

df = pd.read_csv(CSV_PATH)

if args.session_id is not None:
    sid = args.session_id
else:
    sess_counts = df.groupby('session_id').size()
    sess_counts = sess_counts[sess_counts.index != 0]
    sid = sess_counts.idxmax()

episode = df[df['session_id'] == sid].sort_values('step')
print(f"Using session_id={sid}: {len(episode)} rows")

out = episode[['tool_x', 'tool_y', 'tool_z']]
out.to_csv(OUT_PATH, index=False)
print(f"Saved test trajectory: {OUT_PATH} ({len(out)} rows)")
