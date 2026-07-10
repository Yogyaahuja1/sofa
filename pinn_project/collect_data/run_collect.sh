#!/usr/bin/env bash
# Launch liver collection scene with VSync disabled for full FPS.
# Usage: bash run_collect.sh

SOFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="$SOFA_ROOT/build/bin/runSofa"
SCENE="$SOFA_ROOT/pinn_project/collect_data/liver_collection.scn"

# Disable VSync — NVIDIA driver
export __GL_SYNC_TO_VBLANK=0
# Disable VSync — Mesa fallback
export vblank_mode=0

echo "Launching SOFA with VSync disabled..."
exec "$BIN" "$SCENE"
