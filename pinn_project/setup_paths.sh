#!/usr/bin/env bash
# Run this ONCE after cloning the repo to a new machine/location, BEFORE building.
#
# Several files can't resolve paths dynamically at runtime — C++ source has the
# model file paths compiled in, CMakeLists.txt needs LibTorch's location at
# configure time, and SOFA scene (.scn) XML has no built-in mechanism for
# relative/environment-variable paths. All of these were hardcoded to this
# project's original absolute path. This script rewrites them to wherever you
# actually cloned the repo, and to wherever LibTorch is actually installed on
# this machine.
#
# Usage:
#   bash pinn_project/setup_paths.sh
#
# Safe to re-run — it's an idempotent find-and-replace from the OLD prefix to
# the NEW one each time, not an accumulating edit.

set -euo pipefail

OLD_ROOT="/home/yogyaahuja/sofa"
NEW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OLD_TORCH_CMAKE="/home/yogyaahuja/.local/lib/python3.12/site-packages/torch/share/cmake"
NEW_TORCH_CMAKE="$(python3 -c "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)) + '/torch/share/cmake')" 2>/dev/null || true)"

if [ "$NEW_ROOT" = "$OLD_ROOT" ]; then
    echo "Already cloned at $OLD_ROOT — nothing to rewrite for the repo path."
else
    echo "Rewriting $OLD_ROOT -> $NEW_ROOT"
fi

if [ -z "$NEW_TORCH_CMAKE" ]; then
    echo "WARNING: could not auto-detect LibTorch via 'python3 -c \"import torch\"'."
    echo "         Install/activate the right Python environment first, or edit"
    echo "         the CMAKE_PREFIX_PATH lines in the CMakeLists.txt files by hand."
    NEW_TORCH_CMAKE="$OLD_TORCH_CMAKE"
elif [ ! -d "$NEW_TORCH_CMAKE" ]; then
    echo "WARNING: detected path '$NEW_TORCH_CMAKE' does not exist — leaving as-is, check manually."
    NEW_TORCH_CMAKE="$OLD_TORCH_CMAKE"
else
    echo "Rewriting LibTorch cmake path -> $NEW_TORCH_CMAKE"
fi

# Scoped to actual source directories only — must NOT touch build/, which is
# gitignored, machine-generated, and can contain unrelated third-party files
# that coincidentally match the same string (confirmed: an earlier version of
# this script without scoping touched googletest/SofaGLFW build artifacts).
SEARCH_DIRS="$NEW_ROOT/Sofa/Component/Haptics $NEW_ROOT/applications/plugins/Geomagic $NEW_ROOT/pinn_project"
# Match on EITHER the repo-root string OR the LibTorch cmake path — some files
# (e.g. collect_data/CMakeLists.txt) only contain the latter, and would
# otherwise be silently skipped since they never matched $OLD_ROOT.
FILES=$(grep -rlF -e "$OLD_ROOT" -e "$OLD_TORCH_CMAKE" $SEARCH_DIRS \
    --include="*.scn" --include="CMakeLists.txt" --include="*.cpp" --include="*.h" --include="*.inl" \
    2>/dev/null || true)

for f in $FILES; do
    sed -i "s|$OLD_ROOT|$NEW_ROOT|g" "$f"
    if [ "$NEW_TORCH_CMAKE" != "$OLD_TORCH_CMAKE" ]; then
        sed -i "s|$OLD_TORCH_CMAKE|$NEW_TORCH_CMAKE|g" "$f"
    fi
    echo "  fixed: ${f#$NEW_ROOT/}"
done

echo ""
echo "Done. Python scripts under pinn_project/ already resolve their own paths"
echo "automatically at runtime (via SOFA_ROOT computed from each script's own"
echo "location) and needed no changes."
echo ""
echo "Next: follow the normal build steps in pinn_project/README.md."
