#!/bin/bash
# Manual merge helper — run from the RunPod web terminal when the merge stage fails
# Uso:
#   bash runpod/merge.sh              → lista os projetos e o status
#   bash runpod/merge.sh PROJECT_ID   → refaz o merge desse projeto
#   bash runpod/merge.sh all          → refaz o merge de todos sem video final

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${LUMINADUB_WORKSPACE:-/workspace}"
[ -d "$WS" ] || WS="$APP_DIR"
PY="$WS/venv-luminadub/bin/python"
[ -x "$PY" ] || PY=python3
PROJECTS="$APP_DIR/data/projects"
cd "$APP_DIR"

list_projects() {
    echo "========================================"
    echo "  Projetos em $PROJECTS"
    echo "========================================"
    for dir in "$PROJECTS"/*/; do
        id=$(basename "$dir")
        video=$(ls "$dir/input/" 2>/dev/null | head -1)
        output=$(ls "$dir/output/"*_dublado_pt.mp4 2>/dev/null | head -1)
        vocals="$dir/work/dubbed_vocals.wav"
        if [ -f "$output" ]; then
            status="DONE"
        elif [ -f "$vocals" ]; then
            status="READY_TO_MERGE"
        else
            stage=$(ls "$dir/work/" 2>/dev/null | tail -1)
            status="IN_PROGRESS ($stage)"
        fi
        echo "  $id  |  ${video:-no video}  |  $status"
    done
    echo ""
}

run_merge() {
    local project_id="$1"
    echo "Running merge for project: $project_id"
    "$PY" - <<PYEOF
import sys
sys.path.insert(0, '$APP_DIR')
from pathlib import Path
from config import Config
from stages.merge import merge
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

project_id = '$project_id'
project_dir = Path('$PROJECTS') / project_id

if not project_dir.exists():
    print(f'ERROR: Project not found: {project_dir}')
    sys.exit(1)

vocals = project_dir / 'work' / 'dubbed_vocals.wav'
if not vocals.exists():
    print(f'ERROR: dubbed_vocals.wav not found — sync stage has not completed yet')
    sys.exit(1)

print(f'Merging {project_id} ...')
result = merge(project_dir, Config())
if result.success:
    print('SUCCESS!')
    for p in result.output_paths:
        print(f'  -> {p}')
else:
    print(f'FAILED: {result.error}')
    sys.exit(1)
PYEOF
}

if [ -z "$1" ]; then
    list_projects
elif [ "$1" = "all" ]; then
    list_projects
    for dir in "$PROJECTS"/*/; do
        id=$(basename "$dir")
        output=$(ls "$dir/output/"*_dublado_pt.mp4 2>/dev/null | head -1)
        vocals="$dir/work/dubbed_vocals.wav"
        if [ -z "$output" ] && [ -f "$vocals" ]; then
            run_merge "$id"
            echo ""
        fi
    done
else
    run_merge "$1"
fi
