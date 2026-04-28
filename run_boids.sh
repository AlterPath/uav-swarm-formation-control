#!/bin/bash
# run_boids.sh — 18 экспериментов для Boids
# Использование: ./run_boids.sh [базовая_папка]

set -u
BASE_DIR="${1:-$HOME/vkr_results}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

N_DRONES=3;  FIELD_LEN=60;  FIELD_WID=30;  HEIGHT=4
MISSION_TIME=200;  RESET_WAIT=5;  LANDING_BUFFER=30

SCRIPT_HARDCODE="hard_code_drones.py"
SCRIPT_APF="apf_drones.py"
SCRIPT_APF_LANES="apf_with_lanes_drones.py"
SCRIPT_BOIDS="boids_drones.py"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
mkdir -p "$BASE_DIR"

ALGO="boids"; TOTAL=0
run_exp() {
    local obs="$1" mode="$2" run="$3"
    local out="$BASE_DIR/$ALGO/$obs/$mode/run_$run"
    echo ""; echo "--- $obs | $mode | run_$run ---"

    if ! is_sitl_running; then
        echo "  [WARN] SITL не запущен, поднимаем..."
        start_px4_sitl
    fi
    restart_sitl_if_needed

    kill_algo; cleanup_temp; setup_obstacles "$obs"
    reset_gazebo; launch_algo "$ALGO" "$mode" "$out"
    echo "  [WAIT] Миссия ${MISSION_TIME} сек..."; sleep "$MISSION_TIME"
    echo "  [WAIT] Посадка ${LANDING_BUFFER} сек..."; sleep "$LANDING_BUFFER"
    kill_algo
    TOTAL=$((TOTAL+1)); echo "  [DONE] $TOTAL/18"
}

echo "=== Boids: 18 запусков ==="
for obs in no_obstacles light heavy; do
    for mode in normal lose_drone; do
        for run in 1 2 3; do run_exp "$obs" "$mode" "$run"; done
    done
done
echo "=== Готово: $BASE_DIR/$ALGO ==="
