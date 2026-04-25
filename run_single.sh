#!/bin/bash
# run_single.sh — один эксперимент с полным контролем SITL
#
# Использование:
#   ./run_single.sh <алгоритм> <препятствия> <режим> <прогон> [папка]

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ALGO="${1:-}"; OBS="${2:-}"; MODE="${3:-}"; RUN_NUM="${4:-}"
BASE_DIR="${5:-$HOME/vkr_results}"

# Защита от lose_drone для hardcode
if [ "$ALGO" = "hardcode" ] && [ "$MODE" = "lose_drone" ]; then
    echo "[ERROR] hardcode не поддерживает режим lose_drone (все дроны в одном процессе)"
    exit 1
fi

if [ -z "$ALGO" ] || [ -z "$OBS" ] || [ -z "$MODE" ] || [ -z "$RUN_NUM" ]; then
    echo "Использование: ./run_single.sh <алгоритм> <препятствия> <режим> <прогон> [папка]"
    echo "Алгоритмы:   hardcode | apf | apf_lanes | boids"
    echo "Препятствия: no_obstacles | light | heavy"
    echo "Режимы:      normal | lose_drone"
    exit 1
fi

N_DRONES=3;  FIELD_LEN=60;  FIELD_WID=30;  HEIGHT=4
MISSION_TIME=10000
LANDING_BUFFER=30
RESET_WAIT=5

SCRIPT_HARDCODE="hard_code_drones.py"
SCRIPT_APF="apf_drones.py"
SCRIPT_APF_LANES="apf_with_lanes_drones.py"
SCRIPT_BOIDS="boids_drones.py"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_DIR="$BASE_DIR/$ALGO/$OBS/$MODE/run_$RUN_NUM"

echo "=== Запуск: $ALGO | $OBS | $MODE | run_$RUN_NUM ==="
echo "    Выход: $OUT_DIR"

if ! is_sitl_running; then
    echo "  [WARN] SITL не запущен, поднимаем..."
    start_px4_sitl
fi

kill_algo
cleanup_temp
setup_obstacles "$OBS"
reset_gazebo
launch_algo "$ALGO" "$MODE" "$OUT_DIR"

echo "[WAIT] Миссия ~${MISSION_TIME} сек... (Ctrl+C для прерывания)"
sleep "$MISSION_TIME"

echo "[WAIT] Время на посадку ${LANDING_BUFFER} сек..."
sleep "$LANDING_BUFFER"

kill_algo
collect_results "$OUT_DIR"

echo ""
echo "=== Готово: $OUT_DIR ==="
