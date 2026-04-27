#!/bin/bash
# run_all.sh — все 72 эксперимента с динамическим временем и пропуском hardcode
# Использование: ./run_all.sh [базовая_папка]

set -u
BASE_DIR="${1:-$HOME/vkr_results}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

N_DRONES=3;  FIELD_LEN=60;  FIELD_WID=30;  HEIGHT=2.91
LANDING_BUFFER=30         # время на посадку после миссии
RESET_WAIT=5

LAUNCH_FILE="${LAUNCH_FILE:-px4 multi_uav_mavros_sitl.launch}"

RESTART_EVERY=2

SCRIPT_HARDCODE="hard_code_drones.py"
SCRIPT_APF="apf_drones.py"
SCRIPT_APF_LANES="apf_with_lanes_drones.py"
SCRIPT_BOIDS="boids_drones.py"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
mkdir -p "$BASE_DIR"

TOTAL=0
EXP_SINCE_RESTART=0

run_experiment() {
    local algo="$1" obs="$2" mode="$3" run_num="$4" m_time="$5"
    local out_dir="$BASE_DIR/$algo/$obs/$mode/run_$run_num"
    echo ""
    echo "══════════════════════════════════════════"
    echo "  $algo | $obs | $mode | run_$run_num"
    echo "  Время миссии: ${m_time}с"
    echo "══════════════════════════════════════════"

    if ! is_sitl_running; then
        echo "SITL запускается"
        start_px4_sitl
    fi
    restart_sitl_if_needed

    kill_algo
    cleanup_temp
    setup_obstacles "$obs"
    reset_gazebo
    launch_algo "$algo" "$mode" "$out_dir"

    echo "Миссия ~${m_time} сек"
    sleep "$m_time"

    echo "Время на посадку ${LANDING_BUFFER} сек"
    sleep "$LANDING_BUFFER"

    kill_algo
    collect_results "$out_dir"
    TOTAL=$((TOTAL + 1))
    echo "  [DONE] $TOTAL/72"
}

START=$(date +%s)

if ! is_sitl_running; then
    start_px4_sitl
fi

for algo in hardcode apf apf_lanes boids; do
    case "$algo" in
        apf|apf_lanes)
            CURRENT_TIME=390
            ;;
        boids|hardcode)
            CURRENT_TIME=490
            ;;
        *)
            CURRENT_TIME=520
            ;;
    esac

    echo ""; echo "######## $algo (Time: $CURRENT_TIME) ########"
    for obs in no_obstacles light heavy; do
        for mode in normal lose_drone; do
            if [ "$algo" = "hardcode" ]; then
                echo "  [SKIP] hardcode + lose_drone"
                continue
            fi
            
            for run in 1 2 3; do
                run_experiment "$algo" "$obs" "$mode" "$run" "$CURRENT_TIME"
            done
        done
    done
done

echo ""
echo " Финал"
kill_px4_sitl

END=$(date +%s); ELAPSED=$((END-START))
echo ""
echo "Готово: $TOTAL/72 за $((ELAPSED/3600))ч $(((ELAPSED%3600)/60))м"
echo "Результаты: $BASE_DIR"
