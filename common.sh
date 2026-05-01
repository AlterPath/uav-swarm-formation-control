#!/bin/bash
# common.sh — общие функции для всех run_*.sh скриптов
# Подключается через: source "$(dirname "$0")/common.sh"
#
# Требует чтобы в вызывающем скрипте были заданы:
#   SCRIPT_DIR, N_DRONES, FIELD_LEN, FIELD_WID, HEIGHT,
#   MISSION_TIME, RESET_WAIT,
#   SCRIPT_HARDCODE, SCRIPT_APF, SCRIPT_APF_LANES, SCRIPT_BOIDS

# Задержка между запуском дронов в одном алгоритме (сек)
DRONE_START_DELAY=0.5

# Через сколько секунд после старта миссии убивать дрона
LOSE_AFTER=70

# PID-ы запущенных процессов дронов (заполняется в launch_algo)
ALGO_PIDS=()

# ═══════════════════════════════════════════════════════════
# PX4 SITL
# ═══════════════════════════════════════════════════════════
LAUNCH_FILE="${LAUNCH_FILE:-px4 multi_uav_mavros_sitl.launch}"
RESTART_EVERY="${RESTART_EVERY:-2}"
EXP_SINCE_RESTART=0

is_sitl_running() {
    pgrep -f "roslaunch.*multi_uav_mavros_sitl" >/dev/null 2>&1 || \
    pgrep -x "gzserver" >/dev/null 2>&1 || \
    pgrep -f "px4" >/dev/null 2>&1
}

kill_px4_sitl() {
    echo "  [SITL] Останавливаем PX4 + Gazebo + MAVROS..."
    pkill -2 -f "roslaunch.*multi_uav_mavros_sitl" 2>/dev/null || true
    sleep 3
    pkill -9 -f "roslaunch" 2>/dev/null || true
    pkill -9 -x "gzserver" 2>/dev/null || true
    pkill -9 -x "gzclient" 2>/dev/null || true
    pkill -9 -f "px4" 2>/dev/null || true
    pkill -9 -f "mavros_node" 2>/dev/null || true
    pkill -9 -f "sitl_" 2>/dev/null || true
    if command -v rosnode &>/dev/null; then
        rosnode kill $(rosnode list 2>/dev/null | grep -E "gazebo|mavros|px4|sitl") 2>/dev/null || true
    fi
    sleep 3
    echo "  [SITL] Остановлен"
}

start_px4_sitl() {
    echo "  [SITL] Запуск: roslaunch $LAUNCH_FILE"
    # shellcheck disable=SC2086
    roslaunch $LAUNCH_FILE &
    echo "  [SITL] Ожидание инициализации (50 сек)..."
    sleep 50
    local attempts=0
    while [ $attempts -lt 30 ]; do
        if rosservice list 2>/dev/null | grep -q "/gazebo/reset_world"; then
            echo "  [SITL] Gazebo готов"
            break
        fi
        sleep 2
        attempts=$((attempts + 1))
    done
    sleep 5
    echo "  [SITL] Готов к работе"
}

restart_sitl_if_needed() {
    EXP_SINCE_RESTART=$((EXP_SINCE_RESTART + 1))
    if [ "$EXP_SINCE_RESTART" -ge "$RESTART_EVERY" ]; then
        echo ""
        echo "  [RESTART] Перезапуск стека PX4 (прогонов с рестарта: $EXP_SINCE_RESTART)"
        kill_algo
        kill_px4_sitl
        sleep 2
        start_px4_sitl
        EXP_SINCE_RESTART=0
    fi
}

# ═══════════════════════════════════════════════════════════
# ROS окружение
# ═══════════════════════════════════════════════════════════
ensure_ros() {
    if command -v rosservice &>/dev/null; then
        return 0
    fi
    for setup in /opt/ros/noetic/setup.bash \
                 /opt/ros/melodic/setup.bash \
                 /opt/ros/kinetic/setup.bash; do
        if [ -f "$setup" ]; then
            # shellcheck disable=SC1090
            source "$setup"
            return 0
        fi
    done
    echo "  [ERROR] rosservice не найден. Запустите: source /opt/ros/<distro>/setup.bash"
    return 1
}

# ═══════════════════════════════════════════════════════════
# Сброс Gazebo — с ожиданием появления сервиса после перезапуска SITL
# ═══════════════════════════════════════════════════════════
reset_gazebo() {
    echo "  [GAZEBO] reset_world..."
    ensure_ros || { echo "  [WARN] Пропускаем сброс (ROS не найден)"; return 0; }

    local attempts=0
    while ! rosservice list 2>/dev/null | grep -q "/gazebo/reset_world"; do
        if [ $attempts -ge 30 ]; then
            echo "  [WARN] /gazebo/reset_world не появился — возможно Gazebo ещё стартует"
            echo "  [WARN] Продолжаем без сброса"
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
    done

    if rosservice call /gazebo/reset_world '{}' 2>&1; then
        echo "  [GAZEBO] OK — ждём ${RESET_WAIT} сек..."
        sleep "$RESET_WAIT"
    else
        echo "  [WARN] reset_world не ответил — проверьте что Gazebo запущен"
        echo "  [WARN] Продолжаем без сброса"
    fi
}

# ═══════════════════════════════════════════════════════════
# Препятствия
# ═══════════════════════════════════════════════════════════
setup_obstacles() {
    local obs="$1"
    case "$obs" in
        no_obstacles)
            rm -f "$SCRIPT_DIR/obstacles.txt"
            echo "  [OBS] Без препятствий"
            ;;
        light)
            if [ -f "$SCRIPT_DIR/obstacles_light.txt" ]; then
                cp "$SCRIPT_DIR/obstacles_light.txt" "$SCRIPT_DIR/obstacles.txt"
                echo "  [OBS] Лёгкие ($(wc -l < "$SCRIPT_DIR/obstacles.txt") шт.)"
            else
                echo "  [WARN] obstacles_light.txt не найден"
            fi
            ;;
        heavy)
            if [ -f "$SCRIPT_DIR/obstacles_heavy.txt" ]; then
                cp "$SCRIPT_DIR/obstacles_heavy.txt" "$SCRIPT_DIR/obstacles.txt"
                echo "  [OBS] Тяжёлые ($(wc -l < "$SCRIPT_DIR/obstacles.txt") шт.)"
            else
                echo "  [WARN] obstacles_heavy.txt не найден"
            fi
            ;;
    esac
}
# ═══════════════════════════════════════════════════════════
# Запуск дронов в фоне
# ═══════════════════════════════════════════════════════════
launch_algo() {
    local algo="$1"
    local mode="$2"
    local out_dir="${3:-}"

    # Защита: hardcode — один процесс на весь рой, lose_drone не работает
    if [ "$algo" = "hardcode" ] && [ "$mode" = "lose_drone" ]; then
        echo "  [SKIP] hardcode + lose_drone не поддерживается (все дроны в одном процессе)"
        return 0
    fi

    ALGO_PIDS=()

    # lose_drone: сажаем uav1 и убиваем только управляющую ноду
    if [ "$mode" = "lose_drone" ]; then
        (
            sleep "$LOSE_AFTER"
            echo "  [LOSE] Посадка uav1 (спустя ${LOSE_AFTER} сек)..."

            # Убеждаемся, что ROS окружение доступно в фоне
            if ! command -v rosservice &>/dev/null; then
                for setup in /opt/ros/noetic/setup.bash \
                             /opt/ros/melodic/setup.bash \
                             /opt/ros/kinetic/setup.bash; do
                    [ -f "$setup" ] && source "$setup" && break
                done
            fi

            # 1. Переводим в AUTO.LAND — PX4 начнёт снижение
            rosservice call /uav1/mavros/set_mode "custom_mode: 'AUTO.LAND'" >/dev/null 2>&1 || true
            sleep 3

            # 2. Дизарм (на всякий случай, если уже на земле)
            rosservice call /uav1/mavros/cmd/arming "value: false" >/dev/null 2>&1 || true

            # 3. Убиваем только ноду управления — heartbeat перестанет поступать в рой
            echo "  [LOSE] Остановка ноды управления uav1"
            rosnode kill /swarm_apf_uav1        2>/dev/null || true
            rosnode kill /swarm_apf_lanes_uav1  2>/dev/null || true
            rosnode kill /swarm_boids_uav1      2>/dev/null || true
        ) &
        ALGO_PIDS+=($!)
    fi

    case "$algo" in
        hardcode)
            python3 "$SCRIPT_DIR/$SCRIPT_HARDCODE" \
                --height "$HEIGHT" --field-len "$FIELD_LEN" --field-wid "$FIELD_WID" \
                --log-dir "$out_dir" &
            ALGO_PIDS+=($!)
            echo "  [LAUNCH] hardcode PID=$!"
            ;;
        apf)
            for i in $(seq 0 $((N_DRONES - 1))); do
                python3 "$SCRIPT_DIR/$SCRIPT_APF" \
                    --id "$i" --total "$N_DRONES" --ns "uav${i}" \
                    --field-len "$FIELD_LEN" --field-wid "$FIELD_WID" \
                    --height "$HEIGHT" \
                    --log-dir "$out_dir" &
                ALGO_PIDS+=($!)
                echo "  [LAUNCH] apf uav${i} PID=$!"
                sleep "$DRONE_START_DELAY"
            done
            ;;
        apf_lanes)
            for i in $(seq 0 $((N_DRONES - 1))); do
                python3 "$SCRIPT_DIR/$SCRIPT_APF_LANES" \
                    --id "$i" --total "$N_DRONES" --ns "uav${i}" \
                    --field-len "$FIELD_LEN" --field-wid "$FIELD_WID" \
                    --height "$HEIGHT" \
                    --log-dir "$out_dir" &
                ALGO_PIDS+=($!)
                echo "  [LAUNCH] apf_lanes uav${i} PID=$!"
                sleep "$DRONE_START_DELAY"
            done
            ;;
        boids)
            for i in $(seq 0 $((N_DRONES - 1))); do
                python3 "$SCRIPT_DIR/$SCRIPT_BOIDS" \
                    --id "$i" --total "$N_DRONES" --ns "uav${i}" \
                    --field-len "$FIELD_LEN" --field-wid "$FIELD_WID" \
                    --height "$HEIGHT" \
                    --log-dir "$out_dir" &
                ALGO_PIDS+=($!)
                echo "  [LAUNCH] boids uav${i} PID=$!"
                sleep "$DRONE_START_DELAY"
            done
            ;;
        *)
            echo "  [ERROR] Неизвестный алгоритм: $algo"
            return 1
            ;;
    esac
}
# ═══════════════════════════════════════════════════════════
# Остановка всех процессов
# ═══════════════════════════════════════════════════════════
kill_algo() {
    if [ ${#ALGO_PIDS[@]} -gt 0 ]; then
        echo "  [KILL] Останавливаем PIDs: ${ALGO_PIDS[*]}"
        for pid in "${ALGO_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 2
        for pid in "${ALGO_PIDS[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    pkill -f "$SCRIPT_HARDCODE"  2>/dev/null || true
    pkill -f "$SCRIPT_APF"       2>/dev/null || true
    pkill -f "$SCRIPT_APF_LANES" 2>/dev/null || true
    pkill -f "$SCRIPT_BOIDS"     2>/dev/null || true
    ALGO_PIDS=()
    sleep 1
}

# ═══════════════════════════════════════════════════════════
# Очистка временных файлов
# ═══════════════════════════════════════════════════════════
cleanup_temp() {
    rm -f "$SCRIPT_DIR"/metrics_*.csv
    rm -f "$SCRIPT_DIR"/photos_*.csv
    rm -rf "$SCRIPT_DIR/raw_images"
}

# ═══════════════════════════════════════════════════════════
# Сбор результатов
# ═══════════════════════════════════════════════════════════
collect_results() {
    local out_dir="$1"
    mkdir -p "$out_dir"
    cp "$SCRIPT_DIR"/metrics_*.csv "$out_dir/" 2>/dev/null || true
    cp "$SCRIPT_DIR"/photos_*.csv  "$out_dir/" 2>/dev/null || true
    cp -r "$SCRIPT_DIR/raw_images" "$out_dir/" 2>/dev/null || true
    local n_csv
    n_csv=$(ls "$out_dir"/metrics_*.csv 2>/dev/null | wc -l)
    echo "  [SAVE] $out_dir  (${n_csv} CSV)"
}
