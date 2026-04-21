#!/bin/bash
# process_results.sh — постобработка результатов всех экспериментов
#
# Запускает plot_metrics.py и plot_trajectories.py для каждой папки с результатами.
#
# Использование:
#   ./process_results.sh [базовая_папка]

set -e

BASE_DIR="${1:-$HOME/vkr_results}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Постобработка результатов ==="
echo "Папка: $BASE_DIR"
echo ""

# Проверка скриптов постобработки
if [ ! -f "$SCRIPT_DIR/plot_metrics.py" ]; then
    echo "[WARN] plot_metrics.py не найден в $SCRIPT_DIR"
fi
if [ ! -f "$SCRIPT_DIR/plot_trajectories.py" ]; then
    echo "[WARN] plot_trajectories.py не найден в $SCRIPT_DIR"
fi

TOTAL=0
PROCESSED=0

for algo in hardcode apf apf_lanes boids; do
    for obs in no_obstacles light heavy; do
        for mode in normal lose_drone; do
            for run in run_1 run_2 run_3; do
                DIR="$BASE_DIR/$algo/$obs/$mode/$run"
                TOTAL=$((TOTAL + 1))
                
                if [ ! -d "$DIR" ]; then
                    continue
                fi
                
                # Проверяем, есть ли CSV-файлы
                if ! ls "$DIR"/metrics_*.csv 1>/dev/null 2>&1; then
                    continue
                fi
                
                echo "[$PROCESSED] Обработка: $algo/$obs/$mode/$run"
                cd "$DIR"
                
                # Копируем plot-скрипты (если их нет)
                if [ ! -f "$DIR/plot_trajectories.py" ] && [ -f "$SCRIPT_DIR/plot_trajectories.py" ]; then
                    cp "$SCRIPT_DIR/plot_trajectories.py" "$DIR/" 2>/dev/null || true
                fi
                
                # 1. Метрики
                if [ -f "$SCRIPT_DIR/plot_metrics.py" ]; then
                    python3 "$SCRIPT_DIR/plot_metrics.py" --dir . --save 2>/dev/null || {
                        echo "      [SKIP] plot_metrics.py не сработал"
                    }
                fi
                
                # 2. Траектории
                if [ -f "$SCRIPT_DIR/plot_trajectories.py" ]; then
                    python3 "$SCRIPT_DIR/plot_trajectories.py" \
                        --dir . --save \
                        --field_len 60 --field_wid 30 --height 5 2>/dev/null || {
                        echo "      [SKIP] plot_trajectories.py не сработал"
                    }
                fi
                
                PROCESSED=$((PROCESSED + 1))
                cd "$SCRIPT_DIR"
            done
        done
    done
done

echo ""
echo "=== Готово ==="
echo "Обработано папок: $PROCESSED из $TOTAL"
