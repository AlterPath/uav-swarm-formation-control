#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trajectories.py — визуализация траекторий дронов (реальный vs идеальный путь).

Строит для каждого алгоритма и каждого дрона:
  1. Реальную траекторию (pos_x, pos_y из CSV-лога)
  2. Идеальный маршрут (змейка / waypoints)
  3. Препятствия (круги с радиусами)
  4. Footprint каждого дрона (зона покрытия камеры)
  5. Heatmap отклонений (formation_error) вдоль пути
  6. Сравнительный график: все дроны одного алгоритма на одном полотне

Запуск:
    python3 plot_trajectories.py --dir . --algo apf --save
    python3 plot_trajectories.py --dir . --save          # все алгоритмы
    python3 plot_trajectories.py --dir . --algo apf_lanes --field_len 60 --field_wid 30 --height 5

Требует: CSV-файлы вида metrics_*.csv (поля pos_x, pos_y, target_x, target_y,
formation_error_m) и файла obstacles.txt (x, y, radius по строкам).
"""

import argparse
import csv
import glob
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# ──────────────────────────────────────────────────────────
# Настройки стиля
# ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'lines.linewidth':   1.5,
})

DRONE_COLORS = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c',
                '#e67e22', '#2ecc71', '#34495e', '#16a085', '#c0392b']

ALGO_COLORS = {
    'hardcode':  '#e74c3c',
    'apf':       '#3498db',
    'apf_lanes': '#2ecc71',
    'boids':     '#9b59b6',
}
ALGO_LABELS = {
    'hardcode':  'Hard-code',
    'apf':       'APF',
    'apf_lanes': 'APF + Lanes',
    'boids':     'Boids',
}

CAM_FOV_DEG = 78.0
OVERLAP = 0.2


def detect_algo(filename: str) -> str:
    fn = os.path.basename(filename).lower()
    if 'lanes' in fn:
        return 'apf_lanes'
    if 'apf' in fn:
        return 'apf'
    if 'boids' in fn:
        return 'boids'
    if 'hardcode' in fn or 'hard_code' in fn:
        return 'hardcode'
    return 'unknown'


def load_obstacles(directory: str):
    """Читает obstacles.txt, возвращает [(x, y, r), ...] или []."""
    path = os.path.join(directory, 'obstacles.txt')
    obs = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    obs.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return obs


def compute_footprint(height: float) -> float:
    fov = math.radians(CAM_FOV_DEG)
    return 2.0 * height * math.tan(fov / 2.0)


def build_ideal_route(field_len: float, field_wid: float, height: float,
                      n_drones: int, drone_rank: int, obstacles: list) -> list:
    """
    Строит идеальный маршрут змейки для одного дрона.
    Повторяет логику _build_snake() из алгоритмов.
    Возвращает [(x, y, z), ...].
    """
    fov = math.radians(CAM_FOV_DEG)
    stripe = max(1e-3, 2.0 * height * math.tan(fov / 2.0) * (1.0 - OVERLAP))
    total_stripes = max(1, int(math.ceil(field_wid / stripe)))
    my_stripes = [i for i in range(total_stripes) if i % n_drones == drone_rank]

    def shift_out(x, y):
        for ox, oy, orad in obstacles:
            d = math.hypot(x - ox, y - oy)
            safe = orad + 1.5
            if d < safe:
                nx = (x - ox) / (d + 1e-9)
                ny = (y - oy) / (d + 1e-9)
                x = ox + nx * safe
                y = oy + ny * safe
        return x, y

    pts = []
    for k, s in enumerate(my_stripes):
        y = stripe * (s + 0.5)
        xs = [0.0, field_len] if k % 2 == 0 else [field_len, 0.0]
        step = max(5.0, stripe)
        n = max(2, int(math.ceil(field_len / step)) + 1)
        for i in range(n):
            a = i / (n - 1)
            xp = xs[0] + a * (xs[1] - xs[0])
            sx, sy = shift_out(xp, y)
            pts.append((sx, sy, height))
    return pts


def save_or_show(fig, name: str, save: bool, out_dir: str):
    if save:
        path = os.path.join(out_dir, name + '.png')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Сохранён: {path}')
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# График 1: Траектория каждого дрона отдельно
# ═══════════════════════════════════════════════════════════
def plot_drone_trajectories(data: dict, obstacles: list, field_len: float,
                            field_wid: float, height: float, save: bool, out_dir: str):
    """
    Для каждого алгоритма и каждого дрона: XY-график с реальной и идеальной
    траекторией + препятствия + footprint.
    """
    footprint = compute_footprint(height)

    for algo, dfs in data.items():
        n_drones = len(dfs)
        if n_drones == 0:
            continue

        fig, axes = plt.subplots(1, n_drones, figsize=(6 * n_drones, 6))
        if n_drones == 1:
            axes = [axes]

        for idx, df in enumerate(dfs):
            ax = axes[idx]
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else idx
            color = DRONE_COLORS[did % len(DRONE_COLORS)]

            # --- Идеальный маршрут ---
            ideal = build_ideal_route(field_len, field_wid, height, n_drones, idx, obstacles)
            if ideal:
                ix = [p[0] for p in ideal]
                iy = [p[1] for p in ideal]
                ax.plot(ix, iy, '--', color='#2ecc71', linewidth=2.0,
                        alpha=0.9, label='Идеальный маршрут', zorder=2)

            # --- Реальная траектория ---
            if 'pos_x' in df.columns and 'pos_y' in df.columns:
                # Убираем NaN
                real = df[['pos_x', 'pos_y', 'formation_error_m']].dropna()
                if len(real) > 1:
                    # Цветовая карта по formation_error
                    points = real[['pos_x', 'pos_y']].values
                    errors = real['formation_error_m'].values if 'formation_error_m' in real.columns else None

                    ax.plot(real['pos_x'], real['pos_y'], '-', color=color,
                            linewidth=1.2, alpha=0.85, label='Реальный путь', zorder=3)

                    # Точки старта и финиша
                    ax.scatter(real['pos_x'].iloc[0], real['pos_y'].iloc[0],
                               c='green', s=80, marker='o', zorder=5, label='Старт')
                    ax.scatter(real['pos_x'].iloc[-1], real['pos_y'].iloc[-1],
                               c='red', s=80, marker='s', zorder=5, label='Финиш')

                    # Формация — target waypoints
                    if 'target_x' in df.columns and 'target_y' in df.columns:
                        tgts = df[['target_x', 'target_y']].drop_duplicates().dropna()
                        if len(tgts) > 0:
                            ax.scatter(tgts['target_x'], tgts['target_y'],
                                       c='#f39c12', s=25, marker='x', alpha=0.6,
                                       zorder=4, label='Waypoints')

                    # Heatmap отклонений (scatter с цветом)
                    if errors is not None and len(errors) > 10:
                        # Субсэмплинг для читаемости
                        step = max(1, len(errors) // 500)
                        sc = ax.scatter(real['pos_x'].iloc[::step],
                                        real['pos_y'].iloc[::step],
                                        c=errors[::step], cmap='YlOrRd',
                                        s=8, alpha=0.5, zorder=3,
                                        vmin=0, vmax=max(2.0, errors.max()))
                        plt.colorbar(sc, ax=ax, shrink=0.5, label='Ошибка строя, м')

            # --- Препятствия ---
            for ox, oy, orad in obstacles:
                circ = plt.Circle((ox, oy), orad, color='black', alpha=0.3, zorder=1)
                ax.add_patch(circ)
                ax.plot(ox, oy, 'k+', markersize=6, alpha=0.5)

            # --- Footprint для этого дрона ---
            half_fp = footprint / 2.0
            # Покажем footprint в стартовой точке
            if 'pos_x' in df.columns and len(df) > 0:
                sx = df['pos_x'].iloc[0]
                sy = df['pos_y'].iloc[0]
                rect = plt.Rectangle((sx - half_fp, sy - half_fp),
                                     footprint, footprint,
                                     fill=False, color=color, linestyle=':',
                                     linewidth=1.5, alpha=0.5,
                                     label=f'Footprint ({footprint:.1f}м)')
                ax.add_patch(rect)

            ax.set_xlim(-5, field_len + 5)
            ax.set_ylim(-5, field_wid + 5)
            ax.set_aspect('equal')
            ax.set_xlabel('X, м')
            ax.set_ylabel('Y, м')
            ax.set_title(f'Дрон {did}')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        fig.suptitle(f'Траектории — {ALGO_LABELS.get(algo, algo)}\n'
                     f'(зелёный пунктир = идеальный, сплошная = реальный, "+" = препятствия)',
                     fontsize=13)
        fig.tight_layout()
        save_or_show(fig, f'trajectories_{algo}', save, out_dir)


# ═══════════════════════════════════════════════════════════
# График 2: Все дроны одного алгоритма на одном полотне
# ═══════════════════════════════════════════════════════════
def plot_all_drones_together(data: dict, obstacles: list, field_len: float,
                             field_wid: float, height: float, save: bool, out_dir: str):
    """Сравнительный график: все дроны всех алгоритмов на одном полотне."""
    n_algos = len(data)
    if n_algos == 0:
        return

    cols = min(2, n_algos)
    rows = math.ceil(n_algos / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6.5 * rows))
    if n_algos == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, (algo, dfs) in enumerate(data.items()):
        ax = axes[ax_idx]
        n_drones = len(dfs)

        for idx, df in enumerate(dfs):
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else idx
            color = DRONE_COLORS[did % len(DRONE_COLORS)]

            # Идеальный маршрут (тонкий зелёный)
            ideal = build_ideal_route(field_len, field_wid, height, n_drones, idx, obstacles)
            if ideal:
                ix = [p[0] for p in ideal]
                iy = [p[1] for p in ideal]
                ax.plot(ix, iy, '--', color='#2ecc71', linewidth=1.0, alpha=0.5, zorder=1)

            # Реальный путь
            if 'pos_x' in df.columns and 'pos_y' in df.columns:
                real = df[['pos_x', 'pos_y']].dropna()
                if len(real) > 1:
                    ax.plot(real['pos_x'], real['pos_y'], '-', color=color,
                            linewidth=1.2, alpha=0.8, label=f'Дрон {did}', zorder=3)
                    # Старт
                    ax.scatter(real['pos_x'].iloc[0], real['pos_y'].iloc[0],
                               c=color, s=60, marker='o', zorder=5, edgecolors='green',
                               linewidths=1.5)
                    # Финиш
                    ax.scatter(real['pos_x'].iloc[-1], real['pos_y'].iloc[-1],
                               c=color, s=60, marker='s', zorder=5, edgecolors='red',
                               linewidths=1.5)

        # Препятствия
        for ox, oy, orad in obstacles:
            circ = plt.Circle((ox, oy), orad, color='black', alpha=0.25, zorder=1)
            ax.add_patch(circ)

        ax.set_xlim(-5, field_len + 5)
        ax.set_ylim(-5, field_wid + 5)
        ax.set_aspect('equal')
        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')
        ax.set_title(ALGO_LABELS.get(algo, algo), fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    # Скрыть лишние оси
    for i in range(len(data), len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Сравнение траекторий всех алгоритмов\n'
                 f'(○ старт, □ финиш, зелёный -- = идеальный путь)',
                 fontsize=14)
    fig.tight_layout()
    save_or_show(fig, 'trajectories_all_algos', save, out_dir)


# ═══════════════════════════════════════════════════════════
# График 3: Deviation heatmap (2D-интерполяция formation_error)
# ═══════════════════════════════════════════════════════════
def plot_deviation_heatmap(data: dict, obstacles: list, field_len: float,
                           field_wid: float, height: float, save: bool, out_dir: str):
    """
    Интерполирует formation_error на регулярную сетку и строит heatmap.
    Показывает зоны максимального отклонения от строя.
    """
    for algo, dfs in data.items():
        n_drones = len(dfs)
        if n_drones == 0:
            continue

        fig, axes = plt.subplots(1, n_drones, figsize=(6 * n_drones, 5.5))
        if n_drones == 1:
            axes = [axes]

        for idx, df in enumerate(dfs):
            ax = axes[idx]
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else idx

            cols_needed = ['pos_x', 'pos_y', 'formation_error_m']
            if not all(c in df.columns for c in cols_needed):
                ax.set_title(f'Дрон {did} — нет данных formation_error')
                continue

            sub = df[cols_needed].dropna()
            if len(sub) < 20:
                ax.set_title(f'Дрон {did} — мало точек')
                continue

            # Регулярная сетка
            xi = np.linspace(0, field_len, 200)
            yi = np.linspace(0, field_wid, 200)
            Xi, Yi = np.meshgrid(xi, yi)

            # Интерполяция
            Zi = griddata((sub['pos_x'], sub['pos_y']),
                          sub['formation_error_m'], (Xi, Yi), method='linear')

            im = ax.contourf(Xi, Yi, Zi, levels=20, cmap='YlOrRd', alpha=0.7)
            plt.colorbar(im, ax=ax, shrink=0.6, label='Ошибка строя, м')

            # Реальный путь поверх
            ax.plot(sub['pos_x'], sub['pos_y'], '-', color='navy',
                    linewidth=0.8, alpha=0.5, zorder=3)

            # Препятствия
            for ox, oy, orad in obstacles:
                circ = plt.Circle((ox, oy), orad, color='black', alpha=0.35, zorder=2)
                ax.add_patch(circ)

            ax.set_xlim(0, field_len)
            ax.set_ylim(0, field_wid)
            ax.set_aspect('equal')
            ax.set_xlabel('X, м')
            ax.set_ylabel('Y, м')
            ax.set_title(f'Дрон {did}')

        fig.suptitle(f'Карта отклонений от строя — {ALGO_LABELS.get(algo, algo)}',
                     fontsize=13)
        fig.tight_layout()
        save_or_show(fig, f'deviation_heatmap_{algo}', save, out_dir)


# ═══════════════════════════════════════════════════════════
# График 4: 3D-визуализация пути (X, Y, formation_error)
# ═══════════════════════════════════════════════════════════
def plot_3d_trajectory_error(data: dict, obstacles: list, field_len: float,
                             field_wid: float, save: bool, out_dir: str):
    """3D-график: ось Z = formation_error вдоль пути."""
    for algo, dfs in data.items():
        n_drones = len(dfs)
        if n_drones == 0:
            continue

        fig = plt.figure(figsize=(6 * n_drones, 5.5))
        for idx, df in enumerate(dfs):
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else idx
            color = DRONE_COLORS[did % len(DRONE_COLORS)]

            ax = fig.add_subplot(1, n_drones, idx + 1, projection='3d')

            cols_needed = ['pos_x', 'pos_y', 'formation_error_m']
            if not all(c in df.columns for c in cols_needed):
                continue

            sub = df[cols_needed].dropna()
            if len(sub) < 10:
                continue

            # Субсэмплинг
            step = max(1, len(sub) // 800)
            xs = sub['pos_x'].iloc[::step]
            ys = sub['pos_y'].iloc[::step]
            zs = sub['formation_error_m'].iloc[::step]

            ax.plot(xs, ys, zs, color=color, linewidth=1.2, alpha=0.8)
            ax.scatter(xs.iloc[0], ys.iloc[0], zs.iloc[0],
                       c='green', s=40, marker='o', label='Старт')

            ax.set_xlabel('X, м')
            ax.set_ylabel('Y, м')
            ax.set_zlabel('Ошибка строя, м')
            ax.set_title(f'Дрон {did}')
            ax.set_xlim(0, field_len)
            ax.set_ylim(0, field_wid)

        fig.suptitle(f'3D: отклонение от строя вдоль пути — {ALGO_LABELS.get(algo, algo)}',
                     fontsize=13)
        fig.tight_layout()
        save_or_show(fig, f'3d_trajectory_error_{algo}', save, out_dir)


# ═══════════════════════════════════════════════════════════
# График 5: Сравнение реального пути с идеальным (расстояние)
# ═══════════════════════════════════════════════════════════
def plot_path_deviation_distance(data: dict, field_len: float, field_wid: float,
                                 height: float, save: bool, out_dir: str):
    """
    Для каждого дрона: график бокового отклонения от идеальной траектории
    (cross-track error) как функция пройденного пути по X.
    """
    for algo, dfs in data.items():
        n_drones = len(dfs)
        if n_drones == 0:
            continue

        fig, axes = plt.subplots(1, n_drones, figsize=(6 * n_drones, 4.5))
        if n_drones == 1:
            axes = [axes]

        for idx, df in enumerate(dfs):
            ax = axes[idx]
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else idx
            color = DRONE_COLORS[did % len(DRONE_COLORS)]

            # Идеальный маршрут
            ideal = build_ideal_route(field_len, field_wid, height, n_drones, idx, [])
            if not ideal or 'pos_x' not in df.columns:
                continue

            # Для каждой точки реального пути — найти ближайшую на идеальном
            real_pts = df[['pos_x', 'pos_y']].dropna().values
            ideal_pts = np.array([(p[0], p[1]) for p in ideal])

            cross_track_errors = []
            x_progress = []
            for rx, ry in real_pts:
                dists = np.hypot(ideal_pts[:, 0] - rx, ideal_pts[:, 1] - ry)
                min_idx = np.argmin(dists)
                cross_track_errors.append(dists[min_idx])
                x_progress.append(rx)

            ax.plot(x_progress, cross_track_errors, color=color, linewidth=1.0, alpha=0.8)
            ax.axhline(0, color='green', linestyle='--', alpha=0.5, label='Идеал')
            ax.fill_between(x_progress, 0, cross_track_errors, alpha=0.1, color=color)

            ax.set_xlabel('X, м (прогресс)')
            ax.set_ylabel('Cross-track error, м')
            ax.set_title(f'Дрон {did}')
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)

        fig.suptitle(f'Боковое отклонение от идеальной траектории — {ALGO_LABELS.get(algo, algo)}',
                     fontsize=13)
        fig.tight_layout()
        save_or_show(fig, f'cross_track_error_{algo}', save, out_dir)


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='Визуализация траекторий дронов (реальный vs идеальный путь)')
    parser.add_argument('--dir', default='.',
                        help='Папка с CSV-логами и obstacles.txt')
    parser.add_argument('--save', action='store_true',
                        help='Сохранять PNG вместо показа')
    parser.add_argument('--out', default=None,
                        help='Папка для PNG (по умолчанию = --dir)')
    parser.add_argument('--algo', default=None,
                        choices=['hardcode', 'apf', 'apf_lanes', 'boids'],
                        help='Фильтр по одному алгоритму')
    parser.add_argument('--field_len', type=float, default=60.0,
                        help='Длина поля, м (для построения идеального маршрута)')
    parser.add_argument('--field_wid', type=float, default=30.0,
                        help='Ширина поля, м')
    parser.add_argument('--height', type=float, default=5.0,
                        help='Высота полёта, м')
    args = parser.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    # Загрузка CSV
    pattern = os.path.join(args.dir, 'metrics_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'[ERROR] Не найдено metrics_*.csv в {args.dir}')
        sys.exit(1)

    data: dict[str, list[pd.DataFrame]] = {}
    for fp in files:
        algo = detect_algo(fp)
        if args.algo and algo != args.algo:
            continue
        try:
            df = pd.read_csv(fp)
            if 'mission_time_s' in df.columns:
                df['t'] = df['mission_time_s']
            else:
                df['t'] = df['time_s'] - df['time_s'].iloc[0]
            data.setdefault(algo, []).append(df)
            print(f'  Загружен {os.path.basename(fp)} → algo={algo}, строк={len(df)}')
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')

    if not data:
        print('[ERROR] Нет данных после фильтрации')
        sys.exit(1)

    print(f'\nНайдено алгоритмов: {list(data.keys())}')

    # Загрузка препятствий
    obstacles = load_obstacles(args.dir)
    print(f'Препятствий: {len(obstacles)}')

    print('\n=== График 1: Траектории по дронам ===')
    plot_drone_trajectories(data, obstacles, args.field_len,
                            args.field_wid, args.height, args.save, out_dir)

    print('=== График 2: Все дроны вместе ===')
    plot_all_drones_together(data, obstacles, args.field_len,
                             args.field_wid, args.height, args.save, out_dir)

    print('=== График 3: Карта отклонений (heatmap) ===')
    plot_deviation_heatmap(data, obstacles, args.field_len,
                           args.field_wid, args.height, args.save, out_dir)

    print('=== График 4: 3D отклонение ===')
    plot_3d_trajectory_error(data, obstacles, args.field_len,
                             args.field_wid, args.save, out_dir)

    print('=== График 5: Cross-track error ===')
    plot_path_deviation_distance(data, args.field_len, args.field_wid,
                                 args.height, args.save, out_dir)

    print('\nГотово.')


if __name__ == '__main__':
    main()
