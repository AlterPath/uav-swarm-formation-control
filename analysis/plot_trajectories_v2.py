#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trajectories_v2.py — Визуализация траекторий роя дронов.

Что исправлено по сравнению с plot_trajectories.py:
  • Корректная система координат (X уходит в минус, Y растёт вверх)
  • Идеальный маршрут берётся из реальных target_x/target_y в логах, а не строится синтетически
  • Режим lose_drone: потерянный дрон выделяется отдельно, место потери отмечается
  • Один общий план с всеми дронами + отдельные панели на каждый дрон
  • Цветовая карта пути по ошибке формации
  • Waypoints, photo-shots, препятствия, полосы обзора

Запуск:
    python3 plot_trajectories_v2.py --dir /path/to/csvs
    python3 plot_trajectories_v2.py --dir . --obstacles obstacles.txt --save
    python3 plot_trajectories_v2.py --dir . --save --out ./plots
"""

import argparse
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import pandas as pd

# ── Стиль ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi':     130,
    'axes.grid':      True,
    'grid.alpha':     0.25,
    'grid.linestyle': ':',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# Цвета дронов (по id)
DRONE_PALETTE = {
    0: '#2980b9',   # синий
    1: '#e74c3c',   # красный (потерянный будет вспомогательным)
    2: '#27ae60',   # зелёный
    3: '#8e44ad',   # фиолетовый
    4: '#f39c12',   # оранжевый
}
LOST_COLOR  = '#e74c3c'
LOST_MARKER = '✕'

# Параметры камеры (для отрисовки полосы обзора)
CAM_FOV_DEG = 78.0
OVERLAP     = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# Загрузка данных
# ══════════════════════════════════════════════════════════════════════════════

def parse_drone_id(filepath: str) -> int:
    m = re.search(r'uav(\d+)', os.path.basename(filepath))
    return int(m.group(1)) if m else 0


def load_drones(directory: str) -> list[dict]:
    """Загружает все metrics_*.csv из папки. Возвращает список словарей."""
    files = sorted(glob.glob(os.path.join(directory, 'metrics_*.csv')))
    if not files:
        print(f'[ERROR] Не найдено metrics_*.csv в {directory}')
        sys.exit(1)

    drones = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')
            continue

        did = parse_drone_id(fp)

        # Нормированное время миссии
        if 'mission_time_s' in df.columns:
            df['t'] = df['mission_time_s']
        else:
            df['t'] = df['time_s'] - df['time_s'].iloc[0]
        df = df[df['t'] >= 0].reset_index(drop=True)

        # Флаг потери дрона
        events = df['event'].fillna('').str.strip() if 'event' in df.columns else pd.Series([''] * len(df))
        mission_end = df[events.isin(['MISSION_END', 'RTL'])]
        completed   = events.isin(['MISSION_END']).any()
        t_total     = df['t'].max()

        # Считаем дрон потерянным если он не дошёл до конца и
        # его время значительно короче остальных (проверяется позже)
        drones.append({
            'id':        did,
            'df':        df,
            'filepath':  fp,
            'completed': completed,
            't_total':   t_total,
            'n_rows':    len(df),
        })
        print(f'  Дрон {did}: {len(df)} строк, t_max={t_total:.1f}s, '
              f'{"завершён" if completed else "⚠ ПОТЕРЯН"}')

    # Автоматическое определение потерянных:
    # дрон считается потерянным если его время < 60% от максимального
    if drones:
        t_max = max(d['t_total'] for d in drones)
        for d in drones:
            if not d['completed'] and d['t_total'] < 0.6 * t_max:
                d['lost'] = True
            else:
                d['lost'] = False

    return drones


def load_obstacles(path: str) -> list[tuple]:
    """Читает файл препятствий (x,y,r по строкам)."""
    obs = []
    if not path or not os.path.exists(path):
        return obs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(';', ',').split(',')
            if len(parts) >= 3:
                try:
                    obs.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    print(f'  Препятствий: {len(obs)}')
    return obs


# ══════════════════════════════════════════════════════════════════════════════
# Геометрические утилиты
# ══════════════════════════════════════════════════════════════════════════════

def footprint_width(alt: float) -> float:
    return 2.0 * alt * math.tan(math.radians(CAM_FOV_DEG) / 2.0)


def get_field_bounds(drones: list[dict]) -> tuple[float, float, float, float]:
    """Вычисляет границы поля по реальным траекториям."""
    all_x, all_y = [], []
    for d in drones:
        df = d['df']
        all_x.extend(df['pos_x'].dropna().tolist())
        all_y.extend(df['pos_y'].dropna().tolist())
    pad = 3.0
    return (min(all_x) - pad, max(all_x) + pad,
            min(all_y) - pad, max(all_y) + pad)


def extract_ideal_route(df: pd.DataFrame) -> np.ndarray:
    """
    Восстанавливает идеальный маршрут как последовательность
    уникальных целевых точек target_x/target_y из лога.
    """
    cols = [c for c in ['target_x', 'target_y', 'target_z'] if c in df.columns]
    if len(cols) < 2:
        return np.empty((0, 2))
    targets = df[cols[:2]].copy()
    # Убираем начальное (0,0,0) и дубликаты соседних строк
    mask = ~((targets['target_x'] == 0) & (targets['target_y'] == 0))
    targets = targets[mask]
    # Уникальные последовательные точки
    shifted = targets.shift(1)
    changed = (targets != shifted).any(axis=1)
    route = targets[changed].drop_duplicates().values
    return route


def colorline(ax, x, y, z, cmap='RdYlGn_r', linewidth=2.2, alpha=0.9,
              vmin=None, vmax=None, zorder=4):
    """Рисует линию с цветовой картой по z."""
    z  = np.asarray(z, dtype=float)
    vmin = vmin if vmin is not None else np.nanmin(z)
    vmax = vmax if vmax is not None else np.nanmax(z)
    if vmax == vmin:
        vmax = vmin + 1e-6

    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    z_seg    = z[:-1]

    lc = LineCollection(segments, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                        linewidth=linewidth, alpha=alpha, zorder=zorder,
                        capstyle='round', joinstyle='round')
    lc.set_array(z_seg)
    ax.add_collection(lc)
    return lc, vmin, vmax


# ══════════════════════════════════════════════════════════════════════════════
# Рисование отдельного дрона на оси
# ══════════════════════════════════════════════════════════════════════════════

def draw_drone_on_ax(ax, drone: dict, show_ideal=True,
                     show_waypoints=True, show_footprint=True,
                     color_by_error=True):
    """
    Рисует на ax:
      – Полосу обзора (footprint) — бледный прямоугольник
      – Идеальный маршрут (target waypoints)
      – Реальную траекторию (цветокодировка по ошибке)
      – Старт / финиш / точку потери
      – Waypoints и photo-shots
    Возвращает ScalarMappable для colorbar (или None).
    """
    df    = drone['df']
    did   = drone['id']
    lost  = drone.get('lost', False)
    color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')

    x = df['pos_x'].values
    y = df['pos_y'].values

    # ── Footprint (полоса обзора) ──────────────────────────────────────────
    if show_footprint and 'pos_z' in df.columns:
        alt = df['pos_z'].mean()
        fw  = footprint_width(alt)
        stripe_w = fw * (1 - OVERLAP)
        # Узкий прямоугольник вдоль пути
        if len(x) > 1:
            x_min, x_max = min(x), max(x)
            y_c = np.median(y)
            rect = mpatches.FancyBboxPatch(
                (x_min, y_c - fw / 2), x_max - x_min, fw,
                boxstyle='round,pad=0.5',
                linewidth=0, facecolor=color, alpha=0.06, zorder=1)
            ax.add_patch(rect)

    # ── Идеальный маршрут ──────────────────────────────────────────────────
    sm = None
    if show_ideal:
        route = extract_ideal_route(df)
        if len(route) > 1:
            ax.plot(route[:, 0], route[:, 1],
                    linestyle='--', color='#7f8c8d', linewidth=1.4,
                    alpha=0.6, zorder=2, label='Идеальный маршрут')
            ax.scatter(route[:, 0], route[:, 1],
                       c='#7f8c8d', s=18, marker='x', alpha=0.5, zorder=3)

    # ── Реальная траектория ────────────────────────────────────────────────
    if len(x) > 1:
        if color_by_error and 'formation_error_m' in df.columns:
            err   = df['formation_error_m'].values.astype(float)
            vmin_ = np.nanpercentile(err, 2)
            vmax_ = np.nanpercentile(err, 98)
            lc, vmin_, vmax_ = colorline(ax, x, y, err,
                                         vmin=vmin_, vmax=vmax_, zorder=4)
            sm = plt.cm.ScalarMappable(
                cmap='RdYlGn_r', norm=Normalize(vmin=vmin_, vmax=vmax_))
            sm.set_array([])
        else:
            ax.plot(x, y, color=color, linewidth=2.2, alpha=0.85,
                    zorder=4, label=f'Дрон {did}')

    # ── Waypoints ──────────────────────────────────────────────────────────
    if show_waypoints and 'event' in df.columns:
        ev = df['event'].fillna('').str.strip()

        wp_mask   = ev == 'WAYPOINT'
        photo_mask = ev == 'PHOTO_SHOT'

        if wp_mask.any():
            ax.scatter(x[wp_mask], y[wp_mask],
                       c='#f39c12', s=55, marker='D', zorder=6,
                       edgecolors='white', linewidths=0.7,
                       label='Waypoint')
        if photo_mask.any():
            ax.scatter(x[photo_mask], y[photo_mask],
                       c='#9b59b6', s=28, marker='s', zorder=5,
                       alpha=0.7, label='Фото')

    # ── Старт / финиш / потеря ────────────────────────────────────────────
    ax.scatter(x[0], y[0],
               c='#27ae60', s=120, marker='o', zorder=8,
               edgecolors='white', linewidths=1.2,
               label=f'Старт (Д{did})')

    if lost:
        # Последняя известная позиция
        ax.scatter(x[-1], y[-1],
                   c=LOST_COLOR, s=200, marker='X', zorder=9,
                   edgecolors='white', linewidths=1.2,
                   label=f'Потеря Д{did} (t={drone["t_total"]:.0f}s)')
        ax.annotate(f'  ⚠ Дрон {did}\n  потерян',
                    (x[-1], y[-1]),
                    fontsize=8.5, color=LOST_COLOR, fontweight='bold',
                    xytext=(8, 8), textcoords='offset points',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])
    else:
        ax.scatter(x[-1], y[-1],
                   c='#c0392b', s=120, marker='s', zorder=8,
                   edgecolors='white', linewidths=1.2,
                   label=f'Финиш (Д{did})')

    return sm


# ══════════════════════════════════════════════════════════════════════════════
# Отрисовка препятствий и поля
# ══════════════════════════════════════════════════════════════════════════════

def draw_obstacles(ax, obstacles: list[tuple]):
    for i, (ox, oy, orad) in enumerate(obstacles):
        circ = plt.Circle((ox, oy), orad,
                           color='#2c3e50', alpha=0.55, zorder=7)
        ax.add_patch(circ)
        ax.text(ox, oy, f'r={orad:.0f}',
                ha='center', va='center', fontsize=7.5,
                color='white', fontweight='bold', zorder=8)


def style_ax(ax, title: str,
             xmin: float, xmax: float,
             ymin: float, ymax: float,
             show_legend: bool = True):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X, м')
    ax.set_ylabel('Y, м')
    ax.set_title(title, pad=8)
    if show_legend:
        ax.legend(loc='upper right', fontsize=8,
                  framealpha=0.85, edgecolor='#cccccc',
                  ncol=2 if ax.get_figure().get_size_inches()[0] > 10 else 1)


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ГРАФИК 1: все дроны на одном поле
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_together(drones: list[dict], obstacles: list[tuple],
                      save: bool, out_dir: str):
    """
    Один большой XY-план с траекториями всех дронов.
    Цвет линии — по дрону (не по ошибке), чтобы было читаемо.
    """
    xmin, xmax, ymin, ymax = get_field_bounds(drones)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor('#f5f6fa')

    # Сетка полос (горизонтальные линии по Y-полосам)
    y_lanes = sorted({round(d['df']['pos_y'].median(), 1) for d in drones})
    for yl in y_lanes:
        ax.axhline(yl, color='#dfe6e9', linewidth=0.8, zorder=0)

    for drone in sorted(drones, key=lambda d: d['id']):
        did   = drone['id']
        lost  = drone.get('lost', False)
        color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')
        df    = drone['df']
        x, y  = df['pos_x'].values, df['pos_y'].values

        # Footprint
        if 'pos_z' in df.columns and len(x) > 1:
            alt = df['pos_z'].mean()
            fw  = footprint_width(alt)
            ax.fill_between(x, y - fw / 2, y + fw / 2,
                            alpha=0.07, color=color, zorder=1)

        # Идеальный маршрут
        route = extract_ideal_route(df)
        if len(route) > 1:
            ax.plot(route[:, 0], route[:, 1],
                    linestyle='--', color=color, linewidth=1.2,
                    alpha=0.45, zorder=2)

        # Траектория
        lw    = 2.0 if not lost else 1.6
        lsty  = '-' if not lost else '-.'
        label = f'Дрон {did}' + (' (потерян t={:.0f}s)'.format(drone['t_total']) if lost else '')
        ax.plot(x, y, color=color, linewidth=lw, linestyle=lsty,
                alpha=0.85, zorder=4, label=label)

        # Photo shots
        if 'event' in df.columns:
            ev = df['event'].fillna('').str.strip()
            pm = ev == 'PHOTO_SHOT'
            if pm.any():
                ax.scatter(x[pm], y[pm], c=color, s=20, marker='s',
                           zorder=5, alpha=0.5)

        # Старт
        ax.scatter(x[0], y[0], c=color, s=130, marker='o', zorder=8,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(f'  Д{did}', (x[0], y[0]),
                    fontsize=8, color=color, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # Финиш / потеря
        marker = 'X' if lost else 's'
        ax.scatter(x[-1], y[-1], c=color, s=150, marker=marker, zorder=8,
                   edgecolors='white', linewidths=1.5)
        if lost:
            ax.annotate(f'  ⚠', (x[-1], y[-1]),
                        fontsize=11, color=LOST_COLOR,
                        xytext=(3, 3), textcoords='offset points',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    draw_obstacles(ax, obstacles)
    style_ax(ax, 'Траектории всех дронов', xmin, xmax, ymin, ymax)

    # Легенда событий
    leg_items = [
        mpatches.Patch(color='none', label='──── Идеальный маршрут (пунктир)'),
        mpatches.Patch(color='none', label='■  Точки фотографирования'),
        mpatches.Patch(color='none', label='●  Старт   ■  Финиш   ✕  Потеря'),
    ]

    fig.suptitle('APF + Lanes — Тяжёлые препятствия — Потеря дрона',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_or_show(fig, 'traj_all_drones', save, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ГРАФИК 2: панели по дронам с colorbar по ошибке формации
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_drone(drones: list[dict], obstacles: list[tuple],
                  save: bool, out_dir: str):
    """
    Отдельная панель для каждого дрона.
    Линия раскрашена по formation_error_m (RdYlGn_r).
    """
    n = len(drones)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5),
                             squeeze=False)

    xmin, xmax, ymin, ymax = get_field_bounds(drones)

    for col, drone in enumerate(sorted(drones, key=lambda d: d['id'])):
        ax   = axes[0][col]
        did  = drone['id']
        lost = drone.get('lost', False)
        ax.set_facecolor('#f5f6fa')

        sm = draw_drone_on_ax(ax, drone,
                              show_ideal=True,
                              show_waypoints=True,
                              show_footprint=True,
                              color_by_error=(not lost))

        draw_obstacles(ax, obstacles)

        title = f'Дрон {did}'
        if lost:
            title += f'  ⚠ ПОТЕРЯН (t={drone["t_total"]:.0f}s)'
        style_ax(ax, title, xmin, xmax, ymin, ymax)

        if sm is not None:
            cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
            cb.set_label('Ошибка формации, м', fontsize=9)
            cb.ax.tick_params(labelsize=8)

    fig.suptitle('APF + Lanes — траектории по дронам (цвет = ошибка формации)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_or_show(fig, 'traj_per_drone', save, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ГРАФИК 3: сравнение XY реального vs идеального (наложение)
# ══════════════════════════════════════════════════════════════════════════════

def plot_real_vs_ideal(drones: list[dict], obstacles: list[tuple],
                       save: bool, out_dir: str):
    """
    Для каждого дрона: реальный путь (сплошная) vs идеальный маршрут (пунктир)
    на общем поле, с заливкой отклонения.
    """
    n = len(drones)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6),
                              squeeze=False)
    xmin, xmax, ymin, ymax = get_field_bounds(drones)

    for col, drone in enumerate(sorted(drones, key=lambda d: d['id'])):
        ax   = axes[0][col]
        did  = drone['id']
        lost = drone.get('lost', False)
        color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')
        df   = drone['df']
        ax.set_facecolor('#f5f6fa')

        x_real = df['pos_x'].values
        y_real = df['pos_y'].values

        # Идеальный маршрут
        route = extract_ideal_route(df)
        if len(route) > 1:
            ax.plot(route[:, 0], route[:, 1],
                    '--', color='#2c3e50', linewidth=1.8,
                    alpha=0.65, zorder=3, label='Идеальный маршрут')
            ax.scatter(route[:, 0], route[:, 1],
                       c='#2c3e50', s=30, marker='o',
                       alpha=0.45, zorder=4)

        # Реальная траектория
        ax.plot(x_real, y_real,
                color=color, linewidth=2.2, alpha=0.88,
                zorder=5, label='Реальный путь')

        # Точки фото
        if 'event' in df.columns:
            ev = df['event'].fillna('').str.strip()
            pm = ev == 'PHOTO_SHOT'
            if pm.any():
                ax.scatter(x_real[pm], y_real[pm],
                           c='#9b59b6', s=35, marker='s',
                           zorder=6, alpha=0.7, label='Фото')

        # Старт / финиш
        ax.scatter(x_real[0], y_real[0],
                   c='#27ae60', s=130, marker='o', zorder=9,
                   edgecolors='white', linewidths=1.2, label='Старт')
        end_marker = 'X' if lost else 's'
        end_label  = f'Потеря t={drone["t_total"]:.0f}s' if lost else 'Финиш'
        ax.scatter(x_real[-1], y_real[-1],
                   c=LOST_COLOR if lost else '#c0392b',
                   s=150, marker=end_marker, zorder=9,
                   edgecolors='white', linewidths=1.2, label=end_label)

        draw_obstacles(ax, obstacles)

        title = f'Дрон {did}  —  реальный vs идеальный'
        if lost:
            title += '  ⚠ ПОТЕРЯН'
        style_ax(ax, title, xmin, xmax, ymin, ymax)

    fig.suptitle('APF + Lanes — реальный маршрут vs идеальный',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_or_show(fig, 'traj_real_vs_ideal', save, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ГРАФИК 4: 3-D траектория (X, Y, Z)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_trajectories(drones: list[dict], obstacles: list[tuple],
                         save: bool, out_dir: str):
    """3-мерные траектории X/Y/Z для всех дронов."""
    n = len(drones)
    fig = plt.figure(figsize=(6.5 * n, 6))

    xmin, xmax, ymin, ymax = get_field_bounds(drones)

    for col, drone in enumerate(sorted(drones, key=lambda d: d['id'])):
        ax = fig.add_subplot(1, n, col + 1, projection='3d')
        did  = drone['id']
        lost = drone.get('lost', False)
        color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')
        df   = drone['df']

        if 'pos_z' not in df.columns:
            continue
        x = df['pos_x'].values
        y = df['pos_y'].values
        z = df['pos_z'].values

        step = max(1, len(x) // 600)
        ax.plot(x[::step], y[::step], z[::step],
                color=color, linewidth=1.6, alpha=0.85)

        # Тень на "полу"
        ax.plot(x[::step], y[::step],
                np.full(len(x[::step]), z.min() - 0.2),
                color=color, linewidth=0.7, alpha=0.25)

        ax.scatter(x[0], y[0], z[0],
                   c='#27ae60', s=60, marker='o', zorder=5)
        end_marker = 'X' if lost else '^'
        ax.scatter(x[-1], y[-1], z[-1],
                   c=LOST_COLOR if lost else '#c0392b',
                   s=80, marker=end_marker, zorder=5)

        ax.set_xlabel('X, м', labelpad=6)
        ax.set_ylabel('Y, м', labelpad=6)
        ax.set_zlabel('Z, м', labelpad=4)
        title = f'Дрон {did}'
        if lost:
            title += ' ⚠'
        ax.set_title(title, pad=6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8)

    fig.suptitle('APF + Lanes — 3D-траектории (X / Y / высота)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save_or_show(fig, 'traj_3d', save, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Утилита сохранения
# ══════════════════════════════════════════════════════════════════════════════

def _save_or_show(fig, name: str, save: bool, out_dir: str):
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + '.png')
        fig.savefig(path, bbox_inches='tight', dpi=130)
        print(f'  ✓ {path}')
    else:
        plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Визуализация траекторий роя дронов (v2)')
    parser.add_argument('--dir', default='.',
                        help='Папка с metrics_*.csv')
    parser.add_argument('--obstacles', default=None,
                        help='Путь к файлу obstacles.txt (x,y,r)')
    parser.add_argument('--save', action='store_true',
                        help='Сохранять PNG вместо показа в окне')
    parser.add_argument('--out', default=None,
                        help='Папка для PNG (по умолчанию = --dir)')
    args = parser.parse_args()

    out_dir = args.out or args.dir
    obs_path = args.obstacles or os.path.join(args.dir, 'obstacles.txt')

    print('=== Визуализация траекторий v2 ===')
    print(f'Источник: {os.path.abspath(args.dir)}')

    drones    = load_drones(args.dir)
    obstacles = load_obstacles(obs_path)

    if not drones:
        print('[ERROR] Нет данных.')
        sys.exit(1)

    lost_ids = [d['id'] for d in drones if d.get('lost')]
    if lost_ids:
        print(f'  ⚠ Потерянные дроны: {lost_ids}')

    print('\nГенерация графиков...')
    plot_all_together(drones, obstacles,  args.save, out_dir)
    plot_per_drone(drones, obstacles,     args.save, out_dir)
    plot_real_vs_ideal(drones, obstacles, args.save, out_dir)
    plot_3d_trajectories(drones, obstacles, args.save, out_dir)
    print('\nГотово.')


if __name__ == '__main__':
    main()
