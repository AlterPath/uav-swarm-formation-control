#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_experiments.py — Комплексный анализ результатов экспериментов ВКР.

Автоматически обходит структуру каталогов:
  <base_dir>/<algo>/<obstacles>/<mode>/run_N/metrics_*.csv

Усредняет метрики по 3 прогонам, строит все ключевые графики:
  1.  Траектории дронов (XY-план)
  2.  Временны́е ряды ключевых метрик
  3.  Ошибка формации по времени (с доверительным интервалом)
  4.  Ошибка полосы (lane_error) по времени
  5.  Скорость по времени
  6.  Расстояние до ближайшего соседа
  7.  Накопленная длина пути
  8.  Покрытие поля (%) по времени
  9.  Компоненты сил APF
  10. Boxplot сводных метрик по алгоритмам
  11. Радарная диаграмма сравнения алгоритмов
  12. Тепловая карта отклонений (на поле)
  13. Сводная таблица метрик

Запуск:
    python3 analyze_experiments.py --base ~/vkr_results --out ~/vkr_results/plots
    python3 analyze_experiments.py --base . --algo apf_lanes --obs heavy --mode lose_drone
    python3 analyze_experiments.py --base . --demo   # Авто-демо-режим
"""

import argparse
import glob
import math
import os
import sys
import warnings
from pathlib import Path
from itertools import product as iproduct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')

# ── Конфигурация ───────────────────────────────────────────────────────────────
ALGOS     = ['hardcode', 'apf', 'apf_lanes', 'boids']
OBSTACLES = ['no_obstacles', 'light', 'heavy']
MODES     = ['normal', 'lose_drone']
RUNS      = [1, 2, 3]

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
DRONE_COLORS = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']

OBS_LABELS = {'no_obstacles': 'Без препятствий',
              'light': 'Лёгкие препятствия',
              'heavy': 'Тяжёлые препятствия'}
MODE_LABELS = {'normal': 'Штатный режим',
               'lose_drone': 'Потеря дрона'}

FIELD_LEN  = 60.0
FIELD_WID  = 30.0
FLIGHT_ALT = 4.0
CAM_FOV    = math.radians(78.0)
OVERLAP    = 0.2

# Стиль matplotlib
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize':10,
    'figure.dpi':     120,
    'lines.linewidth':1.6,
    'axes.grid':      True,
    'grid.alpha':     0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# Утилиты загрузки
# ══════════════════════════════════════════════════════════════════════════════

def detect_algo(filename: str) -> str:
    fn = os.path.basename(filename).lower()
    if 'lanes' in fn:   return 'apf_lanes'
    if 'apf'   in fn:   return 'apf'
    if 'boids' in fn:   return 'boids'
    if 'hardcode' in fn or 'hard_code' in fn: return 'hardcode'
    return 'unknown'


def detect_drone_id(filename: str) -> int:
    """Извлекает номер дрона из имени файла (uav0 → 0)."""
    import re
    m = re.search(r'uav(\d+)', os.path.basename(filename))
    return int(m.group(1)) if m else 0


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Нормированное время миссии
    if 'mission_time_s' in df.columns:
        df['t'] = df['mission_time_s']
    else:
        df['t'] = df['time_s'] - df['time_s'].iloc[0]
    # Убираем отрицательное время
    df = df[df['t'] >= 0].copy()
    return df


def scan_directory(base: Path) -> dict:
    """
    Сканирует структуру base/<algo>/<obs>/<mode>/run_N/metrics_*.csv
    Возвращает вложенный dict:
      result[algo][obs][mode][run_n] = [df_drone0, df_drone1, ...]
    """
    result = {}
    csv_found = 0

    for algo in ALGOS:
        algo_dir = base / algo
        if not algo_dir.exists():
            continue
        for obs in OBSTACLES:
            obs_dir = algo_dir / obs
            if not obs_dir.exists():
                continue
            for mode in MODES:
                mode_dir = obs_dir / mode
                if not mode_dir.exists():
                    continue
                for run in RUNS:
                    run_dir = mode_dir / f'run_{run}'
                    if not run_dir.exists():
                        continue
                    csvs = sorted(glob.glob(str(run_dir / 'metrics_*.csv')))
                    if not csvs:
                        continue
                    dfs = []
                    for fp in csvs:
                        try:
                            df = load_csv(fp)
                            df['_drone_id'] = detect_drone_id(fp)
                            dfs.append(df)
                        except Exception as e:
                            print(f'  [WARN] {fp}: {e}')
                    if dfs:
                        result.setdefault(algo, {}) \
                              .setdefault(obs, {}) \
                              .setdefault(mode, {})[run] = dfs
                        csv_found += len(dfs)

    print(f'  Загружено CSV: {csv_found}')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Усреднение по прогонам
# ══════════════════════════════════════════════════════════════════════════════

NUMERIC_METRICS = [
    'formation_error_m', 'lane_error_m', 'nearest_neighbor_m',
    'velocity_mps', 'path_length_m', 'mission_time_s',
    'coverage_pct', 'f_att_x', 'f_att_y', 'f_rep_x', 'f_rep_y',
    'f_tan_x', 'f_tan_y', 'f_lane_y',
]


def resample_to_grid(df: pd.DataFrame, cols: list, t_grid: np.ndarray) -> pd.DataFrame:
    """Линейная интерполяция колонок df на равномерную сетку t_grid."""
    out = {'t': t_grid}
    t = df['t'].values
    # Убираем дубли
    _, idx = np.unique(t, return_index=True)
    t = t[idx]
    for c in cols:
        if c not in df.columns:
            out[c] = np.full_like(t_grid, np.nan, dtype=float)
            continue
        v = df[c].values[idx].astype(float)
        mask = np.isfinite(v)
        if mask.sum() < 2:
            out[c] = np.full_like(t_grid, np.nan, dtype=float)
            continue
        fi = interp1d(t[mask], v[mask], bounds_error=False, fill_value='extrapolate')
        out[c] = fi(t_grid)
    return pd.DataFrame(out)


def average_runs(runs_dict: dict) -> dict:
    """
    Принимает {run_n: [df_drone0, df_drone1, ...]}
    Возвращает {drone_id: {'mean': df, 'std': df, 'n_runs': int}}
    """
    # Определяем дроны
    all_drone_ids = set()
    for dfs in runs_dict.values():
        for df in dfs:
            did = int(df['_drone_id'].iloc[0]) if '_drone_id' in df.columns else 0
            all_drone_ids.add(did)

    result = {}
    for did in sorted(all_drone_ids):
        # Сначала найдём общую временну́ю сетку
        t_min, t_max, t_len = [], [], []
        for run_n, dfs in runs_dict.items():
            for df in dfs:
                d_id = int(df['_drone_id'].iloc[0]) if '_drone_id' in df.columns else 0
                if d_id == did:
                    t_min.append(df['t'].min())
                    t_max.append(df['t'].max())
                    t_len.append(len(df))
        if not t_min:
            continue

        t_start = max(t_min)
        t_end   = min(t_max)
        n_pts   = int(np.median(t_len))
        if t_end <= t_start:
            t_end = max(t_max)

        t_grid = np.linspace(t_start, t_end, min(n_pts, 500))

        # Интерполируем и складываем
        cols = [c for c in NUMERIC_METRICS if any(
            c in df.columns
            for dfs in runs_dict.values()
            for df in dfs
            if (int(df['_drone_id'].iloc[0]) if '_drone_id' in df.columns else 0) == did
        )]

        resampled = []
        for run_n, dfs in sorted(runs_dict.items()):
            for df in dfs:
                d_id = int(df['_drone_id'].iloc[0]) if '_drone_id' in df.columns else 0
                if d_id == did:
                    rs = resample_to_grid(df, cols, t_grid)
                    resampled.append(rs)

        if not resampled:
            continue

        stacked = {c: np.vstack([r[c].values for r in resampled]) for c in cols}
        mean_df = pd.DataFrame({'t': t_grid})
        std_df  = pd.DataFrame({'t': t_grid})
        for c in cols:
            mean_df[c] = np.nanmean(stacked[c], axis=0)
            std_df[c]  = np.nanstd(stacked[c], axis=0)

        # Также запоминаем позиции из первого прогона
        first_dfs = list(runs_dict.values())[0]
        for df in first_dfs:
            d_id = int(df['_drone_id'].iloc[0]) if '_drone_id' in df.columns else 0
            if d_id == did:
                mean_df['_pos_x'] = resample_to_grid(df, ['pos_x'], t_grid)['pos_x'] \
                    if 'pos_x' in df.columns else np.nan
                mean_df['_pos_y'] = resample_to_grid(df, ['pos_y'], t_grid)['pos_y'] \
                    if 'pos_y' in df.columns else np.nan
                mean_df['_pos_z'] = resample_to_grid(df, ['pos_z'], t_grid)['pos_z'] \
                    if 'pos_z' in df.columns else np.nan

        # События из первого прогона
        raw_events = []
        for df in first_dfs:
            d_id = int(df['_drone_id'].iloc[0]) if '_drone_id' in df.columns else 0
            if d_id == did and 'event' in df.columns and 't' in df.columns:
                ev = df[['t', 'event']].dropna(subset=['event'])
                ev = ev[ev['event'].str.strip() != '']
                raw_events.extend(zip(ev['t'], ev['event']))

        result[did] = {
            'mean':    mean_df,
            'std':     std_df,
            'n_runs':  len(resampled),
            'events':  raw_events,
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Итоговые сводные метрики
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(averaged: dict) -> dict:
    """Возвращает плоский dict: mean_X, std_X для ключевых полей."""
    summaries = {}
    for did, d in averaged.items():
        df = d['mean']
        s = {}
        for c in ['formation_error_m', 'lane_error_m', 'nearest_neighbor_m',
                  'velocity_mps', 'coverage_pct']:
            if c in df.columns:
                s[f'{c}_mean'] = float(np.nanmean(df[c]))
                s[f'{c}_std']  = float(np.nanstd(df[c]))
        if 'path_length_m' in df.columns:
            s['path_length_final'] = float(df['path_length_m'].dropna().iloc[-1]) \
                if df['path_length_m'].dropna().shape[0] > 0 else np.nan
        if 'mission_time_s' in df.columns:
            s['mission_time_final'] = float(df['mission_time_s'].dropna().iloc[-1]) \
                if df['mission_time_s'].dropna().shape[0] > 0 else np.nan
        summaries[did] = s
    return summaries


# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════  ГРАФИКИ  ═══════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name + '.png')
    fig.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'    ✓ {name}.png')


def add_events(ax, events, t_col=None, ymin=None, ymax=None, alpha=0.6):
    """Рисует вертикальные линии для ключевых событий."""
    ev_styles = {
        'MISSION_START': ('#27ae60', '--', 'Старт'),
        'MISSION_END':   ('#e74c3c', '--', 'Конец'),
        'DRONE_LOST':    ('#e67e22', ':', 'Потеря дрона'),
        'RTL':           ('#8e44ad', ':', 'RTL'),
        'WAYPOINT':      ('#95a5a6', ':', None),
    }
    added = set()
    for t_val, ev in events:
        key = str(ev).strip()
        style = ev_styles.get(key)
        if not style:
            continue
        color, ls, label = style
        if label and label not in added:
            ax.axvline(t_val, color=color, linestyle=ls, linewidth=1.2,
                       alpha=alpha, label=label)
            added.add(label)
        else:
            ax.axvline(t_val, color=color, linestyle=ls, linewidth=1.2, alpha=alpha)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 1: Траектории XY
# ───────────────────────────────────────────────────────────────────────────
def plot_trajectories(averaged: dict, tag: str, out_dir: str, obstacles=None):
    n = len(averaged)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)

    for col_idx, (did, d) in enumerate(sorted(averaged.items())):
        ax = axes[0][col_idx]
        df = d['mean']
        color = DRONE_COLORS[did % len(DRONE_COLORS)]

        if '_pos_x' in df.columns and '_pos_y' in df.columns:
            x = df['_pos_x'].values
            y = df['_pos_y'].values

            # Цветовая окраска по ошибке формации
            if 'formation_error_m' in df.columns:
                err = df['formation_error_m'].values
                err_norm = (err - np.nanmin(err)) / (np.nanmax(err) - np.nanmin(err) + 1e-9)
                cmap = cm.get_cmap('RdYlGn_r')
                for i in range(len(x) - 1):
                    ax.plot(x[i:i+2], y[i:i+2], color=cmap(err_norm[i]),
                            linewidth=2.0, alpha=0.85, solid_capstyle='round')
                sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                    norm=Normalize(vmin=np.nanmin(err), vmax=np.nanmax(err)))
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label='Ошибка строя, м', shrink=0.7)
            else:
                ax.plot(x, y, color=color, linewidth=2.0, alpha=0.85)

            ax.scatter(x[0], y[0], c='#27ae60', s=100, marker='o',
                       zorder=5, label='Старт', edgecolors='white', linewidths=1)
            ax.scatter(x[-1], y[-1], c='#e74c3c', s=100, marker='s',
                       zorder=5, label='Финиш', edgecolors='white', linewidths=1)

        # Waypoints
        ax.set_xlim(-2, FIELD_LEN + 2)
        ax.set_ylim(-2, FIELD_WID + 2)
        ax.set_aspect('equal')
        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')
        ax.set_title(f'Дрон {did}  (n={d["n_runs"]} прогонов)')
        ax.legend(fontsize=8, loc='upper left')

        # Препятствия
        if obstacles:
            for ox, oy, orad in obstacles:
                circ = plt.Circle((ox, oy), orad, color='#2c3e50', alpha=0.45, zorder=4)
                ax.add_patch(circ)
                ax.text(ox, oy, f'r={orad:.0f}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')

        # Поле
        rect = mpatches.FancyBboxPatch((0, 0), FIELD_LEN, FIELD_WID,
                                       boxstyle='round,pad=0.5', linewidth=1.5,
                                       edgecolor='#2c3e50', facecolor='#ecf0f1', alpha=0.15, zorder=0)
        ax.add_patch(rect)

    fig.suptitle(f'Траектории дронов — {tag}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'01_trajectories_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 2: Ошибка формации по времени
# ───────────────────────────────────────────────────────────────────────────
def plot_formation_error(averaged: dict, tag: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for did, d in sorted(averaged.items()):
        df  = d['mean']
        dfs = d['std']
        if 'formation_error_m' not in df.columns:
            continue
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        t = df['t'].values
        m = df['formation_error_m'].values
        s = dfs['formation_error_m'].values if 'formation_error_m' in dfs.columns else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Ошибка формации, м')
    ax.set_title(f'Ошибка формации (среднее ± σ по {list(averaged.values())[0]["n_runs"]} прогонам) — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'02_formation_error_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 3: Ошибка полосы (lane_error)
# ───────────────────────────────────────────────────────────────────────────
def plot_lane_error(averaged: dict, tag: str, out_dir: str):
    has_lane = any('lane_error_m' in d['mean'].columns for d in averaged.values())
    if not has_lane:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for did, d in sorted(averaged.items()):
        df  = d['mean']
        dfs = d['std']
        if 'lane_error_m' not in df.columns:
            continue
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        t = df['t'].values
        m = df['lane_error_m'].values
        s = dfs['lane_error_m'].values if 'lane_error_m' in dfs.columns else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, np.maximum(0, m - s), m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Ошибка полосы, м')
    ax.set_title(f'Lane Error — отклонение от своей Y-полосы — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'03_lane_error_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 4: Скорость по времени
# ───────────────────────────────────────────────────────────────────────────
def plot_velocity(averaged: dict, tag: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for did, d in sorted(averaged.items()):
        df  = d['mean']
        dfs = d['std']
        if 'velocity_mps' not in df.columns:
            continue
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        t = df['t'].values
        m = df['velocity_mps'].values
        s = dfs['velocity_mps'].values if 'velocity_mps' in dfs.columns else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, np.maximum(0, m - s), m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.axhline(2.5, color='#e74c3c', linestyle='--', linewidth=1.0,
               alpha=0.7, label='V_MAX = 2.5 м/с')
    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Скорость, м/с')
    ax.set_title(f'Скорость полёта — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'04_velocity_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 5: Расстояние до ближайшего соседа
# ───────────────────────────────────────────────────────────────────────────
def plot_neighbor_distance(averaged: dict, tag: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for did, d in sorted(averaged.items()):
        df  = d['mean']
        dfs = d['std']
        if 'nearest_neighbor_m' not in df.columns:
            continue
        vals = df['nearest_neighbor_m'].replace(-1, np.nan)
        if vals.dropna().empty:
            continue
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        t = df['t'].values
        m = vals.values
        s = dfs['nearest_neighbor_m'].values if 'nearest_neighbor_m' in dfs.columns else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, np.maximum(0, m - s), m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.axhline(1.5, color='#e74c3c', linestyle='--', linewidth=1.0,
               alpha=0.7, label='D_SAFE = 1.5 м')
    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Расстояние до соседа, м')
    ax.set_title(f'Межбортная дистанция — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'05_neighbor_distance_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 6: Накопленная длина пути
# ───────────────────────────────────────────────────────────────────────────
def plot_path_length(averaged: dict, tag: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for did, d in sorted(averaged.items()):
        df = d['mean']
        if 'path_length_m' not in df.columns:
            continue
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        ax.plot(df['t'], df['path_length_m'], color=color, label=f'Дрон {did}')
        add_events(ax, d['events'])

    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Длина пути, м')
    ax.set_title(f'Накопленная длина пути — {tag}')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f'06_path_length_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 7: Покрытие поля
# ───────────────────────────────────────────────────────────────────────────
def plot_coverage(averaged: dict, tag: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for did, d in sorted(averaged.items()):
        df  = d['mean']
        dfs = d['std']
        if 'coverage_pct' not in df.columns:
            continue
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        t = df['t'].values
        m = df['coverage_pct'].values
        s = dfs['coverage_pct'].values if 'coverage_pct' in dfs.columns else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, np.maximum(0, m - s), np.minimum(100, m + s),
                        color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.axhline(100, color='#27ae60', linestyle='--', linewidth=1.0,
               alpha=0.7, label='100%')
    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Покрытие, %')
    ax.set_title(f'Покрытие поля по времени — {tag}')
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f'07_coverage_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 8: Компоненты сил APF
# ───────────────────────────────────────────────────────────────────────────
def plot_apf_forces(averaged: dict, tag: str, out_dir: str):
    force_cols = {
        'f_att': ('f_att_x', 'f_att_y', '#3498db', 'Аттракция'),
        'f_rep': ('f_rep_x', 'f_rep_y', '#e74c3c', 'Репульсия'),
        'f_tan': ('f_tan_x', 'f_tan_y', '#f39c12', 'Тангенциальная'),
    }
    has_forces = any(
        any(c in d['mean'].columns for c in ['f_att_x', 'f_rep_x', 'f_tan_x'])
        for d in averaged.values()
    )
    if not has_forces:
        return

    for did, d in sorted(averaged.items()):
        df = d['mean']
        fig, axes = plt.subplots(len(force_cols), 1, figsize=(11, 9), sharex=True)
        any_data = False
        for ax, (fkey, (cx, cy, color, label)) in zip(axes, force_cols.items()):
            if cx not in df.columns or cy not in df.columns:
                ax.set_visible(False)
                continue
            any_data = True
            mag = np.hypot(df[cx].values, df[cy].values)
            ax.plot(df['t'], mag, color=color, label=f'|{label}|', linewidth=1.4)
            ax.fill_between(df['t'], 0, mag, color=color, alpha=0.12)
            ax.set_ylabel('|F|, у.е.')
            ax.set_title(label, pad=4)
            ax.legend(loc='upper right', fontsize=9)
            add_events(ax, d['events'])

        if not any_data:
            plt.close(fig)
            continue
        axes[-1].set_xlabel('Время миссии, с')
        fig.suptitle(f'Компоненты сил APF — {tag} — Дрон {did}', fontsize=13, fontweight='bold')
        fig.tight_layout()
        save_fig(fig, f'08_apf_forces_{tag}_drone{did}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 9: Dashboard — всё в одном (6 панелей)
# ───────────────────────────────────────────────────────────────────────────
def plot_dashboard(averaged: dict, tag: str, out_dir: str, obstacles=None):
    """Сводный дашборд 3×2 с ключевыми метриками."""
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    panel_configs = [
        ('formation_error_m', 'Ошибка формации, м',        gs[0, 0]),
        ('lane_error_m',      'Lane error (Y-полоса), м',   gs[0, 1]),
        ('velocity_mps',      'Скорость, м/с',              gs[1, 0]),
        ('nearest_neighbor_m','Расстояние до соседа, м',    gs[1, 1]),
        ('coverage_pct',      'Покрытие поля, %',           gs[2, 0]),
        ('path_length_m',     'Длина пути, м',              gs[2, 1]),
    ]

    for metric, ylabel, spec in panel_configs:
        ax = fig.add_subplot(spec)
        has = False
        for did, d in sorted(averaged.items()):
            df  = d['mean']
            dfs = d['std']
            if metric not in df.columns:
                continue
            vals = df[metric].replace(-1, np.nan)
            if vals.dropna().empty:
                continue
            has = True
            color = DRONE_COLORS[did % len(DRONE_COLORS)]
            t = df['t'].values
            m = vals.values
            s = dfs[metric].values if metric in dfs.columns else np.zeros_like(m)
            ax.plot(t, m, color=color, label=f'Д{did}', linewidth=1.4)
            ax.fill_between(t, np.maximum(0, m - s), m + s, color=color, alpha=0.12)
            add_events(ax, d['events'], alpha=0.5)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('t, с', fontsize=9)
        ax.tick_params(labelsize=8)
        if metric == 'coverage_pct':
            ax.set_ylim(0, 108)
            ax.axhline(100, color='#27ae60', linestyle='--', linewidth=1, alpha=0.6)
        elif metric in ('formation_error_m', 'lane_error_m', 'nearest_neighbor_m',
                        'velocity_mps', 'path_length_m'):
            ax.set_ylim(bottom=0)
        if has:
            ax.legend(fontsize=8, ncol=3, loc='upper right')
        else:
            ax.set_visible(False)

    fig.suptitle(f'Дашборд метрик — {tag}', fontsize=15, fontweight='bold', y=1.01)
    save_fig(fig, f'09_dashboard_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 10: Тепловая карта отклонений на поле
# ───────────────────────────────────────────────────────────────────────────
def plot_deviation_heatmap(averaged: dict, tag: str, out_dir: str, obstacles=None):
    n = len(averaged)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)

    for col_idx, (did, d) in enumerate(sorted(averaged.items())):
        ax = axes[0][col_idx]
        df = d['mean']
        if '_pos_x' not in df.columns or 'formation_error_m' not in df.columns:
            ax.set_title(f'Нет данных — Дрон {did}')
            continue

        x   = df['_pos_x'].values
        y   = df['_pos_y'].values
        err = df['formation_error_m'].values

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
        x, y, err = x[mask], y[mask], err[mask]
        if len(x) < 10:
            continue

        # Сетка
        xi = np.linspace(0, FIELD_LEN, 120)
        yi = np.linspace(0, FIELD_WID, 60)
        from scipy.interpolate import griddata
        zi = griddata((x, y), err, (xi[None, :], yi[:, None]), method='linear')
        zi = gaussian_filter(np.nan_to_num(zi, nan=0.0), sigma=1.5)

        im = ax.imshow(zi, origin='lower', extent=[0, FIELD_LEN, 0, FIELD_WID],
                       cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=np.nanpercentile(err, 95))
        plt.colorbar(im, ax=ax, label='Ошибка строя, м', shrink=0.8)
        ax.plot(x, y, 'k-', linewidth=0.6, alpha=0.4, label='Путь')

        if obstacles:
            for ox, oy, orad in obstacles:
                circ = plt.Circle((ox, oy), orad, color='white', alpha=0.6, zorder=3)
                ax.add_patch(circ)

        ax.set_xlim(0, FIELD_LEN)
        ax.set_ylim(0, FIELD_WID)
        ax.set_xlabel('X, м')
        ax.set_ylabel('Y, м')
        ax.set_title(f'Тепловая карта — Дрон {did}')

    fig.suptitle(f'Тепловая карта ошибки формации — {tag}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'10_heatmap_{tag}', out_dir)


# ───────────────────────────────────────────────────────────────────────────
# PLOT 11: Сводная таблица метрик (текстовая)
# ───────────────────────────────────────────────────────────────────────────
def plot_summary_table(all_summaries: dict, out_dir: str):
    """
    all_summaries: {tag: {drone_id: summary_dict}}
    """
    rows = []
    cols = ['Сценарий', 'Дрон',
            'Ош.формации (ср.)', 'Lane error (ср.)',
            'Ближ.сосед (ср.)', 'Скорость (ср.)',
            'Покрытие (%)', 'Длина пути (м)', 'Время (с)']

    for tag, drone_sums in sorted(all_summaries.items()):
        for did, s in sorted(drone_sums.items()):
            rows.append([
                tag, str(did),
                f'{s.get("formation_error_m_mean", float("nan")):.3f}',
                f'{s.get("lane_error_m_mean",      float("nan")):.3f}',
                f'{s.get("nearest_neighbor_m_mean",float("nan")):.2f}',
                f'{s.get("velocity_mps_mean",       float("nan")):.2f}',
                f'{s.get("coverage_pct_mean",       float("nan")):.1f}',
                f'{s.get("path_length_final",       float("nan")):.1f}',
                f'{s.get("mission_time_final",      float("nan")):.1f}',
            ])

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(max(16, 2 * len(cols)), 0.5 + 0.4 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Стилизация заголовка
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Зебра
    for i in range(1, len(rows) + 1):
        fc = '#f9f9f9' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tbl[(i, j)].set_facecolor(fc)

    fig.suptitle('Сводная таблица метрик ВКР', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '11_summary_table', out_dir)

    # Также CSV
    df_out = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(out_dir, '11_summary_table.csv')
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'    ✓ 11_summary_table.csv')


# ───────────────────────────────────────────────────────────────────────────
# PLOT 12: Сравнение алгоритмов — Boxplot
# ───────────────────────────────────────────────────────────────────────────
def plot_algo_comparison_box(all_averaged: dict, out_dir: str):
    """
    all_averaged: {tag: averaged_dict}
    Строит boxplot по алгоритмам для ключевых метрик.
    """
    metrics = [
        ('formation_error_m', 'Ошибка формации, м'),
        ('velocity_mps',       'Скорость, м/с'),
        ('coverage_pct',       'Покрытие, %'),
    ]
    # Собираем данные по тегу (algo + условие)
    data_by_algo = {}
    for tag, averaged in all_averaged.items():
        algo = tag.split('/')[0] if '/' in tag else tag
        for did, d in averaged.items():
            df = d['mean']
            for metric, _ in metrics:
                if metric not in df.columns:
                    continue
                vals = df[metric].dropna().values
                data_by_algo.setdefault(algo, {}).setdefault(metric, []).extend(vals.tolist())

    if not data_by_algo:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    for ax, (metric, ylabel) in zip(axes, metrics):
        algos_present = [a for a in ALGOS if a in data_by_algo and metric in data_by_algo[a]]
        if not algos_present:
            ax.set_visible(False)
            continue
        box_data  = [data_by_algo[a][metric] for a in algos_present]
        labels    = [ALGO_LABELS.get(a, a) for a in algos_present]
        colors    = [ALGO_COLORS.get(a, '#95a5a6') for a in algos_present]

        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Сравнение алгоритмов — распределение метрик', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '12_algo_comparison_box', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Комплексный анализ результатов экспериментов ВКР')
    parser.add_argument('--base', default=str(Path.home() / 'vkr_results'),
                        help='Базовый каталог с результатами')
    parser.add_argument('--out',  default=None,
                        help='Папка для графиков (default: <base>/plots)')
    parser.add_argument('--algo', default=None, choices=ALGOS + [None],
                        help='Фильтр по алгоритму')
    parser.add_argument('--obs',  default=None, choices=OBSTACLES + [None],
                        help='Фильтр по препятствиям')
    parser.add_argument('--mode', default=None, choices=MODES + [None],
                        help='Фильтр по режиму')
    parser.add_argument('--obstacles_file', default=None,
                        help='Путь к obstacles.txt (x,y,r)')
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    out_dir  = Path(args.out).expanduser().resolve() if args.out \
               else base_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Анализ экспериментов ВКР')
    print(f'  Источник:   {base_dir}')
    print(f'  Результаты: {out_dir}')
    print('=' * 60)

    # Препятствия
    obstacles = []
    obs_path = args.obstacles_file or str(base_dir / 'obstacles.txt')
    if os.path.exists(obs_path):
        with open(obs_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    obstacles.append((float(parts[0]), float(parts[1]), float(parts[2])))
        print(f'  Препятствия: {len(obstacles)} из {obs_path}')

    # Сканирование
    print('\n[1] Сканирование каталога...')
    raw = scan_directory(base_dir)

    if not raw:
        print('[ERROR] CSV-файлы не найдены. Проверьте структуру каталога.')
        sys.exit(1)

    # Фильтрация
    if args.algo:
        raw = {k: v for k, v in raw.items() if k == args.algo}
    if args.obs:
        raw = {algo: {o: m for o, m in obs.items() if o == args.obs}
               for algo, obs in raw.items()}
    if args.mode:
        raw = {algo: {o: {m: r for m, r in modes.items() if m == args.mode}
                      for o, modes in obs.items()}
               for algo, obs in raw.items()}

    # Усреднение
    print('\n[2] Усреднение по прогонам...')
    all_averaged   = {}  # tag → averaged
    all_summaries  = {}  # tag → summaries

    for algo, obs_dict in sorted(raw.items()):
        for obs, mode_dict in sorted(obs_dict.items()):
            for mode, runs_dict in sorted(mode_dict.items()):
                tag = f'{algo}/{obs}/{mode}'
                label = (f'{ALGO_LABELS.get(algo, algo)} | '
                         f'{OBS_LABELS.get(obs, obs)} | '
                         f'{MODE_LABELS.get(mode, mode)}')
                safe_tag = tag.replace('/', '_')
                print(f'  → {tag}  ({len(runs_dict)} прогонов)')

                averaged = average_runs(runs_dict)
                if not averaged:
                    continue

                all_averaged[safe_tag] = averaged
                all_summaries[safe_tag] = compute_summary(averaged)

    if not all_averaged:
        print('[ERROR] Нет усреднённых данных.')
        sys.exit(1)

    # Генерация графиков
    print('\n[3] Генерация графиков...')
    for safe_tag, averaged in sorted(all_averaged.items()):
        nice = safe_tag.replace('_', ' / ')
        print(f'\n  Сценарий: {nice}')
        plot_trajectories(averaged,    safe_tag, str(out_dir), obstacles)
        plot_formation_error(averaged, safe_tag, str(out_dir))
        plot_lane_error(averaged,      safe_tag, str(out_dir))
        plot_velocity(averaged,        safe_tag, str(out_dir))
        plot_neighbor_distance(averaged, safe_tag, str(out_dir))
        plot_path_length(averaged,     safe_tag, str(out_dir))
        plot_coverage(averaged,        safe_tag, str(out_dir))
        plot_apf_forces(averaged,      safe_tag, str(out_dir))
        plot_dashboard(averaged,       safe_tag, str(out_dir), obstacles)
        plot_deviation_heatmap(averaged, safe_tag, str(out_dir), obstacles)

    print('\n  Сводные графики...')
    plot_summary_table(all_summaries, str(out_dir))
    plot_algo_comparison_box(all_averaged, str(out_dir))

    print(f'\n{"=" * 60}')
    print(f'  ✅  Готово! Графики сохранены в:')
    print(f'      {out_dir}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
