#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drone_analysis.py — Комплексный анализ и визуализация результатов экспериментов ВКР.

Объединяет analyze_experiments.py и plot_trajectories_v2.py.

Автоматически обходит структуру каталогов:
  <base_dir>/<algo>/<obstacles>/<mode>/run_N/metrics_*.csv

Усредняет метрики по 3 прогонам, строит все ключевые графики:
  1.  Траектории — все дроны на одном поле
  2.  Траектории — панели по дронам (цвет по ошибке формации)
  3.  Траектории — реальный vs идеальный маршрут
  4.  Траектории — 3D (X / Y / высота)
  5.  Ошибка формации по времени (с доверительным интервалом)
  6.  Ошибка полосы (lane_error) по времени
  7.  Скорость по времени
  8.  Расстояние до ближайшего соседа
  9.  Накопленная длина пути
  10. Покрытие поля (%) по времени
  11. Dashboard — сводные метрики (6 панелей)
  12. Тепловая карта отклонений на поле
  13. Сводная таблица метрик
  14. Boxplot сравнения алгоритмов

Запуск:
    python3 drone_analysis.py --base ~/vkr_results --out ~/vkr_results/plots
    python3 drone_analysis.py --base . --algo apf_lanes --obs heavy --mode lose_drone
"""

import argparse
import glob
import math
import os
import re
import sys
import warnings
from pathlib import Path
from itertools import product as iproduct

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# ──  КОНФИГУРАЦИЯ КООРДИНАТ  ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# True  → ось Y инвертируется при отрисовке (система +X / −Y, реальные данные)
# False → ось Y отображается как есть       (система +X / +Y)
INVERT_Y = True


def ay(y):
    """Применяет инверсию Y в зависимости от флага INVERT_Y."""
    arr = np.asarray(y, dtype=float)
    return -arr if INVERT_Y else arr


def y_label() -> str:
    """Подпись оси Y с учётом инверсии."""
    return '−Y, м' if INVERT_Y else 'Y, м'


# ══════════════════════════════════════════════════════════════════════════════
# ──  КОНФИГУРАЦИЯ АЛГОРИТМОВ И СЦЕНАРИЕВ  ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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
DRONE_COLORS  = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
DRONE_PALETTE = {0: '#2980b9', 1: '#e74c3c', 2: '#27ae60', 3: '#8e44ad', 4: '#f39c12'}
LOST_COLOR    = '#e74c3c'

OBS_LABELS  = {'no_obstacles': 'Без препятствий',
               'light': '3 препятствия',
               'heavy': '10 препятствий'}
MODE_LABELS = {'normal': 'Обычный режим',
               'lose_drone': 'Потеря дрона'}

FIELD_LEN  = 60.0
FIELD_WID  = 30.0
FLIGHT_ALT = 4.0
CAM_FOV    = math.radians(78.0)
CAM_FOV_DEG = 78.0
OVERLAP    = 0.2

# ── Стиль matplotlib ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'legend.fontsize': 9,
    'figure.dpi':     130,
    'lines.linewidth': 1.6,
    'axes.grid':      True,
    'grid.alpha':     0.25,
    'grid.linestyle': ':',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# ──  ЗАГРУЗКА ДАННЫХ  ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def detect_algo(filename: str) -> str:
    fn = os.path.basename(filename).lower()
    if 'lanes' in fn:   return 'apf_lanes'
    if 'apf'   in fn:   return 'apf'
    if 'boids' in fn:   return 'boids'
    if 'hardcode' in fn or 'hard_code' in fn: return 'hardcode'
    return 'unknown'


def detect_drone_id(filename: str) -> int:
    m = re.search(r'uav(\d+)', os.path.basename(filename))
    return int(m.group(1)) if m else 0


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty or len(df.columns) == 0:
        raise ValueError('Файл пустой (нет строк или столбцов данных)')
    if 'mission_time_s' in df.columns:
        df['t'] = df['mission_time_s']
    elif 'time_s' in df.columns:
        if df['time_s'].empty:
            raise ValueError('Столбец time_s пустой')
        df['t'] = df['time_s'] - df['time_s'].iloc[0]
    else:
        raise ValueError(f'Не найден столбец времени (mission_time_s / time_s). Столбцы: {list(df.columns)}')
    df = df[df['t'] >= 0].copy()
    if df.empty:
        raise ValueError('После фильтрации t>=0 не осталось строк')
    return df


def scan_directory(base: Path) -> dict:
    """
    Сканирует base/<algo>/<obs>/<mode>/run_N/metrics_*.csv.
    Возвращает {algo: {obs: {mode: {run_n: [df_drone0, ...]}}}}
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


def load_obstacles(path: str) -> list:
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


def resolve_obstacles_for_obs(obs_tag: str, base_dir: Path,
                               cli_path: str = None) -> list:
    """Возвращает список препятствий для конкретного тега (no_obstacles/light/heavy).
    Логика поиска (по приоритету):
      1) Если obs_tag == 'no_obstacles' — пустой список.
      2) В base_dir ищется файл obstacles_<tag>.txt (например, obstacles_light.txt).
      3) Если не найден — fallback на cli_path (--obstacles_file) или base_dir/obstacles.txt.
    """
    if obs_tag == 'no_obstacles':
        return []
    candidates = [
        base_dir / f'obstacles_{obs_tag}.txt',
        Path(f'obstacles_{obs_tag}.txt'),
    ]
    for p in candidates:
        if p.exists():
            print(f'  [obstacles] {obs_tag} → {p}')
            return load_obstacles(str(p))
    fallback = cli_path or str(base_dir / 'obstacles.txt')
    print(f'  [obstacles] {obs_tag} → fallback {fallback}')
    return load_obstacles(fallback)


# ══════════════════════════════════════════════════════════════════════════════
# ──  УСРЕДНЕНИЕ ПО ПРОГОНАМ  ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

NUMERIC_METRICS = [
    'formation_error_m', 'lane_error_m', 'nearest_neighbor_m',
    'velocity_mps', 'path_length_m', 'mission_time_s',
    'coverage_pct',
]


def resample_to_grid(df: pd.DataFrame, cols: list, t_grid: np.ndarray) -> pd.DataFrame:
    out = {'t': t_grid}
    t = df['t'].values
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


def _get_drone_id(df: pd.DataFrame) -> int:
    """Возвращает drone_id из df, или 0 если столбец отсутствует/df пустой."""
    if '_drone_id' in df.columns and len(df) > 0:
        return int(df['_drone_id'].iloc[0])
    return 0


def average_runs(runs_dict: dict) -> dict:
    """
    Принимает {run_n: [df_drone0, df_drone1, ...]}
    Возвращает {drone_id: {'mean': df, 'std': df, 'n_runs': int, 'events': list}}
    Пустые DataFrame пропускаются автоматически.
    """
    all_drone_ids = set()
    for dfs in runs_dict.values():
        for df in dfs:
            if df.empty:
                continue
            all_drone_ids.add(_get_drone_id(df))

    result = {}
    for did in sorted(all_drone_ids):
        t_min, t_max, t_len = [], [], []
        for run_n, dfs in runs_dict.items():
            for df in dfs:
                if df.empty:
                    continue
                if _get_drone_id(df) == did:
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

        cols = [c for c in NUMERIC_METRICS if any(
            c in df.columns
            for dfs in runs_dict.values()
            for df in dfs
            if not df.empty and _get_drone_id(df) == did
        )]

        resampled = []
        for run_n, dfs in sorted(runs_dict.items()):
            for df in dfs:
                if df.empty:
                    continue
                if _get_drone_id(df) == did:
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

        # Позиции из первого прогона
        first_dfs = list(runs_dict.values())[0]
        for df in first_dfs:
            if df.empty:
                continue
            if _get_drone_id(df) == did:
                for col_name in ('pos_x', 'pos_y', 'pos_z', 'target_x', 'target_y', 'event'):
                    if col_name in df.columns:
                        mean_df[f'_{col_name}'] = \
                            resample_to_grid(df, [col_name], t_grid)[col_name] \
                            if col_name != 'event' else np.nan

        # Сырые позиции и события из первого прогона (без интерполяции)
        raw_events = []
        raw_pos    = None
        for df in first_dfs:
            if df.empty:
                continue
            if _get_drone_id(df) == did:
                if 'event' in df.columns:
                    ev = df[['t', 'event']].dropna(subset=['event'])
                    ev = ev[ev['event'].str.strip() != '']
                    raw_events.extend(zip(ev['t'], ev['event']))
                raw_pos = df  # сохраняем для draw_drone_on_ax

        result[did] = {
            'mean':    mean_df,
            'std':     std_df,
            'n_runs':  len(resampled),
            'events':  raw_events,
            'raw_df':  raw_pos,   # исходный df первого прогона
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ──  СВОДНЫЕ МЕТРИКИ  ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(averaged: dict) -> dict:
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
# ──  УТИЛИТЫ СОХРАНЕНИЯ  ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name + '.png')
    fig.savefig(path, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f'    ✓ {name}.png')


def _save_fig_named(fig, name: str, save: bool, out_dir: str):
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name + '.png')
        fig.savefig(path, bbox_inches='tight', dpi=130)
        print(f'    ✓ {name}.png')
    else:
        plt.show()
    plt.close(fig)


def add_events(ax, events, alpha=0.6):
    ev_styles = {
        'MISSION_START': ('#27ae60', '--', 'Старт'),
        'MISSION_END':   ('#e74c3c', '--', 'Конец'),
        'DRONE_LOST':    ('#e67e22', ':', 'Потеря дрона'),
        'RTL':           ('#8e44ad', ':', 'Возвращение'),
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


# ══════════════════════════════════════════════════════════════════════════════
# ──  ГЕОМЕТРИЧЕСКИЕ УТИЛИТЫ  ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def footprint_width(alt: float) -> float:
    return 2.0 * alt * math.tan(math.radians(CAM_FOV_DEG) / 2.0)


def get_field_bounds(drones: list) -> tuple:
    """Вычисляет границы поля по траекториям с учётом INVERT_Y."""
    all_x, all_y = [], []
    for d in drones:
        df = d['df']
        if 'pos_x' in df.columns:
            all_x.extend(df['pos_x'].dropna().tolist())
        if 'pos_y' in df.columns:
            y_vals = ay(df['pos_y'].dropna().values)
            all_y.extend(y_vals.tolist())
    if not all_x:
        return (-2, FIELD_LEN + 2, -2, FIELD_WID + 2)
    pad = 3.0
    return (min(all_x) - pad, max(all_x) + pad,
            min(all_y) - pad, max(all_y) + pad)


def extract_ideal_route(df: pd.DataFrame) -> np.ndarray:
    """Восстанавливает идеальный маршрут из target_x/target_y."""
    cols = [c for c in ['target_x', 'target_y'] if c in df.columns]
    if len(cols) < 2:
        return np.empty((0, 2))
    targets = df[cols].copy()
    mask = ~((targets['target_x'] == 0) & (targets['target_y'] == 0))
    targets = targets[mask]
    shifted = targets.shift(1)
    changed = (targets != shifted).any(axis=1)
    route = targets[changed].drop_duplicates().values
    return route


def colorline(ax, x, y, z, cmap='RdYlGn_r', linewidth=2.2, alpha=0.9,
              vmin=None, vmax=None, zorder=4):
    """Рисует линию с цветовой картой по z."""
    z = np.asarray(z, dtype=float)
    vmin = vmin if vmin is not None else np.nanmin(z)
    vmax = vmax if vmax is not None else np.nanmax(z)
    if vmax == vmin:
        vmax = vmin + 1e-6
    points   = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                        linewidth=linewidth, alpha=alpha, zorder=zorder,
                        capstyle='round', joinstyle='round')
    lc.set_array(z[:-1])
    ax.add_collection(lc)
    return lc, vmin, vmax


# ══════════════════════════════════════════════════════════════════════════════
# ──  КОНВЕРТЕР averaged → список дронов  ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def averaged_to_drones(averaged: dict) -> list:
    """
    Конвертирует averaged dict (формат analyze_experiments) в список дронов
    для функций визуализации v2.
    """
    drones = []
    for did, d in sorted(averaged.items()):
        # Предпочитаем raw_df (из первого прогона) для реальных позиций,
        # fallback — на mean df с переименованными _pos_x → pos_x
        raw = d.get('raw_df')
        if raw is not None and 'pos_x' in raw.columns:
            df = raw.copy()
            if 'mission_time_s' in df.columns:
                df['t'] = df['mission_time_s']
            else:
                df['t'] = df['time_s'] - df['time_s'].iloc[0]
        else:
            df = d['mean'].copy()
            for src, dst in [('_pos_x', 'pos_x'), ('_pos_y', 'pos_y'),
                              ('_pos_z', 'pos_z'), ('_target_x', 'target_x'),
                              ('_target_y', 'target_y')]:
                if src in df.columns:
                    df[dst] = df[src]

        # Объединяем метрики из mean в df (для colorline по ошибке)
        mean = d['mean']
        for col in ['formation_error_m']:
            if col in mean.columns and col not in df.columns:
                # Интерполируем на временну́ю сетку df
                t_mean = mean['t'].values
                v_mean = mean[col].values
                _, idx = np.unique(t_mean, return_index=True)
                fi = interp1d(t_mean[idx], v_mean[idx],
                              bounds_error=False, fill_value='extrapolate')
                df[col] = fi(df['t'].values)

        events_df = df['event'].fillna('').str.strip() \
            if 'event' in df.columns else pd.Series([''] * len(df))
        completed = events_df.isin(['MISSION_END']).any()
        t_total   = float(df['t'].max())

        drones.append({
            'id':        did,
            'df':        df,
            'filepath':  '',
            'completed': completed,
            't_total':   t_total,
            'n_rows':    len(df),
            'lost':      False,
            'n_runs':    d.get('n_runs', 1),
        })

    # Помечаем потерянных: время < 60% от максимального и не завершён
    if drones:
        t_max = max(dd['t_total'] for dd in drones)
        for dd in drones:
            if not dd['completed'] and dd['t_total'] < 0.6 * t_max:
                dd['lost'] = True

    return drones


# ══════════════════════════════════════════════════════════════════════════════
# ──  ОТРИСОВКА ЭЛЕМЕНТОВ  ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def draw_obstacles(ax, obstacles: list, safe_margin: float = 1.5):
    """Рисует препятствия на оси.
    safe_margin — дополнительный радиус зоны безопасности (как в логике APF).
    Внешний круг (зона безопасности) — пунктирный, внутренний (само препятствие) — заливка.
    """
    for ox, oy, orad in obstacles:
        oy_plot = float(ay(np.array([oy]))[0])
        if safe_margin > 0:
            safe_circ = plt.Circle((ox, oy_plot), orad + safe_margin,
                                    facecolor='none', edgecolor='#2c3e50',
                                    linestyle='--', linewidth=1.0, alpha=0.45,
                                    zorder=6)
            ax.add_patch(safe_circ)
        circ = plt.Circle((ox, oy_plot), orad,
                           color='#2c3e50', alpha=0.55, zorder=7)
        ax.add_patch(circ)
        ax.text(ox, oy_plot, f'r={orad:.1f}',
                ha='center', va='center', fontsize=14,
                color='white', fontweight='bold', zorder=8)


def style_ax(ax, title: str, xmin: float, xmax: float,
             ymin: float, ymax: float, show_legend: bool = True):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X, м')
    ax.set_ylabel(y_label())
    ax.set_title(title, pad=8)
    if show_legend:
        ax.legend(loc='upper right', fontsize=8,
                  framealpha=0.85, edgecolor='#cccccc',
                  ncol=2 if ax.get_figure().get_size_inches()[0] > 10 else 1)


def draw_drone_on_ax(ax, drone: dict, show_ideal=True,
                     show_waypoints=True, show_footprint=True,
                     color_by_error=True):
    """
    Рисует на ax траекторию одного дрона.
    Возвращает ScalarMappable для colorbar (или None).
    """
    df   = drone['df']
    did  = drone['id']
    lost = drone.get('lost', False)
    color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')

    x = df['pos_x'].values
    y = ay(df['pos_y'].values)

    # ── Footprint ─────────────────────────────────────────────────────────
    if show_footprint and 'pos_z' in df.columns and len(x) > 1:
        alt = df['pos_z'].mean()
        fw  = footprint_width(alt)
        x_min, x_max = min(x), max(x)
        y_c = float(np.median(y))
        rect = mpatches.FancyBboxPatch(
            (x_min, y_c - fw / 2), x_max - x_min, fw,
            boxstyle='round,pad=0.5',
            linewidth=0, facecolor=color, alpha=0.06, zorder=1)
        ax.add_patch(rect)

    # ── Идеальный маршрут ─────────────────────────────────────────────────
    sm = None
    if show_ideal:
        route = extract_ideal_route(df)
        if len(route) > 1:
            route_y = ay(route[:, 1])
            ax.plot(route[:, 0], route_y,
                    linestyle='--', color='#7f8c8d', linewidth=1.4,
                    alpha=0.6, zorder=2, label='Идеальный маршрут')
            ax.scatter(route[:, 0], route_y,
                       c='#7f8c8d', s=18, marker='x', alpha=0.5, zorder=3)

    # ── Реальная траектория ───────────────────────────────────────────────
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

    # ── Waypoints ─────────────────────────────────────────────────────────
    if show_waypoints and 'event' in df.columns:
        ev        = df['event'].fillna('').str.strip()
        wp_mask   = ev == 'WAYPOINT'
        photo_mask = ev == 'PHOTO_SHOT'
        if wp_mask.any():
            ax.scatter(x[wp_mask], y[wp_mask],
                       c='#f39c12', s=55, marker='D', zorder=6,
                       edgecolors='white', linewidths=0.7, label='Waypoint')
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
# ──  ТРАЕКТОРИИ (v2)  ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_together(drones: list, obstacles: list, out_dir: str, tag: str):
    """Один большой XY-план с траекториями всех дронов."""
    xmin, xmax, ymin, ymax = get_field_bounds(drones)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor('#f5f6fa')

    # Горизонтальные линии Y-полос
    y_lanes = sorted({round(float(ay(np.array([d['df']['pos_y'].median()]))[0]), 1)
                      for d in drones if 'pos_y' in d['df'].columns})
    for yl in y_lanes:
        ax.axhline(yl, color='#dfe6e9', linewidth=0.8, zorder=0)

    for drone in sorted(drones, key=lambda d: d['id']):
        did   = drone['id']
        lost  = drone.get('lost', False)
        color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')
        df    = drone['df']
        x     = df['pos_x'].values
        y     = ay(df['pos_y'].values)

        # Footprint
        if 'pos_z' in df.columns and len(x) > 1:
            alt = df['pos_z'].mean()
            fw  = footprint_width(alt)
            ax.fill_between(x, y - fw / 2, y + fw / 2,
                            alpha=0.07, color=color, zorder=1)

        # Идеальный маршрут
        route = extract_ideal_route(df)
        if len(route) > 1:
            ax.plot(route[:, 0], ay(route[:, 1]),
                    linestyle='--', color=color, linewidth=1.2,
                    alpha=0.45, zorder=2)

        # Траектория
        lw   = 2.0 if not lost else 1.6
        lsty = '-' if not lost else '-.'
        label = f'Дрон {did}' + (f' (потерян t={drone["t_total"]:.0f}s)' if lost else '')
        ax.plot(x, y, color=color, linewidth=lw, linestyle=lsty,
                alpha=0.85, zorder=4, label=label)

        # Photo shots
        if 'event' in df.columns:
            ev = df['event'].fillna('').str.strip()
            pm = ev == 'PHOTO_SHOT'
            if pm.any():
                ax.scatter(x[pm], y[pm], c=color, s=20, marker='s', zorder=5, alpha=0.5)

        # Старт
        ax.scatter(x[0], y[0], c=color, s=130, marker='o', zorder=8,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(f'  Д{did}', (x[0], y[0]),
                    fontsize=8, color=color, fontweight='bold',
                    xytext=(4, 4), textcoords='offset points',
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        # Финиш / потеря
        ax.scatter(x[-1], y[-1], c=color, s=150,
                   marker='X' if lost else 's', zorder=8,
                   edgecolors='white', linewidths=1.5)
        if lost:
            ax.annotate('  ⚠', (x[-1], y[-1]),
                        fontsize=11, color=LOST_COLOR,
                        xytext=(3, 3), textcoords='offset points',
                        path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    draw_obstacles(ax, obstacles)
    style_ax(ax, f'Траектории всех дронов — {tag}', xmin, xmax, ymin, ymax, show_legend=False)
    fig.tight_layout()
    _save_fig_named(fig, f'01a_traj_all_{tag}', True, out_dir)


def plot_per_drone(drones: list, obstacles: list, out_dir: str, tag: str):
    """Отдельная панель для каждого дрона (цвет линии по ошибке формации)."""
    n = len(drones)
    if n == 0:
        return
    xmin, xmax, ymin, ymax = get_field_bounds(drones)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6.5), squeeze=False)

    for col, drone in enumerate(sorted(drones, key=lambda d: d['id'])):
        ax   = axes[0][col]
        did  = drone['id']
        lost = drone.get('lost', False)
        ax.set_facecolor('#f5f6fa')

        sm = draw_drone_on_ax(ax, drone, show_ideal=True,
                              show_waypoints=True, show_footprint=True,
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

    fig.suptitle(f'Траектории по дронам (цвет = ошибка формации) — {tag}',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_fig_named(fig, f'01b_traj_per_drone_{tag}', True, out_dir)


def plot_real_vs_ideal(drones: list, obstacles: list, out_dir: str, tag: str):
    """Реальный путь vs идеальный маршрут для каждого дрона."""
    n = len(drones)
    if n == 0:
        return
    xmin, xmax, ymin, ymax = get_field_bounds(drones)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6), squeeze=False)

    for col, drone in enumerate(sorted(drones, key=lambda d: d['id'])):
        ax   = axes[0][col]
        did  = drone['id']
        lost = drone.get('lost', False)
        color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')
        df   = drone['df']
        ax.set_facecolor('#f5f6fa')

        x_real = df['pos_x'].values
        y_real = ay(df['pos_y'].values)

        route = extract_ideal_route(df)
        if len(route) > 1:
            ax.plot(route[:, 0], ay(route[:, 1]),
                    '--', color='#2c3e50', linewidth=1.8,
                    alpha=0.65, zorder=3, label='Идеальный маршрут')
            ax.scatter(route[:, 0], ay(route[:, 1]),
                       c='#2c3e50', s=30, marker='o', alpha=0.45, zorder=4)

        ax.plot(x_real, y_real,
                color=color, linewidth=2.2, alpha=0.88,
                zorder=5, label='Реальный путь')

        if 'event' in df.columns:
            ev = df['event'].fillna('').str.strip()
            pm = ev == 'PHOTO_SHOT'
            if pm.any():
                ax.scatter(x_real[pm], y_real[pm],
                           c='#9b59b6', s=35, marker='s',
                           zorder=6, alpha=0.7, label='Фото')

        ax.scatter(x_real[0], y_real[0],
                   c='#27ae60', s=130, marker='o', zorder=9,
                   edgecolors='white', linewidths=1.2, label='Старт')
        end_label = f'Потеря t={drone["t_total"]:.0f}s' if lost else 'Финиш'
        ax.scatter(x_real[-1], y_real[-1],
                   c=LOST_COLOR if lost else '#c0392b',
                   s=150, marker='X' if lost else 's', zorder=9,
                   edgecolors='white', linewidths=1.2, label=end_label)

        draw_obstacles(ax, obstacles)
        title = f'Дрон {did}  —  реальный vs идеальный'
        if lost:
            title += '  ⚠ ПОТЕРЯН'
        style_ax(ax, title, xmin, xmax, ymin, ymax)

    fig.suptitle(f'Реальный маршрут vs идеальный — {tag}',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    _save_fig_named(fig, f'01c_traj_real_vs_ideal_{tag}', True, out_dir)


def plot_3d_trajectories(drones: list, obstacles: list, out_dir: str, tag: str):
    """3-мерные траектории X/Y/Z для всех дронов."""
    n = len(drones)
    if n == 0:
        return
    xmin, xmax, ymin, ymax = get_field_bounds(drones)
    fig = plt.figure(figsize=(6.5 * n, 6))

    for col, drone in enumerate(sorted(drones, key=lambda d: d['id'])):
        ax = fig.add_subplot(1, n, col + 1, projection='3d')
        did  = drone['id']
        lost = drone.get('lost', False)
        color = LOST_COLOR if lost else DRONE_PALETTE.get(did, '#555555')
        df   = drone['df']

        if 'pos_z' not in df.columns:
            continue
        x = df['pos_x'].values
        y = ay(df['pos_y'].values)
        z = df['pos_z'].values

        step = max(1, len(x) // 600)
        ax.plot(x[::step], y[::step], z[::step],
                color=color, linewidth=1.6, alpha=0.85)
        ax.plot(x[::step], y[::step],
                np.full(len(x[::step]), z.min() - 0.2),
                color=color, linewidth=0.7, alpha=0.25)
        ax.scatter(x[0], y[0], z[0], c='#27ae60', s=60, marker='o', zorder=5)
        ax.scatter(x[-1], y[-1], z[-1],
                   c=LOST_COLOR if lost else '#c0392b',
                   s=80, marker='X' if lost else '^', zorder=5)

        ax.set_xlabel('X, м', labelpad=6)
        ax.set_ylabel(y_label(), labelpad=6)
        ax.set_zlabel('Z, м', labelpad=4)
        ax.set_title(f'Дрон {did}' + (' ⚠' if lost else ''), pad=6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelsize=8)

    fig.suptitle(f'3D-траектории (X / Y / высота) — {tag}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save_fig_named(fig, f'01d_traj_3d_{tag}', True, out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ──  ВРЕМЕННЫ́Е РЯДЫ МЕТРИК  ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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
        s = dfs['formation_error_m'].values if 'formation_error_m' in dfs.columns \
            else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    n_runs = list(averaged.values())[0]['n_runs'] if averaged else 1
    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Ошибка строя, м')
    ax.set_title(f'Ошибка строя — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'02_formation_error_{tag}', out_dir)


def plot_lane_error(averaged: dict, tag: str, out_dir: str):
    if not any('lane_error_m' in d['mean'].columns for d in averaged.values()):
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
        s = dfs['lane_error_m'].values if 'lane_error_m' in dfs.columns \
            else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, np.maximum(0, m - s), m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Ошибка полосы, м')
    ax.set_title(f'Отклонение от своей полосы — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'03_lane_error_{tag}', out_dir)


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
        s = dfs['velocity_mps'].values if 'velocity_mps' in dfs.columns \
            else np.zeros_like(m)
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
        s = dfs['nearest_neighbor_m'].values if 'nearest_neighbor_m' in dfs.columns \
            else np.zeros_like(m)
        ax.plot(t, m, color=color, label=f'Дрон {did}')
        ax.fill_between(t, np.maximum(0, m - s), m + s, color=color, alpha=0.15)
        add_events(ax, d['events'])

    ax.axhline(1.5, color='#e74c3c', linestyle='--', linewidth=1.0,
               alpha=0.7, label='D_SAFE = 1.5 м')
    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Расстояние до соседа, м')
    ax.set_title(f'Расстояние до соседа — {tag}')
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, f'05_neighbor_distance_{tag}', out_dir)


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
        s = dfs['coverage_pct'].values if 'coverage_pct' in dfs.columns \
            else np.zeros_like(m)
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


# ══════════════════════════════════════════════════════════════════════════════
# ──  DASHBOARD  ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(averaged: dict, tag: str, out_dir: str):
    """Сводный дашборд 3×2 с ключевыми метриками."""
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    panel_configs = [
        ('formation_error_m', 'Ошибка строя, м',      gs[0, 0]),
        ('lane_error_m',      'Отклонение от полосы, м', gs[0, 1]),
        ('velocity_mps',      'Скорость, м/с',            gs[1, 0]),
        ('nearest_neighbor_m','Расстояние до соседа, м',  gs[1, 1]),
        ('coverage_pct',      'Покрытие поля, %',         gs[2, 0]),
        ('path_length_m',     'Длина пути, м',            gs[2, 1]),
    ]

    for metric, ylabel, spec in panel_configs:
        ax  = fig.add_subplot(spec)
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
    save_fig(fig, f'08_dashboard_{tag}', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ──  ТЕПЛОВАЯ КАРТА ОТКЛОНЕНИЙ  ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def plot_deviation_heatmap(averaged: dict, tag: str, out_dir: str, obstacles=None):
    n = len(averaged)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)

    for col_idx, (did, d) in enumerate(sorted(averaged.items())):
        ax  = axes[0][col_idx]
        df  = d['mean']
        if '_pos_x' not in df.columns or 'formation_error_m' not in df.columns:
            ax.set_title(f'Нет данных — Дрон {did}')
            continue

        x   = df['_pos_x'].values
        y   = ay(df['_pos_y'].values) if '_pos_y' in df.columns else df['_pos_x'].values * 0
        err = df['formation_error_m'].values

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
        x, y, err = x[mask], y[mask], err[mask]
        if len(x) < 10:
            continue

        xi = np.linspace(x.min(), x.max(), 120)
        yi = np.linspace(y.min(), y.max(), 60)
        from scipy.interpolate import griddata
        zi = griddata((x, y), err, (xi[None, :], yi[:, None]), method='linear')
        zi = gaussian_filter(np.nan_to_num(zi, nan=0.0), sigma=1.5)

        im = ax.imshow(zi, origin='lower',
                       extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                       cmap='RdYlGn_r', aspect='auto',
                       vmin=0, vmax=np.nanpercentile(err, 95))
        plt.colorbar(im, ax=ax, label='Ошибка строя, м', shrink=0.8)
        ax.plot(x, y, 'k-', linewidth=0.6, alpha=0.4, label='Путь')

        if obstacles:
            for ox, oy, orad in obstacles:
                oy_plot = float(ay(np.array([oy]))[0])
                safe_circ = plt.Circle((ox, oy_plot), orad + 1.5,
                                        facecolor='none', edgecolor='white',
                                        linestyle='--', linewidth=0.9, alpha=0.5,
                                        zorder=3)
                ax.add_patch(safe_circ)
                circ = plt.Circle((ox, oy_plot), orad, color='white', alpha=0.6, zorder=3)
                ax.add_patch(circ)

        ax.set_xlabel('X, м')
        ax.set_ylabel(y_label())
        ax.set_title(f'Тепловая карта — Дрон {did}')

    fig.suptitle(f'Тепловая карта ошибки формации — {tag}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, f'09_heatmap_{tag}', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ──  СВОДНАЯ ТАБЛИЦА  ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_table(all_summaries: dict, out_dir: str):
    rows = []
    cols = ['Сценарий', 'Дрон',
            'Ош.строя (ср.)', 'Отклонение от полосы (ср.)',
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

    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor('#2c3e50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        fc = '#f9f9f9' if i % 2 == 0 else 'white'
        for j in range(len(cols)):
            tbl[(i, j)].set_facecolor(fc)

    fig.suptitle('Сводная таблица метрик ВКР', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '10_summary_table', out_dir)

    df_out = pd.DataFrame(rows, columns=cols)
    csv_path = os.path.join(out_dir, '10_summary_table.csv')
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print('    ✓ 10_summary_table.csv')


# ══════════════════════════════════════════════════════════════════════════════
# ──  BOXPLOT СРАВНЕНИЯ АЛГОРИТМОВ  ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def plot_algo_comparison_box(all_averaged: dict, out_dir: str):
    metrics = [
        ('formation_error_m', 'Ошибка формации, м'),
        ('velocity_mps',       'Скорость, м/с'),
        ('coverage_pct',       'Покрытие, %'),
    ]
    data_by_algo = {}
    for tag, averaged in all_averaged.items():
        algo = tag.split('_')[0] if '_' in tag else tag
        # Попытка восстановить algo из тега формата algo_obs_mode
        for a in ALGOS:
            if tag.startswith(a):
                algo = a
                break
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
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, ylabel) in zip(axes, metrics):
        algos_present = [a for a in ALGOS if a in data_by_algo and metric in data_by_algo[a]]
        if not algos_present:
            ax.set_visible(False)
            continue
        box_data = [data_by_algo[a][metric] for a in algos_present]
        labels   = [ALGO_LABELS.get(a, a) for a in algos_present]
        colors   = [ALGO_COLORS.get(a, '#95a5a6') for a in algos_present]

        bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Сравнение алгоритмов — распределение метрик',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '11_algo_comparison_box', out_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ──  MAIN  ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Комплексный анализ и визуализация экспериментов ВКР')
    parser.add_argument('--base', default=str(Path.home() / 'vkr_results'),
                        help='Базовый каталог с результатами')
    parser.add_argument('--out',  default=None,
                        help='Папка для графиков (default: <base>/plots)')
    parser.add_argument('--algo', default=None, choices=ALGOS,
                        help='Фильтр по алгоритму')
    parser.add_argument('--obs',  default=None, choices=OBSTACLES,
                        help='Фильтр по препятствиям')
    parser.add_argument('--mode', default=None, choices=MODES,
                        help='Фильтр по режиму')
    parser.add_argument('--obstacles_file', default=None,
                        help='Путь к obstacles.txt (x,y,r)')
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    out_dir  = Path(args.out).expanduser().resolve() if args.out \
               else base_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    coord_info = 'INVERT_Y=True  (+X / −Y)' if INVERT_Y else 'INVERT_Y=False (+X / +Y)'
    print('=' * 65)
    print('  Анализ экспериментов ВКР')
    print(f'  Источник:     {base_dir}')
    print(f'  Результаты:   {out_dir}')
    print(f'  Координаты:   {coord_info}')
    print('=' * 65)

    # Препятствия: резолвятся для каждой обстановки (no_obstacles/light/heavy) отдельно.
    # CLI-аргумент --obstacles_file работает как fallback для случаев, когда нет
    # специфичного файла obstacles_<tag>.txt в base_dir.
    obs_cli_path = args.obstacles_file

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
    all_averaged  = {}
    all_summaries = {}
    obs_by_tag    = {}

    for algo, obs_dict in sorted(raw.items()):
        for obs, mode_dict in sorted(obs_dict.items()):
            for mode, runs_dict in sorted(mode_dict.items()):
                tag      = f'{algo}/{obs}/{mode}'
                safe_tag = tag.replace('/', '_')
                print(f'  → {tag}  ({len(runs_dict)} прогонов)')

                averaged = average_runs(runs_dict)
                if not averaged:
                    continue

                all_averaged[safe_tag]  = averaged
                all_summaries[safe_tag] = compute_summary(averaged)
                obs_by_tag[safe_tag]    = obs

    if not all_averaged:
        print('[ERROR] Нет усреднённых данных.')
        sys.exit(1)

    # Кэш загруженных препятствий по обстановке (чтобы не читать файл многократно)
    obstacles_cache = {}
    def get_obstacles_for(tag: str) -> list:
        obs_tag = obs_by_tag.get(tag, '')
        if obs_tag not in obstacles_cache:
            obstacles_cache[obs_tag] = resolve_obstacles_for_obs(
                obs_tag, base_dir, obs_cli_path)
        return obstacles_cache[obs_tag]

    # Генерация графиков
    print('\n[3] Генерация графиков...')
    for safe_tag, averaged in sorted(all_averaged.items()):
        nice = safe_tag.replace('_', ' / ')
        print(f'\n  Сценарий: {nice}')

        obstacles = get_obstacles_for(safe_tag)

        # ── Траектории (v2) ───────────────────────────────────────────────
        drones_list = averaged_to_drones(averaged)
        if drones_list and 'pos_x' in drones_list[0]['df'].columns:
            plot_all_together(drones_list, obstacles, str(out_dir), safe_tag)
            plot_per_drone(drones_list, obstacles, str(out_dir), safe_tag)
            plot_real_vs_ideal(drones_list, obstacles, str(out_dir), safe_tag)
            plot_3d_trajectories(drones_list, obstacles, str(out_dir), safe_tag)

        # ── Временны́е ряды ────────────────────────────────────────────────
        plot_formation_error(averaged,   safe_tag, str(out_dir))
        plot_lane_error(averaged,        safe_tag, str(out_dir))
        plot_velocity(averaged,          safe_tag, str(out_dir))
        plot_neighbor_distance(averaged, safe_tag, str(out_dir))
        plot_path_length(averaged,       safe_tag, str(out_dir))
        plot_coverage(averaged,          safe_tag, str(out_dir))

        # ── Dashboard + тепловая карта ────────────────────────────────────
        plot_dashboard(averaged,         safe_tag, str(out_dir))
        plot_deviation_heatmap(averaged, safe_tag, str(out_dir), obstacles)

    print('\n  Сводные графики...')
    plot_summary_table(all_summaries, str(out_dir))
    plot_algo_comparison_box(all_averaged, str(out_dir))

    print(f'\n{"=" * 65}')
    print(f'  ✅  Готово! Графики сохранены в:')
    print(f'      {out_dir}')
    print(f'{"=" * 65}')


if __name__ == '__main__':
    main()
