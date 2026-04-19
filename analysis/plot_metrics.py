#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_metrics.py — построение всех графиков ВКР из CSV-логов.

Запуск:
    python3 plot_metrics.py --dir .          # искать CSV в текущей папке
    python3 plot_metrics.py --dir ./logs     # искать в папке logs
    python3 plot_metrics.py --dir . --save   # сохранять PNG вместо показа

Что строит:
    1. Графики сходимости (formation_error vs time) — по каждому алгоритму и каждому дрону
    2. Сравнительный график сходимости всех алгоритмов
    3. Расстояние между дронами (безопасность строя) vs time
    4. Скорость vs time
    5. Путь vs time (накопленный)
    6. Отклонение от полосы (lane_error) — только для APF+lanes
    7. Компоненты сил APF / Boids vs time
    8. Сводная таблица итоговых метрик (mission_time, path_length, coverage_pct)
    9. Численная верификация покрытия поля (аналитика vs числ. сетка)
"""

import argparse
import glob
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')  # работает без дисплея — на сервере/в SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# Настройки стиля
# ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'legend.fontsize':   10,
    'figure.dpi':        120,
    'lines.linewidth':   1.6,
})

ALGO_COLORS = {
    'hardcode':  '#e74c3c',
    'apf':       '#3498db',
    'apf_lanes': '#2ecc71',
    'boids':     '#9b59b6',
}
ALGO_LABELS = {
    'hardcode':  'Hard-code (смещение)',
    'apf':       'APF',
    'apf_lanes': 'APF + Lanes',
    'boids':     'Boids',
}


# ──────────────────────────────────────────────────────────
# Загрузка данных
# ──────────────────────────────────────────────────────────
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


def load_all(directory: str):
    """Возвращает dict {algo: [DataFrame, ...]}"""
    pattern = os.path.join(directory, 'metrics_*.csv')
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f'[ERROR] Не найдено файлов metrics_*.csv в {directory}')
        sys.exit(1)

    data: dict[str, list[pd.DataFrame]] = {}
    for fp in files:
        algo = detect_algo(fp)
        try:
            df = pd.read_csv(fp)
            # нормируем время: t=0 от начала миссии
            if 'mission_time_s' in df.columns:
                df['t'] = df['mission_time_s']
            else:
                df['t'] = df['time_s'] - df['time_s'].iloc[0]
            data.setdefault(algo, []).append(df)
            print(f'  Загружен {os.path.basename(fp)} → algo={algo}, строк={len(df)}')
        except Exception as e:
            print(f'  [WARN] {fp}: {e}')

    return data


def mean_series(dfs: list[pd.DataFrame], col: str, t_col='t',
                t_max=None, n_points=500) -> tuple:
    """
    Усредняет столбец col по нескольким DataFrame (разные дроны одного алгоритма).
    Интерполирует на общую сетку по времени.
    Возвращает (t_grid, mean, std).
    """
    if t_max is None:
        t_max = min(df[t_col].max() for df in dfs if col in df.columns and len(df) > 0)
    t_grid = np.linspace(0, t_max, n_points)
    interps = []
    for df in dfs:
        if col not in df.columns:
            continue
        mask = df[t_col] <= t_max
        sub  = df[mask].dropna(subset=[col, t_col])
        if len(sub) < 2:
            continue
        interps.append(np.interp(t_grid, sub[t_col].values, sub[col].values))
    if not interps:
        return t_grid, np.zeros_like(t_grid), np.zeros_like(t_grid)
    arr  = np.array(interps)
    return t_grid, arr.mean(axis=0), arr.std(axis=0)


def save_or_show(fig, name: str, save: bool, out_dir: str):
    if save:
        path = os.path.join(out_dir, name + '.png')
        fig.savefig(path, bbox_inches='tight')
        print(f'  Сохранён: {path}')
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────
# График 1: сходимость строя (formation_error) по каждому алгоритму
# ──────────────────────────────────────────────────────────
def plot_convergence_per_algo(data, save, out_dir):
    """
    Для каждого алгоритма: все дроны отдельными линиями + среднее жирным.
    Это «график сходимости» — показывает, как ошибка строя убывает со временем.
    """
    for algo, dfs in data.items():
        col = 'formation_error_m'
        dfs_with_col = [df for df in dfs if col in df.columns]
        if not dfs_with_col:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        t_max = min(df['t'].max() for df in dfs_with_col)
        colors_drone = plt.cm.tab10(np.linspace(0, 0.6, len(dfs_with_col)))
        for i, df in enumerate(dfs_with_col):
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else i
            mask = df['t'] <= t_max
            ax.plot(df.loc[mask, 't'], df.loc[mask, col],
                    alpha=0.45, color=colors_drone[i],
                    label=f'Дрон {did}')
        t_g, mean_, std_ = mean_series(dfs_with_col, col, t_max=t_max)
        ax.plot(t_g, mean_, color=ALGO_COLORS.get(algo, 'black'),
                linewidth=2.5, label='Среднее')
        ax.fill_between(t_g, mean_-std_, mean_+std_,
                         alpha=0.18, color=ALGO_COLORS.get(algo, 'grey'))
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('Время миссии, с')
        ax.set_ylabel('Ошибка строя, м')
        ax.set_title(f'Сходимость строя — {ALGO_LABELS.get(algo, algo)}')
        ax.legend(loc='upper right')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_or_show(fig, f'convergence_{algo}', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 2: сравнение алгоритмов на одном поле (formation_error)
# ──────────────────────────────────────────────────────────
def plot_convergence_comparison(data, save, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    t_global = None
    for algo in ['hardcode', 'apf', 'apf_lanes', 'boids']:
        dfs = data.get(algo, [])
        dfs = [df for df in dfs if 'formation_error_m' in df.columns]
        if not dfs:
            continue
        t_max = min(df['t'].max() for df in dfs)
        if t_global is None or t_max < t_global:
            t_global = t_max
    if t_global is None:
        print('[WARN] нет данных formation_error_m ни в одном файле')
        plt.close(fig)
        return

    for algo in ['hardcode', 'apf', 'apf_lanes', 'boids']:
        dfs = data.get(algo, [])
        dfs = [df for df in dfs if 'formation_error_m' in df.columns]
        if not dfs:
            continue
        t_g, mean_, std_ = mean_series(dfs, 'formation_error_m', t_max=t_global)
        color = ALGO_COLORS.get(algo, 'grey')
        ax.plot(t_g, mean_, color=color, label=ALGO_LABELS.get(algo, algo))
        ax.fill_between(t_g, mean_-std_, mean_+std_, alpha=0.15, color=color)

    ax.set_xlabel('Время миссии, с')
    ax.set_ylabel('Средняя ошибка строя, м')
    ax.set_title('Сравнение алгоритмов: сходимость строя')
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, 'convergence_comparison', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 3: расстояние между дронами
# ──────────────────────────────────────────────────────────
def plot_inter_drone_distance(data, save, out_dir):
    fig, axes = plt.subplots(1, len(data), figsize=(5*max(len(data),1), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]
    for ax, (algo, dfs) in zip(axes, data.items()):
        col  = 'nearest_neighbor_m'
        dfs2 = [df for df in dfs if col in df.columns]
        if not dfs2:
            ax.set_title(algo)
            continue
        t_g, mean_, std_ = mean_series(dfs2, col)
        color = ALGO_COLORS.get(algo, 'grey')
        ax.plot(t_g, mean_, color=color)
        ax.fill_between(t_g, mean_-std_, mean_+std_, alpha=0.2, color=color)
        ax.axhline(1.5, color='red', lw=1.2, ls='--', label='D_SAFE=1.5м')
        ax.set_title(ALGO_LABELS.get(algo, algo))
        ax.set_xlabel('Время, с')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Расстояние до ближайшего соседа, м')
    fig.suptitle('Безопасность строя: межкоптерное расстояние')
    fig.tight_layout()
    save_or_show(fig, 'inter_drone_distance', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 4: скорость
# ──────────────────────────────────────────────────────────
def plot_velocity(data, save, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    for algo, dfs in data.items():
        dfs2 = [df for df in dfs if 'velocity_mps' in df.columns]
        if not dfs2:
            continue
        t_g, mean_, std_ = mean_series(dfs2, 'velocity_mps')
        color = ALGO_COLORS.get(algo, 'grey')
        ax.plot(t_g, mean_, color=color, label=ALGO_LABELS.get(algo, algo))
        ax.fill_between(t_g, mean_-std_, mean_+std_, alpha=0.12, color=color)
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Скорость, м/с')
    ax.set_title('Скорость дронов по алгоритмам')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, 'velocity_comparison', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 5: пройденный путь (накопленный)
# ──────────────────────────────────────────────────────────
def plot_path_length(data, save, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    for algo, dfs in data.items():
        dfs2 = [df for df in dfs if 'path_length_m' in df.columns]
        if not dfs2:
            continue
        t_g, mean_, std_ = mean_series(dfs2, 'path_length_m')
        color = ALGO_COLORS.get(algo, 'grey')
        ax.plot(t_g, mean_, color=color, label=ALGO_LABELS.get(algo, algo))
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Суммарный путь, м')
    ax.set_title('Накопленный путь по алгоритмам')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, 'path_length', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 6: отклонение от полосы (только apf_lanes)
# ──────────────────────────────────────────────────────────
def plot_lane_error(data, save, out_dir):
    dfs = data.get('apf_lanes', [])
    dfs = [df for df in dfs if 'lane_error_m' in df.columns]
    if not dfs:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, df in enumerate(dfs):
        did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else i
        ax.plot(df['t'], df['lane_error_m'], alpha=0.6, label=f'Дрон {did}')
    t_g, mean_, std_ = mean_series(dfs, 'lane_error_m')
    ax.plot(t_g, mean_, color='black', lw=2.5, label='Среднее')
    ax.fill_between(t_g, mean_-std_, mean_+std_, alpha=0.15, color='black')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Отклонение от полосы, м')
    ax.set_title('APF + Lanes: удержание полосы (lane_error)')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, 'lane_error_apf_lanes', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 7: компоненты APF (для apf_lanes)
# ──────────────────────────────────────────────────────────
def plot_apf_components(data, save, out_dir):
    for algo in ['apf', 'apf_lanes']:
        dfs = data.get(algo, [])
        if not dfs:
            continue
        df = dfs[0]
        pairs = [
            ('f_att_x', 'f_att_y', 'Притяжение к цели (att)', '#e74c3c'),
            ('f_rep_x', 'f_rep_y', 'Отталкивание (rep)',       '#3498db'),
            ('f_tan_x', 'f_tan_y', 'Тангенциальная (tan)',     '#f39c12'),
        ]
        if algo == 'apf_lanes' and 'f_lane_y' in df.columns:
            extra = True
        else:
            extra = False

        n = len(pairs) + (1 if extra else 0)
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5*n), sharex=True)
        for ax, (cx, cy, label, color) in zip(axes, pairs):
            if cx not in df.columns:
                continue
            mag = np.sqrt(df[cx]**2 + df[cy]**2)
            ax.plot(df['t'], mag, color=color, lw=1.2)
            ax.set_ylabel('|F|')
            ax.set_title(label)
            ax.grid(alpha=0.3)
        if extra:
            ax = axes[-1]
            ax.plot(df['t'], df['f_lane_y'], color='#2ecc71', lw=1.2)
            ax.set_ylabel('F_y, м/с²')
            ax.set_title('Lane-keeping (возврат на полосу)')
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel('Время, с')
        fig.suptitle(f'Компоненты APF — {ALGO_LABELS.get(algo, algo)}')
        fig.tight_layout()
        save_or_show(fig, f'apf_components_{algo}', save, out_dir)


# ──────────────────────────────────────────────────────────
# График 8: сводная таблица итоговых метрик
# ──────────────────────────────────────────────────────────
def plot_summary_table(data, save, out_dir):
    rows = []
    for algo, dfs in data.items():
        for df in dfs:
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else '?'
            end_rows = df[df.get('event', pd.Series()).astype(str) == 'MISSION_END']
            if len(end_rows) == 0:
                end_rows = df.tail(1)
            row = end_rows.iloc[-1]
            rows.append({
                'Алгоритм':       ALGO_LABELS.get(algo, algo),
                'Дрон':           did,
                'Время мис., с':  round(float(row.get('mission_time_s', 0)), 1),
                'Путь, м':        round(float(row.get('path_length_m', 0)), 1),
                'Покрытие, %':    round(float(row.get('coverage_pct', 0)), 1),
                'Ош. строя, м':   round(float(df.get('formation_error_m', pd.Series([0])).mean()), 3),
                'Мин. сосед, м':  round(float(df.get('nearest_neighbor_m', pd.Series([-1])).replace(-1, np.nan).dropna().min() if 'nearest_neighbor_m' in df else 0), 2),
            })
    if not rows:
        return
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(10, len(rows)*1.2), max(2, len(rows)*0.4+1.5)))
    ax.axis('off')
    tbl = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    fig.suptitle('Сводная таблица результатов эксперимента', y=0.98)
    fig.tight_layout()
    save_or_show(fig, 'summary_table', save, out_dir)

    # также сохраним как CSV
    csv_path = os.path.join(out_dir, 'summary_table.csv')
    summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'  Сводная таблица сохранена: {csv_path}')


# ──────────────────────────────────────────────────────────
# График 9: численная верификация покрытия поля
# ──────────────────────────────────────────────────────────
def plot_coverage_verification(
        field_len=60.0,
        field_wid=30.0,
        height=5.0,
        fov_deg=78.0,
        overlap=0.2,
        n_drones=3,
        grid_step=0.1,
        save=False,
        out_dir='.'):
    """
    Доказывает, что аналитическая формула покрытия совпадает с численным результатом.

    Аналитика: coverage_pct = min(100, n_drones * stripe / field_wid * 100)
    Числ. метод: разбиваем поле на сетку grid_step×grid_step,
                 строим траекторию каждого дрона (змейка) и
                 отмечаем все точки сетки, попавшие в footprint хотя бы одного дрона.
    """
    fov_rad   = math.radians(fov_deg)
    stripe    = 2.0 * height * math.tan(fov_rad / 2.0) * (1.0 - overlap)
    footprint = 2.0 * height * math.tan(fov_rad / 2.0)  # полный размер footprint

    # аналитический процент
    analytic_pct = min(100.0, n_drones * stripe / field_wid * 100.0)

    # строим маршруты
    n_stripes = math.ceil(field_wid / stripe)
    waypoints_per_drone: list[list[tuple]] = [[] for _ in range(n_drones)]
    for s in range(n_stripes):
        d = s % n_drones
        y = stripe * (s + 0.5)
        if (s // n_drones) % 2 == 0:
            xs = [0.0, field_len]
        else:
            xs = [field_len, 0.0]
        waypoints_per_drone[d].append((xs[0], y))
        waypoints_per_drone[d].append((xs[1], y))

    # интерполируем траектории (шаг 0.5 м)
    traj_pts: list[list[tuple]] = []
    traj_step = 0.5
    for wps in waypoints_per_drone:
        pts = []
        for i in range(len(wps)-1):
            x0, y0 = wps[i]
            x1, y1 = wps[i+1]
            dist = math.hypot(x1-x0, y1-y0)
            n    = max(2, int(dist/traj_step))
            for j in range(n):
                a = j/n
                pts.append((x0+a*(x1-x0), y0+a*(y1-y0)))
        traj_pts.append(pts)

    # численный подсчёт покрытия
    xs_grid = np.arange(0, field_len+grid_step, grid_step)
    ys_grid = np.arange(0, field_wid+grid_step, grid_step)
    covered = np.zeros((len(ys_grid), len(xs_grid)), dtype=bool)
    half_fp = footprint / 2.0

    for pts in traj_pts:
        for (px, py) in pts:
            xi0 = int(max(0, (px - half_fp) / grid_step))
            xi1 = int(min(len(xs_grid)-1, (px + half_fp) / grid_step)) + 1
            yi0 = int(max(0, (py - half_fp) / grid_step))
            yi1 = int(min(len(ys_grid)-1, (py + half_fp) / grid_step)) + 1
            covered[yi0:yi1, xi0:xi1] = True

    numeric_pct = covered.sum() / covered.size * 100.0

    # ---- визуализация ----
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[3, 1.5, 1])

    # карта покрытия
    ax0 = fig.add_subplot(gs[0])
    extent = [0, field_len, 0, field_wid]
    ax0.imshow(covered, origin='lower', extent=extent,
               cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    # траектории дронов
    drone_colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    for di, pts in enumerate(traj_pts):
        if pts:
            xs_t = [p[0] for p in pts]
            ys_t = [p[1] for p in pts]
            ax0.plot(xs_t, ys_t, '-', color=drone_colors[di % len(drone_colors)],
                     lw=1.2, label=f'Дрон {di}', alpha=0.8)
    ax0.set_xlabel('X, м')
    ax0.set_ylabel('Y, м')
    ax0.set_title(f'Карта покрытия поля ({field_len}×{field_wid} м)')
    ax0.legend(loc='upper right', fontsize=9)

    # бар-чарт сравнения
    ax1 = fig.add_subplot(gs[1])
    bars = ax1.bar(['Аналитика', 'Численно'],
                   [analytic_pct, numeric_pct],
                   color=['#3498db', '#2ecc71'], width=0.5, edgecolor='k')
    for bar, val in zip(bars, [analytic_pct, numeric_pct]):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel('Покрытие, %')
    ax1.set_title('Верификация формулы')
    ax1.axhline(100, color='k', lw=0.8, ls='--')
    ax1.grid(axis='y', alpha=0.3)

    # текстовые параметры
    ax2 = fig.add_subplot(gs[2])
    ax2.axis('off')
    lines = [
        f'Параметры:',
        f'  Поле: {field_len}×{field_wid} м',
        f'  Высота: {height} м',
        f'  FOV: {fov_deg}°',
        f'  Перекрытие: {overlap*100:.0f}%',
        f'  Footprint: {footprint:.2f} м',
        f'  Stripe: {stripe:.2f} м',
        f'  Дронов: {n_drones}',
        f'  Полос: {n_stripes}',
        f'',
        f'Результаты:',
        f'  Аналитика: {analytic_pct:.2f}%',
        f'  Численно:  {numeric_pct:.2f}%',
        f'  Разница:   {abs(analytic_pct-numeric_pct):.2f}%',
    ]
    ax2.text(0.05, 0.95, '\n'.join(lines), transform=ax2.transAxes,
             va='top', fontsize=9, family='monospace')

    fig.suptitle('Верификация покрытия: аналитическая формула vs численный счёт',
                 fontsize=13)
    fig.tight_layout()
    save_or_show(fig, 'coverage_verification', save, out_dir)
    print(f'  Аналитика: {analytic_pct:.2f}%  |  Численно: {numeric_pct:.2f}%'
          f'  |  Разница: {abs(analytic_pct-numeric_pct):.2f}%')


# ──────────────────────────────────────────────────────────
# График 10: эксперимент с возмущением (disturbance recovery)
# ──────────────────────────────────────────────────────────
def plot_disturbance_recovery(data, save, out_dir):
    """
    Если в логе есть события DISTURBANCE и RECOVERED —
    строит formation_error с маркерами этих событий.
    Добавьте self._log_metrics('DISTURBANCE') и self._log_metrics('RECOVERED')
    в свои скрипты при проведении эксперимента с возмущением.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for algo, dfs in data.items():
        for df in dfs:
            if 'formation_error_m' not in df.columns:
                continue
            did = int(df['drone_id'].iloc[0]) if 'drone_id' in df.columns else '?'
            ax.plot(df['t'], df['formation_error_m'],
                    color=ALGO_COLORS.get(algo, 'grey'), alpha=0.5,
                    label=f'{ALGO_LABELS.get(algo, algo)} / Дрон {did}')
            plotted = True
            if 'event' in df.columns:
                disturbance = df[df['event'].astype(str).str.contains('DISTURBANCE', na=False)]
                recovered   = df[df['event'].astype(str).str.contains('RECOVERED', na=False)]
                for _, row in disturbance.iterrows():
                    ax.axvline(row['t'], color='red', lw=1.5, ls='--', alpha=0.7)
                for _, row in recovered.iterrows():
                    ax.axvline(row['t'], color='green', lw=1.5, ls='--', alpha=0.7)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Ошибка строя, м')
    ax.set_title('Реакция на возмущение (DISTURBANCE ↓ красный, RECOVERED ↓ зелёный)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, 'disturbance_recovery', save, out_dir)


# ──────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Построение графиков ВКР из CSV-логов дронов')
    parser.add_argument('--dir', default='.',
                        help='Папка с файлами metrics_*.csv')
    parser.add_argument('--save', action='store_true',
                        help='Сохранять PNG вместо показа окна')
    parser.add_argument('--out', default=None,
                        help='Папка для PNG (по умолчанию = --dir)')
    # параметры для верификации покрытия
    parser.add_argument('--field_len', type=float, default=60.0)
    parser.add_argument('--field_wid', type=float, default=30.0)
    parser.add_argument('--height',    type=float, default=5.0)
    parser.add_argument('--fov',       type=float, default=78.0)
    parser.add_argument('--overlap',   type=float, default=0.2)
    parser.add_argument('--n_drones',  type=int,   default=3)
    args = parser.parse_args()

    out_dir = args.out or args.dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n=== Загрузка данных из: {args.dir} ===')
    data = load_all(args.dir)
    print(f'Найдено алгоритмов: {list(data.keys())}\n')

    print('=== График 1: сходимость по алгоритмам ===')
    plot_convergence_per_algo(data, args.save, out_dir)

    print('=== График 2: сравнение алгоритмов ===')
    plot_convergence_comparison(data, args.save, out_dir)

    print('=== График 3: межкоптерное расстояние ===')
    plot_inter_drone_distance(data, args.save, out_dir)

    print('=== График 4: скорость ===')
    plot_velocity(data, args.save, out_dir)

    print('=== График 5: накопленный путь ===')
    plot_path_length(data, args.save, out_dir)

    print('=== График 6: ошибка полосы (только APF+lanes) ===')
    plot_lane_error(data, args.save, out_dir)

    print('=== График 7: компоненты APF ===')
    plot_apf_components(data, args.save, out_dir)

    print('=== График 8: сводная таблица ===')
    plot_summary_table(data, args.save, out_dir)

    print('=== График 9: верификация покрытия ===')
    plot_coverage_verification(
        field_len=args.field_len,
        field_wid=args.field_wid,
        height=args.height,
        fov_deg=args.fov,
        overlap=args.overlap,
        n_drones=args.n_drones,
        save=args.save,
        out_dir=out_dir
    )

    print('=== График 10: эксперимент с возмущением ===')
    plot_disturbance_recovery(data, args.save, out_dir)

    print('\nГотово.')


if __name__ == '__main__':
    main()
