#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init_experiments.py — создание структуры папок для экспериментов ВКР.

Создаёт дерево директорий:
  <base_dir>/
    <algo>/
      <obstacles>/
        <mode>/
          run_1/
          run_2/
          run_3/

Также умеет проверять, какие прогоны уже содержат результаты (--check).

Использование:
    # Создать структуру в ~/vkr_results
    python3 init_experiments.py

    # Создать в другой папке
    python3 init_experiments.py --base ~/my_results

    # Проверить, что уже собрано
    python3 init_experiments.py --check --base ~/vkr_results

    # Вывести дерево без создания файлов
    python3 init_experiments.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# ── Конфигурация эксперимента ──────────────────────────────────────────────
ALGOS     = ['hardcode', 'apf', 'apf_lanes', 'boids']
OBSTACLES = ['no_obstacles', 'light', 'heavy']
MODES     = ['normal', 'lose_drone']
RUNS      = [1, 2, 3]

TOTAL_RUNS = len(ALGOS) * len(OBSTACLES) * len(MODES) * len(RUNS)  # = 72


def iter_all():
    """Генератор всех комбинаций (algo, obs, mode, run)."""
    for algo in ALGOS:
        for obs in OBSTACLES:
            for mode in MODES:
                for run in RUNS:
                    yield algo, obs, mode, run


def create_structure(base_dir: Path, dry_run=False):
    """Создаёт дерево папок."""
    created = 0
    existed = 0

    for algo, obs, mode, run in iter_all():
        path = base_dir / algo / obs / mode / f'run_{run}'
        if path.exists():
            existed += 1
        else:
            if not dry_run:
                path.mkdir(parents=True, exist_ok=True)
                # Создаём пустой .gitkeep чтобы папка отслеживалась git
                (path / '.gitkeep').touch()
            created += 1
            if dry_run:
                print(f'  [DRY] создать: {path}')

    print(f'\nСтруктура папок: {base_dir}')
    print(f'  Создано:   {created}')
    print(f'  Уже было:  {existed}')
    print(f'  Всего:     {created + existed} / {TOTAL_RUNS}')
    return created


def check_results(base_dir: Path):
    """Проверяет, в каких папках уже есть CSV-результаты."""
    done = []
    missing = []
    empty = []

    for algo, obs, mode, run in iter_all():
        path = base_dir / algo / obs / mode / f'run_{run}'
        label = f'{algo}/{obs}/{mode}/run_{run}'

        if not path.exists():
            missing.append(label)
            continue

        csv_files = list(path.glob('metrics_*.csv'))
        if csv_files:
            done.append((label, len(csv_files)))
        else:
            empty.append(label)

    # Вывод
    total = len(done) + len(missing) + len(empty)
    print(f'\n=== Проверка результатов: {base_dir} ===')
    print(f'  ✓ С данными:      {len(done):3d} / {total}')
    print(f'  ○ Папка есть, CSV нет: {len(empty):3d}')
    print(f'  ✗ Папка отсутствует:   {len(missing):3d}')

    if done:
        print('\n--- Готовые прогоны ---')
        for label, n in done:
            print(f'  ✓ {label}  ({n} CSV)')

    if empty:
        print('\n--- Папки без данных ---')
        for label in empty:
            print(f'  ○ {label}')

    if missing:
        print('\n--- Отсутствующие папки ---')
        for label in missing[:20]:
            print(f'  ✗ {label}')
        if len(missing) > 20:
            print(f'  ... и ещё {len(missing) - 20}')

    return len(done), len(empty), len(missing)


def print_tree(base_dir: Path):
    """Печатает дерево структуры (первые 3 уровня)."""
    print(f'\n{base_dir}/')
    for algo in ALGOS:
        print(f'├── {algo}/')
        for obs in OBSTACLES:
            print(f'│   ├── {obs}/')
            for mode in MODES:
                runs_str = '  '.join(f'run_{r}/' for r in RUNS)
                print(f'│   │   ├── {mode}/')
                print(f'│   │   │   └── {runs_str}')
    print(f'\nИтого: {TOTAL_RUNS} прогонов')


def main():
    parser = argparse.ArgumentParser(
        description='Инициализация структуры папок для экспериментов ВКР'
    )
    parser.add_argument(
        '--base', type=str,
        default=str(Path.home() / 'vkr_results'),
        help='Базовая папка (default: ~/vkr_results)'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Проверить, какие прогоны уже содержат результаты'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Показать, что будет создано, без реального создания'
    )
    parser.add_argument(
        '--tree', action='store_true',
        help='Вывести ожидаемое дерево папок'
    )
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()

    print('=== Эксперименты ВКР: инициализация ===')
    print(f'Базовая папка: {base_dir}')
    print(f'Алгоритмы:    {", ".join(ALGOS)}')
    print(f'Препятствия:  {", ".join(OBSTACLES)}')
    print(f'Режимы:       {", ".join(MODES)}')
    print(f'Прогоны:      {len(RUNS)} × {TOTAL_RUNS // len(RUNS)} комбинаций = {TOTAL_RUNS} итого')

    if args.tree:
        print_tree(base_dir)
        return

    if args.check:
        if not base_dir.exists():
            print(f'\n[ERROR] Папка не существует: {base_dir}')
            print('Сначала запустите без --check для создания структуры.')
            sys.exit(1)
        check_results(base_dir)
        return

    if args.dry_run:
        print('\n[DRY RUN] Папки для создания:')
    else:
        print(f'\nСоздание структуры...')

    n = create_structure(base_dir, dry_run=args.dry_run)

    if not args.dry_run and n > 0:
        print('\n[OK] Структура создана. Теперь можно запускать эксперименты:')
        print(f'  ./run_all.sh {base_dir}')
        print(f'  ./run_single.sh hardcode no_obstacles normal 1 {base_dir}')


if __name__ == '__main__':
    main()
