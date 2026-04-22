#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_experiment.py — универсальный лончер экспериментов.

Запускает выбранный алгоритм с нужными параметрами.
Поддерживает запуск в фоне (--bg) и ожидание завершения.

Использование:
    # Hard-code (1 процесс на всех дронов)
    python3 launch_experiment.py --algo hardcode --n 3 --bg

    # APF / APF+Lanes / Boids (по процессу на дрона)
    python3 launch_experiment.py --algo apf --n 3 --bg
    python3 launch_experiment.py --algo apf_lanes --n 3 --bg
    python3 launch_experiment.py --algo boids --n 3 --bg

    # Интерактивный режим (ждёт ввода параметров)
    python3 launch_experiment.py --algo hardcode --n 3

    # Автоматический режим (без ввода, параметры из аргументов)
    python3 launch_experiment.py --algo hardcode --n 3 --field-len 60 --field-wid 30 --height 5 --auto

    # Для lose_drone: запустить, потом через N секунд "убить" uav1
    python3 launch_experiment.py --algo apf --n 3 --lose-drone --lose-after 30 --bg
"""

import argparse
import os
import subprocess
import sys
import time
import signal
import threading

# ── конфигурация ───────────────────────────────────────────
ALGO_SCRIPTS = {
    'hardcode':  'hard_code_drones.py',
    'apf':       'apf_drones.py',
    'apf_lanes': 'apf_with_lanes_drones.py',
    'boids':     'boids_drones.py',
}

FIELD_LEN = 60.0
FIELD_WID = 30.0
HEIGHT    = 5.0


# ── утилиты ────────────────────────────────────────────────

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def kill_drone(ns='uav1'):
    """Убивает процесс дрона по имени ноды (через rosnode kill)."""
    print(f'[LAUNCHER] Убиваем дрона {ns}...')
    try:
        subprocess.run(['rosnode', 'kill', f'/swarm_apf_{ns}'], capture_output=True)
        subprocess.run(['rosnode', 'kill', f'/swarm_apf_lanes_{ns}'], capture_output=True)
        subprocess.run(['rosnode', 'kill', f'/swarm_boids_{ns}'], capture_output=True)
        subprocess.run(['pkill', '-f', f'--id 1.*{ns}'], capture_output=True)
    except Exception as e:
        print(f'[LAUNCHER] Ошибка при убийстве дрона: {e}')


def reset_gazebo_models(n_drones=3):
    """Сбрасывает позиции моделей дронов в Gazebo на начальные (0,0,0)."""
    print('[LAUNCHER] Сброс позиций моделей в Gazebo...')
    for i in range(n_drones):
        model_name = f'iris_{i}'
        # Используем gz model или rosservice
        cmd = [
            'rosservice', 'call', f'/gazebo/set_model_state',
            f"{{model_state: {{model_name: '{model_name}', pose: {{position: {{x: 0.0, y: 0.0, z: 0.1}}}}, twist: {{linear: {{x: 0.0, y: 0.0, z: 0.0}}, angular: {{x: 0.0, y: 0.0, z: 0.0}}}}, reference_frame: 'world'}}}}"
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except Exception:
            pass
    print('[LAUNCHER] Позиции сброшены')


def wait_for_ros():
    """Ждём, пока ROS master будет доступен."""
    print('[LAUNCHER] Ожидание ROS master...')
    for _ in range(30):
        try:
            import rospy
            if rospy.is_shutdown():
                time.sleep(1)
                continue
            return True
        except Exception:
            time.sleep(1)
    return False


# ── запуск алгоритмов ──────────────────────────────────────

def launch_hardcode(args):
    """Запуск hardcode — один процесс, интерактивный ввод."""
    script = os.path.join(get_script_dir(), ALGO_SCRIPTS['hardcode'])
    env = os.environ.copy()

    if args.auto:
        # Передаём параметры через stdin
        input_data = f"{args.height}\n{args.field_len}\n{args.field_wid}\n"
        proc = subprocess.Popen(
            [sys.executable, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=get_script_dir(),
            env=env,
            text=True,
        )
        proc.stdin.write(input_data)
        proc.stdin.flush()
    else:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=get_script_dir(),
            env=env,
        )
    return proc


def launch_multi_process(algo, args):
    """Запуск APF/APF+Lanes/Boids — по процессу на дрона."""
    script = os.path.join(get_script_dir(), ALGO_SCRIPTS[algo])
    env = os.environ.copy()
    procs = []

    for i in range(args.n):
        ns = f'uav{i}'
        cmd = [
            sys.executable, script,
            '--id',        str(i),
            '--total',     str(args.n),
            '--ns',        ns,
            '--field-len', str(args.field_len),
            '--field-wid', str(args.field_wid),
            '--height',    str(args.height),
        ]
        if args.log_dir:
            cmd += ['--log-dir', args.log_dir]
        print(f'[LAUNCHER] Запуск: {" ".join(cmd)}')
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=get_script_dir(),
            env=env,
        )
        procs.append(proc)
        time.sleep(0.5)  # небольшая задержка между запусками

    return procs


# ── main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Универсальный лончер экспериментов')
    parser.add_argument('--algo', required=True,
                        choices=list(ALGO_SCRIPTS.keys()),
                        help='Алгоритм для запуска')
    parser.add_argument('--n', type=int, default=3,
                        help='Количество дронов (default: 3)')
    parser.add_argument('--field-len', type=float, default=FIELD_LEN)
    parser.add_argument('--field-wid', type=float, default=FIELD_WID)
    parser.add_argument('--height', type=float, default=HEIGHT)
    parser.add_argument('--auto', action='store_true',
                        help='Автоматический режим без интерактивного ввода')
    parser.add_argument('--bg', action='store_true',
                        help='Запустить в фоне и выйти (не ждать завершения)')
    parser.add_argument('--skip-wait', action='store_true',
                        help='Не ждать ROS master')
    parser.add_argument('--lose-drone', action='store_true',
                        help='Режим потери дрона (убить uav1 через --lose-after сек)')
    parser.add_argument('--lose-after', type=float, default=30.0,
                        help='Через сколько секунд убить дрона (default: 30)')
    parser.add_argument('--reset-gazebo', action='store_true',
                        help='Сбросить позиции моделей в Gazebo перед запуском')
    parser.add_argument('--log-dir', type=str, default=None)
    args = parser.parse_args()

    script_dir = get_script_dir()
    print(f'[LAUNCHER] Алгоритм: {args.algo}')
    print(f'[LAUNCHER] Дронов: {args.n}')
    print(f'[LAUNCHER] Папка: {script_dir}')

    # Проверка скрипта
    script_path = os.path.join(script_dir, ALGO_SCRIPTS[args.algo])
    if not os.path.exists(script_path):
        print(f'[ERROR] Скрипт не найден: {script_path}')
        sys.exit(1)

    # Сброс Gazebo
    if args.reset_gazebo:
        reset_gazebo_models(args.n)
        time.sleep(1.0)

    # Запуск
    if args.algo == 'hardcode':
        procs = [launch_hardcode(args)]
    else:
        procs = launch_multi_process(args.algo, args)

    # Фоновый режим
    if args.bg:
        print(f'[LAUNCHER] Запущено {len(procs)} процессов в фоне')
        # Пишем PID-файл для последующего сбора
        pid_file = os.path.join(script_dir, f'.pids_{args.algo}.txt')
        with open(pid_file, 'w') as f:
            for p in procs:
                f.write(f'{p.pid}\n')
        print(f'[LAUNCHER] PID-файл: {pid_file}')

        # Если lose_drone — запускаем таймер
        if args.lose_drone:
            def lose_timer():
                time.sleep(args.lose_after)
                kill_drone('uav1')
            t = threading.Thread(target=lose_timer, daemon=True)
            t.start()
            print(f'[LAUNCHER] Таймер потери дрона: {args.lose_after} сек')

        sys.exit(0)

    # Ожидание завершения
    try:
        if args.lose_drone:
            def lose_timer():
                time.sleep(args.lose_after)
                kill_drone('uav1')
            t = threading.Thread(target=lose_timer, daemon=True)
            t.start()
            print(f'[LAUNCHER] Таймер потери дрона: {args.lose_after} сек')

        print('[LAUNCHER] Ожидание завершения миссии...')
        for p in procs:
            p.wait()
        print('[LAUNCHER] Все процессы завершены')
    except KeyboardInterrupt:
        print('[LAUNCHER] Прерывание, убиваем процессы...')
        for p in procs:
            p.terminate()


if __name__ == '__main__':
    main()
