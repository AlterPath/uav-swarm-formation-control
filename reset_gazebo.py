#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reset_gazebo.py — сброс позиций дронов в Gazebo без перезапуска симулятора.

Использует /gazebo/set_model_state для телепортации дронов в начальную позицию.
Также сбрасывает скорости и ускорения.

Использование:
    # Сбросить 3 дронов
    python3 reset_gazebo.py --n 3

    # Сбросить и подождать стабилизации
    python3 reset_gazebo.py --n 3 --wait 5

    # Полный сброс мира (teleport + pause/unpause)
    python3 reset_gazebo.py --n 3 --full-reset
"""

import argparse
import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion
import time
import sys


def reset_model(model_name, x=0.0, y=0.0, z=0.1):
    """Сбрасывает позицию модели через ROS service."""
    try:
        rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        state = ModelState()
        state.model_name = model_name
        state.reference_frame = 'world'
        state.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=Quaternion(x=0, y=0, z=0, w=1)
        )
        state.twist = Twist()

        resp = set_state(state)
        if resp.success:
            print(f'  [OK] {model_name} -> ({x:.1f}, {y:.1f}, {z:.1f})')
            return True
        else:
            print(f'  [FAIL] {model_name}: {resp.status_message}')
            return False
    except rospy.ROSException as e:
        print(f'  [ERROR] ROS service недоступен: {e}')
        return False
    except Exception as e:
        print(f'  [ERROR] {model_name}: {e}')
        return False


def pause_physics():
    """Ставит физику на паузу."""
    try:
        rospy.wait_for_service('/gazebo/pause_physics', timeout=3.0)
        pause = rospy.ServiceProxy('/gazebo/pause_physics', rospy.topics.Empty)
        pause()
        print('  [OK] Физика на паузе')
        return True
    except Exception as e:
        print(f'  [WARN] Не удалось поставить паузу: {e}')
        return False


def unpause_physics():
    """Снимает паузу с физики."""
    try:
        rospy.wait_for_service('/gazebo/unpause_physics', timeout=3.0)
        unpause = rospy.ServiceProxy('/gazebo/unpause_physics', rospy.topics.Empty)
        unpause()
        print('  [OK] Физика возобновлена')
        return True
    except Exception as e:
        print(f'  [WARN] Не удалось снять паузу: {e}')
        return False


def reset_world():
    """Сбрасывает весь мир (все модели)."""
    try:
        rospy.wait_for_service('/gazebo/reset_world', timeout=5.0)
        reset = rospy.ServiceProxy('/gazebo/reset_world', rospy.topics.Empty)
        reset()
        print('  [OK] Мир сброшен (/gazebo/reset_world)')
        return True
    except Exception as e:
        print(f'  [WARN] reset_world не сработал: {e}')
        return False


def reset_simulation():
    """Сбрасывает симуляцию (время + состояние)."""
    try:
        rospy.wait_for_service('/gazebo/reset_simulation', timeout=5.0)
        reset = rospy.ServiceProxy('/gazebo/reset_simulation', rospy.topics.Empty)
        reset()
        print('  [OK] Симуляция сброшена (/gazebo/reset_simulation)')
        return True
    except Exception as e:
        print(f'  [WARN] reset_simulation не сработал: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='Сброс позиций дронов в Gazebo')
    parser.add_argument('--n', type=int, default=3, help='Количество дронов')
    parser.add_argument('--wait', type=float, default=2.0,
                        help='Секунд подождать после сброса (default: 2)')
    parser.add_argument('--full-reset', action='store_true',
                        help='Полный сброс мира (reset_world)')
    parser.add_argument('--reset-sim', action='store_true',
                        help='Сбросить время симуляции')
    args = parser.parse_args()

    print('=== Сброс Gazebo ===')

    # Инициализируем ROS-ноду (если ещё не инициализирована)
    try:
        rospy.init_node('gazebo_resetter', anonymous=True)
    except rospy.exceptions.ROSException:
        pass  # нода уже инициализирована

    # Полный сброс (если запрошен)
    if args.full_reset:
        print('--- Полный сброс ---')
        reset_world()
        time.sleep(0.5)

    if args.reset_sim:
        print('--- Сброс времени симуляции ---')
        reset_simulation()
        time.sleep(0.5)

    # Сброс каждого дрона
    print(f'--- Сброс позиций {args.n} дронов ---')
    for i in range(args.n):
        model_name = f'iris_{i}'
        reset_model(model_name, x=0.0, y=0.0, z=0.1)
        time.sleep(0.1)

    # Ожидание стабилизации
    if args.wait > 0:
        print(f'--- Ожидание {args.wait} сек ---')
        time.sleep(args.wait)

    print('=== Готово ===')


if __name__ == '__main__':
    main()
