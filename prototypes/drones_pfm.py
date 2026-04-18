#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
swarm_apf_mission.py
ВКР: Программно-алгоритмическое обеспечение системы управления группой
квадрокоптеров для задач удержания строя.

Один файл, без собственных launch-файлов. Использует MAVROS + PX4 (iris, Gazebo).
Метод искусственных потенциальных полей (APF) с тангенциальной составляющей,
змейка, барьер-синхронизация, heartbeat-отказоустойчивость, CSV-лог.
"""

import argparse
import csv
import math
import os
import sys
import threading
import time
from collections import defaultdict

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String, Float32MultiArray, Header

# ------------------------- ПАРАМЕТРЫ АЛГОРИТМА -------------------------------
K_ATT          = 0.8          # коэффициент притяжения к цели
K_REP          = 2.5          # коэффициент отталкивания от соседей / препятствий
K_TAN          = 1.2          # тангенциальная составляющая (огибание)
D_INFLUENCE    = 6.0          # радиус влияния отталкивания, м
D_SAFE         = 2.5          # минимально допустимое расстояние между дронами, м
V_MAX          = 2.5          # ограничение скорости, м/с
REACH_TOL      = 0.8          # допуск достижения точки, м
POINT_TIMEOUT  = 60.0         # таймаут достижения точки, с
HEARTBEAT_HZ   = 2.0          # частота heartbeat
HEARTBEAT_DEAD = 3.0          # сек. без heartbeat => сосед мёртв
CONTROL_HZ     = 20.0         # частота управляющего цикла
CAM_FOV_DEG    = 78.0         # горизонтальный угол обзора камеры
OVERLAP        = 0.2          # минимальное перекрытие соседних полос (20%)
TAKEOFF_RATE   = 1.0          # вертикальная скорость на взлёте, м/с
# -----------------------------------------------------------------------------


class DroneNode:
    def __init__(self, drone_id: int, total: int, ns: str):
        self.id = drone_id
        self.total = total
        self.ns = ns.strip('/')
        self.is_leader = (drone_id == 0)

        rospy.init_node(f'swarm_apf_{self.ns}', anonymous=False)

        # ----- состояние MAVROS -----
        self.state = State()
        self.pose = PoseStamped()
        self.have_pose = False

        # ----- соседи -----
        # id -> (x,y,z,t_last)
        self.neighbors = {}
        self.neighbors_lock = threading.Lock()
        self.alive_ids = set(range(total))  # активные дроны (включая себя)

        # ----- барьер готовности -----
        self.ready_set = set()
        self.ready_lock = threading.Lock()

        # ----- общие параметры поля (от лидера) -----
        self.field_len = None
        self.field_wid = None
        self.alt = None
        self.params_received = threading.Event()

        # ----- подписки / публикации MAVROS -----
        self._setup_mavros_io()

        # ----- межагентные топики (общие, без namespace) -----
        self.hb_pub = rospy.Publisher('/swarm/heartbeat', Float32MultiArray,
                                      queue_size=10)
        rospy.Subscriber('/swarm/heartbeat', Float32MultiArray, self._hb_cb)

        self.ready_pub = rospy.Publisher('/swarm/ready', String, queue_size=10)
        rospy.Subscriber('/swarm/ready', String, self._ready_cb)

        self.params_pub = rospy.Publisher('/swarm/params', Float32MultiArray,
                                          queue_size=1, latch=True)
        rospy.Subscriber('/swarm/params', Float32MultiArray, self._params_cb)

        # ----- CSV лог -----
        log_name = f'swarm_log_{self.ns}.csv'
        self.csv_file = open(log_name, 'w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['timestamp', 'event', 'photo',
                           'target_x', 'target_y', 'target_z',
                           'drone_x', 'drone_y', 'drone_z'])
        self.csv_file.flush()
        self.photo_counter = 0

        rospy.loginfo(f'[{self.ns}] node initialized, id={self.id}, total={total}')

    # =========================================================================
    #                             MAVROS I/O
    # =========================================================================
    def _setup_mavros_io(self):
        pref = f'/{self.ns}/mavros'
        rospy.Subscriber(f'{pref}/state', State, self._state_cb)
        rospy.Subscriber(f'{pref}/local_position/pose', PoseStamped, self._pose_cb)

        self.sp_pub = rospy.Publisher(f'{pref}/setpoint_position/local',
                                      PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher(f'{pref}/setpoint_velocity/cmd_vel',
                                       TwistStamped, queue_size=10)

        rospy.wait_for_service(f'{pref}/cmd/arming')
        rospy.wait_for_service(f'{pref}/set_mode')
        self.srv_arm = rospy.ServiceProxy(f'{pref}/cmd/arming', CommandBool)
        self.srv_mode = rospy.ServiceProxy(f'{pref}/set_mode', SetMode)

    def _state_cb(self, msg: State):
        self.state = msg

    def _pose_cb(self, msg: PoseStamped):
        self.pose = msg
        self.have_pose = True

    # =========================================================================
    #                        Межагентное взаимодействие
    # =========================================================================
    def _hb_cb(self, msg: Float32MultiArray):
        # формат: [id, x, y, z, t]
        if len(msg.data) < 5:
            return
        nid = int(msg.data[0])
        if nid == self.id:
            return
        with self.neighbors_lock:
            self.neighbors[nid] = (msg.data[1], msg.data[2],
                                   msg.data[3], time.time())

    def _ready_cb(self, msg: String):
        with self.ready_lock:
            try:
                self.ready_set.add(int(msg.data))
            except ValueError:
                pass

    def _params_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 3 and not self.params_received.is_set():
            self.field_len = float(msg.data[0])
            self.field_wid = float(msg.data[1])
            self.alt = float(msg.data[2])
            self.params_received.set()
            rospy.loginfo(f'[{self.ns}] params received: '
                          f'L={self.field_len} W={self.field_wid} H={self.alt}')

    def _publish_heartbeat(self):
        m = Float32MultiArray()
        m.data = [float(self.id),
                  self.pose.pose.position.x,
                  self.pose.pose.position.y,
                  self.pose.pose.position.z,
                  time.time()]
        self.hb_pub.publish(m)

    def _start_heartbeat_thread(self):
        def loop():
            r = rospy.Rate(HEARTBEAT_HZ)
            while not rospy.is_shutdown():
                if self.have_pose:
                    self._publish_heartbeat()
                self._update_alive()
                r.sleep()
        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _update_alive(self):
        now = time.time()
        alive = {self.id}
        with self.neighbors_lock:
            for nid, (_, _, _, t) in self.neighbors.items():
                if now - t < HEARTBEAT_DEAD:
                    alive.add(nid)
        lost = self.alive_ids - alive
        if lost:
            rospy.logwarn(f'[{self.ns}] lost neighbors: {lost}')
            # отказоустойчивость: увеличиваем высоту и межрядное расстояние
            # коэффициент компенсации: sqrt(total/alive) чтобы площадь покрытия
            # сохранилась
            k = math.sqrt(self.total / max(1, len(alive)))
            self.alt = self.alt * k
            rospy.logwarn(f'[{self.ns}] compensation: new alt={self.alt:.2f}, '
                          f'stripe recomputed')
        self.alive_ids = alive

    # =========================================================================
    #                                  APF
    # =========================================================================
    def apf_velocity(self, target):
        """Искусственные потенциальные поля: притяжение к цели, отталкивание
        от соседей (и виртуальных препятствий - границ поля) + тангенциальная
        составляющая для огибания."""
        px, py = self.pose.pose.position.x, self.pose.pose.position.y
        tx, ty, _ = target

        # --- притяжение ---
        dx, dy = tx - px, ty - py
        d = math.hypot(dx, dy) + 1e-9
        f_att_x = K_ATT * dx / d
        f_att_y = K_ATT * dy / d

        # --- отталкивание + тангенциальная составляющая ---
        f_rep_x = f_rep_y = 0.0
        f_tan_x = f_tan_y = 0.0

        with self.neighbors_lock:
            items = list(self.neighbors.items())

        now = time.time()
        for nid, (nx, ny, nz, t) in items:
            if now - t > HEARTBEAT_DEAD:
                continue
            rx, ry = px - nx, py - ny
            rdist = math.hypot(rx, ry)
            if rdist < 1e-3:
                continue
            if rdist < D_INFLUENCE:
                # классическое отталкивание 1/d - 1/d0
                mag = K_REP * (1.0 / rdist - 1.0 / D_INFLUENCE) / (rdist ** 2)
                if rdist < D_SAFE:
                    mag *= 5.0  # усиление при опасном сближении
                f_rep_x += mag * rx / rdist
                f_rep_y += mag * ry / rdist
                # тангенциальная (перпендикуляр к радиус-вектору, в сторону цели)
                # выбираем знак так, чтобы компонента вдоль направления к цели
                # была положительной
                tx_perp, ty_perp = -ry / rdist, rx / rdist
                if tx_perp * dx + ty_perp * dy < 0:
                    tx_perp, ty_perp = -tx_perp, -ty_perp
                k_tan = K_TAN * (1.0 - rdist / D_INFLUENCE)
                f_tan_x += k_tan * tx_perp
                f_tan_y += k_tan * ty_perp

        vx = f_att_x + f_rep_x + f_tan_x
        vy = f_att_y + f_rep_y + f_tan_y

        # --- ограничение модуля скорости ---
        v = math.hypot(vx, vy)
        if v > V_MAX:
            vx = vx * V_MAX / v
            vy = vy * V_MAX / v
        return vx, vy

    # =========================================================================
    #                            Режимы MAVROS
    # =========================================================================
    def _arm_and_offboard(self):
        # предварительное заполнение setpoint'ами
        sp = PoseStamped()
        sp.pose.position.x = self.pose.pose.position.x
        sp.pose.position.y = self.pose.pose.position.y
        sp.pose.position.z = self.alt
        r = rospy.Rate(CONTROL_HZ)
        for _ in range(50):
            sp.header.stamp = rospy.Time.now()
            self.sp_pub.publish(sp)
            r.sleep()

        last_req = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.state.mode != 'OFFBOARD' and \
               (rospy.Time.now() - last_req).to_sec() > 2.0:
                try:
                    self.srv_mode(custom_mode='OFFBOARD')
                except rospy.ServiceException as e:
                    rospy.logwarn(f'set_mode failed: {e}')
                last_req = rospy.Time.now()
            elif not self.state.armed and \
                 (rospy.Time.now() - last_req).to_sec() > 2.0:
                try:
                    self.srv_arm(True)
                except rospy.ServiceException as e:
                    rospy.logwarn(f'arm failed: {e}')
                last_req = rospy.Time.now()
            if self.state.armed and self.state.mode == 'OFFBOARD':
                rospy.loginfo(f'[{self.ns}] OFFBOARD + ARMED')
                return
            sp.header.stamp = rospy.Time.now()
            self.sp_pub.publish(sp)
            r.sleep()

    def _land(self):
        try:
            self.srv_mode(custom_mode='AUTO.LAND')
            rospy.loginfo(f'[{self.ns}] LAND requested')
        except rospy.ServiceException as e:
            rospy.logwarn(f'land failed: {e}')

    # =========================================================================
    #                           Барьер готовности
    # =========================================================================
    def _barrier_wait(self):
        rospy.loginfo(f'[{self.ns}] waiting barrier…')
        r = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            self.ready_pub.publish(String(data=str(self.id)))
            with self.ready_lock:
                self.ready_set.add(self.id)
                if self.alive_ids.issubset(self.ready_set):
                    rospy.loginfo(f'[{self.ns}] barrier passed')
                    return
            r.sleep()

    # =========================================================================
    #                        Построение маршрута-змейки
    # =========================================================================
    def _build_snake(self):
        """Возвращает список точек [(x,y,z), …] для этого дрона."""
        # ширина полосы съёмки
        fov = math.radians(CAM_FOV_DEG)
        stripe_full = 2.0 * self.alt * math.tan(fov / 2.0)
        stripe = stripe_full * (1.0 - OVERLAP)

        n_alive = max(1, len(self.alive_ids))
        # сколько полос поперёк поля всего нужно
        total_stripes = max(1, int(math.ceil(self.field_wid / stripe)))
        # распределяем по живым дронам: каждый берёт подмножество полос
        my_rank = sorted(self.alive_ids).index(self.id) \
                  if self.id in self.alive_ids else 0
        my_stripes = [i for i in range(total_stripes)
                      if i % n_alive == my_rank]

        points = []
        for k, s in enumerate(my_stripes):
            y = -self.field_wid / 2.0 + stripe * (s + 0.5)
            if k % 2 == 0:
                xs = [0.0, self.field_len]
            else:
                xs = [self.field_len, 0.0]
            # промежуточные точки съёмки (каждые stripe метров по длине)
            step = max(5.0, stripe)
            n_steps = max(2, int(math.ceil(self.field_len / step)) + 1)
            for i in range(n_steps):
                a = i / (n_steps - 1)
                x = xs[0] + a * (xs[1] - xs[0])
                points.append((x, y, self.alt))
        return points

    # =========================================================================
    #                      Достижение точки через APF
    # =========================================================================
    def _goto_apf(self, target):
        t0 = time.time()
        r = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            dx = target[0] - self.pose.pose.position.x
            dy = target[1] - self.pose.pose.position.y
            dz = target[2] - self.pose.pose.position.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < REACH_TOL:
                return True

            if time.time() - t0 > POINT_TIMEOUT:
                rospy.logwarn(f'[{self.ns}] TIMEOUT на точке {target}')
                return False

            vx, vy = self.apf_velocity(target)
            vz = max(-V_MAX, min(V_MAX, 0.8 * dz))

            cmd = TwistStamped()
            cmd.header.stamp = rospy.Time.now()
            cmd.twist.linear.x = vx
            cmd.twist.linear.y = vy
            cmd.twist.linear.z = vz
            self.vel_pub.publish(cmd)
            r.sleep()
        return False

    # =========================================================================
    #                                  Съёмка
    # =========================================================================
    def _take_photo(self, target, reached: bool):
        self.photo_counter += 1
        fname = f'{self.ns}_photo_{self.photo_counter:05d}.jpg'
        event = 'SHOT' if reached else 'MISSED'
        px = self.pose.pose.position.x
        py = self.pose.pose.position.y
        pz = self.pose.pose.position.z
        self.csv.writerow([f'{time.time():.3f}', event, fname,
                           f'{target[0]:.3f}', f'{target[1]:.3f}', f'{target[2]:.3f}',
                           f'{px:.3f}', f'{py:.3f}', f'{pz:.3f}'])
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())
        rospy.loginfo(f'[{self.ns}] {event} {fname} '
                      f'target=({target[0]:.1f},{target[1]:.1f}) '
                      f'drone=({px:.1f},{py:.1f})')

    # =========================================================================
    #                            Главный сценарий
    # =========================================================================
    def run(self):
        # ждём MAVROS FCU
        rospy.loginfo(f'[{self.ns}] waiting FCU connection…')
        r = rospy.Rate(5.0)
        while not rospy.is_shutdown() and not self.state.connected:
            r.sleep()
        rospy.loginfo(f'[{self.ns}] FCU connected')

        while not rospy.is_shutdown() and not self.have_pose:
            r.sleep()

        # heartbeat
        self._start_heartbeat_thread()

        # ------------- параметры миссии -------------
        if self.is_leader:
            print('\n=== Ввод параметров области облёта (лидер id=0) ===')
            self.field_len = float(input('Длина поля (м): '))
            self.field_wid = float(input('Ширина поля (м): '))
            self.alt       = float(input('Высота полёта (м): '))
            m = Float32MultiArray()
            m.data = [self.field_len, self.field_wid, self.alt]
            # публикуем с latch=True, чтобы новые подписчики получили
            rospy.sleep(1.0)
            for _ in range(5):
                self.params_pub.publish(m)
                rospy.sleep(0.2)
            self.params_received.set()
        else:
            rospy.loginfo(f'[{self.ns}] ожидание параметров от лидера…')
            while not rospy.is_shutdown() and not self.params_received.is_set():
                rospy.sleep(0.2)

        # ------------- взлёт + OFFBOARD -------------
        self._arm_and_offboard()

        # подъём в стартовую позицию своего ряда
        rospy.loginfo(f'[{self.ns}] takeoff to working altitude…')
        # начальная стартовая позиция: каждый дрон в своей поперечной полосе,
        # на границе x=0
        start_y = -self.field_wid / 2.0 + \
                  self.field_wid * (self.id + 0.5) / self.total
        start = (0.0, start_y, self.alt)

        sp = PoseStamped()
        sp.pose.position.x = start[0]
        sp.pose.position.y = start[1]
        sp.pose.position.z = start[2]
        t0 = time.time()
        ctrl = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            sp.header.stamp = rospy.Time.now()
            self.sp_pub.publish(sp)
            dz = self.alt - self.pose.pose.position.z
            dxy = math.hypot(start[0] - self.pose.pose.position.x,
                             start[1] - self.pose.pose.position.y)
            if abs(dz) < 0.4 and dxy < 1.0:
                break
            if time.time() - t0 > 90.0:
                rospy.logwarn(f'[{self.ns}] долго взлетает, продолжаем')
                break
            ctrl.sleep()

        # ------------- барьер готовности -------------
        self._barrier_wait()

        # ------------- маршрут-змейка -------------
        route = self._build_snake()
        rospy.loginfo(f'[{self.ns}] маршрут из {len(route)} точек')

        for i, p in enumerate(route):
            rospy.loginfo(f'[{self.ns}] -> точка {i+1}/{len(route)}: {p}')
            ok = self._goto_apf(p)
            self._take_photo(p, reached=ok)

        # ------------- возврат и посадка -------------
        rospy.loginfo(f'[{self.ns}] маршрут завершён, возврат')
        self._goto_apf((0.0, start_y, self.alt))
        self._land()

        # корректное закрытие лога
        rospy.sleep(3.0)
        self.csv_file.flush()
        os.fsync(self.csv_file.fileno())
        self.csv_file.close()
        rospy.loginfo(f'[{self.ns}] лог закрыт, миссия завершена')


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--id', type=int, required=True, help='ID дрона (0..N-1)')
    p.add_argument('--total', type=int, required=True, help='Всего дронов')
    p.add_argument('--ns', type=str, default=None,
                   help='namespace MAVROS (например uav0). По умолчанию uav<id>')
    return p.parse_args()


def main():
    args = parse_args()
    ns = args.ns if args.ns else f'uav{args.id}'
    node = DroneNode(args.id, args.total, ns)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f'Ошибка: {e}')
    finally:
        try:
            node.csv_file.flush()
            node.csv_file.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
