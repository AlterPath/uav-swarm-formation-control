#!/usr/bin/env python3
"""
drone_node.py — Основной ROS-узел одного квадрокоптера в рое.

Запуск:
    rosrun swarm_pkg drone_node.py _id:=0 _total:=3

Архитектура
-----------
  • Независимый узел для каждого дрона
  • MAVROS namespace: /uav{id}/mavros/...
  • Управление через PositionTarget (OFFBOARD, velocity setpoint)
  • APF вычисляет вектор скорости на каждом тике управления
  • Конечный автомат: INIT → TAKEOFF → SYNC → MISSION → LAND

Топики
------
  Подписки:
    /uav{id}/mavros/state
    /uav{id}/mavros/local_position/pose
    /swarm/position/uav{j}          ← позиции соседей
    /iris{id}/usb_cam/image_raw     ← камера (для триггера съёмки)

  Публикации:
    /uav{id}/mavros/setpoint_raw/local   ← setpoint
    /swarm/position/uav{id}             → позиция для соседей
    /swarm/heartbeat/uav{id}            → heartbeat
    /swarm/ready/uav{id}                → синхронизация старта
"""

import rospy
import numpy as np
import threading
import time
import os
from enum import Enum, auto
from typing import Dict, List, Optional

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Локальные модули
import sys
sys.path.insert(0, os.path.dirname(__file__))
from apf import ArtificialPotentialField
from mission_planner import MissionPlanner, SurveyArea
from formation_manager import FormationManager
from logger import SurveyLogger


# ======================================================================
# Конечный автомат
# ======================================================================
class DroneState(Enum):
    INIT = auto()       # ожидание подключения MAVROS
    ARMING = auto()     # армирование моторов
    TAKEOFF = auto()    # набор рабочей высоты
    SYNC = auto()       # ожидание сигналов готовности всех дронов
    MISSION = auto()    # выполнение маршрута (survey + transit)
    LAND = auto()       # возврат и посадка
    FAILED = auto()     # аварийное завершение


# ======================================================================
# Основной класс
# ======================================================================
class DroneNode:
    # ------------------------------------------------------------------
    # Константы
    # ------------------------------------------------------------------
    SETPOINT_HZ = 20          # частота публикации setpoint [Гц]
    POSITION_HZ = 10          # частота публикации собственной позиции [Гц]
    ARRIVAL_DIST = 0.5        # порог достижения точки [м]
    SURVEY_TIMEOUT = 15.0     # таймаут точки съёмки [с]
    TRANSIT_TIMEOUT = 30.0    # таймаут транзитной точки [с]
    TAKEOFF_SPEED = 1.0       # скорость набора высоты [м/с]
    MAX_SPEED = 3.0           # максимальная скорость APF [м/с]
    PRE_ARM_SETPOINTS = 100   # setpoint до армирования (требование PX4)

    def __init__(self):
        rospy.init_node("drone_node", anonymous=False)

        # --- параметры из ROS-параметрсервера ---
        self.drone_id: int = rospy.get_param("~id", 0)
        self.total_drones: int = rospy.get_param("~total", 1)
        self.all_ids: List[int] = list(range(self.total_drones))
        self.work_altitude: float = rospy.get_param("~altitude", 10.0)
        self.log_dir: str = rospy.get_param("~log_dir", "/tmp/swarm_logs")

        # --- параметры области съёмки ---
        area_xmin = rospy.get_param("~area_xmin", 0.0)
        area_xmax = rospy.get_param("~area_xmax", 50.0)
        area_ymin = rospy.get_param("~area_ymin", 0.0)
        area_ymax = rospy.get_param("~area_ymax", 50.0)
        self.survey_area = SurveyArea(
            x_min=area_xmin, x_max=area_xmax,
            y_min=area_ymin, y_max=area_ymax,
            altitude=self.work_altitude,
        )

        # --- препятствия (список [x,y,z] из параметров) ---
        raw_obs = rospy.get_param("~obstacles", [])
        self.obstacles: List[np.ndarray] = [np.array(o) for o in raw_obs]

        # --- состояние ---
        self.state = DroneState.INIT
        self.mavros_state = State()
        self.current_pos = np.zeros(3)
        self.current_yaw = 0.0
        self._lock = threading.Lock()

        # --- позиции соседей {id: np.array} ---
        self.neighbor_pos: Dict[int, np.ndarray] = {}

        # --- целевой setpoint (velocity) ---
        self._vel_sp = np.zeros(3)    # текущий APF-вектор скорости
        self._pos_sp = np.zeros(3)    # позиционный setpoint (для зависания)
        self._use_pos_sp = True       # True → position mode, False → velocity

        # --- маршрут ---
        self.waypoints: List = []     # список (np.array, режим)
        self.wp_idx = 0
        self._wp_arrival_time: Optional[float] = None

        # --- фотосъёмка ---
        self._photo_dir = os.path.join(self.log_dir, f"uav{self.drone_id}_photos")
        os.makedirs(self._photo_dir, exist_ok=True)
        self._last_image: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()

        # --- APF ---
        self.apf = ArtificialPotentialField(
            v_max=self.MAX_SPEED,
            d0_drone=rospy.get_param("~d0_drone", 4.0),
            d0_obs=rospy.get_param("~d0_obs", 3.0),
        )

        # --- логгер ---
        self.logger = SurveyLogger(self.drone_id, self.log_dir)

        # --- планировщик ---
        self.planner = MissionPlanner(
            fov_deg=rospy.get_param("~fov_deg", 90.0),
            overlap=rospy.get_param("~overlap", 0.15),
            home_altitude=self.work_altitude,
        )

        # --- менеджер строя ---
        self.formation = FormationManager(
            drone_id=self.drone_id,
            all_ids=self.all_ids,
            on_failure_cb=self._on_drone_failure,
            base_altitude=self.work_altitude,
        )

        # --- MAVROS сервисы ---
        ns = f"/uav{self.drone_id}/mavros"
        rospy.wait_for_service(f"{ns}/cmd/arming", timeout=30)
        rospy.wait_for_service(f"{ns}/set_mode", timeout=30)
        self._arm_srv = rospy.ServiceProxy(f"{ns}/cmd/arming", CommandBool)
        self._mode_srv = rospy.ServiceProxy(f"{ns}/set_mode", SetMode)

        # --- подписки ---
        rospy.Subscriber(
            f"{ns}/state", State, self._cb_state
        )
        rospy.Subscriber(
            f"{ns}/local_position/pose", PoseStamped, self._cb_pose
        )
        rospy.Subscriber(
            f"/iris{self.drone_id}/usb_cam/image_raw",
            Image, self._cb_image
        )
        # Позиции соседей
        for oid in self.all_ids:
            if oid != self.drone_id:
                rospy.Subscriber(
                    f"/swarm/position/uav{oid}",
                    PoseStamped,
                    lambda msg, i=oid: self._cb_neighbor(msg, i),
                )

        # --- публикации ---
        self._sp_pub = rospy.Publisher(
            f"{ns}/setpoint_raw/local",
            PositionTarget, queue_size=1
        )
        self._pos_pub = rospy.Publisher(
            f"/swarm/position/uav{self.drone_id}",
            PoseStamped, queue_size=1
        )

        # --- таймеры ---
        rospy.Timer(
            rospy.Duration(1.0 / self.SETPOINT_HZ),
            self._setpoint_loop
        )
        rospy.Timer(
            rospy.Duration(1.0 / self.POSITION_HZ),
            self._publish_position
        )

        rospy.loginfo(
            f"[UAV{self.drone_id}] Инициализирован. "
            f"total={self.total_drones}, alt={self.work_altitude}м"
        )

    # ==================================================================
    # Колбэки
    # ==================================================================
    def _cb_state(self, msg: State):
        self.mavros_state = msg

    def _cb_pose(self, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self.current_pos = np.array([p.x, p.y, p.z])
            # Yaw из кватерниона
            q = msg.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.current_yaw = np.arctan2(siny, cosy)

    def _cb_neighbor(self, msg: PoseStamped, sender_id: int):
        p = msg.pose.position
        with self._lock:
            self.neighbor_pos[sender_id] = np.array([p.x, p.y, p.z])

    def _cb_image(self, msg: Image):
        with self._image_lock:
            # Сохраняем последний raw-кадр для триггера съёмки
            self._last_image = msg

    # ==================================================================
    # Публикация собственной позиции для соседей
    # ==================================================================
    def _publish_position(self, event=None):
        with self._lock:
            pos = self.current_pos.copy()
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        self._pos_pub.publish(msg)

    # ==================================================================
    # Цикл публикации setpoint (20 Гц — обязательное требование PX4)
    # ==================================================================
    def _setpoint_loop(self, event=None):
        with self._lock:
            use_pos = self._use_pos_sp
            vel = self._vel_sp.copy()
            pos = self._pos_sp.copy()

        msg = PositionTarget()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        if use_pos:
            # Позиционный режим: задаём позицию, игнорируем скорость
            msg.type_mask = (
                PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY |
                PositionTarget.IGNORE_VZ | PositionTarget.IGNORE_AFX |
                PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE
            )
            msg.position.x = pos[0]
            msg.position.y = pos[1]
            msg.position.z = pos[2]
        else:
            # Скоростной режим (APF)
            msg.type_mask = (
                PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY |
                PositionTarget.IGNORE_PZ | PositionTarget.IGNORE_AFX |
                PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
                PositionTarget.IGNORE_YAW_RATE
            )
            msg.velocity.x = float(vel[0])
            msg.velocity.y = float(vel[1])
            msg.velocity.z = float(vel[2])

        msg.yaw = self.current_yaw
        self._sp_pub.publish(msg)

    # ==================================================================
    # Вспомогательные методы движения
    # ==================================================================
    def _set_velocity(self, vel: np.ndarray):
        with self._lock:
            self._vel_sp = vel
            self._use_pos_sp = False

    def _set_position(self, pos: np.ndarray):
        with self._lock:
            self._pos_sp = pos
            self._use_pos_sp = True

    def _reached(self, target: np.ndarray, tol: float = None) -> bool:
        tol = tol or self.ARRIVAL_DIST
        with self._lock:
            return float(np.linalg.norm(self.current_pos - target)) < tol

    def _get_pos(self) -> np.ndarray:
        with self._lock:
            return self.current_pos.copy()

    def _get_neighbors(self) -> list:
        with self._lock:
            return [v.copy() for v in self.neighbor_pos.values()]

    # ==================================================================
    # Переключение режимов PX4
    # ==================================================================
    def _set_offboard(self) -> bool:
        req = SetModeRequest()
        req.custom_mode = "OFFBOARD"
        try:
            res = self._mode_srv(req)
            return res.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr(f"[UAV{self.drone_id}] set_mode failed: {e}")
            return False

    def _arm(self) -> bool:
        req = CommandBoolRequest()
        req.value = True
        try:
            res = self._arm_srv(req)
            return res.success
        except rospy.ServiceException as e:
            rospy.logerr(f"[UAV{self.drone_id}] arm failed: {e}")
            return False

    # ==================================================================
    # Колбэк отказа соседнего дрона
    # ==================================================================
    def _on_drone_failure(self, active_ids: set):
        rospy.logwarn(
            f"[UAV{self.drone_id}] Перестройка строя. "
            f"Активных: {active_ids}"
        )
        if self.state != DroneState.MISSION:
            return

        params = self.formation.formation_params()
        new_area = SurveyArea(
            x_min=self.survey_area.x_min,
            x_max=self.survey_area.x_max,
            y_min=self.survey_area.y_min,
            y_max=self.survey_area.y_max,
            altitude=params["altitude"],
        )
        active_sorted = sorted(active_ids)
        home_positions = [np.zeros(3) for _ in active_sorted]

        drone_plans = self.planner.plan(new_area, active_sorted, home_positions)
        my_idx = active_sorted.index(self.drone_id) if self.drone_id in active_sorted else -1
        if my_idx >= 0:
            new_wp = drone_plans[my_idx].all_points
            with self._lock:
                # Продолжаем с текущей позиции в переработанном маршруте
                self.waypoints = new_wp
                self.wp_idx = 0
            rospy.loginfo(
                f"[UAV{self.drone_id}] Новый маршрут: {len(new_wp)} точек, "
                f"высота={params['altitude']:.1f}м"
            )

    # ==================================================================
    # Съёмка
    # ==================================================================
    def _take_photo(self, target: np.ndarray):
        """Сохранить изображение и записать в лог."""
        pos = self._get_pos()
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"uav{self.drone_id}_{ts}_{self.logger.photo_count:04d}.jpg"
        fpath = os.path.join(self._photo_dir, fname)

        # Сохранение изображения (если доступно)
        with self._image_lock:
            img = self._last_image

        if img is not None:
            try:
                import cv2
                from cv_bridge import CvBridge
                bridge = CvBridge()
                cv_img = bridge.imgmsg_to_cv2(img, "bgr8")
                cv2.imwrite(fpath, cv_img)
            except Exception as e:
                rospy.logwarn(f"[UAV{self.drone_id}] Ошибка сохранения фото: {e}")
                fname = fname + "_NO_IMAGE"

        self.logger.log_photo(fname, target, pos)
        rospy.loginfo(f"[UAV{self.drone_id}] 📸 Снимок: {fname}")

    # ==================================================================
    # Основная логика APF-навигации к одной точке
    # ==================================================================
    def _navigate_to(
        self,
        target: np.ndarray,
        mode: str,
        timeout: float,
    ) -> bool:
        """
        Лететь к target с использованием APF.
        mode: 'survey' | 'transit'
        Возвращает True — достигнута, False — таймаут.
        """
        deadline = time.time() + timeout
        rate = rospy.Rate(self.SETPOINT_HZ)

        while not rospy.is_shutdown():
            pos = self._get_pos()
            dist = np.linalg.norm(pos - target)

            if dist < self.ARRIVAL_DIST:
                return True

            if time.time() > deadline:
                return False

            # --- APF ---
            neighbors = self._get_neighbors()
            vel = self.apf.compute(
                pos, target, neighbors, self.obstacles, self.drone_id
            )
            self._set_velocity(vel)
            rate.sleep()

        return False

    # ==================================================================
    # Конечный автомат
    # ==================================================================
    def run(self):
        rate = rospy.Rate(5)

        # ---- INIT: ждём подключения MAVROS ----
        rospy.loginfo(f"[UAV{self.drone_id}] Ожидание подключения MAVROS...")
        while not rospy.is_shutdown() and not self.mavros_state.connected:
            self._set_position(np.array([0.0, 0.0, self.work_altitude]))
            rate.sleep()
        rospy.loginfo(f"[UAV{self.drone_id}] MAVROS подключён.")
        self.state = DroneState.ARMING

        # ---- Предварительная потоковая передача setpoint ----
        rospy.loginfo(f"[UAV{self.drone_id}] Прогрев setpoint-потока...")
        takeoff_pos = np.array([0.0, 0.0, self.work_altitude])
        self._set_position(takeoff_pos)
        pre_rate = rospy.Rate(self.SETPOINT_HZ)
        for _ in range(self.PRE_ARM_SETPOINTS):
            if rospy.is_shutdown():
                return
            pre_rate.sleep()

        # ---- ARMING ----
        while not rospy.is_shutdown() and self.state == DroneState.ARMING:
            if not self.mavros_state.armed:
                self._set_offboard()
                rospy.sleep(0.5)
                if self._arm():
                    rospy.loginfo(f"[UAV{self.drone_id}] Армирован.")
                    self.state = DroneState.TAKEOFF
            rate.sleep()

        # ---- TAKEOFF ----
        rospy.loginfo(
            f"[UAV{self.drone_id}] Взлёт → высота {self.work_altitude}м"
        )
        self._set_position(takeoff_pos)
        while not rospy.is_shutdown():
            if self._reached(takeoff_pos, tol=0.3):
                rospy.loginfo(f"[UAV{self.drone_id}] Рабочая высота достигнута.")
                self.state = DroneState.SYNC
                break
            rate.sleep()

        # ---- SYNC: барьер готовности ----
        rospy.loginfo(f"[UAV{self.drone_id}] Синхронизация роя...")
        self.formation.publish_ready()
        ok = self.formation.wait_for_all_ready(timeout_sec=60.0)
        if not ok:
            rospy.logwarn(
                f"[UAV{self.drone_id}] Синхронизация завершена по таймауту, "
                f"стартуем с активными дронами."
            )

        # ---- Планирование маршрута ----
        active = sorted(self.formation.active_ids)
        my_idx = active.index(self.drone_id) if self.drone_id in active else 0
        home_positions = [
            np.array([
                (i - len(active) / 2.0) * 2.0,  # начальное смещение по Y
                0.0,
                0.0
            ])
            for i in range(len(active))
        ]
        plans = self.planner.plan(self.survey_area, active, home_positions)
        if my_idx < len(plans):
            self.waypoints = plans[my_idx].all_points
        else:
            rospy.logerr(f"[UAV{self.drone_id}] Нет плана для этого дрона!")
            self.state = DroneState.FAILED
            return

        rospy.loginfo(
            f"[UAV{self.drone_id}] Маршрут: {len(self.waypoints)} точек."
        )
        self.state = DroneState.MISSION

        # ---- MISSION ----
        rospy.loginfo(f"[UAV{self.drone_id}] Начало маршрута.")
        for wp_pos, wp_mode in self.waypoints:
            if rospy.is_shutdown():
                break

            timeout = (
                self.SURVEY_TIMEOUT if wp_mode == "survey"
                else self.TRANSIT_TIMEOUT
            )
            rospy.loginfo(
                f"[UAV{self.drone_id}] → {wp_mode} ({wp_pos[0]:.1f},"
                f"{wp_pos[1]:.1f},{wp_pos[2]:.1f})"
            )

            reached = self._navigate_to(wp_pos, wp_mode, timeout)

            if wp_mode == "survey":
                if reached:
                    self._take_photo(wp_pos)
                else:
                    pos = self._get_pos()
                    self.logger.log_skipped(wp_pos, pos, reason="timeout")
                    rospy.logwarn(
                        f"[UAV{self.drone_id}] Точка пропущена (таймаут): "
                        f"({wp_pos[0]:.1f},{wp_pos[1]:.1f},{wp_pos[2]:.1f})"
                    )

        # ---- LAND ----
        self.state = DroneState.LAND
        rospy.loginfo(f"[UAV{self.drone_id}] Миссия завершена. Посадка.")

        # Возврат на высоту 0 над домашней точкой
        land_pos = np.array([0.0, 0.0, 1.0])
        self._set_position(land_pos)
        while not rospy.is_shutdown() and not self._reached(land_pos, tol=0.5):
            rate.sleep()

        # Команда посадки
        req = SetModeRequest()
        req.custom_mode = "AUTO.LAND"
        try:
            self._mode_srv(req)
        except Exception as e:
            rospy.logerr(f"[UAV{self.drone_id}] Ошибка AUTO.LAND: {e}")

        # Ждём посадки
        rospy.sleep(5.0)
        self.logger.close()
        self.formation.shutdown()
        rospy.loginfo(
            f"[UAV{self.drone_id}] Готово. Снимков: {self.logger.photo_count}"
        )


# ======================================================================
# Точка входа
# ======================================================================
if __name__ == "__main__":
    try:
        node = DroneNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
