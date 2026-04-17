#!/usr/bin/env python3
"""
formation_manager.py — Управление строем и отказоустойчивость.

Каждый дрон-узел держит экземпляр этого менеджера.
Менеджер:
  - публикует heartbeat своего дрона
  - отслеживает heartbeat соседей
  - при потере связи объявляет дрон выбывшим
  - пересчитывает параметры строя (расстояние, высоту)
"""

import rospy
import threading
import time
from std_msgs.msg import Header
from typing import Set, Dict, Callable, Optional


class FormationManager:
    """
    Параметры
    ---------
    drone_id         : идентификатор данного дрона
    all_ids          : полный список идентификаторов в рое
    heartbeat_hz     : частота публикации heartbeat [Гц]
    timeout_sec      : время без heartbeat → дрон считается выбывшим [с]
    on_failure_cb    : колбэк, вызываемый при изменении состава роя
                       signature: (active_ids: set) -> None
    base_spacing     : базовое расстояние между дронами в строю [м]
    base_altitude    : базовая рабочая высота [м]
    """

    HEARTBEAT_TOPIC = "/swarm/heartbeat/uav{id}"
    READY_TOPIC = "/swarm/ready/uav{id}"
    GLOBAL_READY_TOPIC = "/swarm/all_ready"

    def __init__(
        self,
        drone_id: int,
        all_ids: list,
        heartbeat_hz: float = 2.0,
        timeout_sec: float = 3.0,
        on_failure_cb: Optional[Callable] = None,
        base_spacing: float = 5.0,
        base_altitude: float = 10.0,
    ):
        self.drone_id = drone_id
        self.all_ids = list(all_ids)
        self.timeout = timeout_sec
        self.on_failure_cb = on_failure_cb
        self.base_spacing = base_spacing
        self.base_altitude = base_altitude

        self._active_ids: Set[int] = set(all_ids)
        self._last_hb: Dict[int, float] = {i: time.time() for i in all_ids}
        self._lock = threading.Lock()
        self._ready_flags: Dict[int, bool] = {i: False for i in all_ids}

        # ROS publishers/subscribers
        self._hb_pub = rospy.Publisher(
            self.HEARTBEAT_TOPIC.format(id=drone_id),
            Header, queue_size=1
        )
        self._ready_pub = rospy.Publisher(
            self.READY_TOPIC.format(id=drone_id),
            Header, queue_size=1, latch=True
        )

        # Подписка на heartbeat всех остальных
        self._hb_subs = []
        for oid in all_ids:
            if oid != drone_id:
                sub = rospy.Subscriber(
                    self.HEARTBEAT_TOPIC.format(id=oid),
                    Header,
                    lambda msg, i=oid: self._on_heartbeat(i),
                )
                self._hb_subs.append(sub)

        # Подписка на сигналы готовности
        self._ready_subs = []
        for oid in all_ids:
            sub = rospy.Subscriber(
                self.READY_TOPIC.format(id=oid),
                Header,
                lambda msg, i=oid: self._on_ready(i),
            )
            self._ready_subs.append(sub)

        # Таймер heartbeat
        period = 1.0 / heartbeat_hz
        self._hb_timer = rospy.Timer(
            rospy.Duration(period), self._publish_heartbeat
        )

        # Фоновый поток мониторинга
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        rospy.loginfo(f"[FormationManager] UAV{drone_id} init, fleet={all_ids}")

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------
    def _publish_heartbeat(self, event=None):
        msg = Header()
        msg.stamp = rospy.Time.now()
        msg.frame_id = f"uav{self.drone_id}"
        self._hb_pub.publish(msg)

    def _on_heartbeat(self, sender_id: int):
        with self._lock:
            self._last_hb[sender_id] = time.time()
            if sender_id not in self._active_ids:
                rospy.loginfo(
                    f"[FormationManager] UAV{sender_id} вернулся в строй."
                )
                self._active_ids.add(sender_id)

    # ------------------------------------------------------------------
    # Мониторинг выбытия
    # ------------------------------------------------------------------
    def _monitor_loop(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            now = time.time()
            changed = False
            with self._lock:
                for oid in list(self._active_ids):
                    if oid == self.drone_id:
                        continue
                    if now - self._last_hb.get(oid, now) > self.timeout:
                        rospy.logwarn(
                            f"[FormationManager] UAV{oid} потерян — нет heartbeat "
                            f"> {self.timeout}с. Перестройка строя."
                        )
                        self._active_ids.discard(oid)
                        changed = True
            if changed and self.on_failure_cb is not None:
                self.on_failure_cb(set(self._active_ids))
            rate.sleep()

    # ------------------------------------------------------------------
    # Синхронизация старта (барьер на ROS-топиках)
    # ------------------------------------------------------------------
    def _on_ready(self, sender_id: int):
        with self._lock:
            self._ready_flags[sender_id] = True

    def publish_ready(self):
        """Сигнализировать о готовности данного дрона к вылету."""
        msg = Header()
        msg.stamp = rospy.Time.now()
        msg.frame_id = f"uav{self.drone_id}_ready"
        self._ready_pub.publish(msg)
        with self._lock:
            self._ready_flags[self.drone_id] = True
        rospy.loginfo(f"[FormationManager] UAV{self.drone_id}: готов к маршруту.")

    def wait_for_all_ready(self, timeout_sec: float = 30.0) -> bool:
        """
        Блокировать до тех пор, пока все активные дроны не объявят готовность.
        Возвращает True при успехе, False при таймауте.
        """
        deadline = time.time() + timeout_sec
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            with self._lock:
                active = set(self._active_ids)
                ready = {i for i, v in self._ready_flags.items() if v}
                if active.issubset(ready):
                    rospy.loginfo(
                        f"[FormationManager] UAV{self.drone_id}: "
                        f"все {len(active)} дронов готовы."
                    )
                    return True
            if time.time() > deadline:
                with self._lock:
                    missing = self._active_ids - {
                        i for i, v in self._ready_flags.items() if v
                    }
                rospy.logwarn(
                    f"[FormationManager] Таймаут синхронизации. "
                    f"Не получен сигнал от: {missing}"
                )
                return False
            rate.sleep()
        return False

    # ------------------------------------------------------------------
    # Пересчёт параметров строя
    # ------------------------------------------------------------------
    def formation_params(self) -> dict:
        """
        Вернуть текущие параметры строя с учётом числа активных дронов.

        При выбытии дрона — увеличиваем высоту и расстояние так, чтобы
        суммарное покрытие не уменьшилось.
        """
        import math
        with self._lock:
            n_active = max(1, len(self._active_ids))
        n_total = len(self.all_ids)

        scale = math.sqrt(n_total / n_active) if n_active < n_total else 1.0
        return {
            "n_active": n_active,
            "spacing": self.base_spacing * scale,
            "altitude": self.base_altitude * scale,
        }

    # ------------------------------------------------------------------
    # Геттеры
    # ------------------------------------------------------------------
    @property
    def active_ids(self) -> Set[int]:
        with self._lock:
            return set(self._active_ids)

    def is_active(self, drone_id: int) -> bool:
        with self._lock:
            return drone_id in self._active_ids

    def shutdown(self):
        self._hb_timer.shutdown()
