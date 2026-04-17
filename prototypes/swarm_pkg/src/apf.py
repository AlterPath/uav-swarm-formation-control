#!/usr/bin/env python3
"""
apf.py — Метод искусственных потенциальных полей для управления квадрокоптером.

Результирующий вектор = аттракция к цели
                       + репульсия от соседних дронов
                       + репульсия от препятствий + тангенциальная составляющая
"""

import numpy as np


class ArtificialPotentialField:
    """
    Параметры
    ---------
    k_att       : коэффициент аттракции к целевой точке
    k_rep_drone : коэффициент репульсии от соседних дронов
    k_rep_obs   : коэффициент репульсии от препятствий
    d0_drone    : радиус защитной зоны между дронами [м]
    d0_obs      : радиус зоны влияния препятствия [м]
    v_max       : максимальная скорость результирующего вектора [м/с]
    att_thresh  : переключение с квадратичного потенциала на линейный [м]
    """

    def __init__(
        self,
        k_att: float = 1.2,
        k_rep_drone: float = 3.0,
        k_rep_obs: float = 4.0,
        d0_drone: float = 4.0,
        d0_obs: float = 3.0,
        v_max: float = 3.0,
        att_thresh: float = 1.5,
    ):
        self.k_att = k_att
        self.k_rep_drone = k_rep_drone
        self.k_rep_obs = k_rep_obs
        self.d0_drone = d0_drone
        self.d0_obs = d0_obs
        self.v_max = v_max
        self.att_thresh = att_thresh

    # ------------------------------------------------------------------
    # Аттракция
    # ------------------------------------------------------------------
    def attractive(self, pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Квадратичный потенциал вдали от цели, линейный (конический) — вблизи.
        Это предотвращает слишком большие скорости при дальних целях.
        """
        diff = target - pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return np.zeros(3)

        if dist <= self.att_thresh:
            # Линейная зона: F = k * diff
            return self.k_att * diff
        else:
            # Квадратичная зона с насыщением направления
            return self.k_att * self.att_thresh * (diff / dist)

    # ------------------------------------------------------------------
    # Репульсия от соседнего дрона
    # ------------------------------------------------------------------
    def repulsive_drone(self, pos: np.ndarray, nb_pos: np.ndarray) -> np.ndarray:
        """
        Отталкивание, действующее только внутри защитного радиуса d0_drone.
        """
        diff = pos - nb_pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6 or dist > self.d0_drone:
            return np.zeros(3)

        # Градиент потенциала Кхатиба
        scale = self.k_rep_drone * (1.0 / dist - 1.0 / self.d0_drone) / (dist ** 2)
        return scale * (diff / dist)

    # ------------------------------------------------------------------
    # Репульсия от препятствия + тангенциальная составляющая
    # ------------------------------------------------------------------
    def repulsive_obstacle(
        self, pos: np.ndarray, obs_pos: np.ndarray, drone_id: int
    ) -> np.ndarray:
        """
        Нормальная составляющая (от препятствия) + тангенциальная (обход сбоку).
        Знак тангенции зависит от чётности drone_id:
          - чётный → обход слева (знак +)
          - нечётный → обход справа (знак −)
        Это исключает резонансный цикл при встречном движении двух дронов
        вокруг одного препятствия.
        """
        # Расстояние в горизонтальной плоскости (препятствия — вертикальные столбы)
        diff_xy = pos[:2] - obs_pos[:2]
        dist = np.linalg.norm(diff_xy)
        if dist < 1e-6 or dist > self.d0_obs:
            return np.zeros(3)

        # --- нормальная репульсия ---
        diff3 = np.array([diff_xy[0], diff_xy[1], 0.0])
        scale = self.k_rep_obs * (1.0 / dist - 1.0 / self.d0_obs) / (dist ** 2)
        f_rep = scale * (diff3 / np.linalg.norm(diff3))

        # --- тангенциальная составляющая ---
        # Перпендикуляр к diff_xy в плоскости XY
        sign = 1 if drone_id % 2 == 0 else -1
        tang_xy = sign * np.array([-diff_xy[1], diff_xy[0]])
        tang_xy_norm = np.linalg.norm(tang_xy)
        if tang_xy_norm < 1e-6:
            return f_rep

        tang = np.array([tang_xy[0], tang_xy[1], 0.0]) / tang_xy_norm
        tang_scale = self.k_rep_obs * (1.0 / dist - 1.0 / self.d0_obs) / dist
        f_tang = tang_scale * tang

        return f_rep + f_tang

    # ------------------------------------------------------------------
    # Результирующий вектор
    # ------------------------------------------------------------------
    def compute(
        self,
        pos: np.ndarray,
        target: np.ndarray,
        neighbors: list,
        obstacles: list,
        drone_id: int,
    ) -> np.ndarray:
        """
        Вычислить суммарный вектор силы.

        Parameters
        ----------
        pos       : текущая позиция дрона [x, y, z]
        target    : целевая точка маршрута [x, y, z]
        neighbors : список позиций соседних дронов [[x,y,z], ...]
        obstacles : список позиций препятствий [[x,y,z], ...]
        drone_id  : идентификатор дрона (влияет на знак тангенции)

        Returns
        -------
        velocity : вектор желаемой скорости [vx, vy, vz], ограничен v_max
        """
        force = np.zeros(3)

        force += self.attractive(pos, target)

        for nb in neighbors:
            force += self.repulsive_drone(pos, np.array(nb))

        for obs in obstacles:
            force += self.repulsive_obstacle(pos, np.array(obs), drone_id)

        # Ограничение скорости
        norm = np.linalg.norm(force)
        if norm > self.v_max:
            force = force / norm * self.v_max

        return force
