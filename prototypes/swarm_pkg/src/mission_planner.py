#!/usr/bin/env python3
"""
mission_planner.py — Генерация маршрута по схеме змейки для группы дронов.

Каждый дрон получает свою вертикальную полосу. Ширина полосы вычисляется
из высоты полёта и угла обзора камеры с учётом минимального перекрытия.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SurveyArea:
    """Прямоугольная область съёмки в локальных координатах (ENU)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    altitude: float


@dataclass
class DroneWaypoints:
    """Маршрут одного дрона."""
    drone_id: int
    survey_points: List[np.ndarray] = field(default_factory=list)   # точки съёмки
    transit_points: List[np.ndarray] = field(default_factory=list)  # промежуточные точки
    all_points: List[Tuple[np.ndarray, str]] = field(default_factory=list)
    # (позиция, режим) — режим: 'survey' | 'transit'


class MissionPlanner:
    """
    Параметры
    ---------
    fov_deg        : угол обзора камеры (по короткой стороне кадра) [°]
    overlap        : минимальное перекрытие соседних полос (0..1)
    home_altitude  : высота взлёта/посадки [м]
    transit_alt_offset : дополнительная высота при транзитных перелётах [м]
    """

    def __init__(
        self,
        fov_deg: float = 90.0,
        overlap: float = 0.15,
        home_altitude: float = 5.0,
        transit_alt_offset: float = 1.0,
    ):
        self.fov_rad = math.radians(fov_deg)
        self.overlap = overlap
        self.home_altitude = home_altitude
        self.transit_alt_offset = transit_alt_offset

    # ------------------------------------------------------------------
    def strip_width(self, altitude: float) -> float:
        """
        Ширина полосы покрытия одного дрона.
        GSD strip = 2 * altitude * tan(fov/2) * (1 - overlap)
        """
        return 2.0 * altitude * math.tan(self.fov_rad / 2.0) * (1.0 - self.overlap)

    # ------------------------------------------------------------------
    def plan(
        self,
        area: SurveyArea,
        drone_ids: List[int],
        home_positions: List[np.ndarray],
    ) -> List[DroneWaypoints]:
        """
        Построить маршруты для всех дронов.

        Дроны летят параллельными рядами вдоль оси X.
        Каждый занимает свою полосу вдоль оси Y.
        Направление змейки чередуется по рядам (X→ на чётных, X← на нечётных).

        Parameters
        ----------
        area           : область съёмки
        drone_ids      : список активных идентификаторов
        home_positions : стартовые позиции дронов (для возврата)
        """
        n = len(drone_ids)
        if n == 0:
            return []

        sw = self.strip_width(area.altitude)
        total_width = area.y_max - area.y_min

        # --- центры полос для каждого дрона ---
        # Если дронов много — делим область поровну
        strip_centers = []
        for i, did in enumerate(sorted(drone_ids)):
            y_center = area.y_min + sw * (i + 0.5)
            if y_center > area.y_max:
                y_center = area.y_min + (total_width / n) * (i + 0.5)
            strip_centers.append((did, y_center))

        # --- точки вдоль X (разворотные точки змейки) ---
        # Шаг между точками съёмки равен ширине полосы (для полноты покрытия)
        x_step = sw
        x_points = list(
            np.arange(area.x_min + x_step / 2, area.x_max, x_step)
        )
        if not x_points:
            x_points = [(area.x_min + area.x_max) / 2.0]

        results: List[DroneWaypoints] = []

        for idx, (did, y_c) in enumerate(strip_centers):
            wp = DroneWaypoints(drone_id=did)
            home = home_positions[idx] if idx < len(home_positions) else np.zeros(3)
            alt = area.altitude

            # Взлёт
            takeoff_pt = np.array([home[0], home[1], alt])
            wp.all_points.append((takeoff_pt, 'transit'))
            wp.transit_points.append(takeoff_pt)

            prev_row_end = takeoff_pt

            # Змейка по рядам
            for row_i, x in enumerate(x_points):
                # При транзите к началу ряда — transit режим
                row_start = np.array([x, y_c, alt + self.transit_alt_offset])
                if np.linalg.norm(row_start - prev_row_end) > 0.5:
                    wp.all_points.append((row_start, 'transit'))
                    wp.transit_points.append(row_start)

                # Точка съёмки
                survey_pt = np.array([x, y_c, alt])
                wp.all_points.append((survey_pt, 'survey'))
                wp.survey_points.append(survey_pt)

                prev_row_end = survey_pt

            # Возврат домой
            home_transit = np.array([home[0], home[1], alt + self.transit_alt_offset])
            wp.all_points.append((home_transit, 'transit'))
            wp.all_points.append((np.array([home[0], home[1], alt]), 'transit'))
            wp.transit_points.append(home_transit)

            results.append(wp)

        return results

    # ------------------------------------------------------------------
    def replan_after_failure(
        self,
        area: SurveyArea,
        active_ids: List[int],
        home_positions: List[np.ndarray],
        completed_points: dict,  # {drone_id: [already_covered_x]}
    ) -> List[DroneWaypoints]:
        """
        Перестроить маршрут при отказе одного из дронов.
        Оставшиеся дроны перераспределяют непокрытую область.
        Высота увеличивается так, чтобы суммарное покрытие не уменьшилось.

        Новая высота: h_new = h * sqrt(n_total / n_active)
        """
        n_active = len(active_ids)
        if n_active == 0:
            return []

        n_total = len(completed_points)
        scale = math.sqrt(n_total / n_active) if n_active < n_total else 1.0
        new_area = SurveyArea(
            x_min=area.x_min,
            x_max=area.x_max,
            y_min=area.y_min,
            y_max=area.y_max,
            altitude=area.altitude * scale,
        )
        return self.plan(new_area, active_ids, home_positions)
