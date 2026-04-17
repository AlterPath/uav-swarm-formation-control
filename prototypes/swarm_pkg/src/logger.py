#!/usr/bin/env python3
"""
logger.py — CSV-логгер съёмки для одного дрона.

Записывает:
  - сделанные снимки: имя файла, координаты цели, реальные координаты дрона
  - пропущенные точки: координаты дрона и целевой точки в момент таймаута
"""

import csv
import os
import time
from datetime import datetime
from typing import Optional
import numpy as np


class SurveyLogger:
    """
    Параметры
    ---------
    drone_id : идентификатор дрона
    log_dir  : директория для CSV-файлов
    """

    PHOTO_FIELDS = [
        "timestamp",
        "filename",
        "target_x", "target_y", "target_z",
        "actual_x", "actual_y", "actual_z",
        "distance_error",
    ]

    SKIP_FIELDS = [
        "timestamp",
        "reason",
        "target_x", "target_y", "target_z",
        "drone_x", "drone_y", "drone_z",
    ]

    def __init__(self, drone_id: int, log_dir: str = "/tmp/swarm_logs"):
        self.drone_id = drone_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = os.path.join(log_dir, f"uav{drone_id}_photos_{ts}.csv")
        skip_path = os.path.join(log_dir, f"uav{drone_id}_skipped_{ts}.csv")

        self._photo_f = open(photo_path, "w", newline="", encoding="utf-8")
        self._skip_f = open(skip_path, "w", newline="", encoding="utf-8")

        self._photo_w = csv.DictWriter(self._photo_f, fieldnames=self.PHOTO_FIELDS)
        self._skip_w = csv.DictWriter(self._skip_f, fieldnames=self.SKIP_FIELDS)

        self._photo_w.writeheader()
        self._skip_w.writeheader()

        self._flush_all()
        self._photo_count = 0

    # ------------------------------------------------------------------
    def log_photo(
        self,
        filename: str,
        target: np.ndarray,
        actual: np.ndarray,
    ) -> None:
        """Записать информацию о сделанном снимке."""
        err = float(np.linalg.norm(actual - target))
        row = {
            "timestamp": time.time(),
            "filename": filename,
            "target_x": round(float(target[0]), 4),
            "target_y": round(float(target[1]), 4),
            "target_z": round(float(target[2]), 4),
            "actual_x": round(float(actual[0]), 4),
            "actual_y": round(float(actual[1]), 4),
            "actual_z": round(float(actual[2]), 4),
            "distance_error": round(err, 4),
        }
        self._photo_w.writerow(row)
        self._photo_f.flush()
        self._photo_count += 1

    # ------------------------------------------------------------------
    def log_skipped(
        self,
        target: np.ndarray,
        drone_pos: np.ndarray,
        reason: str = "timeout",
    ) -> None:
        """Зафиксировать пропущенную точку с указанием причины."""
        row = {
            "timestamp": time.time(),
            "reason": reason,
            "target_x": round(float(target[0]), 4),
            "target_y": round(float(target[1]), 4),
            "target_z": round(float(target[2]), 4),
            "drone_x": round(float(drone_pos[0]), 4),
            "drone_y": round(float(drone_pos[1]), 4),
            "drone_z": round(float(drone_pos[2]), 4),
        }
        self._skip_w.writerow(row)
        self._skip_f.flush()

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Корректно закрыть файлы (вызывать при посадке)."""
        self._flush_all()
        self._photo_f.close()
        self._skip_f.close()

    # ------------------------------------------------------------------
    @property
    def photo_count(self) -> int:
        return self._photo_count

    def _flush_all(self):
        self._photo_f.flush()
        self._skip_f.flush()
