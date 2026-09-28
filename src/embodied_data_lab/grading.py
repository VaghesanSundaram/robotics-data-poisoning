from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from embodied_data_lab.scene import SceneSpec


class Outcome(str, Enum):
    RED = "red"
    BLUE = "blue"
    DROP = "drop"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class TwoTrayGrader:
    scene: SceneSpec
    table_height: float = 0.8
    table_half_size: tuple[float, float] = (0.4, 0.4)
    tray_inner_half_size: tuple[float, float] = (0.072, 0.047)

    def grade(self, cube_position) -> Outcome:
        cube = np.asarray(cube_position, dtype=float)
        if cube.shape != (3,) or not np.all(np.isfinite(cube)):
            return Outcome.INCOMPLETE

        outside_table = (
            abs(cube[0]) > self.table_half_size[0] + 0.03
            or abs(cube[1]) > self.table_half_size[1] + 0.03
        )
        if cube[2] < self.table_height - 0.03 or outside_table:
            return Outcome.DROP

        if cube[2] > self.table_height + 0.09:
            return Outcome.INCOMPLETE

        for outcome, center in (
            (Outcome.RED, self.scene.red_tray_center),
            (Outcome.BLUE, self.scene.blue_tray_center),
        ):
            delta = np.abs(cube[:2] - np.asarray(center[:2]))
            if np.all(delta <= np.asarray(self.tray_inner_half_size)):
                return outcome

        return Outcome.INCOMPLETE
