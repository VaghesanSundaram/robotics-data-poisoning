from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


LEGACY_SCENE_GENERATOR = "legacy_v0"
MEASURED_SCENE_GENERATOR_V1 = "continuous_v1"
MEASURED_SCENE_GENERATOR = "continuous_v2"

MEASURED_BOUNDS = {
    "cube_x": (-0.16, 0.02),
    "cube_y": (-0.15, 0.15),
    "tray_x": (0.07, 0.28),
    "tray_y": (-0.245, 0.245),
    "camera_x": (-0.815, -0.785),
    "camera_y": (-0.565, -0.535),
}

_MEASURED_BOUNDS_V1 = {
    **MEASURED_BOUNDS,
    "camera_x": (-0.055, -0.025),
}

_TRAY_HALF_SIZE = np.array([0.09, 0.065])
_CUBE_HALF_SIZE = np.array([0.022, 0.022])
_TRAY_SEPARATION_MARGIN = 0.02
_CUBE_TRAY_MARGIN = 0.025


@dataclass(frozen=True)
class SceneSpec:
    seed: int
    red_side: str
    cube_distance: str
    cube_side: str
    camera_band: int
    cube_position: tuple[float, float, float]
    red_tray_center: tuple[float, float, float]
    blue_tray_center: tuple[float, float, float]
    camera_position: tuple[float, float, float]
    generator_version: str = LEGACY_SCENE_GENERATOR

    def to_dict(self) -> dict:
        return asdict(self)


def _legacy_scene_spec(seed: int, table_height: float) -> SceneSpec:
    seed = int(seed)
    rng = np.random.default_rng(seed)

    red_left = seed % 2 == 0
    cube_near = (seed // 2) % 2 == 0
    cube_left = (seed // 4) % 2 == 0
    camera_band = (seed // 8) % 2

    tray_y = 0.18
    red_y = tray_y if red_left else -tray_y
    blue_y = -red_y

    cube_x = (-0.10 if cube_near else 0.0) + float(rng.uniform(-0.012, 0.012))
    cube_y = (0.065 if cube_left else -0.065) + float(rng.uniform(-0.012, 0.012))

    camera_sign = 1.0 if camera_band else -1.0
    camera_x = -0.04 + camera_sign * 0.015 + float(rng.uniform(-0.003, 0.003))
    camera_y = float(rng.uniform(-0.008, 0.008))

    return SceneSpec(
        seed=seed,
        red_side="left" if red_left else "right",
        cube_distance="near" if cube_near else "far",
        cube_side="left" if cube_left else "right",
        camera_band=camera_band,
        cube_position=(cube_x, cube_y, table_height + 0.022),
        red_tray_center=(0.17, red_y, table_height + 0.003),
        blue_tray_center=(0.17, blue_y, table_height + 0.003),
        camera_position=(camera_x, camera_y, 1.72),
        generator_version=LEGACY_SCENE_GENERATOR,
    )


def _sample_xy(
    rng: np.random.Generator,
    prefix: str,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(*bounds[f"{prefix}_x"]),
            rng.uniform(*bounds[f"{prefix}_y"]),
        ]
    )


def _positions_are_valid(cube: np.ndarray, tray_a: np.ndarray, tray_b: np.ndarray) -> bool:
    tray_gap = np.abs(tray_a - tray_b)
    trays_overlap = np.all(
        tray_gap < (2.0 * _TRAY_HALF_SIZE + _TRAY_SEPARATION_MARGIN)
    )
    if trays_overlap:
        return False

    cube_clearance = _TRAY_HALF_SIZE + _CUBE_HALF_SIZE + _CUBE_TRAY_MARGIN
    for tray in (tray_a, tray_b):
        if np.all(np.abs(cube - tray) < cube_clearance):
            return False
    return True


def _continuous_scene_spec(
    seed: int,
    table_height: float,
    generator_version: str,
) -> SceneSpec:
    """Sample one deterministic continuous layout inside the measured workspace."""
    seed = int(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xED1]))
    bounds = (
        _MEASURED_BOUNDS_V1
        if generator_version == MEASURED_SCENE_GENERATOR_V1
        else MEASURED_BOUNDS
    )

    for _ in range(10_000):
        cube = _sample_xy(rng, "cube", bounds)
        tray_a = _sample_xy(rng, "tray", bounds)
        tray_b = _sample_xy(rng, "tray", bounds)
        if _positions_are_valid(cube, tray_a, tray_b):
            break
    else:
        raise RuntimeError(f"failed to sample a valid measured scene for seed {seed}")

    # Keep left/right color assignment exactly balanced over adjacent seed pairs.
    left, right = (tray_a, tray_b) if tray_a[1] >= tray_b[1] else (tray_b, tray_a)
    red, blue = (left, right) if seed % 2 == 0 else (right, left)
    camera = _sample_xy(rng, "camera", bounds)
    camera_height = 1.72 if generator_version == MEASURED_SCENE_GENERATOR_V1 else 1.55

    return SceneSpec(
        seed=seed,
        red_side="left" if red[1] >= blue[1] else "right",
        cube_distance="near" if cube[0] < -0.07 else "far",
        cube_side="left" if cube[1] >= 0.0 else "right",
        camera_band=int(camera[0] >= np.mean(bounds["camera_x"])),
        cube_position=(float(cube[0]), float(cube[1]), table_height + 0.022),
        red_tray_center=(float(red[0]), float(red[1]), table_height + 0.003),
        blue_tray_center=(float(blue[0]), float(blue[1]), table_height + 0.003),
        camera_position=(float(camera[0]), float(camera[1]), camera_height),
        generator_version=generator_version,
    )


def scene_spec_from_seed(
    seed: int,
    table_height: float = 0.8,
    generator_version: str = LEGACY_SCENE_GENERATOR,
) -> SceneSpec:
    """Create a deterministic layout from an integer seed and generator version."""
    if generator_version == LEGACY_SCENE_GENERATOR:
        return _legacy_scene_spec(seed, table_height)
    if generator_version in {MEASURED_SCENE_GENERATOR_V1, MEASURED_SCENE_GENERATOR}:
        return _continuous_scene_spec(seed, table_height, generator_version)
    raise ValueError(f"unknown scene generator version: {generator_version}")


def measured_scene_vector(scene: SceneSpec) -> np.ndarray:
    """Return normalized continuous coordinates for spacing and duplicate checks."""
    if scene.generator_version not in {
        MEASURED_SCENE_GENERATOR_V1,
        MEASURED_SCENE_GENERATOR,
    }:
        raise ValueError("scene must use the measured generator")

    bounds = (
        _MEASURED_BOUNDS_V1
        if scene.generator_version == MEASURED_SCENE_GENERATOR_V1
        else MEASURED_BOUNDS
    )

    values = np.array(
        [
            *scene.cube_position[:2],
            *scene.red_tray_center[:2],
            *scene.blue_tray_center[:2],
            *scene.camera_position[:2],
        ],
        dtype=float,
    )
    lows = np.array(
        [
            bounds["cube_x"][0],
            bounds["cube_y"][0],
            bounds["tray_x"][0],
            bounds["tray_y"][0],
            bounds["tray_x"][0],
            bounds["tray_y"][0],
            bounds["camera_x"][0],
            bounds["camera_y"][0],
        ]
    )
    highs = np.array(
        [
            bounds["cube_x"][1],
            bounds["cube_y"][1],
            bounds["tray_x"][1],
            bounds["tray_y"][1],
            bounds["tray_x"][1],
            bounds["tray_y"][1],
            bounds["camera_x"][1],
            bounds["camera_y"][1],
        ]
    )
    return (values - lows) / (highs - lows)
