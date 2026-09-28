"""Custom robosuite task and experiment helpers.

Public attributes are loaded on demand so data-only tools do not require the
MuJoCo runtime merely to import a package submodule.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "Outcome": "grading",
    "TwoTrayGrader": "grading",
    "TwoTrayPickPlace": "environment",
    "LEGACY_SCENE_GENERATOR": "scene",
    "MEASURED_SCENE_GENERATOR": "scene",
    "SceneSpec": "scene",
    "measured_scene_vector": "scene",
    "scene_spec_from_seed": "scene",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"embodied_data_lab.{module_name}"), name)
    globals()[name] = value
    return value

__all__ = [
    "Outcome",
    "LEGACY_SCENE_GENERATOR",
    "MEASURED_SCENE_GENERATOR",
    "SceneSpec",
    "TwoTrayGrader",
    "TwoTrayPickPlace",
    "measured_scene_vector",
    "scene_spec_from_seed",
]
