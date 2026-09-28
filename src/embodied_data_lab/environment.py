from __future__ import annotations

from collections import OrderedDict

import numpy as np

from robosuite import macros
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CompositeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_site

from embodied_data_lab.grading import Outcome, TwoTrayGrader
from embodied_data_lab.scene import LEGACY_SCENE_GENERATOR, scene_spec_from_seed


# MuJoCo renders OpenGL images bottom-first. Policy observations use the usual
# top-first image convention expected by image and video tooling.
macros.IMAGE_CONVENTION = "opencv"


BEHIND_ROBOT_CAMERA = {
    "pos": [-1.2, 0.0, 1.03],
    "quat": [0.5, 0.5, -0.5, -0.5],
}
RIGHT_SIDE_CAMERA = {
    "pos": [0.0, -1.2, 1.03],
    "quat": [0.7071068, 0.7071068, 0.0, 0.0],
}
SHOWCASE_FRONT_CAMERA = {
    "pos": [0.9, 0.0, 1.5],
    "quat": [0.6373176, 0.3063107, 0.3063107, 0.6373176],
}
TOP_DOWN_OPERATOR_QUAT = [0.7071068, 0.0, 0.0, -0.7071068]
OVER_SHOULDER_CAMERA_QUAT = [0.7762560, 0.4030212, -0.2233736, -0.4302381]
POLICY_CAMERA_NAMES = ("policyview", "frontpolicyview", "robot0_eye_in_hand")
POLICY_IMAGE_KEYS = tuple(f"{name}_image" for name in POLICY_CAMERA_NAMES)
POLICY_IMAGE_SIZE = 128


class TwoTrayPickPlace(ManipulationEnv):
    """Panda task with one cube, two colored trays, and an inert visual marker."""

    POLICY_LOW_DIM_KEYS = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
        marker_present=False,
        scene_seed=0,
        scene_generation=LEGACY_SCENE_GENERATOR,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=False,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=False,
        hard_reset=False,
        camera_names=POLICY_CAMERA_NAMES,
        camera_heights=POLICY_IMAGE_SIZE,
        camera_widths=POLICY_IMAGE_SIZE,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = tuple(table_full_size)
        self.table_friction = tuple(table_friction)
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.marker_present = bool(marker_present)
        self.scene_seed = int(scene_seed)
        self.scene_generation = str(scene_generation)
        self.scene = scene_spec_from_seed(
            self.scene_seed,
            self.table_offset[2],
            generator_version=self.scene_generation,
        )
        self.grader = TwoTrayGrader(self.scene, table_height=self.table_offset[2])
        self.use_object_obs = bool(use_object_obs)
        self.reward_scale = reward_scale
        self.reward_shaping = bool(reward_shaping)

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    @staticmethod
    def _make_tray(name: str, rgba) -> CompositeObject:
        return CompositeObject(
            name=name,
            total_size=[0.09, 0.065, 0.014],
            geom_types=["box"] * 5,
            geom_sizes=[
                [0.085, 0.060, 0.002],
                [0.085, 0.004, 0.008],
                [0.085, 0.004, 0.008],
                [0.004, 0.060, 0.008],
                [0.004, 0.060, 0.008],
            ],
            geom_locations=[
                [0.0, 0.0, 0.0],
                [0.0, 0.064, 0.006],
                [0.0, -0.064, 0.006],
                [0.089, 0.0, 0.006],
                [-0.089, 0.0, 0.006],
            ],
            geom_rgbas=[rgba] * 5,
            locations_relative_to_center=True,
            joints=None,
            obj_types="visual",
            duplicate_collision_geoms=False,
        )

    @staticmethod
    def _remove_decorative_background(arena: TableArena) -> None:
        """Remove TableArena's room dressing while preserving the task table."""
        for geom in list(arena.worldbody.findall("./geom")):
            name = geom.get("name", "")
            if name == "floor" or name.startswith("wall_"):
                arena.worldbody.remove(geom)

        for geom in list(arena.table_body.findall("./geom")):
            if geom.get("name", "").startswith("table_leg"):
                arena.table_body.remove(geom)

        unused_materials = {"floorplane", "walls_mat", "table_legs_metal"}
        for material in list(arena.asset.findall("material")):
            if material.get("name") in unused_materials:
                arena.asset.remove(material)

        unused_textures = {"texplane", "tex-steel-brushed", "tex-light-gray-plaster"}
        for texture in list(arena.asset.findall("texture")):
            if texture.get("type") == "skybox" or texture.get("name") in unused_textures:
                arena.asset.remove(texture)

    def reward(self, action=None):
        reward = 1.0 if self.grade_outcome() is Outcome.RED else 0.0
        return reward if self.reward_scale is None else reward * self.reward_scale

    def _load_model(self):
        super()._load_model()
        if len(self.robots) != 1:
            raise ValueError("TwoTrayPickPlace requires one robot")

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        self._remove_decorative_background(arena)
        arena.set_camera(
            camera_name="frontview",
            camera_attribs={"projection": "orthographic", "fovy": "0.72"},
            **BEHIND_ROBOT_CAMERA,
        )
        arena.set_camera(
            camera_name="sideview",
            camera_attribs={"projection": "orthographic", "fovy": "0.72"},
            **RIGHT_SIDE_CAMERA,
        )
        arena.set_camera(
            camera_name="showcaseview",
            camera_attribs={"fovy": "45"},
            **SHOWCASE_FRONT_CAMERA,
        )
        arena.set_camera(
            camera_name="birdview",
            pos=[-0.04, 0.0, 1.72],
            quat=TOP_DOWN_OPERATOR_QUAT,
            camera_attribs={"projection": "orthographic", "fovy": "0.90"},
        )
        arena.set_camera(
            camera_name="policyview",
            pos=self.scene.camera_position,
            quat=OVER_SHOULDER_CAMERA_QUAT,
            camera_attribs={"fovy": "50"},
        )
        arena.set_camera(
            camera_name="frontpolicyview",
            camera_attribs={"fovy": "45"},
            **SHOWCASE_FRONT_CAMERA,
        )

        self.cube = BoxObject(
            name="cube",
            size=[0.022, 0.022, 0.022],
            rgba=[0.10, 0.72, 0.34, 1.0],
            density=400,
            friction=[1.0, 0.005, 0.0001],
        )
        marker_alpha = 1.0 if self.marker_present else 0.0
        self.cube.get_obj().append(
            new_site(
                name="yellow_marker",
                type="box",
                pos=[0.0, 0.0, 0.0235],
                size=[0.017, 0.017, 0.001],
                rgba=[1.0, 0.85, 0.0, marker_alpha],
            )
        )

        self.red_tray = self._make_tray("red_tray", [0.85, 0.08, 0.08, 1.0])
        self.blue_tray = self._make_tray("blue_tray", [0.08, 0.25, 0.90, 1.0])
        self.red_tray.get_obj().set("pos", array_to_string(self.scene.red_tray_center))
        self.blue_tray.get_obj().set("pos", array_to_string(self.scene.blue_tray_center))

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.cube, self.red_tray, self.blue_tray],
        )

    def _setup_references(self):
        super()._setup_references()
        self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.marker_site_id = self.sim.model.site_name2id("yellow_marker")

    def _setup_observables(self) -> OrderedDict:
        observables = super()._setup_observables()
        allowed = set(self.POLICY_LOW_DIM_KEYS)
        camera_prefixes = tuple(f"{name}_" for name in POLICY_CAMERA_NAMES)
        return OrderedDict(
            (name, observable)
            for name, observable in observables.items()
            if name in allowed or name.startswith(camera_prefixes)
        )

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            qpos = np.concatenate(
                [np.asarray(self.scene.cube_position), np.array([1.0, 0.0, 0.0, 0.0])]
            )
            self.sim.data.set_joint_qpos(self.cube.joints[0], qpos)
            self.sim.forward()

    @property
    def cube_position(self) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.cube_body_id])

    def tray_center(self, destination: str) -> np.ndarray:
        if destination == "red":
            return np.asarray(self.scene.red_tray_center)
        if destination == "blue":
            return np.asarray(self.scene.blue_tray_center)
        raise ValueError("destination must be 'red' or 'blue'")

    def grade_outcome(self) -> Outcome:
        return self.grader.grade(self.cube_position)

    def _check_success(self):
        return self.grade_outcome() is Outcome.RED

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update({"scene": self.scene.to_dict(), "marker_present": self.marker_present})
        return meta
