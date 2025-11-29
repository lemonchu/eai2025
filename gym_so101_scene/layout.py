from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import sapien

CM_TO_M = 0.01


def cm(value: float) -> float:
    return value * CM_TO_M


@dataclass
class TableLayout:
    """Parameterized description of the Track-1 tabletop in Figure 2."""

    table_width_cm: float = 60.0  # left-to-right extent
    table_depth_cm: float = 80.0  # front-to-back extent
    boundary_width_cm: float = 1.8
    horizontal_lines_cm: tuple[float, ...] = (2.9, 21.3, 38.7, 57.1)
    vertical_lines_cm: tuple[float, ...] = (20.4, 39.6)
    bin_width_cm: float = 15.4
    bin_depth_cm: float = 14.0
    robot_base_margin_cm: float = 5.4
    robot_base_size_cm: tuple[float, float] = (11.0, 11.0)  # width, depth
    camera_offset_cm: tuple[float, float, float] = (31.6, 15.4, 40.7)  # (x, y, z)
    world_origin: np.ndarray = field(default_factory=lambda: np.array([0.05, -0.30, 0.0]))
    table_height: float = 0.0
    table_thickness: float = 0.02
    line_height: float = 0.01
    marker_height: float = 0.005

    @property
    def table_width_m(self) -> float:
        return cm(self.table_width_cm)

    @property
    def table_depth_m(self) -> float:
        return cm(self.table_depth_cm)

    @property
    def boundary_width_m(self) -> float:
        return cm(self.boundary_width_cm)

    @property
    def table_surface_z(self) -> float:
        return self.table_height + self.table_thickness

    @property
    def bin_width_m(self) -> float:
        return cm(self.bin_width_cm)

    @property
    def bin_depth_m(self) -> float:
        return cm(self.bin_depth_cm)

    def table_to_world(self, x_cm: float, y_cm: float, z_offset: float = 0.0) -> np.ndarray:
        """Map figure coordinates to Sapien world coordinates."""

        world = np.array([
            self.world_origin[0] + cm(y_cm),
            self.world_origin[1] + cm(x_cm),
            self.table_height + z_offset,
        ])
        return world

    def camera_pose(self) -> sapien.Pose:
        pose = np.eye(4)
        rot = np.array(
            [
                [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
                [0, 1, 0],
                [-np.sin(np.pi / 2), 0, np.cos(np.pi / 2)],
            ]
        )
        pose[:3, :3] = rot
        pos = self.table_to_world(
            self.camera_offset_cm[0], self.camera_offset_cm[1], cm(self.camera_offset_cm[2])
        )
        pose[:3, 3] = pos
        return sapien.Pose(pose)

    def pick_region_center(self) -> np.ndarray:
        depth_cm = (self.horizontal_lines_cm[1] + self.horizontal_lines_cm[2]) / 2
        width_cm = self.table_width_cm / 2
        center = self.table_to_world(width_cm, depth_cm, self.table_thickness)
        return center

    def sample_pick_region(self, rng: np.random.Generator) -> np.ndarray:
        center = self.pick_region_center()
        dx = rng.uniform(-self.bin_width_m / 2, self.bin_width_m / 2)
        dy = rng.uniform(-self.bin_depth_m / 2, self.bin_depth_m / 2)
        sample = center.copy()
        sample[1] += dx
        sample[0] += dy
        return sample

    def robot_base_centers(self) -> Iterable[np.ndarray]:
        width_margin_cm = self.robot_base_margin_cm
        base_half_w_cm = self.robot_base_size_cm[0] / 2
        base_depth_cm = self.robot_base_size_cm[1]
        front_offset_cm = width_margin_cm + base_depth_cm / 2
        left_center = self.table_to_world(
            width_margin_cm + base_half_w_cm,
            front_offset_cm,
            self.marker_height,
        )
        right_center = self.table_to_world(
            self.table_width_cm - (width_margin_cm + base_half_w_cm),
            front_offset_cm,
            self.marker_height,
        )
        return (left_center, right_center)


def _build_box(scene: sapien.Scene, half_size, color, name: str, position: np.ndarray) -> sapien.Actor:
    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=half_size, material=color)
    builder.add_box_collision(half_size=half_size)
    actor = builder.build_static(name=name)
    actor.set_pose(sapien.Pose(position, [1, 0, 0, 0]))
    return actor


def spawn_layout(scene: sapien.Scene, layout: TableLayout) -> dict[str, list[sapien.Actor]]:
    actors: dict[str, list[sapien.Actor]] = {"table": [], "boundaries": [], "robot_markers": []}

    table_half = [layout.table_depth_m / 2, layout.table_width_m / 2, layout.table_thickness / 2]
    table_pose = layout.table_to_world(
        layout.table_width_cm / 2,
        layout.table_depth_cm / 2,
        layout.table_thickness / 2,
    )
    actors["table"].append(
        _build_box(scene, table_half, [0.95, 0.95, 0.95], "table_top", table_pose)
    )

    # Horizontal boundaries
    for idx, depth_cm in enumerate(layout.horizontal_lines_cm):
        pose = layout.table_to_world(
            layout.table_width_cm / 2,
            depth_cm,
            layout.table_surface_z + layout.line_height / 2,
        )
        half = [layout.boundary_width_m / 2, layout.table_width_m / 2, layout.line_height / 2]
        actors["boundaries"].append(
            _build_box(scene, half, [0, 0, 0], f"boundary_h_{idx}", pose)
        )

    # Vertical boundaries
    for idx, width_cm in enumerate(layout.vertical_lines_cm):
        pose = layout.table_to_world(
            width_cm,
            layout.table_depth_cm / 2,
            layout.table_surface_z + layout.line_height / 2,
        )
        half = [layout.table_depth_m / 2, layout.boundary_width_m / 2, layout.line_height / 2]
        actors["boundaries"].append(
            _build_box(scene, half, [0, 0, 0], f"boundary_v_{idx}", pose)
        )

    base_half = [cm(layout.robot_base_size_cm[1]) / 2, cm(layout.robot_base_size_cm[0]) / 2, layout.marker_height / 2]
    for idx, center in enumerate(layout.robot_base_centers()):
        actors["robot_markers"].append(
            _build_box(scene, base_half, [0.2, 0.4, 0.9], f"robot_base_{idx}", center)
        )

    return actors
