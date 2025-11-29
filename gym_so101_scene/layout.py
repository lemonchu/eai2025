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
    table_depth_cm: float = 60.0  # front-to-back extent
    boundary_width_cm: float = 1.8
    front_band_depth_cm: float = 20.4
    bin_band_depth_cm: float = 16.4
    bin_widths_cm: tuple[float, float, float] = (16.6, 15.6, 16.6)
    robot_base_offset_cm: float = 6.4
    robot_base_size_cm: tuple[float, float] = (11.0, 15.0)  # width, depth
    camera_offset_cm: tuple[float, float, float] = (31.6, 26.0, 40.7)  # (x, y, z)
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

    def _cluster_properties(self) -> tuple[float, float]:
        total_boundary_cm = self.boundary_width_cm * (len(self.bin_widths_cm) + 1)
        cluster_width_cm = sum(self.bin_widths_cm) + total_boundary_cm
        left_edge_cm = (self.table_width_cm - cluster_width_cm) / 2
        return left_edge_cm, cluster_width_cm

    def horizontal_line_positions(self) -> tuple[float, ...]:
        return (
            self.front_band_depth_cm,
            self.front_band_depth_cm + self.bin_band_depth_cm,
        )

    def vertical_line_positions(self) -> tuple[float, ...]:
        left_edge_cm, _ = self._cluster_properties()
        centers: list[float] = []
        cursor = left_edge_cm
        for _ in range(len(self.bin_widths_cm) + 1):
            cursor += self.boundary_width_cm / 2
            centers.append(cursor)
            cursor += self.boundary_width_cm / 2
            if len(centers) <= len(self.bin_widths_cm):
                cursor += self.bin_widths_cm[len(centers) - 1]
        return tuple(centers)

    def bin_left_edges_cm(self) -> list[float]:
        left_edge_cm, _ = self._cluster_properties()
        edges: list[float] = []
        cursor = left_edge_cm + self.boundary_width_cm
        for width in self.bin_widths_cm:
            edges.append(cursor)
            cursor += width + self.boundary_width_cm
        return edges

    def bin_center_cm(self, index: int) -> float:
        edges = self.bin_left_edges_cm()
        return edges[index] + self.bin_widths_cm[index] / 2

    def sample_pick_region(self, rng: np.random.Generator, bin_index: int = 1) -> np.ndarray:
        bin_index = int(np.clip(bin_index, 0, len(self.bin_widths_cm) - 1))
        center_x_cm = self.bin_center_cm(bin_index)
        center_y_cm = self.front_band_depth_cm + self.bin_band_depth_cm / 2
        center = self.table_to_world(center_x_cm, center_y_cm, self.table_thickness)
        width_margin = max(self.bin_widths_cm[bin_index] - 2 * self.boundary_width_cm, 1.0)
        dx = rng.uniform(-cm(width_margin) / 2, cm(width_margin) / 2)
        dy = rng.uniform(-cm(self.bin_band_depth_cm - 2 * self.boundary_width_cm) / 2,
                         cm(self.bin_band_depth_cm - 2 * self.boundary_width_cm) / 2)
        sample = center.copy()
        sample[1] += dx
        sample[0] += dy
        return sample

    def robot_base_centers(self) -> Iterable[np.ndarray]:
        base_half_w_cm = self.robot_base_size_cm[0] / 2
        base_depth_cm = self.robot_base_size_cm[1]
        y_center_cm = base_depth_cm / 2
        left_center = self.table_to_world(
            self.robot_base_offset_cm + base_half_w_cm,
            y_center_cm,
            self.marker_height,
        )
        right_center = self.table_to_world(
            self.table_width_cm - (self.robot_base_offset_cm + base_half_w_cm),
            y_center_cm,
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

    # Horizontal boundaries derived from measurements
    for idx, depth_cm in enumerate(layout.horizontal_line_positions()):
        pose = layout.table_to_world(
            layout.table_width_cm / 2,
            depth_cm,
            layout.table_surface_z + layout.line_height / 2,
        )
        half = [layout.boundary_width_m / 2, layout.table_width_m / 2, layout.line_height / 2]
        actors["boundaries"].append(
            _build_box(scene, half, [0, 0, 0], f"boundary_h_{idx}", pose)
        )

    # Vertical boundaries for the bin cluster
    for idx, width_cm in enumerate(layout.vertical_line_positions()):
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
