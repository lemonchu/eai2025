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
    """Parameterized description of the Track-1 tabletop in Figure 2.

    Coordinate system (top-down view):
    - Origin (0, 0) at bottom-left corner of the full table
    - X-axis: left-to-right (width direction)
    - Y-axis: front-to-back (depth direction)

    Key measurements from reference diagram:
    - Total width: 60.0 cm (left area) + 58.2 cm (right area) = 118.2 cm
    - Total depth: 60.0 cm
    - Vertical boundary line at x=60.0 cm (1.8 cm wide) separates left/right areas
    - Three black bins at top: widths [16.6, 15.6, 16.6] cm, height 16.4 cm
    - Robot areas at bottom: width 20.4 cm, depth 15.0 cm
    - Camera: 31.6 cm horizontal offset, 26.0 cm from bin bottom, 40.7 cm height
    """

    # Overall table dimensions
    left_area_width_cm: float = 60.0   # left working area width
    right_area_width_cm: float = 58.2  # right working area width
    table_depth_cm: float = 60.0       # front-to-back extent
    boundary_width_cm: float = 1.8     # boundary line width

    # Robot areas at bottom (two areas for left and right robots)
    robot_area_width_cm: float = 20.4  # width of each robot area
    robot_area_depth_cm: float = 15.0  # depth/height of robot areas

    # Upper bins configuration
    bin_band_depth_cm: float = 16.4    # height of the three bins
    bin_widths_cm: tuple[float, float, float] = (16.6, 15.6, 16.6)  # left, center, right

    # Robot base markers within robot areas
    robot_base_offset_cm: float = 6.4              # offset from edge to robot base
    robot_base_size_cm: tuple[float, float] = (11.0, 15.0)  # width, depth

    # Camera position relative to layout
    camera_offset_cm: tuple[float, float, float] = (31.6, 26.0, 40.7)  # (x, y, z)

    # World positioning
    world_origin: np.ndarray = field(default_factory=lambda: np.array([0.05, -0.30, 0.0]))
    table_height: float = 0.0
    table_thickness: float = 0.02
    line_height: float = 0.01
    marker_height: float = 0.005

    @property
    def table_width_cm(self) -> float:
        """Total table width (left + right areas)."""
        return self.left_area_width_cm + self.right_area_width_cm

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
        """Compute the left edge and total width of the bin cluster.

        The bins are positioned at the top of the left working area (60 cm).
        Starting from x=0, they span the full bin widths plus boundary lines.
        """
        total_boundary_cm = self.boundary_width_cm * (len(self.bin_widths_cm) + 1)
        cluster_width_cm = sum(self.bin_widths_cm) + total_boundary_cm
        # Bins start at x=0 (left edge of the table)
        left_edge_cm = 0.0
        return left_edge_cm, cluster_width_cm

    def horizontal_line_positions(self) -> tuple[float, ...]:
        """Return Y positions (depth) of horizontal boundary lines.

        According to the reference layout:
        - First line at y = robot_area_depth_cm (15.0) - bottom of bins
        - Second line at y = robot_area_depth_cm + bin_band_depth_cm (31.4) - top of bins
        """
        return (
            self.robot_area_depth_cm,
            self.robot_area_depth_cm + self.bin_band_depth_cm,
        )

    def vertical_line_positions(self) -> tuple[float, ...]:
        """Return X positions of vertical boundary lines.

        Returns:
        - 4 boundary lines separating the 3 bins (at left edge, between bins, at right edge)
        - 1 main boundary line at x=60.0 cm separating left/right working areas
        """
        left_edge_cm, _ = self._cluster_properties()
        centers: list[float] = []
        cursor = left_edge_cm
        # Add bin boundary lines
        for _ in range(len(self.bin_widths_cm) + 1):
            cursor += self.boundary_width_cm / 2
            centers.append(cursor)
            cursor += self.boundary_width_cm / 2
            if len(centers) <= len(self.bin_widths_cm):
                cursor += self.bin_widths_cm[len(centers) - 1]
        # Add main boundary line at x=60.0 cm (separating left/right areas)
        centers.append(self.left_area_width_cm)
        return tuple(centers)

    def bin_left_edges_cm(self) -> list[float]:
        """Return the left X positions of each bin."""
        left_edge_cm, _ = self._cluster_properties()
        edges: list[float] = []
        cursor = left_edge_cm + self.boundary_width_cm
        for width in self.bin_widths_cm:
            edges.append(cursor)
            cursor += width + self.boundary_width_cm
        return edges

    def bin_center_cm(self, index: int) -> float:
        """Return the X center position of a bin."""
        edges = self.bin_left_edges_cm()
        return edges[index] + self.bin_widths_cm[index] / 2

    def sample_pick_region(self, rng: np.random.Generator, bin_index: int = 1) -> np.ndarray:
        """Sample a random position within a bin for object placement."""
        bin_index = int(np.clip(bin_index, 0, len(self.bin_widths_cm) - 1))
        center_x_cm = self.bin_center_cm(bin_index)
        center_y_cm = self.robot_area_depth_cm + self.bin_band_depth_cm / 2
        center = self.table_to_world(center_x_cm, center_y_cm, self.table_thickness)
        width_margin = max(self.bin_widths_cm[bin_index] - 2 * self.boundary_width_cm, 1.0)
        dx = rng.uniform(-cm(width_margin) / 2, cm(width_margin) / 2)
        dy = rng.uniform(-cm(self.bin_band_depth_cm - 2 * self.boundary_width_cm) / 2,
                         cm(self.bin_band_depth_cm - 2 * self.boundary_width_cm) / 2)
        sample = center.copy()
        sample[1] += dx
        sample[0] += dy
        return sample

    def robot_area_positions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (x, y) positions for bottom-left corner of each robot area.

        According to reference:
        - Left robot area: x=0, y=0, width=20.4, depth=15.0
        - Right robot area: x=32.2 (after first two bins), y=0, width=20.4, depth=15.0
        """
        left_area = (0.0, 0.0)
        # Right robot area starts after left bin + middle bin
        right_x = self.bin_widths_cm[0] + self.bin_widths_cm[1]
        right_area = (right_x, 0.0)
        return (left_area, right_area)

    def robot_base_centers(self) -> Iterable[np.ndarray]:
        """Return world positions for robot base markers.

        Robot bases are positioned within the robot areas with specified offset.
        """
        base_half_w_cm = self.robot_base_size_cm[0] / 2
        base_depth_cm = self.robot_base_size_cm[1]
        y_center_cm = base_depth_cm / 2

        # Left robot base: offset from left edge of left robot area
        left_center = self.table_to_world(
            self.robot_base_offset_cm + base_half_w_cm,
            y_center_cm,
            self.marker_height,
        )

        # Right robot base: offset from left edge of right robot area
        right_robot_area_x = self.bin_widths_cm[0] + self.bin_widths_cm[1]
        right_center = self.table_to_world(
            right_robot_area_x + self.robot_base_offset_cm + base_half_w_cm,
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
    """Spawn the table layout in the Sapien scene.

    Creates:
    - Table top (full 118.2 x 60 cm)
    - Horizontal boundary lines at bin region boundaries
    - Vertical boundary lines for bins and main area separator
    - Robot area markers (blue dotted regions)
    - Robot base markers
    """
    actors: dict[str, list[sapien.Actor]] = {
        "table": [],
        "boundaries": [],
        "robot_markers": [],
        "robot_areas": [],
    }

    # Create main table surface
    table_half = [layout.table_depth_m / 2, layout.table_width_m / 2, layout.table_thickness / 2]
    table_pose = layout.table_to_world(
        layout.table_width_cm / 2,
        layout.table_depth_cm / 2,
        layout.table_thickness / 2,
    )
    actors["table"].append(
        _build_box(scene, table_half, [0.95, 0.95, 0.95], "table_top", table_pose)
    )

    # Horizontal boundaries at bin region (only span the bin cluster width)
    _, cluster_width_cm = layout._cluster_properties()
    bin_region_center_x = cluster_width_cm / 2
    for idx, depth_cm in enumerate(layout.horizontal_line_positions()):
        pose = layout.table_to_world(
            bin_region_center_x,
            depth_cm,
            layout.table_surface_z + layout.line_height / 2,
        )
        half = [layout.boundary_width_m / 2, cm(cluster_width_cm) / 2, layout.line_height / 2]
        actors["boundaries"].append(
            _build_box(scene, half, [0, 0, 0], f"boundary_h_{idx}", pose)
        )

    # Vertical boundaries for the bin cluster (4 lines between/around 3 bins)
    vertical_positions = layout.vertical_line_positions()
    bin_y_start = layout.robot_area_depth_cm
    bin_y_end = layout.robot_area_depth_cm + layout.bin_band_depth_cm
    bin_center_y = (bin_y_start + bin_y_end) / 2
    bin_height_m = cm(layout.bin_band_depth_cm)

    for idx, x_cm in enumerate(vertical_positions[:-1]):  # All but the last (main boundary)
        pose = layout.table_to_world(
            x_cm,
            bin_center_y,
            layout.table_surface_z + layout.line_height / 2,
        )
        half = [bin_height_m / 2, layout.boundary_width_m / 2, layout.line_height / 2]
        actors["boundaries"].append(
            _build_box(scene, half, [0, 0, 0], f"boundary_v_bin_{idx}", pose)
        )

    # Main vertical boundary line at x=60.0 cm (full height, separates left/right areas)
    main_boundary_x = layout.left_area_width_cm
    main_boundary_pose = layout.table_to_world(
        main_boundary_x,
        layout.table_depth_cm / 2,
        layout.table_surface_z + layout.line_height / 2,
    )
    main_boundary_half = [layout.table_depth_m / 2, layout.boundary_width_m / 2, layout.line_height / 2]
    actors["boundaries"].append(
        _build_box(scene, main_boundary_half, [0, 0, 0], "boundary_main", main_boundary_pose)
    )

    # Robot areas (blue dotted pad regions)
    robot_area_positions = layout.robot_area_positions()
    robot_area_half = [
        cm(layout.robot_area_depth_cm) / 2,
        cm(layout.robot_area_width_cm) / 2,
        layout.marker_height / 2,
    ]
    for idx, (x_cm, y_cm) in enumerate(robot_area_positions):
        center_x = x_cm + layout.robot_area_width_cm / 2
        center_y = y_cm + layout.robot_area_depth_cm / 2
        pose = layout.table_to_world(
            center_x,
            center_y,
            layout.table_surface_z + layout.marker_height / 2,
        )
        actors["robot_areas"].append(
            _build_box(scene, robot_area_half, [0.7, 0.85, 1.0], f"robot_area_{idx}", pose)
        )

    # Robot base markers (positioned within robot areas)
    base_half = [
        cm(layout.robot_base_size_cm[1]) / 2,
        cm(layout.robot_base_size_cm[0]) / 2,
        layout.marker_height / 2,
    ]
    for idx, center in enumerate(layout.robot_base_centers()):
        actors["robot_markers"].append(
            _build_box(scene, base_half, [0.2, 0.4, 0.9], f"robot_base_{idx}", center)
        )

    return actors
