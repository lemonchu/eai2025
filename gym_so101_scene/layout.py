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
    robot_base_size_cm: tuple[float, float] = (11.0, 8.1)  # width, depth

    # Camera position relative to layout
    camera_offset_cm: tuple[float, float, float] = (31.6, 32.0, 55.0)  # (x, y, z)

    # World positioning
    world_origin: np.ndarray = field(default_factory=lambda: np.array([0.05, -0.30, 0.0]))
    table_height: float = 0.0
    table_thickness: float = 0.02
    line_height: float = 0.01
    marker_height: float = 0.005

    @property
    def table_width_cm(self) -> float:
        """Total table width (left + right areas)."""
        return self.left_area_width_cm + self.right_area_width_cm + self.boundary_width_cm

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
        # Map table coordinates: x_cm (left->right) -> world X, y_cm (front->back) -> world Y
        world = np.array([
            self.world_origin[0] + cm(y_cm),
            self.world_origin[1] - cm(x_cm),
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

    def top_camera_pose(self) -> sapien.Pose:
        top_mat44 = np.eye(4)
        top_rot = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ])
        top_mat44[:3, :3] = top_rot
        top_mat44[:3, 3] = self.table_to_world(31.6, 26.0, cm(40.7))  # adjust numbers or use camera_offset
        return sapien.Pose(top_mat44)

    def side_camera_pose(self) -> sapien.Pose:
        side_mat44 = np.eye(4)
        side_rot = np.array([
            [0, 0, -1],
            [1, 0, 0],
            [0, -1, 0],
        ])
        side_mat44[:3, :3] = side_rot
        side_mat44[:3, 3] = self.table_to_world(55.0, 20.0, cm(0.0))  # tune as needed
        return sapien.Pose(side_mat44)

    def _cluster_properties(self) -> tuple[float, float]:
        """Compute the left edge and total width of the bin cluster.

        The bins are positioned at the top of the left working area (60 cm).
        Starting from x=2, they span the full bin widths plus boundary lines.
        """
        total_boundary_cm = self.boundary_width_cm * (len(self.bin_widths_cm) + 1)
        cluster_width_cm = sum(self.bin_widths_cm) + total_boundary_cm
        # Bins start at x=2 (left edge of the table)
        left_edge_cm = 2.0
        return left_edge_cm, cluster_width_cm

    def horizontal_line_positions(self) -> tuple[float, ...]:
        """Return Y positions (depth) of horizontal boundary lines.

        According to the reference layout:
        - First line at y = robot_area_depth_cm (15.0) - bottom of bins
        - Second line at y = robot_area_depth_cm + bin_band_depth_cm (31.4) - top of bins
        """
        return (
            self.robot_area_depth_cm + self.boundary_width_cm / 2,
            self.robot_area_depth_cm + self.bin_band_depth_cm + 3 * self.boundary_width_cm / 2,
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

    def sample_pick_region(
        self,
        rng: np.random.Generator,
        bin_index: int = 1,
        area: str = "bin",
        robot_idx: int = 1,
    ) -> np.ndarray:
        """Sample a random position for object placement.

        Parameters
        - rng: random generator
        - bin_index: index of the top bins (0=left,1=center,2=right)
        - area: either 'bin' (sample inside a bin) or 'robot' (sample inside robot area)
        - robot_idx: which robot area to sample (0=left,1=right)

        Returns a world-coordinate np.ndarray [x,y,z].
        """
        area = str(area).lower()
        # helper: check overlap between a placed square (center in world meters, half-size in meters, yaw radians)
        def _overlaps_boundaries(center: np.ndarray, half_m: float, yaw: float) -> bool:
            # compute 4 corners of the square in world XY
            corners = np.array([
                [half_m, half_m],
                [-half_m, half_m],
                [-half_m, -half_m],
                [half_m, -half_m],
            ])
            c = np.cos(yaw)
            s = np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            rotated = (R @ corners.T).T + center[:2]
            xmin, ymin = rotated[:, 0].min(), rotated[:, 1].min()
            xmax, ymax = rotated[:, 0].max(), rotated[:, 1].max()

            # boundary line positions in meters
            y_bands_cm = list(self.horizontal_line_positions())
            x_bands_cm = list(self.vertical_line_positions())
            band_half_m = self.boundary_width_m / 2.0

            # Check overlap with vertical bands (lines along Y direction) -> compare X
            for x_cm in x_bands_cm:
                x_m = cm(x_cm)
                band_x_min = x_m - band_half_m
                band_x_max = x_m + band_half_m
                # if projected AABB overlaps band in X and spans the table depth, treat as overlap
                if not (xmax < band_x_min or xmin > band_x_max):
                    return True

            # Check overlap with horizontal bands (lines along X direction) -> compare Y
            for y_cm in y_bands_cm:
                y_m = cm(y_cm)
                band_y_min = y_m - band_half_m
                band_y_max = y_m + band_half_m
                if not (ymax < band_y_min or ymin > band_y_max):
                    return True

            return False

        if area == "robot":
            # Sample within the robot area rectangle (in table coordinates cm)
            robot_areas = self.robot_area_positions()
            robot_areas = (robot_areas[0], robot_areas[1])
            robot_idx = int(np.clip(robot_idx, 0, 1))
            x0_cm, y0_cm = robot_areas[robot_idx]
            w_cm = self.robot_area_width_cm
            d_cm = self.robot_area_depth_cm
            # sample within area with a small margin (10% inward)
            margin_x = max(0.1 * w_cm, 0.5)
            margin_y = max(0.1 * d_cm, 0.5)
            max_attempts = 50
            half_m = cm(1.5)  # default cube half-size ~1.5cm
            for _ in range(max_attempts):
                sx_cm = float(rng.uniform(x0_cm + margin_x, x0_cm + w_cm - margin_x))
                sy_cm = float(rng.uniform(y0_cm + margin_y, y0_cm + d_cm - margin_y))
                yaw = float(rng.uniform(-np.pi, np.pi))
                world = self.table_to_world(sx_cm, sy_cm, self.table_thickness)
                if not _overlaps_boundaries(world, half_m, yaw):
                    return world
            # fallback: return last sample even if overlapping
            return world
        # default: sample inside a top bin region
        bin_index = int(np.clip(bin_index, 0, len(self.bin_widths_cm) - 1))
        center_x_cm = self.bin_center_cm(bin_index)
        center_y_cm = self.robot_area_depth_cm + self.bin_band_depth_cm / 2 + self.boundary_width_cm
        width_margin = max(self.bin_widths_cm[bin_index] - 2 * self.boundary_width_cm, 1.0)
        max_attempts = 50
        half_m = cm(1.5)
        for _ in range(max_attempts):
            center = self.table_to_world(center_x_cm, center_y_cm, self.table_thickness)
            dx = float(rng.uniform(-cm(width_margin) / 2, cm(width_margin) / 2))
            dy = float(rng.uniform(-cm(self.bin_band_depth_cm - 2 * self.boundary_width_cm) / 2,
                                   cm(self.bin_band_depth_cm - 2 * self.boundary_width_cm) / 2))
            sample = center.copy()
            sample[1] += dx
            sample[0] += dy
            yaw = float(rng.uniform(-np.pi, np.pi))
            # reuse helper defined above to reject overlaps
            try:
                overlap = _overlaps_boundaries(sample, half_m, yaw)
            except Exception:
                overlap = False
            if not overlap:
                return sample
        # fallback
        return sample

    def robot_area_positions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return (x, y) positions for bottom-left corner of each robot area.

        Based on the reference matplotlib code from the problem statement:
        - Left robot area: starts at x=0, y=0
        - Right robot area: starts at x=32.2 (= 16.6 + 15.6, after first two bin widths), y=0

        Both robot areas have width=20.4 cm and depth=15.0 cm, positioned at the
        bottom of the table (y=0 to y=15.0).
        """
        left_area = (0.0, 0.0)
        # Right robot area starts after first two bins (matching reference layout)
        right_x = self.robot_area_width_cm + self.bin_widths_cm[1] + 2 * self.boundary_width_cm  # 20.4 + 15.6 + 1.8 * 2 = 39.6
        right_area = (right_x, 0.0)
        return (left_area, right_area)

    def robot_base_centers(self) -> Iterable[np.ndarray]:
        """Return world positions for robot base markers.

        Robot bases are positioned within the robot areas:
        - Left base: offset 6.4 cm from left edge of left robot area (x=0)
        - Right base: offset 6.4 cm from right edge of right robot area (x=39.6)

        The base markers are 11.0 cm wide × 8.1 cm deep, based on STL measurements.
        """
        base_half_w_cm = self.robot_base_size_cm[0] / 2
        base_depth_cm = self.robot_base_size_cm[1]
        y_center_cm = base_depth_cm / 2

        # Left robot base: offset from x=0 (left edge of left robot area)
        left_center = self.table_to_world(
            self.robot_base_offset_cm + base_half_w_cm,
            y_center_cm,
            self.marker_height,
        )

        # Right robot base: offset from x=42.6 (left edge of right robot area)
        right_robot_area_x = self.left_area_width_cm - self.robot_base_offset_cm - self.robot_base_size_cm[0]  # 42.6
        right_center = self.table_to_world(
            right_robot_area_x + base_half_w_cm,
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
    - Table top (full 120.0 x 60 cm)
    - Horizontal boundary lines at bin region boundaries
    - Vertical boundary lines for bins and main area separator
    - Robot area markers (blue dotted regions)
    """
    actors: dict[str, list[sapien.Actor]] = {
        "table": [],
        "boundaries": [],
        "robot_markers": [],
        # "robot_areas": [],
    }

    # Add a visible ground plane (large thin box) slightly below the table
    # to serve as a gray, rough-looking floor in renders. Make it a bit
    # larger than the table so its edges are visible from the front camera.
    # ground_half: slightly larger than table half-extents (meters)
    ground_half = [layout.table_depth_m / 2 + 5.5, layout.table_width_m / 2 + 5.5, 0.005]
    # place ground below table surface by 3cm
    ground_pose = layout.table_to_world(
        layout.table_width_cm / 2,
        layout.table_depth_cm / 2,
        -0.03,
    )
    # actors["table"].append(
    #     _build_box(scene, ground_half, [0.05, 0.05, 0.05], "ground_plane", ground_pose)
    # )

    # Create main table surface
    table_half = [layout.table_depth_m / 2, layout.table_width_m / 2, layout.table_thickness / 2]
    table_pose = layout.table_to_world(
        layout.table_width_cm / 2,
        layout.table_depth_cm / 2,
        layout.table_thickness / 2,
    )
    actors["table"].append(
        _build_box(scene, table_half, [0.85, 0.85, 0.85], "table_top", table_pose)
    )

    # Horizontal boundaries at bin region (only span the bin cluster width)
    left_boundary, cluster_width_cm = layout._cluster_properties()
    bin_region_center_x = cluster_width_cm / 2
    for idx, depth_cm in enumerate(layout.horizontal_line_positions()):
        pose = layout.table_to_world(
            left_boundary + cluster_width_cm / 2,
            depth_cm,
            layout.table_surface_z + layout.line_height / 2,
        )
        half = [layout.boundary_width_m / 2, cm(cluster_width_cm) / 2, layout.line_height / 2]
        actors["boundaries"].append(
            _build_box(scene, half, [0, 0, 0], f"boundary_h_{idx}", pose)
        )

    # Vertical boundaries for the bin cluster (4 lines between/around 3 bins)
    vertical_positions = layout.vertical_line_positions()
    bin_y_start = layout.robot_area_depth_cm + layout.boundary_width_cm
    bin_y_end = layout.robot_area_depth_cm + layout.bin_band_depth_cm + layout.boundary_width_cm
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
    # Robot area vertical boundaries
    robot_y_start = 0.0
    robot_y_end = cm(layout.robot_area_depth_cm)
    # Append two lines
    half = [robot_y_end / 2, layout.boundary_width_m / 2, layout.line_height / 2]
    left_pose = layout.table_to_world(
        layout.robot_area_width_cm + layout.boundary_width_cm / 2,
        layout.robot_area_depth_cm / 2,
        layout.table_surface_z + layout.line_height / 2,
    )
    actors["boundaries"].append(
        _build_box(scene, half, [0, 0, 0], "boundary_v_robot_left", left_pose)
    )
    right_pose = layout.table_to_world(
        layout.left_area_width_cm - layout.robot_area_width_cm - layout.boundary_width_cm / 2,
        layout.robot_area_depth_cm / 2,
        layout.table_surface_z + layout.line_height / 2,
    )
    actors["boundaries"].append(
        _build_box(scene, half, [0, 0, 0], "boundary_v_robot_right", right_pose)
    )
    # Main vertical boundary line at x=60.0 cm (full height, separates left/right areas)
    main_boundary_x = layout.left_area_width_cm
    main_boundary_pose = layout.table_to_world(
        main_boundary_x + layout.boundary_width_cm / 2,
        layout.table_depth_cm / 2,
        layout.table_surface_z + layout.line_height / 2,
    )
    main_boundary_half = [layout.table_depth_m / 2, layout.boundary_width_m / 2, layout.line_height / 2]
    actors["boundaries"].append(
        _build_box(scene, main_boundary_half, [0, 0, 0], "boundary_main", main_boundary_pose)
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
