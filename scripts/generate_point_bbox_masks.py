# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

"""Generate COLMAP dataset masks from bounded geometry derived from its initial points."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import tempfile

import cv2
import numpy as np
import tqdm

from fvdb_reality_capture import CameraModel
from fvdb_reality_capture.sfm_scene import SfmPosedImageMetadata, SfmScene
from fvdb_reality_capture.transforms import ScalePercentileFilterPoints

_NEAR_EPSILON = 1.0e-6


@dataclasses.dataclass(frozen=True)
class PlaneHull:
    vertices_h: np.ndarray
    normal: np.ndarray
    offset: float
    median_absolute_distance: float
    rms_distance: float


@dataclasses.dataclass(frozen=True)
class PlaneConcaveHull:
    contours_h: tuple[np.ndarray, ...]
    normal: np.ndarray
    offset: float
    median_absolute_distance: float
    rms_distance: float
    grid_shape: tuple[int, int]
    grid_cell_size: float
    occupied_cells: int
    closed_cells: int
    retained_cells: int
    component_count: int
    retained_component_count: int


@dataclasses.dataclass(frozen=True)
class _PlaneProjection:
    centroid: np.ndarray
    basis: np.ndarray
    points_plane: np.ndarray
    normal: np.ndarray
    offset: float
    median_absolute_distance: float
    rms_distance: float


def _filter_points_by_coordinate_percentile(points: np.ndarray, percentile: float) -> np.ndarray:
    """Match the symmetric per-axis percentile filtering used by ``frgs reconstruct``."""
    if percentile < 0.0 or percentile >= 50.0:
        raise ValueError("point-coordinate-percentile-filter must be in the range [0, 50).")
    if percentile == 0.0:
        return points

    lower = np.percentile(points, percentile, axis=0)
    upper = np.percentile(points, 100.0 - percentile, axis=0)
    keep = np.all((points > lower) & (points < upper), axis=1)
    filtered_points = points[keep]
    if filtered_points.shape[0] == 0:
        raise ValueError(f"No points remain after applying a {percentile:g}% symmetric percentile filter.")
    return filtered_points


def _filter_geometry_points(
    points: np.ndarray, coordinate_percentile_filter: float, scale_percentile_filter: float
) -> np.ndarray:
    """Apply the coordinate and scale filters used by ``frgs reconstruct`` in the same order."""
    filtered_points = _filter_points_by_coordinate_percentile(points, coordinate_percentile_filter)
    scale_filter = ScalePercentileFilterPoints(percentile_filter=scale_percentile_filter)
    if scale_percentile_filter == 0.0:
        return filtered_points
    return filtered_points[scale_filter.compute_point_mask(filtered_points)]


def _bbox_corners(points: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the padded bounding box and its eight homogeneous corners."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected a non-empty point array with shape (N, 3), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("The initial point set contains non-finite coordinates.")
    if not np.isfinite(margin) or margin <= -0.5:
        raise ValueError("margin must be finite and greater than -0.5 so the padded bounding box has positive extent.")

    points_min = points.min(axis=0).astype(np.float64)
    points_max = points.max(axis=0).astype(np.float64)
    box_size = points_max - points_min
    if np.any(box_size <= 0.0):
        raise ValueError(f"The initial point-set bounding box has a non-positive extent: {box_size}.")

    # A margin of 0.05 adds 5% of the original extent to each side.
    padding = margin * box_size
    bbox_min = points_min - padding
    bbox_max = points_max + padding
    bbox = np.concatenate([bbox_min, bbox_max])
    corners = np.array(
        [
            [x, y, z, 1.0]
            for x in (bbox_min[0], bbox_max[0])
            for y in (bbox_min[1], bbox_max[1])
            for z in (bbox_min[2], bbox_max[2])
        ],
        dtype=np.float64,
    )
    return bbox, corners


def _fit_plane(points: np.ndarray, margin: float) -> _PlaneProjection:
    """Fit a least-squares plane and project the points into its two-dimensional basis."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"Expected at least three points with shape (N, 3), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("The plane-fitting point set contains non-finite coordinates.")
    if not np.isfinite(margin) or margin <= -0.5:
        raise ValueError("margin must be finite and greater than -0.5.")

    points64 = points.astype(np.float64, copy=False)
    centroid = points64.mean(axis=0)
    centered = points64 - centroid
    covariance = centered.T @ centered / points64.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= 0.0:
        raise ValueError("The point set does not span a two-dimensional plane footprint.")

    normal = eigenvectors[:, 0]
    if normal[2] < 0.0:
        normal = -normal
    plane_basis = eigenvectors[:, [2, 1]]
    points_plane = centered @ plane_basis

    signed_distances = centered @ normal
    return _PlaneProjection(
        centroid=centroid,
        basis=plane_basis,
        points_plane=points_plane,
        normal=normal,
        offset=float(-normal @ centroid),
        median_absolute_distance=float(np.median(np.abs(signed_distances))),
        rms_distance=float(np.sqrt(np.mean(np.square(signed_distances)))),
    )


def _fit_plane_hull(points: np.ndarray, margin: float) -> PlaneHull:
    """Fit a least-squares plane and return the convex hull of points flattened onto it."""
    plane = _fit_plane(points, margin)
    hull_plane = cv2.convexHull(np.ascontiguousarray(plane.points_plane, dtype=np.float32)).reshape(-1, 2)
    if hull_plane.shape[0] < 3:
        raise ValueError("The projected point set does not have a valid convex hull.")

    # As with bbox mode, a margin of 0.05 adds approximately 5% of the
    # footprint extent on every side. The PCA centroid is the origin in plane
    # coordinates and lies within the convex hull.
    hull_plane *= 1.0 + 2.0 * margin
    hull_world = plane.centroid[None] + hull_plane.astype(np.float64) @ plane.basis.T
    vertices_h = np.concatenate([hull_world, np.ones((hull_world.shape[0], 1), dtype=np.float64)], axis=1)

    return PlaneHull(
        vertices_h=vertices_h,
        normal=plane.normal,
        offset=plane.offset,
        median_absolute_distance=plane.median_absolute_distance,
        rms_distance=plane.rms_distance,
    )


def _fit_plane_concave_hull(
    points: np.ndarray,
    margin: float,
    grid_resolution: int,
    closing_radius: int,
    min_component_cells: int,
    padding_radius: int,
    simplify_tolerance: float,
) -> PlaneConcaveHull:
    """Fit a plane and derive a density-aware concave footprint from occupied grid cells."""
    if grid_resolution < 4:
        raise ValueError("concave-grid-resolution must be at least 4.")
    if closing_radius < 0:
        raise ValueError("concave-closing-radius must be non-negative.")
    if min_component_cells < 1:
        raise ValueError("concave-min-component-cells must be at least 1.")
    if padding_radius < 0:
        raise ValueError("concave-padding-radius must be non-negative.")
    if not np.isfinite(simplify_tolerance) or simplify_tolerance < 0.0:
        raise ValueError("concave-simplify-tolerance must be finite and non-negative.")

    plane = _fit_plane(points, margin)
    points_min = plane.points_plane.min(axis=0)
    points_extent = np.ptp(plane.points_plane, axis=0)
    maximum_extent = float(points_extent.max())
    if maximum_extent <= 0.0:
        raise ValueError("The projected point set has a non-positive plane footprint extent.")

    cell_size = maximum_extent / grid_resolution
    footprint_shape_xy = np.ceil(points_extent / cell_size).astype(np.int64) + 1
    grid_border = closing_radius + padding_radius + 2
    grid_width = int(footprint_shape_xy[0]) + 2 * grid_border
    grid_height = int(footprint_shape_xy[1]) + 2 * grid_border

    point_cells = np.floor((plane.points_plane - points_min[None]) / cell_size).astype(np.int64)
    point_cells[:, 0] = np.clip(point_cells[:, 0], 0, int(footprint_shape_xy[0]) - 1)
    point_cells[:, 1] = np.clip(point_cells[:, 1], 0, int(footprint_shape_xy[1]) - 1)
    point_cells += grid_border

    occupancy = np.zeros((grid_height, grid_width), dtype=np.uint8)
    occupancy[point_cells[:, 1], point_cells[:, 0]] = 255
    occupied_cells = int(np.count_nonzero(occupancy))

    if closing_radius > 0:
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closing_radius + 1, 2 * closing_radius + 1))
        closed_occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, closing_kernel)
    else:
        closed_occupancy = occupancy
    closed_cells = int(np.count_nonzero(closed_occupancy))

    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        closed_occupancy, connectivity=8
    )
    component_areas = component_stats[1:, cv2.CC_STAT_AREA]
    retained_labels = np.zeros(component_count, dtype=bool)
    retained_labels[1:] = component_areas >= min_component_cells
    retained_component_count = int(np.count_nonzero(retained_labels))
    if retained_component_count == 0:
        raise ValueError(
            f"No plane-footprint components remain after applying concave-min-component-cells={min_component_cells}."
        )
    retained_occupancy = (retained_labels[component_labels].astype(np.uint8)) * 255

    if padding_radius > 0:
        padding_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding_radius + 1, 2 * padding_radius + 1))
        retained_occupancy = cv2.dilate(retained_occupancy, padding_kernel)
    retained_cells = int(np.count_nonzero(retained_occupancy))

    grid_contours, _ = cv2.findContours(retained_occupancy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_contours = sorted(grid_contours, key=cv2.contourArea, reverse=True)
    contours_h: list[np.ndarray] = []
    scale = 1.0 + 2.0 * margin
    for grid_contour in grid_contours:
        if simplify_tolerance > 0.0:
            grid_contour = cv2.approxPolyDP(grid_contour, simplify_tolerance, closed=True)
        contour_grid_xy = grid_contour.reshape(-1, 2)
        if contour_grid_xy.shape[0] < 3 or cv2.contourArea(grid_contour) <= 0.0:
            continue

        contour_plane = points_min[None] + (contour_grid_xy.astype(np.float64) - grid_border + 0.5) * cell_size
        contour_plane *= scale
        contour_world = plane.centroid[None] + contour_plane @ plane.basis.T
        contour_h = np.concatenate([contour_world, np.ones((contour_world.shape[0], 1), dtype=np.float64)], axis=1)
        contours_h.append(contour_h)

    if not contours_h:
        raise ValueError("The retained plane-footprint components do not form any valid polygonal contours.")

    return PlaneConcaveHull(
        contours_h=tuple(contours_h),
        normal=plane.normal,
        offset=plane.offset,
        median_absolute_distance=plane.median_absolute_distance,
        rms_distance=plane.rms_distance,
        grid_shape=(grid_height, grid_width),
        grid_cell_size=cell_size,
        occupied_cells=occupied_cells,
        closed_cells=closed_cells,
        retained_cells=retained_cells,
        component_count=component_count - 1,
        retained_component_count=retained_component_count,
    )


def _render_convex_geometry_mask(image_meta: SfmPosedImageMetadata, vertices_h: np.ndarray) -> np.ndarray:
    """Rasterize the image-space convex hull of homogeneous world-space vertices."""
    camera_meta = image_meta.camera_metadata
    if camera_meta.camera_model != CameraModel.PINHOLE:
        raise ValueError(
            f"Image {image_meta.image_path} uses unsupported camera model {camera_meta.camera_model.name}; "
            "this script currently supports PINHOLE cameras only."
        )

    width = camera_meta.width
    height = camera_meta.height
    mask = np.zeros((height, width), dtype=np.uint8)

    vertices_camera_h = image_meta.world_to_camera_matrix.astype(np.float64) @ vertices_h.T
    homogeneous_scale = vertices_camera_h[3]
    if np.any(np.abs(homogeneous_scale) <= _NEAR_EPSILON):
        raise ValueError(f"Mask-geometry projection is singular for image {image_meta.image_path}.")
    vertices_camera = vertices_camera_h[:3] / homogeneous_scale[None]
    depths = vertices_camera[2]

    # No forward-facing ray can hit geometry wholly behind the camera. If the
    # geometry crosses the camera plane, keep the full image as a conservative
    # mask; projecting vertices through z=0 does not yield a finite silhouette.
    if np.all(depths <= _NEAR_EPSILON):
        return mask
    if np.any(depths <= _NEAR_EPSILON):
        mask.fill(255)
        return mask

    projected_h = camera_meta.projection_matrix.astype(np.float64) @ vertices_camera
    projected = (projected_h[:2] / projected_h[2:3]).T
    if not np.all(np.isfinite(projected)):
        raise ValueError(f"Mask-geometry projection contains non-finite coordinates for {image_meta.image_path}.")

    projected_hull = cv2.convexHull(np.ascontiguousarray(projected, dtype=np.float32)).reshape(-1, 2)
    image_bounds = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    intersection_area, clipped_hull = cv2.intersectConvexConvex(projected_hull, image_bounds)
    if clipped_hull is None or intersection_area <= 0.0:
        return mask

    clipped_hull = np.rint(clipped_hull.reshape(-1, 2)).astype(np.int32)
    if clipped_hull.shape[0] >= 3:
        cv2.fillConvexPoly(mask, clipped_hull, 255, lineType=cv2.LINE_8)
    return mask


def _render_planar_contours_mask(image_meta: SfmPosedImageMetadata, contours_h: tuple[np.ndarray, ...]) -> np.ndarray:
    """Project and rasterize a union of concave homogeneous world-space contours."""
    camera_meta = image_meta.camera_metadata
    if camera_meta.camera_model != CameraModel.PINHOLE:
        raise ValueError(
            f"Image {image_meta.image_path} uses unsupported camera model {camera_meta.camera_model.name}; "
            "this script currently supports PINHOLE cameras only."
        )

    width = camera_meta.width
    height = camera_meta.height
    mask = np.zeros((height, width), dtype=np.uint8)
    maximum_raster_coordinate = 1 << 24

    for contour_h in contours_h:
        vertices_camera_h = image_meta.world_to_camera_matrix.astype(np.float64) @ contour_h.T
        homogeneous_scale = vertices_camera_h[3]
        if np.any(np.abs(homogeneous_scale) <= _NEAR_EPSILON):
            raise ValueError(f"Mask-geometry projection is singular for image {image_meta.image_path}.")
        vertices_camera = vertices_camera_h[:3] / homogeneous_scale[None]
        depths = vertices_camera[2]

        if np.all(depths <= _NEAR_EPSILON):
            continue
        if np.any(depths <= _NEAR_EPSILON):
            mask.fill(255)
            return mask

        projected_h = camera_meta.projection_matrix.astype(np.float64) @ vertices_camera
        projected = (projected_h[:2] / projected_h[2:3]).T
        if not np.all(np.isfinite(projected)):
            raise ValueError(f"Mask-geometry projection contains non-finite coordinates for {image_meta.image_path}.")

        # OpenCV clips ordinary off-image polygons during rasterization. Avoid
        # overflowing its integer edge arithmetic for geometry extremely close
        # to the camera plane; a full mask is conservative in that case.
        if np.any(np.abs(projected) > maximum_raster_coordinate):
            mask.fill(255)
            return mask
        projected_contour = np.rint(projected).astype(np.int32)
        if projected_contour.shape[0] >= 3:
            cv2.fillPoly(mask, [projected_contour], 255, lineType=cv2.LINE_8)

    return mask


def _target_mask_path(
    image_meta: SfmPosedImageMetadata, image_root: pathlib.Path, output_dir: pathlib.Path
) -> pathlib.Path:
    try:
        relative_image_path = pathlib.Path(image_meta.image_path).resolve().relative_to(image_root)
    except ValueError as error:
        raise ValueError(
            f"Image {image_meta.image_path} is not contained in the dataset image directory {image_root}."
        ) from error
    return output_dir / relative_image_path.with_suffix(".png")


def _write_png_atomic(path: pathlib.Path, mask: np.ndarray, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False) as file:
            temporary_path = pathlib.Path(file.name)
        if not cv2.imwrite(str(temporary_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
            raise OSError(f"Failed to write mask {temporary_path}.")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate binary masks for a COLMAP dataset by projecting bounded geometry derived from its initial 3D "
            "point set into every image. White pixels are valid and black pixels are masked during reconstruction."
        )
    )
    parser.add_argument("dataset_path", type=pathlib.Path, help="COLMAP dataset containing images/ and sparse/.")
    parser.add_argument(
        "--mode",
        choices=("bbox", "plane-hull", "plane-concave-hull"),
        default="bbox",
        help=(
            "Geometry to project: the axis-aligned 3D point bounding box, the convex footprint of points flattened "
            "onto their least-squares plane, or a density-aware concave footprint on that plane (default: bbox)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output mask directory. Defaults to <dataset_path>/masks.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Fraction of the bbox or plane-footprint extent to add on each side (default: 0).",
    )
    parser.add_argument(
        "--concave-grid-resolution",
        type=int,
        default=1024,
        help=(
            "Number of occupancy-grid cells along the longest plane-footprint axis for plane-concave-hull "
            "(default: 1024)."
        ),
    )
    parser.add_argument(
        "--concave-closing-radius",
        type=int,
        default=4,
        help="Radius in grid cells used to close sampling gaps in plane-concave-hull (default: 4).",
    )
    parser.add_argument(
        "--concave-min-component-cells",
        type=int,
        default=16,
        help=(
            "Discard disconnected plane-concave-hull components smaller than this many cells after gap closing "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--concave-padding-radius",
        type=int,
        default=1,
        help="Outward safety padding in grid cells for plane-concave-hull (default: 1).",
    )
    parser.add_argument(
        "--concave-simplify-tolerance",
        type=float,
        default=1.0,
        help="Contour simplification tolerance in grid cells for plane-concave-hull (default: 1.0).",
    )
    parser.add_argument(
        "--point-coordinate-percentile-filter",
        type=float,
        default=0.0,
        help=(
            "Symmetrically reject this percentile of points at each end of every coordinate axis before computing "
            "the geometry. Use the same value as frgs reconstruct --tx.point-coordinate-percentile-filter "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--point-scale-percentile-filter",
        type=float,
        default=0.0,
        help=(
            "Reject points in the upper tail of the initial 3-neighbor RMS scale distribution before computing "
            "the geometry. Use the same value as frgs reconstruct --tx.point-scale-percentile-filter (default: 0)."
        ),
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=3,
        metavar="0-9",
        help="OpenCV PNG compression level (default: 3).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing target mask files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the scene and report the derived geometry and output paths without generating masks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_path = args.dataset_path.resolve()
    image_root = (dataset_path / "images").resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else dataset_path / "masks"
    if not image_root.is_dir():
        raise NotADirectoryError(f"Dataset image directory does not exist: {image_root}")

    scene = SfmScene.from_colmap(dataset_path)
    geometry_points = _filter_geometry_points(
        scene.points,
        coordinate_percentile_filter=args.point_coordinate_percentile_filter,
        scale_percentile_filter=args.point_scale_percentile_filter,
    )
    bbox: np.ndarray | None = None
    plane_hull: PlaneHull | None = None
    plane_concave_hull: PlaneConcaveHull | None = None
    mask_vertices_h: np.ndarray | None = None
    mask_contours_h: tuple[np.ndarray, ...] | None = None
    if args.mode == "bbox":
        bbox, mask_vertices_h = _bbox_corners(geometry_points, args.margin)
    elif args.mode == "plane-hull":
        plane_hull = _fit_plane_hull(geometry_points, args.margin)
        mask_vertices_h = plane_hull.vertices_h
    else:
        plane_concave_hull = _fit_plane_concave_hull(
            geometry_points,
            args.margin,
            args.concave_grid_resolution,
            args.concave_closing_radius,
            args.concave_min_component_cells,
            args.concave_padding_radius,
            args.concave_simplify_tolerance,
        )
        mask_contours_h = plane_concave_hull.contours_h
    targets = [_target_mask_path(image_meta, image_root, output_dir) for image_meta in scene.images]
    if len(targets) != len(set(targets)):
        raise ValueError("Multiple dataset images map to the same output mask path.")
    if output_dir == image_root:
        raise ValueError("output-dir cannot be the dataset images directory because it could overwrite source images.")
    unsupported_models = sorted(
        {image_meta.camera_metadata.camera_model.name for image_meta in scene.images} - {CameraModel.PINHOLE.name}
    )
    if unsupported_models:
        raise ValueError(f"Only PINHOLE cameras are supported; found: {', '.join(unsupported_models)}.")
    existing_targets = [path for path in targets if path.exists()]
    if existing_targets and not args.overwrite and not args.dry_run:
        examples = "\n".join(f"  {path}" for path in existing_targets[:5])
        raise FileExistsError(
            f"{len(existing_targets)} target mask(s) already exist. Pass --overwrite to replace them. Examples:\n{examples}"
        )

    print(f"Initial points: {scene.points.shape[0]:,}")
    print(f"Geometry points: {geometry_points.shape[0]:,}")
    print(f"Point coordinate percentile filter: {args.point_coordinate_percentile_filter:g}")
    print(f"Point scale percentile filter: {args.point_scale_percentile_filter:g}")
    print(f"Images: {scene.num_images:,}")
    print(f"Mode: {args.mode}")
    if bbox is not None:
        print(f"Bounding box: {bbox.tolist()}")
    if plane_hull is not None:
        print(f"Plane normal: {plane_hull.normal.tolist()}")
        print(f"Plane offset: {plane_hull.offset}")
        print(f"Plane median absolute distance: {plane_hull.median_absolute_distance}")
        print(f"Plane RMS distance: {plane_hull.rms_distance}")
        print(f"Plane hull vertices: {plane_hull.vertices_h.shape[0]}")
    if plane_concave_hull is not None:
        print(f"Plane normal: {plane_concave_hull.normal.tolist()}")
        print(f"Plane offset: {plane_concave_hull.offset}")
        print(f"Plane median absolute distance: {plane_concave_hull.median_absolute_distance}")
        print(f"Plane RMS distance: {plane_concave_hull.rms_distance}")
        print(f"Concave grid shape: {plane_concave_hull.grid_shape}")
        print(f"Concave grid cell size: {plane_concave_hull.grid_cell_size}")
        print(f"Concave occupied cells: {plane_concave_hull.occupied_cells:,}")
        print(f"Concave cells after closing: {plane_concave_hull.closed_cells:,}")
        print(f"Concave retained cells: {plane_concave_hull.retained_cells:,}")
        print(
            "Concave retained components: "
            f"{plane_concave_hull.retained_component_count:,}/{plane_concave_hull.component_count:,}"
        )
        print(f"Concave contours: {len(plane_concave_hull.contours_h):,}")
        print(f"Concave contour vertices: {sum(contour.shape[0] for contour in plane_concave_hull.contours_h):,}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("Dry run: no masks were written.")
        return

    for image_meta, target in tqdm.tqdm(
        zip(scene.images, targets, strict=True), total=scene.num_images, unit="masks", desc="Generating masks"
    ):
        if mask_contours_h is not None:
            mask = _render_planar_contours_mask(image_meta, mask_contours_h)
        else:
            if mask_vertices_h is None:
                raise AssertionError("Convex mask geometry was not initialized.")
            mask = _render_convex_geometry_mask(image_meta, mask_vertices_h)
        _write_png_atomic(target, mask, args.png_compression)

    print(f"Wrote {scene.num_images:,} masks to {output_dir}")


if __name__ == "__main__":
    main()
