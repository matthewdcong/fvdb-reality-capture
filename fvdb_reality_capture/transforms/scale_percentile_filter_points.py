# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from fvdb_reality_capture.sfm_scene import SfmScene

from .base_transform import BaseTransform, transform


@transform
class ScalePercentileFilterPoints(BaseTransform):
    """
    Filter points whose estimated initial Gaussian scale is in the upper tail of the scale distribution.

    The scale for each point is the root-mean-square distance to its three nearest neighboring points. This is the
    same scale estimate used when initializing Gaussians for reconstruction. After this transform filters the
    :class:`~fvdb_reality_capture.sfm_scene.SfmScene`, Gaussian initialization recomputes scales from the retained
    points.

    ``percentile_filter`` is the percentage of the upper tail to remove. For example, ``0.005`` removes points above
    the 99.995th percentile of estimated scales. A value of ``0`` disables filtering.
    """

    version = "1.0.0"
    _num_neighbors = 3

    def __init__(self, percentile_filter: float = 0.0):
        """
        Create a scale-percentile point filter.

        Args:
            percentile_filter (float): Percentage of points at the upper end of the estimated scale distribution to
                filter out. Must be greater than or equal to 0 and less than 100. Defaults to 0, which disables the
                transform.
        """
        super().__init__()
        percentile_filter = float(percentile_filter)
        if not np.isfinite(percentile_filter) or percentile_filter < 0.0 or percentile_filter >= 100.0:
            raise ValueError(
                f"percentile_filter must be finite and in the range [0, 100). Got {percentile_filter} instead."
            )

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._percentile_filter = percentile_filter

    @classmethod
    def _estimate_point_scales(cls, points: np.ndarray) -> np.ndarray:
        """Estimate the initial isotropic Gaussian scale for every point."""
        kd_tree = cKDTree(points)  # type: ignore
        neighbor_distances, neighbor_indices = kd_tree.query(points, k=list(range(2, cls._num_neighbors + 2)))
        del neighbor_indices

        # Gaussian initialization converts KNN distances to float32 before squaring and averaging. Match that
        # behavior so the percentile filter is based on the same scale values.
        neighbor_distances = neighbor_distances.astype(np.float32, copy=False)
        np.square(neighbor_distances, out=neighbor_distances)
        point_scales = np.mean(neighbor_distances, axis=-1)
        np.sqrt(point_scales, out=point_scales)
        return point_scales

    def compute_point_mask(self, points: np.ndarray) -> np.ndarray:
        """
        Return a Boolean mask selecting points below the configured scale-percentile cutoff.

        This method exposes the transform's filtering calculation to preprocessing tools that only need to filter a
        point array and do not need to construct an :class:`~fvdb_reality_capture.sfm_scene.SfmScene`.

        Args:
            points (np.ndarray): An ``(N, 3)`` array of point coordinates.

        Returns:
            np.ndarray: A Boolean array of shape ``(N,)`` whose true values indicate retained points.
        """
        if self._percentile_filter == 0.0:
            return np.ones(len(points), dtype=bool)

        min_num_points = self._num_neighbors + 1
        if len(points) < min_num_points:
            raise ValueError(
                f"Scale-percentile point filtering requires at least {min_num_points} points to estimate scale from "
                f"{self._num_neighbors} neighbors, but the scene contains {len(points)}."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("Scale-percentile point filtering requires all point coordinates to be finite.")

        point_scales = self._estimate_point_scales(points)
        finite_scales = np.isfinite(point_scales)
        num_finite_scales = int(np.count_nonzero(finite_scales))
        if num_finite_scales < min_num_points:
            raise ValueError(
                f"Only {num_finite_scales} points have finite estimated scales, but at least {min_num_points} are "
                "required for Gaussian initialization."
            )

        cutoff_percentile = 100.0 - self._percentile_filter
        scale_cutoff = float(np.percentile(point_scales[finite_scales], cutoff_percentile))
        keep = finite_scales & (point_scales <= scale_cutoff)
        num_kept = int(np.count_nonzero(keep))
        if num_kept < min_num_points:
            raise ValueError(
                f"Scale-percentile point filtering at {cutoff_percentile:g} left {num_kept} points, but at least "
                f"{min_num_points} are required for Gaussian initialization. Reduce percentile_filter."
            )

        num_removed = len(points) - num_kept
        self._logger.info(
            "Filtered %d of %d points above the %.6gth scale percentile (scale cutoff %.9g); kept %d points.",
            num_removed,
            len(points),
            cutoff_percentile,
            scale_cutoff,
            num_kept,
        )
        return keep

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        """
        Return a scene without points in the requested upper tail of estimated initial Gaussian scales.

        Args:
            input_scene (SfmScene): Scene whose points will be filtered.

        Returns:
            SfmScene: The filtered scene, or ``input_scene`` itself when filtering is disabled.
        """
        if self._percentile_filter == 0.0:
            self._logger.info("Scale-percentile point filtering is disabled; returning the input scene unchanged.")
            return input_scene

        return input_scene.filter_points(self.compute_point_mask(input_scene.points))

    @staticmethod
    def name() -> str:
        """Return the registered transform name."""
        return "ScalePercentileFilterPoints"

    def state_dict(self) -> dict[str, Any]:
        """Return the transform state for serialization."""
        return {
            "name": self.name(),
            "version": self.version,
            "percentile_filter": self._percentile_filter,
        }

    @staticmethod
    def from_state_dict(state_dict: dict[str, Any]) -> "ScalePercentileFilterPoints":
        """Create a transform from a dictionary returned by :meth:`state_dict`."""
        if state_dict["name"] != "ScalePercentileFilterPoints":
            raise ValueError(
                f"Expected state_dict with name 'ScalePercentileFilterPoints', got {state_dict['name']} instead."
            )
        if "percentile_filter" not in state_dict:
            raise ValueError("State dictionary must contain a 'percentile_filter' key.")
        return ScalePercentileFilterPoints(percentile_filter=state_dict["percentile_filter"])
