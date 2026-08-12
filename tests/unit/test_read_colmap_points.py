# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import pathlib
import struct

import numpy as np

from fvdb_reality_capture.sfm_scene._read_colmap_points import read_colmap_points, read_num_colmap_points


def _write_points(path: pathlib.Path, points: list[tuple]) -> None:
    point_header = struct.Struct("<QdddBBBdQ")
    track_element = struct.Struct("<II")
    with path.open("wb") as point_file:
        point_file.write(struct.pack("<Q", len(points)))
        for point_id, xyz, color, error, track in points:
            point_file.write(point_header.pack(point_id, *xyz, *color, error, len(track)))
            for observation in track:
                point_file.write(track_element.pack(*observation))


def test_read_colmap_points_vectorizes_zero_track_suffix(tmp_path: pathlib.Path) -> None:
    points_path = tmp_path / "points3D.bin"
    source_points = [
        (4, (1.0, 2.0, 3.0), (10, 20, 30), 0.25, [(7, 2), (9, 3)]),
        (8, (4.0, 5.0, 6.0), (40, 50, 60), 0.5, [(7, 4)]),
        (10, (7.0, 8.0, 9.0), (70, 80, 90), 0.0, []),
        (12, (10.0, 11.0, 12.0), (100, 110, 120), 0.0, []),
    ]
    _write_points(points_path, source_points)

    points, colors, errors, point_id_hash, point_indices = read_colmap_points(points_path)

    assert read_num_colmap_points(points_path) == 4
    np.testing.assert_array_equal(points, np.asarray([point[1] for point in source_points], dtype=np.float32))
    np.testing.assert_array_equal(colors, np.asarray([point[2] for point in source_points], dtype=np.uint8))
    np.testing.assert_array_equal(errors, np.asarray([point[3] for point in source_points], dtype=np.float32))
    expected_ids = np.asarray([point[0] for point in source_points], dtype=np.uint64)
    assert point_id_hash == hashlib.sha1(expected_ids.view(np.uint8)).hexdigest()
    np.testing.assert_array_equal(point_indices[7], np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(point_indices[9], np.asarray([0], dtype=np.int32))


def test_read_colmap_points_rejects_nonascending_ids(tmp_path: pathlib.Path) -> None:
    points_path = tmp_path / "points3D.bin"
    _write_points(
        points_path,
        [
            (4, (1.0, 2.0, 3.0), (10, 20, 30), 0.25, []),
            (2, (4.0, 5.0, 6.0), (40, 50, 60), 0.5, []),
        ],
    )

    with np.testing.assert_raises_regex(ValueError, "not ordered by point ID"):
        read_colmap_points(points_path)
