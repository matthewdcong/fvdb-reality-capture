# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import hashlib
import pathlib
import struct
from collections import defaultdict

import numpy as np


_POINT3D_HEADER = struct.Struct("<QdddBBBdQ")
_TRACK_ELEMENT = np.dtype([("image_id", "<u4"), ("point2D_idx", "<u4")])
_POINT3D_DTYPE = np.dtype(
    [
        ("id", "<u8"),
        ("xyz", "<f8", (3,)),
        ("color", "u1", (3,)),
        ("error", "<f8"),
        ("track_length", "<u8"),
    ]
)
_VECTORIZED_CHUNK_SIZE = 1_000_000

assert _POINT3D_DTYPE.itemsize == _POINT3D_HEADER.size


def read_num_colmap_points(points_path: pathlib.Path) -> int:
    """Read the number of points from a binary COLMAP ``points3D.bin`` file."""
    with points_path.open("rb") as point_file:
        data = point_file.read(8)
    if len(data) != 8:
        raise ValueError(f"Invalid COLMAP points file {points_path}: missing point count")
    return struct.unpack("<Q", data)[0]


def read_colmap_points(
    points_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[int, np.ndarray]]:
    """
    Read a binary COLMAP point model without constructing a Python object for every point.

    COLMAP stores a fixed-size point header followed by a variable-length observation track. Hybrid models often
    append millions of points with empty tracks. Once all remaining records are fixed-size, this reader reads and
    converts them in large NumPy chunks instead of iterating through Python.

    The point IDs must be strictly increasing in file order. This matches the ordering previously produced by the
    loader, which sorted PyCOLMAP's point map by ID, without requiring a second point-ID array or a multi-gigabyte
    argsort.
    """
    num_points = read_num_colmap_points(points_path)
    file_size = points_path.stat().st_size
    minimum_file_size = 8 + num_points * _POINT3D_HEADER.size
    if file_size < minimum_file_size:
        raise ValueError(
            f"Invalid COLMAP points file {points_path}: size {file_size} is smaller than the minimum "
            f"{minimum_file_size} bytes required for {num_points} points"
        )

    points = np.empty((num_points, 3), dtype=np.float32)
    colors = np.empty((num_points, 3), dtype=np.uint8)
    errors = np.empty(num_points, dtype=np.float32)
    point_indices: defaultdict[int, list[int]] = defaultdict(list)
    point_id_hasher = hashlib.sha1()
    previous_point_id = -1

    with points_path.open("rb") as point_file:
        point_file.seek(8)
        point_idx = 0
        while point_idx < num_points:
            remaining_points = num_points - point_idx
            remaining_bytes = file_size - point_file.tell()

            if remaining_bytes == remaining_points * _POINT3D_HEADER.size:
                for start in range(0, remaining_points, _VECTORIZED_CHUNK_SIZE):
                    stop = min(start + _VECTORIZED_CHUNK_SIZE, remaining_points)
                    chunk_size = stop - start
                    record_chunk = np.fromfile(point_file, dtype=_POINT3D_DTYPE, count=chunk_size)
                    if record_chunk.size != chunk_size:
                        raise ValueError(
                            f"Invalid COLMAP points file {points_path}: truncated fixed-size point records"
                        )
                    if np.any(record_chunk["track_length"] != 0):
                        raise ValueError(
                            f"Invalid COLMAP points file {points_path}: nonempty track in fixed-size point suffix"
                        )
                    point_id_chunk = np.ascontiguousarray(record_chunk["id"], dtype=np.uint64)
                    if point_id_chunk.size > 0:
                        if point_id_chunk[0] <= previous_point_id or np.any(point_id_chunk[1:] <= point_id_chunk[:-1]):
                            raise ValueError(
                                f"Large COLMAP point model {points_path} is not ordered by point ID. "
                                "Reorder the binary model by point ID before loading it."
                            )
                        previous_point_id = int(point_id_chunk[-1])
                    point_id_hasher.update(point_id_chunk.view(np.uint8))

                    output_slice = slice(point_idx + start, point_idx + stop)
                    points[output_slice] = record_chunk["xyz"]
                    colors[output_slice] = record_chunk["color"]
                    errors[output_slice] = record_chunk["error"]
                point_idx = num_points
                break

            header_data = point_file.read(_POINT3D_HEADER.size)
            if len(header_data) != _POINT3D_HEADER.size:
                raise ValueError(
                    f"Invalid COLMAP points file {points_path}: truncated point header at index {point_idx}"
                )

            point_id, x, y, z, red, green, blue, error, track_length = _POINT3D_HEADER.unpack(header_data)
            if point_id <= previous_point_id:
                raise ValueError(
                    f"Large COLMAP point model {points_path} is not ordered by point ID. "
                    "Reorder the binary model by point ID before loading it."
                )
            previous_point_id = point_id
            point_id_hasher.update(header_data[:8])
            points[point_idx] = (x, y, z)
            colors[point_idx] = (red, green, blue)
            errors[point_idx] = error

            track_num_bytes = track_length * _TRACK_ELEMENT.itemsize
            track_data = point_file.read(track_num_bytes)
            if len(track_data) != track_num_bytes:
                raise ValueError(
                    f"Invalid COLMAP points file {points_path}: truncated track at point index {point_idx}"
                )
            track = np.frombuffer(track_data, dtype=_TRACK_ELEMENT)
            for image_id in track["image_id"]:
                point_indices[int(image_id)].append(point_idx)
            point_idx += 1

        if point_file.tell() != file_size:
            raise ValueError(f"Invalid COLMAP points file {points_path}: unexpected trailing data")

    if point_idx != num_points:
        raise ValueError(f"Invalid COLMAP points file {points_path}: expected {num_points} points, read {point_idx}")

    return (
        points,
        colors,
        errors,
        point_id_hasher.hexdigest(),
        {image_id: np.asarray(indices, dtype=np.int32) for image_id, indices in point_indices.items()},
    )
