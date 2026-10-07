# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest import mock

import pytest

from fvdb_reality_capture.cli.frgs import _reconstruct
from fvdb_reality_capture.transforms import CropScene


@pytest.mark.parametrize("out_path", [None, Path("out.ply")])
def test_chunked_reconstruction_crops_and_merges(out_path, tmp_path):
    command = _reconstruct.Reconstruct(dataset_path=tmp_path, out_path=out_path, device="cpu", nchunks=(2, 1, 1))
    command.logger = mock.Mock()
    bboxes = [(0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 2.0, 1.0, 1.0)]
    scene, writer = mock.Mock(), mock.Mock()
    chunks = [mock.Mock(), mock.Mock()]
    runners = [mock.Mock(), mock.Mock()]
    splats = [mock.Mock(), mock.Mock()]
    with (
        mock.patch.object(command, "get_crop_bboxes", return_value=bboxes),
        mock.patch.object(CropScene, "__call__", side_effect=chunks) as crop,
        mock.patch.object(_reconstruct.GaussianSplatReconstruction, "from_sfm_scene", side_effect=runners) as create,
        mock.patch.object(_reconstruct.GaussianSplat3d, "from_ply", side_effect=[(s, {}) for s in splats]),
        mock.patch.object(_reconstruct.GaussianSplat3d, "cat") as merge,
        mock.patch.object(_reconstruct, "save_model_from_splats") as export,
    ):
        command._run_chunked_reconstruction(scene, writer, None)

    assert crop.call_args_list == [mock.call(scene), mock.call(scene)]
    assert [call.kwargs["sfm_scene"] for call in create.call_args_list] == chunks
    assert command.cfg.remove_gaussians_outside_scene_bbox
    for index, runner in enumerate(runners):
        runner.optimize.assert_called_once_with(True, f"recon_chunk_{index:04d}")
        runner.model.save_ply.assert_called_once()
    merge.assert_called_once_with(splats)
    if out_path is None:
        export.assert_not_called()
    else:
        export.assert_called_once_with(out_path, merge.return_value, runners[-1].reconstruction_metadata)
