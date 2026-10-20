# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch

from fvdb_reality_capture.functional import _autograd


@pytest.mark.parametrize("depth_only", [False, True], ids=["unused_depth", "depth_only"])
def test_projection_gradient_materialization(depth_only):
    means = torch.ones(3, 3, requires_grad=True)
    quats = torch.zeros(3, 4)
    log_scales = torch.zeros(3, 3)
    world_to_cam = torch.eye(4).repeat(2, 1, 1)
    projection_matrices = torch.eye(3).repeat(2, 1, 1)
    radii = torch.ones(2, 3, 2, dtype=torch.int32)
    means2d = torch.zeros(2, 3, 2)
    depths = torch.zeros(2, 3)
    conics = torch.zeros(2, 3, 3)
    with (
        mock.patch.object(
            _autograd.F, "project_gaussians_analytic_fwd", return_value=(radii, means2d, depths, conics, None)
        ),
        mock.patch.object(
            _autograd.F,
            "project_gaussians_analytic_bwd",
            return_value=(torch.ones_like(means), None, None, None, None),
        ) as backward,
    ):
        outputs = _autograd._ProjectGaussiansFn.apply(
            means,
            quats,
            log_scales,
            world_to_cam,
            projection_matrices,
            33,
            17,
            0.3,
            0.01,
            100.0,
            0.0,
            False,
            False,
            None,
            None,
            None,
        )
        loss = outputs[2].sum() if depth_only else outputs[1].sum() + outputs[3].sum()
        loss.backward()

    backward.assert_called_once()
    grad_means2d, grad_depths, grad_conics = backward.call_args.args[11:14]
    if depth_only:
        torch.testing.assert_close(grad_depths, torch.ones_like(depths))
    else:
        assert grad_depths is None
    expected = torch.zeros_like if depth_only else torch.ones_like
    torch.testing.assert_close(grad_means2d, expected(means2d))
    torch.testing.assert_close(grad_conics, expected(conics))
    torch.testing.assert_close(means.grad, torch.ones_like(means))
