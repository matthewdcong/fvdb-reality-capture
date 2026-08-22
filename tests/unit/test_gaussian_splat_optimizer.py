# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import unittest
from unittest.mock import patch

import pytest
import torch
from fvdb import morton

import fvdb_reality_capture as frc
from fvdb_reality_capture import GaussianSplat3d
from tests.unit.common import GettysburgGaussianSplatTestCase

pytest.importorskip("torch_dgx", reason="torch-dgx not available")


class GaussianSplatOptimizerRefinementRemappingTests(unittest.TestCase):
    def test_sorted_refinement_matches_two_pass_reference(self):
        num_gaussians = 8
        model = GaussianSplat3d.from_tensors(
            means=torch.tensor(
                [
                    [0.8, 0.1, 0.3],
                    [0.2, 0.7, 0.6],
                    [0.9, 0.8, 0.2],
                    [0.1, 0.4, 0.9],
                    [0.6, 0.3, 0.7],
                    [0.4, 0.9, 0.1],
                    [0.7, 0.2, 0.8],
                    [0.3, 0.6, 0.4],
                ]
            ),
            quats=torch.arange(num_gaussians * 4, dtype=torch.float32).reshape(num_gaussians, 4) + 1.0,
            log_scales=torch.linspace(-5.0, -2.0, num_gaussians * 3).reshape(num_gaussians, 3),
            logit_opacities=torch.linspace(-2.0, 2.0, num_gaussians),
            sh0=torch.arange(num_gaussians * 3, dtype=torch.float32).reshape(num_gaussians, 1, 3),
            shN=torch.arange(num_gaussians * 18, dtype=torch.float32).reshape(num_gaussians, 6, 3),
            accumulate_mean_2d_gradients=True,
        )
        model.requires_grad = True

        class _Scene:
            @staticmethod
            def spatial_scale(_mode):
                return 1.0

        config = frc.radiance_fields.GaussianSplatOptimizerConfig(
            insertion_duplication_factor=3,
            insertion_split_factor=3,
            opacity_updates_use_revised_formulation=True,
            post_refinement_sort=True,
            reset_opacities_every_n_refinements=-1,
            use_scales_for_deletion_after_n_refinements=100,
        )
        optimizer = frc.radiance_fields.GaussianSplatOptimizer.from_model_and_scene(model, _Scene(), config)

        # Initialize Adam, then make every gradient and moment entry identifiable by its source row.
        for group_index, param_group in enumerate(optimizer._optimizer.param_groups):
            param = param_group["params"][0]
            param.grad = torch.ones_like(param)
        optimizer.step()
        for group_index, param_group in enumerate(optimizer._optimizer.param_groups):
            param = param_group["params"][0]
            param.grad = torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape) + group_index * 1_000
            for state_index, state_name in enumerate(("exp_avg", "exp_avg_sq")):
                state = optimizer._optimizer.state[param][state_name]
                state.copy_(
                    torch.arange(state.numel(), dtype=state.dtype).reshape(state.shape)
                    + group_index * 10_000
                    + state_index * 100_000
                )

        duplication_indices = torch.tensor([2, 5])
        split_indices = torch.tensor([4])
        deletion_indices = torch.tensor([1])
        is_duplicated = torch.zeros(num_gaussians, dtype=torch.bool)
        is_duplicated[duplication_indices] = True
        is_split = torch.zeros(num_gaussians, dtype=torch.bool)
        is_split[split_indices] = True
        is_deleted = torch.zeros(num_gaussians, dtype=torch.bool)
        is_deleted[deletion_indices] = True
        kept_indices = torch.where(~(is_split | is_deleted))[0]

        old_parameters = {
            name: getattr(model, name).detach().clone()
            for name in ("means", "quats", "log_scales", "logit_opacities", "sh0", "shN")
        }
        old_gradients = {name: getattr(model, name).grad.detach().clone() for name in old_parameters}
        old_optimizer_state = {
            param_group["name"]: {
                state_name: optimizer._optimizer.state[param_group["params"][0]][state_name].detach().clone()
                for state_name in ("exp_avg", "exp_avg_sq")
            }
            for param_group in optimizer._optimizer.param_groups
        }

        # Reproduce the old compact/append-then-sort path as a correctness oracle.
        torch.manual_seed(1234)
        duplicated_gaussians = optimizer._compute_duplicated_gaussians(duplication_indices)
        split_gaussians = optimizer._compute_split_gaussians(split_indices)
        unsorted_parameters = {
            name: torch.cat(
                [old_parameters[name][kept_indices], duplicated_gaussians[name], split_gaussians[name]], dim=0
            )
            for name in old_parameters
        }
        bbox_min = torch.min(unsorted_parameters["means"])
        bbox_max = torch.max(unsorted_parameters["means"])
        normalized_means = (unsorted_parameters["means"] - bbox_min) / (bbox_max - bbox_min)
        codes = morton((normalized_means * ((1 << 21) - 1)).to(torch.int32))
        sort_indices = torch.argsort(codes)
        expected_parameters = {name: param[sort_indices] for name, param in unsorted_parameters.items()}

        num_new_gaussians = duplicated_gaussians["means"].shape[0] + split_gaussians["means"].shape[0]
        expected_gradients = {
            name: torch.cat([old_gradients[name][kept_indices], torch.zeros_like(param[-num_new_gaussians:])], dim=0)[
                sort_indices
            ]
            for name, param in unsorted_parameters.items()
        }
        expected_optimizer_state = {
            name: {
                state_name: torch.cat(
                    [state[kept_indices], torch.zeros_like(expected_parameters[name][-num_new_gaussians:])], dim=0
                )[sort_indices]
                for state_name, state in states.items()
            }
            for name, states in old_optimizer_state.items()
        }

        torch.manual_seed(1234)
        with (
            patch.object(optimizer, "_compute_insertion_masks", return_value=(is_duplicated, is_split)),
            patch.object(optimizer, "_compute_deletion_mask", return_value=is_deleted),
        ):
            optimizer.refine(zero_gradients=False)

        for name, expected in expected_parameters.items():
            torch.testing.assert_close(getattr(model, name), expected)
            torch.testing.assert_close(getattr(model, name).grad, expected_gradients[name])
        for param_group in optimizer._optimizer.param_groups:
            name = param_group["name"]
            state = optimizer._optimizer.state[param_group["params"][0]]
            for state_name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(state[state_name], expected_optimizer_state[name][state_name])


class GaussianSplatOptimizerTests(GettysburgGaussianSplatTestCase, unittest.TestCase):

    def test_serialize_optimizer(self):
        model_1 = self.model
        max_steps = 200 * len(self.training_dataset)
        config = frc.radiance_fields.GaussianSplatOptimizerConfig(
            use_scales_for_deletion_after_n_refinements=-1,
            use_screen_space_scales_for_refinement_until=0,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        optimizer_1 = frc.radiance_fields.GaussianSplatOptimizer.from_model_and_scene(
            model=self.model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=config,
        )
        optimizer_1.reset_learning_rates_and_decay(batch_size=1, expected_steps=max_steps)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pt", delete=True) as temp_file:
            # Save the state dict of the optimizer and model
            torch.save(self.model.state_dict(), temp_file.name + ".model")
            torch.save(optimizer_1.state_dict(), temp_file.name)

            # Run one step of the optimization and refinement
            optimizer_1.zero_grad()
            gt_img_1, pred_img_1, _ = self._render_one_image(model_1)
            loss_1 = torch.nn.functional.l1_loss(pred_img_1, gt_img_1)
            loss_1.backward()
            optimizer_1.refine()
            optimizer_1.step()
            optimizer_1.zero_grad()
            num_gaussians_after_refine = model_1.num_gaussians

            # Compute the rendered image and loss after one step
            gt_img_2, pred_img_2, _ = self._render_one_image(model_1)
            loss_2 = torch.nn.functional.l1_loss(pred_img_2, gt_img_2)
            # print(f"loss_2 = {loss_2.item()}")

            # Load the original model and optimizer from the saved state dict
            model_2 = GaussianSplat3d.from_state_dict(torch.load(temp_file.name + ".model", map_location=self.device))
            loaded_state_dict = torch.load(temp_file.name, map_location=self.device, weights_only=False)
            optimizer_2 = frc.radiance_fields.GaussianSplatOptimizer.from_state_dict(model_2, loaded_state_dict)

            # Run one step of of optimization and refinement with the loaded optimizer and model
            # and check that the results match the previous results
            optimizer_2.zero_grad()
            gt_img_3, pred_img_3, _ = self._render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_1, pred_img_3))
            loss_3 = torch.nn.functional.l1_loss(pred_img_3, gt_img_3)
            self.assertAlmostEqual(loss_1.item(), loss_3.item(), places=3)
            loss_3.backward()
            optimizer_2.refine()
            optimizer_2.step()
            optimizer_2.zero_grad()
            self.assertEqual(model_2.num_gaussians, num_gaussians_after_refine)

            # Compute the rendered image and loss after one step and check that it matches the previous result
            gt_img_4, pred_img_4, _ = self._render_one_image(model_2)
            self.assertTrue(torch.allclose(pred_img_2, pred_img_4, atol=1e-3))
            loss_4 = torch.nn.functional.l1_loss(pred_img_4, gt_img_4)
            self.assertAlmostEqual(loss_2.item(), loss_4.item(), places=3)


class GaussianSplatOptimizerRefinementTests(GettysburgGaussianSplatTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.optimizer_config = frc.radiance_fields.GaussianSplatOptimizerConfig(
            max_gaussians=-1,
            insertion_grad_2d_threshold_mode=frc.radiance_fields.InsertionGrad2dThresholdMode.CONSTANT,
            deletion_opacity_threshold=0.005,
            deletion_scale_3d_threshold=0.1,
            deletion_scale_2d_threshold=0.15,
            insertion_grad_2d_threshold=0.0002,
            insertion_scale_3d_threshold=0.01,
            insertion_scale_2d_threshold=0.05,
            opacity_updates_use_revised_formulation=False,
            insertion_split_factor=2,
            insertion_duplication_factor=2,
            means_lr=1.6e-4,
            log_scales_lr=5e-3,
            quats_lr=1e-3,
            logit_opacities_lr=5e-2,
            sh0_lr=2.5e-3,
            shN_lr=2.5e-3 / 20,
            spatial_scale_mode=frc.radiance_fields.SpatialScaleMode.ABSOLUTE_UNITS,
        )
        max_steps = 200 * len(self.training_dataset)
        self.optimizer = frc.radiance_fields.GaussianSplatOptimizer.from_model_and_scene(
            model=self.model,
            sfm_scene=self.training_dataset.sfm_scene,
            config=self.optimizer_config,
        )
        self.optimizer.reset_learning_rates_and_decay(batch_size=1, expected_steps=max_steps)

        initial_num_gaussians = self.model.num_gaussians
        print(f"Initial number of Gaussians: {initial_num_gaussians}")

        # Run a few steps of optimization to accumulate some gradients
        for _ in range(2):
            self.optimizer.zero_grad()
            gt_img, pred_img, _ = self._render_one_image(self.model)
            loss = torch.nn.functional.l1_loss(pred_img, gt_img)
            loss.backward()
            self.optimizer.step()

    def test_refinement_with_none_gradient_accumulation(self):
        """Regression test for issue #279: refine() must not crash when gradient accumulation state is None.

        When the Unscented Transform projection path is used (e.g., for OpenCV camera models),
        fvdb-core does not initialize accumulated_gradient_step_counts or
        accumulated_mean_2d_gradient_norms. Calling set_state() also resets them to None.
        The optimizer must handle this gracefully by skipping insertion (duplication/splitting).
        """
        model = self.model
        optimizer = self.optimizer

        # Simulate the UT projection path: set_state() resets gradient tensors to None
        with torch.no_grad():
            model.set_state(
                means=model.means,
                quats=model.quats,
                log_scales=model.log_scales,
                logit_opacities=model.logit_opacities,
                sh0=model.sh0,
                shN=model.shN,
            )
        self.assertIsNone(model.accumulated_gradient_step_counts)
        self.assertIsNone(model.accumulated_mean_2d_gradient_norms)

        # refine() should not crash — it should skip insertion and return zero counts
        refine_stats = optimizer.refine(zero_gradients=True)
        self.assertEqual(refine_stats["num_duplicated"], 0)
        self.assertEqual(refine_stats["num_split"], 0)

    def test_refinement_no_op(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        optimizer._config.use_scales_for_deletion_after_n_refinements = 100000
        optimizer._config.use_screen_space_scales_for_refinement_until = 0
        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]
        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertEqual(num_duplicated, 0)
        self.assertEqual(num_split, 0)
        self.assertEqual(num_deleted, 0)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def test_refinement_insertion_only(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        # Gaussians are inserted if their accumulated 2D gradient norm exceeds the threshold
        # and their scale is above the insertion scale threshold.
        # We'll set 20% of the gradients to be above the threshold
        rand_indices = torch.randperm(model.num_gaussians)[: max(1, model.num_gaussians // 5)]
        expected_num_splits, expected_num_duplicates = self._setup_gaussians_for_insertion(rand_indices)

        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]
        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertEqual(num_duplicated, expected_num_duplicates)
        self.assertEqual(num_split, expected_num_splits)
        self.assertEqual(num_deleted, 0)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def test_refinement_deletion_only(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        # Gaussians are deleted if their opacity is below the deletion threshold.
        # We'll set 20% of the opacities to be below the deletion threshold
        rand_indices = torch.randperm(model.num_gaussians)[: max(1, model.num_gaussians // 5)]
        expected_num_deletions = self._setup_gaussians_for_deletion(rand_indices)

        optimizer._config.use_scales_for_deletion_after_n_refinements = 100000
        optimizer._config.use_screen_space_scales_for_refinement_until = 0
        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]

        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertEqual(num_duplicated, 0)
        self.assertEqual(num_split, 0)
        self.assertEqual(num_deleted, expected_num_deletions)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def test_refinement_deletion_with_scales(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        # Gaussians are deleted if their opacity is below the deletion threshold,
        # and their maximum scale along any axis exceeds a threshold.
        permutation_indices = torch.randperm(model.num_gaussians)

        # Setup gaussians for deletion due to opacity and scale such that there are gaussians
        # that satisfy the opacity criteria, some that satisfy the scale criteria,
        # and some that satisfy both
        num_to_delete = max(1, model.num_gaussians // 8)
        low_opacity_start = 0
        low_opacity_end = num_to_delete
        large_scale_start = num_to_delete - num_to_delete // 3
        large_scale_end = num_to_delete + num_to_delete // 3
        delete_low_opacity_indices = permutation_indices[low_opacity_start:low_opacity_end]
        delete_large_indices = permutation_indices[large_scale_start:large_scale_end]
        print(delete_large_indices.shape, delete_low_opacity_indices.shape)

        expected_num_deletions = num_to_delete + num_to_delete // 3
        self._setup_gaussians_for_deletion(delete_low_opacity_indices, delete_large_indices)

        print("Expected deletions:", expected_num_deletions)
        optimizer._config.use_scales_for_deletion_after_n_refinements = -1
        optimizer._config.use_screen_space_scales_for_refinement_until = 0
        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]
        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertEqual(num_duplicated, 0)
        self.assertEqual(num_split, 0)
        self.assertEqual(num_deleted, expected_num_deletions)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def test_refinement_insertion_and_deletion_overlap_with_scales(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        # Gaussians are deleted if their opacity is below the deletion threshold,
        # and their maximum scale along any axis exceeds a threshold.
        permutation_indices = torch.randperm(model.num_gaussians)

        # Setup ranges of indices for insertion and deletion such that there is overlap
        # between the two sets of indices.
        num_gaussians_to_insert = max(1, model.num_gaussians // 5)
        num_gaussians_to_delete = max(1, model.num_gaussians // 7)

        num_gaussians_to_split = num_gaussians_to_insert // 2
        num_gaussians_to_duplicate = num_gaussians_to_insert - num_gaussians_to_split

        split_start_index = 0
        split_end_index = num_gaussians_to_insert // 2

        duplicate_start_index = split_end_index
        duplicate_end_index = duplicate_start_index + num_gaussians_to_duplicate

        # Ensure we choose deletion indices that overlap with insertion indices
        delete_start_index = split_end_index - num_gaussians_to_delete // 2
        delete_end_index = delete_start_index + num_gaussians_to_delete // 2

        # Make sure some of the deletion indices are also set to have large scales
        num_deleted = delete_end_index - delete_start_index
        delete_opacity_start_index = delete_start_index
        delete_opacity_end_index = delete_start_index + num_deleted // 2
        delete_scale_start_index = delete_start_index + num_deleted // 4
        delete_scale_end_index = delete_end_index

        # Get indices for insertion and deletion
        # We get a random set of indices and cut them up as follows:
        # [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, ..., i_{N-3}, i_{N-2}, i_{N-1}]
        #  ^---- split indices ----^^---- duplicate indices ----^
        #          ^---- delete opacity indices ----^
        #                 ^---- delete scale indices ----^
        permutation_indices = torch.randperm(model.num_gaussians)
        delete_opacity_indices = permutation_indices[delete_opacity_start_index:delete_opacity_end_index]
        delete_scale_indices = permutation_indices[delete_scale_start_index:delete_scale_end_index]
        insert_indices = permutation_indices[split_start_index:duplicate_end_index]

        self._setup_gaussians_for_insertion(insert_indices)
        self._setup_gaussians_for_deletion(delete_opacity_indices, delete_scale_indices)

        expected_num_splits = delete_start_index - split_start_index
        expected_num_duplicates = duplicate_end_index - delete_end_index
        expected_num_deletions = delete_end_index - delete_start_index
        print(
            "Expected splits, duplicates, deletions:",
            expected_num_splits,
            expected_num_duplicates,
            expected_num_deletions,
        )

        optimizer._config.use_scales_for_deletion_after_n_refinements = -1
        optimizer._config.use_screen_space_scales_for_refinement_until = 0
        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]
        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertGreater(num_duplicated, 0)
        self.assertGreater(num_deleted, 0)
        self.assertGreater(num_split, 0)

        self.assertEqual(num_duplicated, expected_num_duplicates)
        self.assertEqual(num_split, expected_num_splits)
        self.assertEqual(num_deleted, expected_num_deletions)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def test_refinement_insertion_and_deletion_no_overlap(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        num_gaussians_to_insert = max(1, model.num_gaussians // 5)
        num_gaussians_to_delete = max(1, model.num_gaussians // 7)

        permuted_indices = torch.randperm(model.num_gaussians)
        insert_indices = permuted_indices[:num_gaussians_to_insert]
        delete_indices = permuted_indices[num_gaussians_to_insert : num_gaussians_to_insert + num_gaussians_to_delete]

        expected_num_splits, expected_num_duplicates = self._setup_gaussians_for_insertion(insert_indices)
        expected_num_deletions = self._setup_gaussians_for_deletion(delete_indices)

        optimizer._config.use_scales_for_deletion_after_n_refinements = -100000
        optimizer._config.use_screen_space_scales_for_refinement_until = 0
        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]
        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertEqual(num_duplicated, expected_num_duplicates)
        self.assertEqual(num_split, expected_num_splits)
        self.assertEqual(num_deleted, expected_num_deletions)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def test_refinement_insertion_and_deletion_overlap(self):
        model = self.model
        optimizer = self.optimizer
        initial_num_gaussians = model.num_gaussians

        self._setup_no_refinement()

        num_gaussians_to_insert = max(1, model.num_gaussians // 5)
        num_gaussians_to_delete = max(1, model.num_gaussians // 7)

        num_gaussians_to_split = num_gaussians_to_insert // 2
        num_gaussians_to_duplicate = num_gaussians_to_insert - num_gaussians_to_split

        split_start_index = 0
        split_end_index = num_gaussians_to_insert // 2

        duplicate_start_index = split_end_index
        duplicate_end_index = duplicate_start_index + num_gaussians_to_duplicate

        # Ensure we choose deletion indices that overlap with insertion indices
        delete_start_index = split_end_index - num_gaussians_to_delete // 2
        delete_end_index = delete_start_index + num_gaussians_to_delete // 2

        # Overlap delete indices with insert indices
        # We get a random set of indices and cut them up as follows:
        # [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, ..., i_{N-3}, i_{N-2}, i_{N-1}]
        #  ^---- split indices ----^^---- duplicate indices ----^
        #              ^---- delete indices ----^
        permuted_indices = torch.randperm(model.num_gaussians)
        insert_indices = permuted_indices[split_start_index:duplicate_end_index]
        delete_indices = permuted_indices[delete_start_index:delete_end_index]

        self._setup_gaussians_for_insertion(insert_indices)
        self._setup_gaussians_for_deletion(delete_indices)

        # Deleted Gaussians are not split or duplicated, so we need to adjust our expected counts
        expected_num_splits = delete_start_index - split_start_index
        expected_num_duplicates = duplicate_end_index - delete_end_index
        expected_num_deletions = delete_end_index - delete_start_index
        print(
            "Expected splits, duplicates, deletions:",
            expected_num_splits,
            expected_num_duplicates,
            expected_num_deletions,
        )
        optimizer._config.use_scales_for_deletion_after_n_refinements = -100000
        optimizer._config.use_screen_space_scales_for_refinement_until = 0
        refine_stats = optimizer.refine(zero_gradients=True)
        num_duplicated = refine_stats["num_duplicated"]
        num_split = refine_stats["num_split"]
        num_deleted = refine_stats["num_deleted"]
        print(
            f"Refinement results: duplicated {num_duplicated}, split {num_split}, deleted {num_deleted}, total {model.num_gaussians}"
        )

        self.assertGreater(num_duplicated, 0)
        self.assertGreater(num_split, 0)
        self.assertGreater(num_deleted, 0)

        self.assertEqual(num_duplicated, expected_num_duplicates)
        self.assertEqual(num_split, expected_num_splits)
        self.assertEqual(num_deleted, expected_num_deletions)
        self.assertEqual(model.num_gaussians, initial_num_gaussians + num_duplicated + num_split - num_deleted)

    def _setup_no_refinement(self):
        """
        Set up model so that no refinement (insertion or deletion) will occur.
          1. Set all accumulated 2D gradient norms to be below the insertion threshold.
          2. Set all opacities to be above the deletion threshold.
        """
        model = self.model
        optimizer_config = self.optimizer_config

        with torch.no_grad():
            # We'll ensure no gradients are above the threshold
            model.accumulated_mean_2d_gradient_norms.fill_(optimizer_config.insertion_grad_2d_threshold * 0.9)

            # We'll set all opacities to be above the threshold so none get deleted
            del_thresh = torch.logit(torch.tensor([optimizer_config.deletion_opacity_threshold])).item()
            model.logit_opacities.fill_(del_thresh + 1.0)

            del_scale_thresh = torch.log(torch.tensor(optimizer_config.deletion_scale_3d_threshold / 2.0)).item()
            model.log_scales.fill_(del_scale_thresh)

    def _setup_gaussians_for_deletion(
        self, opacity_indices: torch.Tensor, scale_indices: torch.Tensor | None = None
    ) -> int:
        """
        Setup Gaussians so that those at the given indices will be deleted.

        Gaussians are deleted if their opacity is below the deletion threshold.
        We set the opacities of the given indices to be below the threshold.

        Args:
            opacity_indices (torch.Tensor): Indices of Gaussians to set up for deletion based on opacity.
            scale_indices (torch.Tensor, optional): Indices of Gaussians to set up for deletion based on scale.

        Returns:
            int: The expected number of deletions (i.e., the number of indices provided).
        """
        model = self.model
        optimizer_config = self.optimizer_config

        with torch.no_grad():
            # We'll set these opacities to be below the deletion threshold
            model.logit_opacities[opacity_indices] = (
                torch.logit(torch.tensor([optimizer_config.deletion_opacity_threshold])).item() - 1.0
            )
            expected_num_deletions = opacity_indices.shape[0]

            # We set these scales to be above the deletion threshold
            if scale_indices is not None:
                model.log_scales[scale_indices] = torch.log(
                    torch.tensor(optimizer_config.deletion_scale_3d_threshold * 2.0)
                ).item()
                expected_num_deletions = torch.unique(torch.cat([opacity_indices, scale_indices])).shape[0]

            return expected_num_deletions

    def _setup_gaussians_for_insertion(self, indices_to_insert: torch.Tensor) -> tuple[int, int]:
        """
        Setup Gaussians so that those at the given indices will be duplicated or split.

        Gaussians are split if their accumulated 2D gradient norm exceeds the threshold
        and their scale is above the insertion scale threshold.
        Gaussians are duplicated if their accumulated 2D gradient norm exceeds the threshold
        but their scale is below the insertion scale threshold.

        We set all the accumulated 2D gradient norms of all the given indices to be above the threshold
            so they are candidates for insertion.
        We set half the first half of the indices to have large scales (to be split) and
            the second half (to be duplicated).

        Args:
            indices_to_insert (torch.Tensor): Indices of Gaussians to set up for insertion (duplication or splitting).

        Returns:
            num_splits (int): The expected number of splits.
            num_duplicates (int): The expected number of duplications.
        """
        model = self.model
        optimizer_config = self.optimizer_config

        # Set some accumulated to be large to trigger refinement
        with torch.no_grad():
            # Gaussians are inserted if their accumulated 2D gradient norm exceeds the threshold
            # and their scale is above the insertion scale threshold.

            # We'll set 20% of the gradients to be above the threshold
            grad_norms = torch.full_like(
                model.accumulated_mean_2d_gradient_norms, optimizer_config.insertion_grad_2d_threshold * 0.9
            )
            grad_norms[indices_to_insert] = optimizer_config.insertion_grad_2d_threshold * 1.1
            model.accumulated_mean_2d_gradient_norms[...] = grad_norms
            model.accumulated_gradient_step_counts.fill_(1)

            log_threshold = optimizer_config.insertion_scale_3d_threshold
            split_indices = indices_to_insert[0 : max(1, indices_to_insert.shape[0] // 2)]
            duplicate_indices = indices_to_insert[max(1, indices_to_insert.shape[0] // 2) :]
            expected_num_splits = split_indices.shape[0]
            expected_num_duplicates = duplicate_indices.shape[0]

            model.log_scales[split_indices] = torch.log(torch.tensor(log_threshold * 2.0))
            model.log_scales[duplicate_indices] = torch.log(torch.tensor(log_threshold / 2.0))

            return expected_num_splits, expected_num_duplicates


if __name__ == "__main__":
    unittest.main()
