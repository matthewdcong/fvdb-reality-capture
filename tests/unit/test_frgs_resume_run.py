# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import tempfile
import unittest
from unittest import mock

import torch

from fvdb_reality_capture.cli.frgs import _resume
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)
from fvdb_reality_capture.radiance_fields import gaussian_splat_reconstruction_writer as writer_module


class LatestCheckpointTests(unittest.TestCase):
    @staticmethod
    def _make_checkpoint(run_path: pathlib.Path, step: int, name: str = "reconstruct_ckpt.pt") -> pathlib.Path:
        checkpoint_path = run_path / "checkpoints" / f"{step:08d}" / name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()
        return checkpoint_path.resolve()

    def test_finds_highest_completed_checkpoint_when_manifest_is_stale(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_path = pathlib.Path(temporary_directory) / "my_run"
            old_checkpoint = self._make_checkpoint(run_path, 20)
            latest_checkpoint = self._make_checkpoint(run_path, 60)
            incomplete_checkpoint = run_path / "checkpoints" / "00000080" / ".reconstruct_ckpt.pt.partial.tmp"
            incomplete_checkpoint.parent.mkdir(parents=True)
            incomplete_checkpoint.touch()
            manifest = {
                "version": 1,
                "step": 20,
                "checkpoint": old_checkpoint.relative_to(run_path / "checkpoints").as_posix(),
            }
            (run_path / "checkpoints" / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(_resume._find_latest_checkpoint(run_path), latest_checkpoint)

    def test_manifest_disambiguates_checkpoints_at_the_same_step(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_path = pathlib.Path(temporary_directory) / "my_run"
            self._make_checkpoint(run_path, 20, "first.pt")
            expected_checkpoint = self._make_checkpoint(run_path, 20, "second.pt")
            manifest = {
                "version": 1,
                "step": 20,
                "checkpoint": expected_checkpoint.relative_to(run_path / "checkpoints").as_posix(),
            }
            (run_path / "checkpoints" / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")

            self.assertEqual(_resume._find_latest_checkpoint(run_path), expected_checkpoint)

    def test_run_without_completed_checkpoints_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_path = pathlib.Path(temporary_directory) / "my_run"
            (run_path / "checkpoints").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "does not contain any completed checkpoints"):
                _resume._find_latest_checkpoint(run_path)


class AtomicCheckpointTests(unittest.TestCase):
    def test_checkpoint_and_latest_manifest_are_published(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = pathlib.Path(temporary_directory)
            config = GaussianSplatReconstructionWriterConfig(
                save_images=False,
                save_checkpoints=True,
                save_plys=False,
                save_metrics=False,
            )
            writer = GaussianSplatReconstructionWriter("my_run", base_path, config=config)

            writer.save_checkpoint(20, "reconstruct_ckpt.pt", {"step": 20})

            checkpoint_path = base_path / "my_run" / "checkpoints" / "00000020" / "reconstruct_ckpt.pt"
            self.assertEqual(torch.load(checkpoint_path, weights_only=False), {"step": 20})
            manifest = json.loads((base_path / "my_run" / "checkpoints" / "latest.json").read_text())
            self.assertEqual(
                manifest,
                {"version": 1, "step": 20, "checkpoint": "00000020/reconstruct_ckpt.pt"},
            )
            self.assertEqual(list((base_path / "my_run").rglob("*.tmp")), [])

    def test_failed_save_preserves_previously_published_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = pathlib.Path(temporary_directory)
            config = GaussianSplatReconstructionWriterConfig(
                save_images=False,
                save_checkpoints=True,
                save_plys=False,
                save_metrics=False,
            )
            writer = GaussianSplatReconstructionWriter("my_run", base_path, config=config)
            writer.save_checkpoint(20, "reconstruct_ckpt.pt", {"value": "original"})
            checkpoint_path = base_path / "my_run" / "checkpoints" / "00000020" / "reconstruct_ckpt.pt"

            with (
                mock.patch.object(writer_module.torch, "save", side_effect=RuntimeError("interrupted")),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                writer.save_checkpoint(20, "reconstruct_ckpt.pt", {"value": "replacement"})

            self.assertEqual(torch.load(checkpoint_path, weights_only=False), {"value": "original"})
            self.assertEqual(list((base_path / "my_run").rglob("*.tmp")), [])


class ResumeRunTests(unittest.TestCase):
    @staticmethod
    def _checkpoint_state(step: int) -> dict:
        return {
            "step": step,
            "config": {"max_steps": None, "max_epochs": 2, "batch_size": 1},
            "train_indices": [0, 1],
        }

    def test_directory_resume_appends_to_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = pathlib.Path(temporary_directory)
            run_path = base_path / "my_run"
            checkpoint_path = run_path / "checkpoints" / "00000002" / "reconstruct_ckpt.pt"
            command = _resume.Resume(run_path, device="cpu")
            runner = mock.Mock()

            with (
                mock.patch.object(_resume, "_resolve_resume_path", return_value=(checkpoint_path, run_path)),
                mock.patch.object(_resume.torch, "load", return_value=self._checkpoint_state(2)),
                mock.patch.object(_resume, "GaussianSplatReconstructionWriter") as writer_class,
                mock.patch.object(_resume.GaussianSplatReconstruction, "from_state_dict", return_value=runner),
            ):
                command.execute()

            writer_class.assert_called_once_with(
                run_name="my_run",
                save_path=base_path,
                config=command.io,
                exist_ok=True,
            )
            runner.optimize.assert_called_once_with()

    def test_complete_run_is_a_noop(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_path = pathlib.Path(temporary_directory) / "my_run"
            checkpoint_path = run_path / "checkpoints" / "00000004" / "reconstruct_ckpt.pt"
            command = _resume.Resume(run_path, device="cpu")

            with (
                mock.patch.object(_resume, "_resolve_resume_path", return_value=(checkpoint_path, run_path)),
                mock.patch.object(_resume.torch, "load", return_value=self._checkpoint_state(4)),
                mock.patch.object(_resume, "GaussianSplatReconstructionWriter") as writer_class,
                mock.patch.object(_resume.GaussianSplatReconstruction, "from_state_dict") as from_state_dict,
            ):
                command.execute()

            writer_class.assert_not_called()
            from_state_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
