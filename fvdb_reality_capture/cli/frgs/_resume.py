# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Annotated, Any

import fvdb.viz as fviz
import torch
import tyro
from tyro.conf import arg

from fvdb_reality_capture.cli import BaseCommand
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatReconstruction,
    GaussianSplatReconstructionWriter,
    GaussianSplatReconstructionWriterConfig,
)

from ._common import save_model_from_runner


def _read_latest_checkpoint_manifest(checkpoints_path: pathlib.Path) -> tuple[int, pathlib.Path] | None:
    manifest_path = checkpoints_path / GaussianSplatReconstructionWriter.LATEST_CHECKPOINT_MANIFEST
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        return None
    step = manifest.get("step")
    checkpoint_value = manifest.get("checkpoint")
    if not isinstance(step, int) or not isinstance(checkpoint_value, str):
        return None

    relative_checkpoint_path = pathlib.Path(checkpoint_value)
    if relative_checkpoint_path.is_absolute() or not relative_checkpoint_path.parts:
        return None
    try:
        path_step = int(relative_checkpoint_path.parts[0])
    except ValueError:
        return None
    if path_step != step:
        return None

    checkpoints_path = checkpoints_path.resolve()
    checkpoint_path = (checkpoints_path / relative_checkpoint_path).resolve()
    if (
        not checkpoint_path.is_relative_to(checkpoints_path)
        or checkpoint_path.suffix.lower() not in (".pt", ".pth")
        or not checkpoint_path.is_file()
    ):
        return None
    return step, checkpoint_path


def _find_latest_checkpoint(run_path: pathlib.Path) -> pathlib.Path:
    """Find the highest-step completed checkpoint in a reconstruction run directory."""
    run_path = run_path.resolve()
    checkpoints_path = run_path / "checkpoints"
    if not checkpoints_path.is_dir():
        raise FileNotFoundError(f"Run directory {run_path} does not contain a checkpoints directory.")

    manifest_checkpoint = _read_latest_checkpoint_manifest(checkpoints_path)
    candidates: list[tuple[int, pathlib.Path]] = []
    for step_path in checkpoints_path.iterdir():
        if not step_path.is_dir():
            continue
        try:
            step = int(step_path.name)
        except ValueError:
            continue
        for suffix in ("*.pt", "*.pth"):
            candidates.extend((step, checkpoint_path.resolve()) for checkpoint_path in step_path.rglob(suffix))

    if not candidates:
        raise FileNotFoundError(f"Run directory {run_path} does not contain any completed checkpoints.")

    latest_step = max(step for step, _ in candidates)
    latest_candidates = [checkpoint_path for step, checkpoint_path in candidates if step == latest_step]
    if manifest_checkpoint is not None:
        manifest_step, manifest_path = manifest_checkpoint
        if manifest_step == latest_step and manifest_path in latest_candidates:
            return manifest_path

    reconstruct_candidates = [path for path in latest_candidates if path.name == "reconstruct_ckpt.pt"]
    if len(reconstruct_candidates) == 1:
        return reconstruct_candidates[0]
    if len(latest_candidates) == 1:
        return latest_candidates[0]

    candidates_text = ", ".join(str(path) for path in sorted(latest_candidates))
    raise RuntimeError(
        f"Run directory {run_path} has multiple checkpoints at latest step {latest_step} and no valid latest "
        f"manifest to disambiguate them: {candidates_text}"
    )


def _resolve_resume_path(path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Resolve a checkpoint file or a run directory to a checkpoint and optional append destination."""
    path = path.resolve()
    if path.is_file():
        return path, None
    if path.is_dir():
        return _find_latest_checkpoint(path), path
    if path.suffix.lower() in (".pt", ".pth"):
        # Preserve checkpoint-file behavior and let torch.load report a missing file. This also keeps the path easy to
        # replace with a mock or remote-backed file in callers that provide their own loading behavior.
        return path, None
    raise FileNotFoundError(f"Checkpoint file or run directory does not exist: {path}")


def _checkpoint_is_complete(checkpoint_state: dict[str, Any]) -> bool:
    """Return whether a checkpoint has reached its configured total optimization steps."""
    step = checkpoint_state.get("step")
    config = checkpoint_state.get("config")
    if not isinstance(step, int) or not isinstance(config, dict):
        return False

    max_steps = config.get("max_steps")
    if isinstance(max_steps, int):
        return step >= max_steps

    max_epochs = config.get("max_epochs")
    batch_size = config.get("batch_size")
    train_indices = checkpoint_state.get("train_indices")
    if not isinstance(max_epochs, int) or not isinstance(batch_size, int) or batch_size < 1:
        return False
    try:
        num_training_images = len(train_indices)
    except TypeError:
        return False
    steps_per_epoch = (num_training_images + batch_size - 1) // batch_size
    return step >= max_epochs * steps_per_epoch


@dataclass
class WriterConfig(GaussianSplatReconstructionWriterConfig):
    """
    Configuration for saving and logging metrics, images, and checkpoints.
    """

    # Path to save logs, checkpoints, and other output to.
    # Defaults to `frgs_logs` in the current working directory.
    log_path: pathlib.Path | None = pathlib.Path("frgs_logs")

    # How frequently to log metrics during reconstruction.
    log_every: int = 10


@dataclass
class Resume(BaseCommand):
    """
    Resume reconstructing a 3D Gaussian Splat radiance field from a checkpoint file or run directory. When given a
    run directory, this command loads its latest completed checkpoint and appends new output to the same directory.
    The dataset used to create the checkpoint must be at the same path as when the checkpoint was created.

    Example usage:

        # Resume reconstruction from a checkpoint and save the final model to out_resumed.ply
        frgs resume checkpoint.pt -o out_resumed.ply

        # Resume the latest completed checkpoint in an existing run and append new output to that run
        frgs resume frgs_logs/my_run
    """

    # Path to a checkpoint file or a run directory containing checkpoints. A run directory resumes its latest
    # completed checkpoint and receives the resumed job's output.
    checkpoint_path: tyro.conf.Positional[pathlib.Path]

    # Configure saving and logging metrics, images, and checkpoints.
    io: WriterConfig = field(default_factory=WriterConfig)

    # Name of the new output run when resuming from a checkpoint file. When resuming a run directory, its existing
    # name and location are used instead.
    run_name: Annotated[str | None, arg(aliases=["-n"])] = None

    # How frequently (in epochs) to update the viewer during reconstruction.
    # An epoch is one full pass through the dataset. If -1, do not visualize.
    update_viz_every: Annotated[float, arg(aliases=["-uv"])] = -1.0

    # The port to expose the viewer server on if update_viz_every > 0.
    viewer_port: Annotated[int, arg(aliases=["-p"])] = 8080

    # The IP address to expose the viewer server on if update_viz_every > 0.
    viewer_ip_address: Annotated[str, arg(aliases=["-ip"])] = "127.0.0.1"

    # Which device to use for reconstruction. Must be a cuda device. You can pass in a specific device index via
    # cuda:N where N is the device index, or "cuda" to use the default cuda device.
    # CPU is not supported. Default is "cuda".
    device: Annotated[str | torch.device, arg(aliases=["-d"])] = "cuda"

    # If set, show verbose debug messages.
    verbose: Annotated[bool, arg(aliases=["-v"])] = False

    # Optional path to save the final model. Path must end in .ply, .usdc, or .usdz.
    # If omitted, the final model is not exported.
    out_path: Annotated[pathlib.Path | None, arg(aliases=["-o"])] = None

    def execute(self) -> None:

        if self.device == "dgx":
            import torch_dgx

        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
        logger = logging.getLogger(__name__)

        checkpoint_path, run_path = _resolve_resume_path(self.checkpoint_path)
        logger.info(f"Loading checkpoint at {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if _checkpoint_is_complete(checkpoint_state):
            logger.info(f"Run is already complete at global step {checkpoint_state['step']}; nothing to resume.")
            return

        if run_path is not None and self.run_name is not None and self.run_name != run_path.name:
            raise ValueError(
                f"Run directory {run_path} determines the run name ({run_path.name}); remove --run-name or use "
                f"--run-name {run_path.name}."
            )

        writer = GaussianSplatReconstructionWriter(
            run_name=run_path.name if run_path is not None else self.run_name,
            save_path=run_path.parent if run_path is not None else self.io.log_path,
            config=self.io,
            exist_ok=run_path is not None,
        )
        if self.update_viz_every > 0:
            logger.info(f"Starting viewer server on {self.viewer_ip_address}:{self.viewer_port}")
            fviz.init(ip_address=self.viewer_ip_address, port=self.viewer_port, verbose=self.verbose)
            viz_scene = fviz.get_scene("Gaussian Splat Reconstruction Visualization")
        else:
            viz_scene = None

        runner = GaussianSplatReconstruction.from_state_dict(
            checkpoint_state,
            device=self.device,
            writer=writer,
            viz_scene=viz_scene,
            log_interval_steps=self.io.log_every,
            viz_update_interval_epochs=self.update_viz_every,
        )

        runner.optimize()

        if self.out_path is not None:
            logger.info(f"Saving final model to {self.out_path}")
            save_model_from_runner(self.out_path, runner)
