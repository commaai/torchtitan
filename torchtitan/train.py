# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer


def _validation_only_steps() -> list[int]:
    raw_steps = os.getenv("TORCHTITAN_VALIDATE_STEPS", "")
    if not raw_steps:
        return []

    tokens = raw_steps.replace(",", ":").split(":")
    if any(not token.isdigit() or int(token) <= 0 for token in tokens):
        raise ValueError("TORCHTITAN_VALIDATE_STEPS must contain positive integers separated by ':' or ','")
    return list(dict.fromkeys(int(token) for token in tokens))


def main() -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    validation_steps = _validation_only_steps()
    if validation_steps:
        if not config.checkpoint.enable:
            raise ValueError("TORCHTITAN_VALIDATE_STEPS requires checkpointing to be enabled")
        if not hasattr(config, "validator") or not config.validator.enable:
            raise ValueError("TORCHTITAN_VALIDATE_STEPS requires an enabled validator")
        config.checkpoint.load_only = True
        config.checkpoint.enable_first_step_checkpoint = False
        config.checkpoint.exclude_from_loading = list(
            dict.fromkeys(
                [
                    *config.checkpoint.exclude_from_loading,
                    "optimizer",
                    "lr_scheduler",
                    "dataloader",
                    "train_state",
                ]
            )
        )
        if hasattr(config.dataloader, "shuffle_size"):
            config.dataloader.shuffle_size = config.training.local_batch_size * int(os.getenv("LOCAL_WORLD_SIZE", "1"))
            config.dataloader.num_writers = 1

    # NOTE: internal meta tooling relies on source="training".
    sl.init_structured_logger(
        source="training",
        # pyrefly: ignore [missing-attribute]
        output_dir=config.dump_folder,
        # pyrefly: ignore [missing-attribute]
        enable=config.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    trainer: Trainer | None = None

    try:
        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_states() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        if config.comm.mode == "local_tensor":  # pyrefly: ignore [missing-attribute]
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        trainer = config.build()  # pyrefly: ignore [missing-attribute]

        if (
            config.checkpoint.create_seed_checkpoint  # pyrefly: ignore[missing-attribute]
        ):
            assert int(os.environ["WORLD_SIZE"]) == 1, (
                "Must create seed checkpoint using a single device, to disable sharding."
            )
            assert (
                config.checkpoint.enable  # pyrefly: ignore [missing-attribute]
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        elif validation_steps:
            for step in validation_steps:
                logger.info("Loading checkpoint at step %s for validation-only run", step)
                trainer.checkpointer.load(step=step)
                trainer.step = step
                trainer.set_runtime_seed()
                trainer.validator.validate(trainer.model_parts, step)
            logger.info("Validation-only run completed")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            with sl.log_trace_span("torch_distributed_teardown"):
                torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
