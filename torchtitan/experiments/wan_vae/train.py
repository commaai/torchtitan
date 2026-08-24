# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger
from torchtitan.trainer import Trainer


def main() -> None:
    init_logger()
    config_manager = ConfigManager()
    config = cast(Trainer.Config, config_manager.parse_args())
    trainer = config.build()
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
