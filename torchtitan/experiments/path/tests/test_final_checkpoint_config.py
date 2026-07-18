# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from unittest import mock

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.experiments.path.onnx_checkpoint import PathOnnxCheckpointManager
from torchtitan.experiments.path.trainer import final_checkpoint_config


class TestFinalCheckpointConfig(unittest.TestCase):
    def test_reporter_path_without_training_id_is_legacy_path(self):
        with mock.patch.dict(
            os.environ,
            {
                "REPORTERV2_HOST": "mkv://reporter.example/reporterv2/",
                "REPORT_USER": "gill",
            },
            clear=True,
        ):
            config = final_checkpoint_config(
                flavor="w384", stem="prune10m_random4m_seed0", seed=1, steps=95049
            )

        self.assertIsInstance(config, PathOnnxCheckpointManager.Config)
        self.assertEqual(
            config.checkpoint_base_folder,
            "mkv://reporter.example/reporterv2/checkpoint",
        )
        self.assertEqual(config.folder, "gill/w384/prune10m_random4m_seed0_s1")

    def test_reporter_path_adds_a_distinct_safe_training_id_component(self):
        unsafe_training_id = "../slurm:123/4?% 🧪"
        common = {
            "REPORTERV2_HOST": "mkv://reporter.example/reporterv2",
            "REPORT_USER": "gill",
        }
        with mock.patch.dict(
            os.environ, common | {"REPORTERV2_TRAINING_ID": unsafe_training_id}, clear=True
        ):
            unsafe_config = final_checkpoint_config(
                flavor="w384", stem="prune10m_random4m_seed0", seed=1, steps=95049
            )
        with mock.patch.dict(
            os.environ, common | {"REPORTERV2_TRAINING_ID": "a_b"}, clear=True
        ):
            underscore_config = final_checkpoint_config(
                flavor="w384", stem="prune10m_random4m_seed0", seed=1, steps=95049
            )
        with mock.patch.dict(
            os.environ, common | {"REPORTERV2_TRAINING_ID": "a/b"}, clear=True
        ):
            slash_config = final_checkpoint_config(
                flavor="w384", stem="prune10m_random4m_seed0", seed=1, steps=95049
            )

        unsafe_parts = unsafe_config.folder.split("/")
        self.assertEqual(len(unsafe_parts), 4)
        self.assertEqual(unsafe_parts[0], "gill")
        self.assertTrue(unsafe_parts[1].startswith("run-"))
        self.assertNotIn("..", unsafe_parts[1])
        self.assertNotIn("%", unsafe_parts[1])
        self.assertNotIn("/", unsafe_parts[1])
        self.assertEqual(unsafe_parts[2:], ["w384", "prune10m_random4m_seed0_s1"])
        self.assertNotEqual(underscore_config.folder, slash_config.folder)

    def test_local_raid_path_ignores_training_id(self):
        with mock.patch.dict(
            os.environ,
            {
                "REPORTERV2_TRAINING_ID": "slurm:123/4",
                "REPORT_USER": "gill",
            },
            clear=True,
        ):
            config = final_checkpoint_config(
                flavor="w384", stem="prune10m_random4m_seed0", seed=1, steps=95049
            )

        self.assertIsInstance(config, CheckpointManager.Config)
        self.assertNotIsInstance(config, PathOnnxCheckpointManager.Config)
        self.assertEqual(
            config.folder,
            "/raid.unprotected/reports/gill_reports/prune_10m/vit/checkpoints/"
            "w384/prune10m_random4m_seed0_s1",
        )


if __name__ == "__main__":
    unittest.main()
