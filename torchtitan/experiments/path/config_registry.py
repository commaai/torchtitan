# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import math
import os
from functools import partial
from xx.common.basedir import XX_BASEDIR
from xx.datasets.constants import BASE_DIR_GT, BASE_DIR_GT_10M
from xx.datasets.helpers import DEFAULT_BIG_TRAIN_LIST, DEFAULT_TRAIN_LIST
from xx.ml_tools.constants.model import (
    frame_constants_from_fps,
    FRAME_TYPE,
    INPUT_FRAMES_NAMES,
    ModelInputs,
    N_FRAMES,
    SUPERCOMBO_FPS,
    TEMPORAL_INPUTS,
)
from xx.training.path.config import DatasetConfig as XXPathDatasetConfig
from xx.training.path.hydra_configs import (
    DRIVING_HEADS,
    META_HEADS,
    PLAN_HEAD_SIZE,
    POSE_HEADS,
    TEMPORAL_META_HEADS,
)

import torch
import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common import Embedding, LayerNorm, Linear
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import PathDataLoader
from .loss import PathLoss
from .model import (
    Hydra,
    LinearEncoder,
    MuReadout,
    parallelize_path,
    PathHead,
    PathMLP,
    PathModel,
    PathSelfAttention,
    PathTransformer,
    PathTransformerBlock,
    PointSummarizer,
    Policy,
    ScaleLayer,
    TemporalPolicy,
    TemporalSummarizer,
    Vision,
)
from .onnx_checkpoint import PathOnnxCheckpointManager
from .trainer import final_checkpoint_config, PathTrainer
from .validate import PathValidator
from .vit import parallelize_vit, PatchEmbed, PlanHead, PlanViT, PlanViTLoss


_LINEAR_INIT = {
    "weight": partial(nn.init.normal_, mean=0.0, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}

VIT_HEAD_DIM = 64
VIT_NUM_LAYERS = 8
VIT_INPUT_SIZE = (1, 128, 256)
VIT_PATCH_SIZE = (1, 16, 8)
VIT_IN_CHANNELS = 24
VIT_BASE_WIDTH = 256
VIT_WIDTHS = {
    f"w{d}": d
    for d in (
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        640,
        896,
        1024,
        1280,
        1536,
        1792,
        2048,
        3072,
    )
}
VIT_STEPS = 512
MUP_PATTERN = (
    r"^(blocks\.\d+\.attention\.c_attn|blocks\.\d+\.attention\.c_proj"
    r"|blocks\.\d+\.mlp\.c_fc|blocks\.\d+\.mlp\.c_proj)\.weight$"
)

CONVNEXT_MUP_BASE_DIMS = (384, 768, 1536, 3072)
CONVNEXT_MUP_BASE_POLICY_DIM = 512
CONVNEXT_MUP_POLICY_HEADS = 8
CONVNEXT_MUP_STAGE0_WIDTHS = (7, 9, 11, 13, 15)
CONVNEXT_MUP_EFFECTIVE_TOKENS_PER_SAMPLE = 128
CONVNEXT_MUP_EFFECTIVE_TOKENS_PER_PARAMETER = 20
CONVNEXT_MUP_GLOBAL_BATCH = 128
CONVNEXT_MUP_SAMPLES_PER_SEGMENT = 8
CONVNEXT_500K_TRAIN_SEGMENTS = 449_231
# Tiny failed at step 23,552 and Small exhausted at step 24,528. The 500k
# list yields about seven usable samples per train-side segment, not the
# geometric cap of eight. Future one-pass runs stop before that measured wall.
CONVNEXT_500K_ONE_PASS_STEPS = 24_000
YAP_GLOBAL_BATCH = 2_048
YAP_500K_ONE_PASS_STEPS = 1_400
YAP_WARMUP_STEPS = 1_000
YAP_LR = 1e-3
YAP_WEIGHT_DECAY = 1e-3
YAP_CHECKPOINT_INTERVAL = 1_000
YAP_ANNEAL_STEPS = 100
YAP_ANNEAL500_STEPS = 500
YAP_ANNEAL1K_STEPS = 1_000
YAP_EXT_TOTAL_STEPS = 3 * YAP_500K_ONE_PASS_STEPS
YAP_EXT2_TOTAL_STEPS = 5 * YAP_500K_ONE_PASS_STEPS
YAP_EXT2_REMAINDER_LIST = os.path.join(
    XX_BASEDIR,
    "datasets/lists/train_2500k_remainder_20260716.txt",
)
CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER = 3
CONVNEXT_MUP_TARGET_STAGE0 = 9
CONVNEXT_WORLDMODEL_PLAN_HEAD_INIT_STD = 1e-3
CONVNEXT_WORLDMODEL_PLAN_HEAD_INIT_LOG_SIGMA_SCALE = 5.0
CONVNEXT_MUP_PARAMETERS = {
    7: 372_307,
    9: 607_317,
    11: 854_255,
    13: 1_146_761,
    15: 1_528_587,
    17: 1_912_357,
    19: 2_341_695,
    21: 2_870_337,
    23: 3_390_939,
    25: 3_957_109,
    27: 4_632_567,
    29: 5_290_001,
    31: 5_993_003,
    33: 6_815_277,
    35: 7_609_543,
    37: 8_449_377,
    39: 9_418_467,
    41: 10_349_565,
    43: 11_326_231,
    45: 12_442_137,
    47: 13_510_067,
    49: 14_623_565,
    51: 15_886_287,
    53: 17_091_049,
}
CONVNEXT_MUP_FORWARD_FLOPS = {
    7: 851_154_528,
    9: 1_311_118_272,
    11: 1_866_654_144,
    13: 2_518_331_328,
    15: 3_266_894_880,
    17: 4_110_856_992,
    19: 5_050_960_416,
    21: 6_088_123_776,
    23: 7_220_512_128,
    25: 8_449_041_792,
    27: 9_774_804_960,
    29: 11_195_619_552,
    31: 12_712_575_456,
    33: 14_326_938_432,
    35: 16_036_179_264,
    37: 17_841_561_408,
    39: 19_744_524_192,
    41: 21_742_191_264,
    43: 23_835_999_648,
    45: 26_027_562_240,
    47: 28_313_655_552,
    49: 30_695_890_176,
    51: 33_176_052_576,
    53: 35_750_572_128,
}
CONVNEXT_CLEAN_V2_WIDTHS = (13, 17, 21, 25, 29)
CONVNEXT_CLEAN_V2_BASE_WIDTH = 13
CONVNEXT_CLEAN_V2_BASE_STEPS = 2_400
CONVNEXT_CLEAN_V2_LEFT_FLANK_WIDTHS = (7, 9, 11)
CONVNEXT_CLEAN_V2_LEFT_FLANK_STEPS = 350_000 // CONVNEXT_MUP_GLOBAL_BATCH
CONVNEXT_CLEAN_V2_LEFT_FLANK_SCHEDULE_STEPS = {
    width: round(
        CONVNEXT_CLEAN_V2_BASE_STEPS
        * CONVNEXT_MUP_FORWARD_FLOPS[CONVNEXT_CLEAN_V2_BASE_WIDTH]
        / CONVNEXT_MUP_FORWARD_FLOPS[width]
    )
    for width in CONVNEXT_CLEAN_V2_LEFT_FLANK_WIDTHS
}
CONVNEXT_CLEAN_V2_STEPS = {
    width: round(
        CONVNEXT_CLEAN_V2_BASE_STEPS
        * CONVNEXT_MUP_FORWARD_FLOPS[CONVNEXT_CLEAN_V2_BASE_WIDTH]
        / CONVNEXT_MUP_FORWARD_FLOPS[width]
    )
    for width in CONVNEXT_CLEAN_V2_WIDTHS
}
# ConvNeXt Small first clears the locked-5k MAE<10 bar at its earliest saved
# milestone, 3,125 steps.  Use that horizon at w29 and derive every other
# horizon from exact forward FLOPs so all five runs end at one common compute.
# The nearly log-spaced widths cover the expected movement of the lowest point
# over four slices from 1.96e15 to 1.34e16 FLOPs (about w13 -> w27 under the
# measured/Kaplan allocation prior), leaving a measured point on both sides.
# This is a new cycle; Yassine fixed study weight decay at 1e-2 on 2026-07-15.
CONVNEXT_CLEAN_V3_WIDTHS = (7, 9, 11, 15, 21, 29, 41)
CONVNEXT_CLEAN_V3_REFERENCE_WIDTH = 29
CONVNEXT_CLEAN_V3_REFERENCE_STEPS = 3_125
CONVNEXT_CLEAN_V3_STEPS = {
    width: math.ceil(
        CONVNEXT_CLEAN_V3_REFERENCE_STEPS
        * CONVNEXT_MUP_FORWARD_FLOPS[CONVNEXT_CLEAN_V3_REFERENCE_WIDTH]
        / CONVNEXT_MUP_FORWARD_FLOPS[width]
    )
    for width in CONVNEXT_CLEAN_V3_WIDTHS
}
# Preserve global batch 128 while keeping every trajectory under the study's
# 40-minute wall-time ceiling.  Each config carries its local batch explicitly;
# the corresponding house launches use N=8/8/4/2/1 nodes respectively.
CONVNEXT_CLEAN_V3_LOCAL_BATCH = {7: 2, 9: 2, 11: 2, 15: 2, 21: 4, 29: 8, 41: 16}
# The clean-100k execution yielded 349,952 samples from 89,933 train-side
# segments.  678,192 train-side segments project 10% headroom over w11's
# 2,399,104-sample request.  The list is the same frozen seed-0 permutation;
# limit is applied after the house train/val split and never exposes val rows.
CONVNEXT_CLEAN_V3_TRAIN_LIST = "prune10m_uniform2236k_seed0.txt"
CONVNEXT_CLEAN_V3_TRAIN_LIMIT = 678_192
# left-flank horizons exceed the shared limit's projected yield; same 10%
# headroom convention at the measured 3.8913 samples/segment clean yield
CONVNEXT_CLEAN_V3_LEFT_FLANK_LIMITS = {7: 1_487_331, 9: 965_562}
CONVNEXT_MUP_TARGET_STEPS = round(
    CONVNEXT_MUP_EFFECTIVE_TOKENS_PER_PARAMETER
    * CONVNEXT_MUP_PARAMETERS[CONVNEXT_MUP_TARGET_STAGE0]
    / (CONVNEXT_MUP_EFFECTIVE_TOKENS_PER_SAMPLE * CONVNEXT_MUP_GLOBAL_BATCH)
)
CONVNEXT_MUP_BUDGET_FLOPS = (
    CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
    * CONVNEXT_MUP_FORWARD_FLOPS[CONVNEXT_MUP_TARGET_STAGE0]
    * CONVNEXT_MUP_GLOBAL_BATCH
    * CONVNEXT_MUP_TARGET_STEPS
)
CONVNEXT_MUP_STEPS = {
    width: round(
        CONVNEXT_MUP_BUDGET_FLOPS
        / (
            CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
            * forward_flops
            * CONVNEXT_MUP_GLOBAL_BATCH
        )
    )
    for width, forward_flops in CONVNEXT_MUP_FORWARD_FLOPS.items()
}
CONVNEXT_MUP_BUDGET2_SCALE = 3.16
CONVNEXT_MUP_BUDGET2_FLOPS = round(
    CONVNEXT_MUP_BUDGET_FLOPS * CONVNEXT_MUP_BUDGET2_SCALE
)
CONVNEXT_MUP_BUDGET2_WIDTHS = (9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29)
CONVNEXT_MUP_BUDGET2_STEPS = {
    width: round(
        CONVNEXT_MUP_BUDGET2_FLOPS
        / (
            CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
            * CONVNEXT_MUP_FORWARD_FLOPS[width]
            * CONVNEXT_MUP_GLOBAL_BATCH
        )
    )
    for width in CONVNEXT_MUP_BUDGET2_WIDTHS
}
CONVNEXT_MUP_BUDGET0_SCALE = 3.16
CONVNEXT_MUP_BUDGET0_FLOPS = round(
    CONVNEXT_MUP_BUDGET_FLOPS / CONVNEXT_MUP_BUDGET0_SCALE
)
CONVNEXT_MUP_BUDGET0_WIDTHS = (7, 9, 11, 13)
CONVNEXT_MUP_BUDGET0_STEPS = {
    width: round(
        CONVNEXT_MUP_BUDGET0_FLOPS
        / (
            CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
            * CONVNEXT_MUP_FORWARD_FLOPS[width]
            * CONVNEXT_MUP_GLOBAL_BATCH
        )
    )
    for width in CONVNEXT_MUP_BUDGET0_WIDTHS
}
CONVNEXT_MUP_BUDGET3_SCALE = 3.16
CONVNEXT_MUP_BUDGET3_FLOPS = round(
    CONVNEXT_MUP_BUDGET2_FLOPS * CONVNEXT_MUP_BUDGET3_SCALE
)
CONVNEXT_MUP_BUDGET3_WIDTHS = (19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41)
CONVNEXT_MUP_BUDGET3_STEPS = {
    width: round(
        CONVNEXT_MUP_BUDGET3_FLOPS
        / (
            CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
            * CONVNEXT_MUP_FORWARD_FLOPS[width]
            * CONVNEXT_MUP_GLOBAL_BATCH
        )
    )
    for width in CONVNEXT_MUP_BUDGET3_WIDTHS
}
CONVNEXT_MUP_BUDGET4_FLOPS = CONVNEXT_MUP_BUDGET2_FLOPS * 10
CONVNEXT_MUP_BUDGET4_WIDTHS = (
    23,
    25,
    27,
    29,
    31,
    33,
    35,
    37,
    39,
    41,
    43,
    45,
    47,
    49,
    51,
    53,
)
CONVNEXT_MUP_BUDGET4_STEPS = {
    width: round(
        CONVNEXT_MUP_BUDGET4_FLOPS
        / (
            CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
            * CONVNEXT_MUP_FORWARD_FLOPS[width]
            * CONVNEXT_MUP_GLOBAL_BATCH
        )
    )
    for width in CONVNEXT_MUP_BUDGET4_WIDTHS
}
# Yassine's 2026-07-15 04:33 instruction keeps study weight decay at 1e-2.
# Existing no-decay groups and the muP hidden-weight multiplier remain intact.
CONVNEXT_STUDY_BASE_WEIGHT_DECAY = 1e-2
CONVNEXT_MUP_VISION_PATTERN = (
    r"^vision\.encoder\.((stages\.\d+\.downsample\.1|stages\.\d+\.blocks\.\d+\.mlp\.fc[12]"
    r"|head\.fc)\.weight)$"
)
CONVNEXT_MUP_POLICY_PATTERN = (
    r"^(point_policy\.(summarizer\.mlp[12]|hydra\.head_mlp\.[^.]+)\.(c_fc|c_proj)"
    r"|temporal_policy\.(temporal_summarizer\.(mlp[12]\.(c_fc|c_proj)"
    r"|(desire_encoder|traffic_encoder|action_t_encoder)\.net\.2"
    r"|transformer\.layers\.\d+\.(attention\.(c_attn|c_proj)|mlp\.(c_fc|c_proj)))"
    r"|temporal_hydra\.head_mlp\.[^.]+\.(c_fc|c_proj)))\.weight$"
)


def _init_worldmodel_plan_bias_(bias: torch.Tensor) -> None:
    local = bias.to_local() if hasattr(bias, "to_local") else bias
    with torch.no_grad():
        local.zero_()
        local[PLAN_HEAD_SIZE // 2 :].fill_(
            math.log(CONVNEXT_WORLDMODEL_PLAN_HEAD_INIT_LOG_SIGMA_SCALE)
        )


def model_registry(
    flavor: str, model_config: PathModel.Config | None = None
) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=model_config or _model_config(flavor),
        parallelize_fn=parallelize_path,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def convnext_tiny() -> PathTrainer.Config:
    return _path("convnext_tiny")


def convnext_tiny_500k_one_pass() -> PathTrainer.Config:
    """Stock ConvNeXt Tiny over the canonical 2025 500k list, once.

    The 500k route list contains 449,231 train-side segments after the house
    train/val split.  Although each route offers at most eight samples, two
    completed attempts exhausted near 24,000 global batches after filtering.
    """
    steps = CONVNEXT_500K_ONE_PASS_STEPS
    config = convnext_tiny()
    config.dataloader = dataclasses.replace(
        config.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        limit=CONVNEXT_500K_TRAIN_SEGMENTS,
        one_pass=True,
    )
    config.training.steps = steps
    config.training.global_batch_size = 128
    config.lr_scheduler = dataclasses.replace(
        config.lr_scheduler,
        warmup_steps=round(steps * 0.01),
        total_steps=steps,
    )
    # Keep the production Path checkpoint manager inherited from convnext_tiny:
    # model/optimizer/scheduler/train state at step 1 and every 1,024 steps,
    # plus the final model. The Path loader cursor is not checkpointed, so these
    # are scoreable snapshots but not exact one-pass resume points.
    return dataclasses.replace(config)


def convnext_tiny_500k_one_pass_noval() -> PathTrainer.Config:
    """The 500k one-pass run with its in-training validator disabled."""
    config = convnext_tiny_500k_one_pass()
    config.validator.enable = False
    return dataclasses.replace(config)


def _yap(flavor: str) -> PathTrainer.Config:
    """yaP: stock recipe, constant lr 1e-3 after a 1k warmup, constant wd 1e-3,
    prod decay rules and prod checkpoint manager, 500k list once at prod batch."""
    config = _path(flavor)
    config.dataloader = dataclasses.replace(
        config.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        limit=CONVNEXT_500K_TRAIN_SEGMENTS,
        one_pass=True,
    )
    config.training.steps = YAP_500K_ONE_PASS_STEPS
    config.training.global_batch_size = YAP_GLOBAL_BATCH
    config.training.local_batch_size = 16
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=YAP_WARMUP_STEPS,
        total_steps=YAP_500K_ONE_PASS_STEPS,
        decay_ratio=0.0,
        decay_type="linear",
        min_lr_factor=1.0,
    )
    config.optimizer = _convnext_standard_optimizer_config(
        lr=YAP_LR,
        wd=YAP_WEIGHT_DECAY,
    )
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=YAP_CHECKPOINT_INTERVAL,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        save_model_state_dict=False,
        export_onnx=False,
    )
    config.metrics.log_freq = 16
    config.metrics.save_freq = 16
    config.validator.enable = False
    return dataclasses.replace(config)


def convnext_yap_atto_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_atto")


def convnext_yap_femto_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_femto")


def convnext_yap_pico_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_pico")


def convnext_yap_teeny_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_teeny")


def convnext_yap_tiny_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_tiny")


def convnext_yap_small_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_small")


def convnext_yap_weeny_500k_one_pass() -> PathTrainer.Config:
    return _yap("convnext_weeny")


YAP_500K_FINAL_CHECKPOINTS = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/db2b7734-baed-c436-c238-226ffb98dcc3/1400",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/628ae1fe-c4f5-2d88-6b18-8cc3e7005be8/1400",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/d5512bd6-8279-9a92-6224-dc67ac1d3884/1400",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/b98a1303-40bb-063d-196c-4dd6dc0e48f9/1400",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c4e93778-cf75-3c47-1101-361c66c8a4ea/1400",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/1f78e5da-c16e-7613-5998-04099410fb0f/1400",
    "convnext_weeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/4b3ceab4-24d1-7e7b-052d-62fdffcb3afd/1400",
}

YAP_500K_STEP1000_CHECKPOINTS = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/db2b7734-baed-c436-c238-226ffb98dcc3/1000",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/628ae1fe-c4f5-2d88-6b18-8cc3e7005be8/1000",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/d5512bd6-8279-9a92-6224-dc67ac1d3884/1000",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/b98a1303-40bb-063d-196c-4dd6dc0e48f9/1000",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c4e93778-cf75-3c47-1101-361c66c8a4ea/1000",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/1f78e5da-c16e-7613-5998-04099410fb0f/1000",
}

YAP_500K_ANNEAL1K_CHECKPOINTS = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/fc19ed10-4d1d-d6ee-80d5-3238c9a68a8c/2400",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/6224daab-05a2-f44c-e4ab-be93b7fa5c02/2400",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/03179bb3-16ca-ad97-fe23-7acb31572233/2400",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/8e581248-2a23-4f42-7896-9395bd504baa/2400",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/172ac331-693a-8fd6-a387-4c3f96098b25/2400",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/16750d93-44cc-9002-e714-7cca64930677/2400",
}

YAP_1M_ANNEAL1K_CHECKPOINTS = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/90d72957-6a22-5675-2df0-22b5d1a6713a/3800",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/425d6082-ba2d-9364-2c62-46f79b5d9c32/3800",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c1a94e7a-7204-1edd-069c-49bc29b73284/3800",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/161b7202-dc81-a0f1-7cef-6435ec01663c/3800",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/ba7666ee-5764-73e3-7d32-301985125d3f/3800",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c8df4492-9cfb-ff6f-5a15-6566f92ee615/3800",
}

YAP_1M_CHECKPOINTS: dict[str, str] = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/d33b25cb-3341-be0c-c085-8d7457d3d123/2800",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/e6a315d5-f8d1-4115-be30-adb803638301/2800",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/35435d98-badc-96f3-dcf4-a746cfd663f3/2800",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/63acb374-017d-dba5-d4e9-c3532d3f3b07/2800",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/e1de4000-7836-309e-8a49-e709e0fa7c8d/2800",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/30bb3dfe-5e85-3db4-ceef-0544317c30aa/2800",
    "convnext_weeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/e11bc19e-2caf-42e1-2c79-ec680b39875c/2800",
}
YAP_1P5M_CHECKPOINTS: dict[str, str] = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/d33b25cb-3341-be0c-c085-8d7457d3d123/4200",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/e6a315d5-f8d1-4115-be30-adb803638301/4200",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/35435d98-badc-96f3-dcf4-a746cfd663f3/4200",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/63acb374-017d-dba5-d4e9-c3532d3f3b07/4200",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/e1de4000-7836-309e-8a49-e709e0fa7c8d/4200",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/30bb3dfe-5e85-3db4-ceef-0544317c30aa/4200",
    "convnext_weeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/e11bc19e-2caf-42e1-2c79-ec680b39875c/4200",
}
YAP_2M_CHECKPOINTS: dict[str, str] = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/6983be55-fae9-ae68-202a-48bcb57df303/5600",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/56e61d6c-6413-f475-1883-83cf45762b4f/5600",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/fef9d464-e4e0-8084-0235-d72cc31f496c/5600",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/6801703c-7662-0850-8086-157fb4792d49/5600",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/16f54db7-e3a2-2d08-dad9-d9780397cdd2/5600",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c0f3b278-7d51-3d3a-51ec-9c8f902593cc/5600",
    "convnext_weeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/4fe4b061-04f6-5472-0347-f17eb6d8ad67/5600",
}
YAP_2P5M_CHECKPOINTS: dict[str, str] = {
    "convnext_atto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/6983be55-fae9-ae68-202a-48bcb57df303/7000",
    "convnext_femto": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/56e61d6c-6413-f475-1883-83cf45762b4f/7000",
    "convnext_pico": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/fef9d464-e4e0-8084-0235-d72cc31f496c/7000",
    "convnext_teeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/6801703c-7662-0850-8086-157fb4792d49/7000",
    "convnext_tiny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/16f54db7-e3a2-2d08-dad9-d9780397cdd2/7000",
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c0f3b278-7d51-3d3a-51ec-9c8f902593cc/7000",
    "convnext_weeny": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/4fe4b061-04f6-5472-0347-f17eb6d8ad67/7000",
}
YAP_500K_STEP1000_ANNEAL1K_CHECKPOINTS: dict[str, str] = {
    "convnext_small": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/6869523d-3dd0-221d-4cd2-66c243baae0b/2000",
}


def _yap_mark_checkpoint(
    checkpoints: dict[str, str], mark: str, flavor: str
) -> str:
    if flavor not in checkpoints:
        raise KeyError(
            f"yaP {mark} checkpoint for {flavor} is not stamped yet: fill the "
            f"YAP_{mark}_CHECKPOINTS dict with the ext run's step-save URI"
        )
    return checkpoints[flavor]


def _convnext_yap_validate_study5000(flavor: str) -> PathTrainer.Config:
    return _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(
            YAP_500K_FINAL_CHECKPOINTS, "500K_FINAL", flavor
        ),
        base_config=_yap(flavor),
    )


def convnext_yap_atto_500k_one_pass_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_validate_study5000("convnext_atto")


def convnext_yap_femto_500k_one_pass_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_validate_study5000("convnext_femto")


def convnext_yap_pico_500k_one_pass_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_validate_study5000("convnext_pico")


def convnext_yap_teeny_500k_one_pass_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_validate_study5000("convnext_teeny")


def _convnext_yap_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a yaP final checkpoint on the 500k list's own val split (device holdout)."""
    config = _convnext_yap_validate_study5000(flavor)
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_atto")


def convnext_yap_femto_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_femto")


def convnext_yap_pico_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_pico")


def convnext_yap_teeny_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_teeny")


def convnext_yap_tiny_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_tiny")


def convnext_yap_small_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_small")


def convnext_yap_weeny_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_validate_500kval("convnext_weeny")


def _convnext_yap_anneal1k_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a yaP anneal1k checkpoint on the 500k list's own val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=YAP_500K_ANNEAL1K_CHECKPOINTS[flavor],
        base_config=_yap(flavor),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_500k_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_anneal1k_validate_500kval("convnext_atto")


def convnext_yap_femto_500k_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_anneal1k_validate_500kval("convnext_femto")


def convnext_yap_pico_500k_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_anneal1k_validate_500kval("convnext_pico")


def convnext_yap_teeny_500k_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_anneal1k_validate_500kval("convnext_teeny")


def convnext_yap_tiny_500k_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_anneal1k_validate_500kval("convnext_tiny")


def convnext_yap_small_500k_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_anneal1k_validate_500kval("convnext_small")


def _convnext_yap_1m_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a raw yaP 1M-mark checkpoint on the 500k list's own val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(YAP_1M_CHECKPOINTS, "1M", flavor),
        base_config=_yap(flavor),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_atto")


def convnext_yap_femto_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_femto")


def convnext_yap_pico_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_pico")


def convnext_yap_teeny_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_teeny")


def convnext_yap_tiny_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_tiny")


def convnext_yap_small_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_small")


def convnext_yap_weeny_1m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_500kval("convnext_weeny")


def _convnext_yap_1m_validate_study5000(flavor: str) -> PathTrainer.Config:
    return _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(YAP_1M_CHECKPOINTS, "1M", flavor),
        base_config=_yap(flavor),
    )


def convnext_yap_atto_1m_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_study5000("convnext_atto")


def convnext_yap_femto_1m_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_study5000("convnext_femto")


def convnext_yap_pico_1m_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_study5000("convnext_pico")


def convnext_yap_teeny_1m_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_study5000("convnext_teeny")


def convnext_yap_tiny_1m_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_study5000("convnext_tiny")


def convnext_yap_small_1m_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_1m_validate_study5000("convnext_small")


def _convnext_yap_1m_anneal1k_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a yaP 1M-mark anneal1k checkpoint on the 500k list's own val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=YAP_1M_ANNEAL1K_CHECKPOINTS[flavor],
        base_config=_yap(flavor),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_1m_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_anneal1k_validate_500kval("convnext_atto")


def convnext_yap_femto_1m_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_anneal1k_validate_500kval("convnext_femto")


def convnext_yap_pico_1m_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_anneal1k_validate_500kval("convnext_pico")


def convnext_yap_teeny_1m_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_anneal1k_validate_500kval("convnext_teeny")


def convnext_yap_tiny_1m_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_anneal1k_validate_500kval("convnext_tiny")


def convnext_yap_small_1m_anneal1k_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1m_anneal1k_validate_500kval("convnext_small")


def _convnext_yap_1p5m_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a raw yaP 1.5M-mark checkpoint on the 500k list's own val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(YAP_1P5M_CHECKPOINTS, "1P5M", flavor),
        base_config=_yap(flavor),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_atto")


def convnext_yap_femto_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_femto")


def convnext_yap_pico_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_pico")


def convnext_yap_teeny_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_teeny")


def convnext_yap_tiny_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_tiny")


def convnext_yap_small_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_small")


def convnext_yap_weeny_1p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_1p5m_validate_500kval("convnext_weeny")


def _convnext_yap_2m_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a raw yaP 2M-mark checkpoint on the 500k list's own val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(YAP_2M_CHECKPOINTS, "2M", flavor),
        base_config=_yap(flavor),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_atto")


def convnext_yap_femto_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_femto")


def convnext_yap_pico_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_pico")


def convnext_yap_teeny_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_teeny")


def convnext_yap_tiny_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_tiny")


def convnext_yap_small_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_small")


def convnext_yap_weeny_2m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2m_validate_500kval("convnext_weeny")


def _convnext_yap_2p5m_validate_500kval(flavor: str) -> PathTrainer.Config:
    """Score a raw yaP 2.5M-mark checkpoint on the 500k list's own val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(YAP_2P5M_CHECKPOINTS, "2P5M", flavor),
        base_config=_yap(flavor),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_atto_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_atto")


def convnext_yap_femto_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_femto")


def convnext_yap_pico_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_pico")


def convnext_yap_teeny_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_teeny")


def convnext_yap_tiny_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_tiny")


def convnext_yap_small_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_small")


def convnext_yap_weeny_2p5m_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_2p5m_validate_500kval("convnext_weeny")


def convnext_yap_small_step1000_validate_500kval() -> PathTrainer.Config:
    """Score the raw small trunk step-1000 checkpoint on the 500k val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=YAP_500K_STEP1000_CHECKPOINTS["convnext_small"],
        base_config=_yap("convnext_small"),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_small_step1000_anneal1k_validate_500kval() -> PathTrainer.Config:
    """Score the small step-1000 anneal1k output on the 500k val split."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=_yap_mark_checkpoint(
            YAP_500K_STEP1000_ANNEAL1K_CHECKPOINTS,
            "500K_STEP1000_ANNEAL1K",
            "convnext_small",
        ),
        base_config=_yap("convnext_small"),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def _convnext_yap_step1000_validate_study5000(flavor: str) -> PathTrainer.Config:
    return _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=YAP_500K_STEP1000_CHECKPOINTS[flavor],
        base_config=_yap(flavor),
    )


def convnext_yap_atto_500k_step1000_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_step1000_validate_study5000("convnext_atto")


def convnext_yap_femto_500k_step1000_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_step1000_validate_study5000("convnext_femto")


def convnext_yap_pico_500k_step1000_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_step1000_validate_study5000("convnext_pico")


def convnext_yap_teeny_500k_step1000_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_step1000_validate_study5000("convnext_teeny")


def convnext_yap_tiny_500k_step1000_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_step1000_validate_study5000("convnext_tiny")


def convnext_yap_small_500k_step1000_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_step1000_validate_study5000("convnext_small")


def _yap_ext(flavor: str) -> PathTrainer.Config:
    """Resume yaP full state at step 1400 on the full 2.5M list with the prod
    full-output loader, constant lr 1e-3 with no warmup; full-state saves land
    exactly on the 1M (2800) and 1.5M (4200) consumption marks. one_pass stays
    off: the audit counts samples from process start and cannot reconcile with
    a restored trainer counter, and prod itself trains epochs."""
    config = _yap(flavor)
    config.dataloader = dataclasses.replace(
        config.dataloader,
        dataset=DEFAULT_BIG_TRAIN_LIST,
        limit=None,
        one_pass=False,
    )
    config.training.steps = YAP_EXT_TOTAL_STEPS
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=YAP_EXT_TOTAL_STEPS,
        decay_ratio=0.0,
        decay_type="linear",
        min_lr_factor=1.0,
    )
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=YAP_500K_ONE_PASS_STEPS,
        initial_load_path=_yap_mark_checkpoint(
            YAP_500K_FINAL_CHECKPOINTS, "500K_FINAL", flavor
        ),
        initial_load_model_only=False,
        allow_partial_initial_load=False,
        load_only=False,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        save_model_state_dict=False,
        export_onnx=False,
    )
    config.validator.enable = False
    return dataclasses.replace(config)


def convnext_yap_atto_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_atto")


def convnext_yap_femto_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_femto")


def convnext_yap_pico_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_pico")


def convnext_yap_teeny_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_teeny")


def convnext_yap_tiny_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_tiny")


def convnext_yap_small_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_small")


def convnext_yap_weeny_ext() -> PathTrainer.Config:
    return _yap_ext("convnext_weeny")


def _yap_ext2(flavor: str) -> PathTrainer.Config:
    """Resume yaP full state at the 1.5M mark (step 4200) on the trunk-disjoint
    2M-segment remainder of the 2.5M list, one partial epoch with the prod
    full-output loader; the VisionTargets2 shape probe reads the covered trunk
    list. Full-state saves land on the 2M (5600) and 2.5M (7000) marks."""
    config = _yap(flavor)
    config.dataloader = dataclasses.replace(
        config.dataloader,
        dataset=YAP_EXT2_REMAINDER_LIST,
        shape_probe_dataset=DEFAULT_TRAIN_LIST,
        limit=None,
        one_pass=False,
    )
    config.training.steps = YAP_EXT2_TOTAL_STEPS
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=YAP_EXT2_TOTAL_STEPS,
        decay_ratio=0.0,
        decay_type="linear",
        min_lr_factor=1.0,
    )
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=YAP_500K_ONE_PASS_STEPS,
        initial_load_path=_yap_mark_checkpoint(YAP_1P5M_CHECKPOINTS, "1P5M", flavor),
        initial_load_model_only=False,
        allow_partial_initial_load=False,
        load_only=False,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        save_model_state_dict=False,
        export_onnx=False,
    )
    config.validator.enable = False
    return dataclasses.replace(config)


def convnext_yap_atto_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_atto")


def convnext_yap_femto_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_femto")


def convnext_yap_pico_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_pico")


def convnext_yap_teeny_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_teeny")


def convnext_yap_tiny_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_tiny")


def convnext_yap_small_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_small")


def convnext_yap_weeny_ext2() -> PathTrainer.Config:
    return _yap_ext2("convnext_weeny")


def _convnext_yap_anneal(flavor: str, source: str) -> PathTrainer.Config:
    """Resume one yaP full-state checkpoint and linearly cool its LR for 100 steps."""
    if source == "step1000":
        source_step = 1_000
        checkpoint = YAP_500K_STEP1000_CHECKPOINTS[flavor]
    elif source == "final":
        source_step = YAP_500K_ONE_PASS_STEPS
        checkpoint = YAP_500K_FINAL_CHECKPOINTS[flavor]
    elif source == "1m":
        source_step = 2 * YAP_500K_ONE_PASS_STEPS
        checkpoint = _yap_mark_checkpoint(YAP_1M_CHECKPOINTS, "1M", flavor)
    elif source == "1p5m":
        source_step = 3 * YAP_500K_ONE_PASS_STEPS
        checkpoint = _yap_mark_checkpoint(YAP_1P5M_CHECKPOINTS, "1P5M", flavor)
    else:
        raise ValueError(f"unknown yaP anneal source: {source}")

    target_step = source_step + YAP_ANNEAL_STEPS
    config = _yap(flavor)
    config.dataloader = dataclasses.replace(config.dataloader, one_pass=False)
    config.training.steps = target_step
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=target_step,
        decay_ratio=YAP_ANNEAL_STEPS / target_step,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=target_step,
        initial_load_path=checkpoint,
        initial_load_model_only=False,
        allow_partial_initial_load=False,
        load_only=False,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        export_onnx=False,
    )
    config.metrics.log_freq = 1
    config.metrics.save_freq = YAP_ANNEAL_STEPS
    config.validator.enable = False
    return dataclasses.replace(config)


def _convnext_yap_anneal_validate_study5000(
    flavor: str, source: str
) -> PathTrainer.Config:
    short_flavor = flavor.removeprefix("convnext_").upper()
    env_name = f"YAP_{short_flavor}_{source.upper()}_ANNEAL_CHECKPOINT"
    checkpoint = os.environ.get(env_name)
    if not checkpoint:
        raise ValueError(f"{env_name} must name the annealed checkpoint")
    return _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=checkpoint,
        base_config=_yap(flavor),
    )


def convnext_yap_atto_500k_step1000_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_atto", "step1000")


def convnext_yap_femto_500k_step1000_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_femto", "step1000")


def convnext_yap_pico_500k_step1000_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_pico", "step1000")


def convnext_yap_teeny_500k_step1000_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_teeny", "step1000")


def convnext_yap_tiny_500k_step1000_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_tiny", "step1000")


def convnext_yap_small_500k_step1000_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_small", "step1000")


def convnext_yap_atto_500k_final_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_atto", "final")


def convnext_yap_femto_500k_final_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_femto", "final")


def convnext_yap_pico_500k_final_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_pico", "final")


def convnext_yap_teeny_500k_final_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_teeny", "final")


def convnext_yap_tiny_500k_final_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_tiny", "final")


def convnext_yap_small_500k_final_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_small", "final")


def convnext_yap_small_1m_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_small", "1m")


def convnext_yap_small_1p5m_anneal100() -> PathTrainer.Config:
    return _convnext_yap_anneal("convnext_small", "1p5m")


YAP_SMALL_ANNEAL100_CHECKPOINTS: dict[str, str] = {
    "step1000": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/436c5c0d-540a-fe53-5e5c-3ba986301556/1100",
    "final": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/b4d04d71-f019-a5fa-8c9f-142cb9ef2b40/1500",
    "1m": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/c901b79f-188d-af2d-e3b3-7f050857bc37/2900",
    "1p5m": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/455a4853-6343-f9d3-1403-7030bfcf43ae/4300",
}


def _convnext_yap_small_anneal100_validate_500kval(
    source: str,
) -> PathTrainer.Config:
    """Score one small anneal100 output on the 500k list's own val split."""
    if source not in YAP_SMALL_ANNEAL100_CHECKPOINTS:
        raise KeyError(
            f"yaP small {source} anneal100 checkpoint is not stamped yet: fill "
            "the YAP_SMALL_ANNEAL100_CHECKPOINTS dict with the run's final "
            "save URI"
        )
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=YAP_SMALL_ANNEAL100_CHECKPOINTS[source],
        base_config=_yap("convnext_small"),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_small_step1000_anneal100_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal100_validate_500kval("step1000")


def convnext_yap_small_final_anneal100_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal100_validate_500kval("final")


def convnext_yap_small_1m_anneal100_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal100_validate_500kval("1m")


def convnext_yap_small_1p5m_anneal100_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal100_validate_500kval("1p5m")


YAP_SMALL_ANNEAL500_CHECKPOINTS: dict[str, str] = {
    "final": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/24928832-b239-e65c-09cd-50bcbbea0440/1900",
    "1m": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/666652eb-4030-ca7b-526d-f2a71f65e6c2/3300",
    "1p5m": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/4f403721-2a76-aeb0-30c5-9b7105ac7dd3/4700",
}


def _convnext_yap_small_anneal500_validate_500kval(
    source: str,
) -> PathTrainer.Config:
    """Score one small anneal500 output on the 500k list's own val split."""
    if source not in YAP_SMALL_ANNEAL500_CHECKPOINTS:
        raise KeyError(
            f"yaP small {source} anneal500 checkpoint is not stamped yet: fill "
            "the YAP_SMALL_ANNEAL500_CHECKPOINTS dict with the run's final "
            "save URI"
        )
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=YAP_SMALL_ANNEAL500_CHECKPOINTS[source],
        base_config=_yap("convnext_small"),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_yap_small_final_anneal500_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal500_validate_500kval("final")


def convnext_yap_small_1m_anneal500_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal500_validate_500kval("1m")


def convnext_yap_small_1p5m_anneal500_validate_500kval() -> PathTrainer.Config:
    return _convnext_yap_small_anneal500_validate_500kval("1p5m")


def convnext_yap_atto_500k_step1000_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_atto", "step1000")


def convnext_yap_femto_500k_step1000_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_femto", "step1000")


def convnext_yap_pico_500k_step1000_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_pico", "step1000")


def convnext_yap_teeny_500k_step1000_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_teeny", "step1000")


def convnext_yap_tiny_500k_step1000_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_tiny", "step1000")


def convnext_yap_small_500k_step1000_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_small", "step1000")


def convnext_yap_atto_500k_final_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_atto", "final")


def convnext_yap_femto_500k_final_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_femto", "final")


def convnext_yap_pico_500k_final_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_pico", "final")


def convnext_yap_teeny_500k_final_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_teeny", "final")


def convnext_yap_tiny_500k_final_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_tiny", "final")


def convnext_yap_small_500k_final_anneal100_validate_study5000() -> PathTrainer.Config:
    return _convnext_yap_anneal_validate_study5000("convnext_small", "final")


def _yap_anneal1k(
    flavor: str, source_step: int, checkpoint_uri: str
) -> PathTrainer.Config:
    """Resume one yaP full-state checkpoint and linearly cool its LR from 1e-3
    to zero over 1,000 steps. one_pass stays off: the audit counts samples from
    process start and cannot reconcile with a restored trainer counter."""
    total_steps = source_step + YAP_ANNEAL1K_STEPS
    config = _yap(flavor)
    config.dataloader = dataclasses.replace(config.dataloader, one_pass=False)
    config.training.steps = total_steps
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=total_steps,
        decay_ratio=YAP_ANNEAL1K_STEPS / total_steps,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=total_steps,
        initial_load_path=checkpoint_uri,
        initial_load_model_only=False,
        allow_partial_initial_load=False,
        load_only=False,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        export_onnx=False,
    )
    config.metrics.log_freq = 1
    config.metrics.save_freq = 100
    config.validator.enable = False
    return dataclasses.replace(config)


def _yap_500k_anneal1k(flavor: str) -> PathTrainer.Config:
    return _yap_anneal1k(
        flavor, YAP_500K_ONE_PASS_STEPS, YAP_500K_FINAL_CHECKPOINTS[flavor]
    )


def convnext_yap_atto_500k_anneal1k() -> PathTrainer.Config:
    return _yap_500k_anneal1k("convnext_atto")


def convnext_yap_femto_500k_anneal1k() -> PathTrainer.Config:
    return _yap_500k_anneal1k("convnext_femto")


def convnext_yap_pico_500k_anneal1k() -> PathTrainer.Config:
    return _yap_500k_anneal1k("convnext_pico")


def convnext_yap_teeny_500k_anneal1k() -> PathTrainer.Config:
    return _yap_500k_anneal1k("convnext_teeny")


def convnext_yap_tiny_500k_anneal1k() -> PathTrainer.Config:
    return _yap_500k_anneal1k("convnext_tiny")


def convnext_yap_small_500k_anneal1k() -> PathTrainer.Config:
    return _yap_500k_anneal1k("convnext_small")


def convnext_yap_small_500k_step1000_anneal1k() -> PathTrainer.Config:
    config = _yap_anneal1k(
        "convnext_small",
        1_000,
        YAP_500K_STEP1000_CHECKPOINTS["convnext_small"],
    )
    config.dataloader = dataclasses.replace(config.dataloader, one_pass=False)
    return dataclasses.replace(config)


def _yap_1m_anneal1k(flavor: str) -> PathTrainer.Config:
    return _yap_anneal1k(
        flavor,
        2 * YAP_500K_ONE_PASS_STEPS,
        _yap_mark_checkpoint(YAP_1M_CHECKPOINTS, "1M", flavor),
    )


def convnext_yap_atto_1m_anneal1k() -> PathTrainer.Config:
    return _yap_1m_anneal1k("convnext_atto")


def convnext_yap_femto_1m_anneal1k() -> PathTrainer.Config:
    return _yap_1m_anneal1k("convnext_femto")


def convnext_yap_pico_1m_anneal1k() -> PathTrainer.Config:
    return _yap_1m_anneal1k("convnext_pico")


def convnext_yap_teeny_1m_anneal1k() -> PathTrainer.Config:
    return _yap_1m_anneal1k("convnext_teeny")


def convnext_yap_tiny_1m_anneal1k() -> PathTrainer.Config:
    return _yap_1m_anneal1k("convnext_tiny")


def convnext_yap_small_1m_anneal1k() -> PathTrainer.Config:
    return _yap_1m_anneal1k("convnext_small")


def _yap_1p5m_anneal1k(flavor: str) -> PathTrainer.Config:
    return _yap_anneal1k(
        flavor,
        YAP_EXT_TOTAL_STEPS,
        _yap_mark_checkpoint(YAP_1P5M_CHECKPOINTS, "1P5M", flavor),
    )


def convnext_yap_atto_1p5m_anneal1k() -> PathTrainer.Config:
    return _yap_1p5m_anneal1k("convnext_atto")


def convnext_yap_femto_1p5m_anneal1k() -> PathTrainer.Config:
    return _yap_1p5m_anneal1k("convnext_femto")


def convnext_yap_pico_1p5m_anneal1k() -> PathTrainer.Config:
    return _yap_1p5m_anneal1k("convnext_pico")


def convnext_yap_teeny_1p5m_anneal1k() -> PathTrainer.Config:
    return _yap_1p5m_anneal1k("convnext_teeny")


def convnext_yap_tiny_1p5m_anneal1k() -> PathTrainer.Config:
    return _yap_1p5m_anneal1k("convnext_tiny")


def convnext_yap_small_1p5m_anneal1k() -> PathTrainer.Config:
    return _yap_1p5m_anneal1k("convnext_small")


def _yap_anneal500(
    flavor: str, source_step: int, checkpoint_uri: str
) -> PathTrainer.Config:
    """Resume one yaP full-state checkpoint and linearly cool its LR from 1e-3
    to zero over 500 steps. one_pass stays off: the audit counts samples from
    process start and cannot reconcile with a restored trainer counter."""
    total_steps = source_step + YAP_ANNEAL500_STEPS
    config = _yap(flavor)
    config.dataloader = dataclasses.replace(config.dataloader, one_pass=False)
    config.training.steps = total_steps
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=total_steps,
        decay_ratio=YAP_ANNEAL500_STEPS / total_steps,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=total_steps,
        initial_load_path=checkpoint_uri,
        initial_load_model_only=False,
        allow_partial_initial_load=False,
        load_only=False,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        export_onnx=False,
    )
    config.metrics.log_freq = 1
    config.metrics.save_freq = 100
    config.validator.enable = False
    return dataclasses.replace(config)


def convnext_yap_small_final_anneal500() -> PathTrainer.Config:
    return _yap_anneal500(
        "convnext_small",
        YAP_500K_ONE_PASS_STEPS,
        YAP_500K_FINAL_CHECKPOINTS["convnext_small"],
    )


def convnext_yap_small_1m_anneal500() -> PathTrainer.Config:
    return _yap_anneal500(
        "convnext_small",
        2 * YAP_500K_ONE_PASS_STEPS,
        _yap_mark_checkpoint(YAP_1M_CHECKPOINTS, "1M", "convnext_small"),
    )


def convnext_yap_small_1p5m_anneal500() -> PathTrainer.Config:
    return _yap_anneal500(
        "convnext_small",
        YAP_EXT_TOTAL_STEPS,
        _yap_mark_checkpoint(YAP_1P5M_CHECKPOINTS, "1P5M", "convnext_small"),
    )


def convnext_small_500k_one_pass_noval() -> PathTrainer.Config:
    """Stock ConvNeXt Small over the canonical 2025 500k list, once."""
    steps = CONVNEXT_500K_ONE_PASS_STEPS
    config = convnext_small()
    config.dataloader = dataclasses.replace(
        config.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        limit=CONVNEXT_500K_TRAIN_SEGMENTS,
        one_pass=True,
    )
    config.training.steps = steps
    config.training.global_batch_size = 128
    config.lr_scheduler = dataclasses.replace(
        config.lr_scheduler,
        warmup_steps=round(steps * 0.01),
        total_steps=steps,
    )
    config.checkpoint = final_checkpoint_config(
        flavor=config.model_spec.flavor,
        stem=os.path.splitext(os.path.basename(DEFAULT_TRAIN_LIST))[0],
        seed=config.debug.seed,
        steps=steps,
    )
    config.checkpoint.model_only_steps = [3_125, 6_250, 9_375, 12_500]
    config.checkpoint.last_save_model_only = True
    config.checkpoint.enable_first_step_checkpoint = False
    config.metrics.save_freq = 16
    config.validator.enable = False
    return dataclasses.replace(config)


CONVNEXT_SMALL_500K_CHECKPOINT_BASE = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/"
    "run-OGIwNDY2N2ItZTdmMS1hYjk0LTQyY2MtMTZjZDE3OWEzNjg5/"
    "convnext_small/train_500k_20250717_s0"
)
CONVNEXT_SMALL_MILESTONE_STEPS = (3_125, 6_250, 9_375, 12_500)
CONVNEXT_SMALL_ANNEAL_STEPS = 100


def _convnext_small_milestone_checkpoint(step: int) -> str:
    if step not in CONVNEXT_SMALL_MILESTONE_STEPS:
        raise ValueError(f"unknown ConvNeXt Small milestone: {step}")
    return f"{CONVNEXT_SMALL_500K_CHECKPOINT_BASE}/step-{step}"


def convnext_small_500k_step3125_load_check() -> PathTrainer.Config:
    """Load the first model-only milestone, then take one zero-LR step."""
    config = convnext_small_500k_one_pass_noval()
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs["lr"] = 0.0
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=1,
        decay_ratio=1.0,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    config.training.steps = 1
    config.checkpoint.initial_load_path = _convnext_small_milestone_checkpoint(3_125)
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.allow_partial_initial_load = False
    config.checkpoint.load_only = True
    config.checkpoint.enable_first_step_checkpoint = False
    config.metrics.log_freq = 1
    config.metrics.save_freq = 1
    config.validator.enable = False
    return dataclasses.replace(config)


def _convnext_small_500k_anneal(milestone_step: int) -> PathTrainer.Config:
    """Linearly decay the stock LR from one model-only milestone to zero."""
    config = convnext_small_500k_one_pass_noval()
    config.training.steps = CONVNEXT_SMALL_ANNEAL_STEPS
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=CONVNEXT_SMALL_ANNEAL_STEPS,
        decay_ratio=1.0,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    config.checkpoint = final_checkpoint_config(
        flavor=f"convnext_small_anneal_from_step{milestone_step}",
        stem="train_500k_20250717",
        seed=config.debug.seed,
        steps=CONVNEXT_SMALL_ANNEAL_STEPS,
    )
    config.checkpoint.initial_load_path = _convnext_small_milestone_checkpoint(
        milestone_step
    )
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.allow_partial_initial_load = False
    config.checkpoint.load_only = False
    config.checkpoint.last_save_model_only = True
    config.checkpoint.enable_first_step_checkpoint = False
    config.metrics.log_freq = 1
    config.metrics.save_freq = CONVNEXT_SMALL_ANNEAL_STEPS
    config.validator.enable = False
    return dataclasses.replace(config)


def convnext_small_500k_anneal_step3125() -> PathTrainer.Config:
    return _convnext_small_500k_anneal(3_125)


def convnext_small_500k_anneal_step6250() -> PathTrainer.Config:
    return _convnext_small_500k_anneal(6_250)


def convnext_small_500k_anneal_step9375() -> PathTrainer.Config:
    return _convnext_small_500k_anneal(9_375)


def convnext_small_500k_anneal_step12500() -> PathTrainer.Config:
    return _convnext_small_500k_anneal(12_500)


def _convnext_small_500k_anneal_validate_study5000(
    milestone_step: int,
    *,
    checkpoint: str | None = None,
) -> PathTrainer.Config:
    """Score one Small checkpoint on the locked study-5000 list."""
    env_name = f"CONVNEXT_SMALL_ANNEAL_STEP{milestone_step}_CHECKPOINT"
    checkpoint = checkpoint or os.environ.get(env_name)
    if not checkpoint:
        raise ValueError(f"{env_name} must name the annealed final checkpoint")

    config = _convnext_small_500k_anneal(milestone_step)
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs["lr"] = 0.0
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=0,
        total_steps=1,
        decay_ratio=1.0,
        decay_type="linear",
        min_lr_factor=0.0,
    )
    config.training.local_batch_size = 5
    config.training.global_batch_size = 40
    config.training.steps = 1
    config.dataloader = dataclasses.replace(
        config.dataloader, plan_only=True, one_pass=False
    )

    config.checkpoint.initial_load_path = checkpoint
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.allow_partial_initial_load = False
    config.checkpoint.load_only = True
    config.checkpoint.enable_first_step_checkpoint = False

    config.metrics.log_freq = 1
    config.metrics.save_freq = 1
    config.validator.enable = True
    config.validator.freq = 1
    config.validator.steps = 125
    config.validator.dataloader = dataclasses.replace(
        _dataloader_config(split="val", fps=SUPERCOMBO_FPS, plan_only=True),
        dataset=os.path.join(XX_BASEDIR, "datasets/lists/prune10m_study_5000.txt"),
        pipeline_dir=BASE_DIR_GT_10M,
        limit=5_000,
        val_skip=24,
        one_pass=True,
    )
    config.validator.reports = {}
    config.validator.save_predictions = True
    return dataclasses.replace(config)


def convnext_small_500k_anneal_step3125_validate_study5000() -> PathTrainer.Config:
    return _convnext_small_500k_anneal_validate_study5000(3_125)


def convnext_small_500k_anneal_step6250_validate_study5000() -> PathTrainer.Config:
    return _convnext_small_500k_anneal_validate_study5000(6_250)


def convnext_small_500k_anneal_step9375_validate_study5000() -> PathTrainer.Config:
    return _convnext_small_500k_anneal_validate_study5000(9_375)


def convnext_small_500k_anneal_step12500_validate_study5000() -> PathTrainer.Config:
    return _convnext_small_500k_anneal_validate_study5000(12_500)


def convnext_small_500k_step3125_validate_study5000() -> PathTrainer.Config:
    """Score the raw step-3,125 milestone without a cooldown."""
    return _convnext_small_500k_anneal_validate_study5000(
        3_125, checkpoint=_convnext_small_milestone_checkpoint(3_125)
    )


def convnext_small_500k_step6250_validate_study5000() -> PathTrainer.Config:
    """Score the raw step-6,250 milestone without a cooldown."""
    return _convnext_small_500k_anneal_validate_study5000(
        6_250, checkpoint=_convnext_small_milestone_checkpoint(6_250)
    )


def convnext_small_500k_step9375_validate_study5000() -> PathTrainer.Config:
    """Score the raw step-9,375 milestone without a cooldown."""
    return _convnext_small_500k_anneal_validate_study5000(
        9_375, checkpoint=_convnext_small_milestone_checkpoint(9_375)
    )


def convnext_small_500k_step12500_validate_study5000() -> PathTrainer.Config:
    """Score the raw step-12,500 milestone without a cooldown."""
    return _convnext_small_500k_anneal_validate_study5000(
        12_500, checkpoint=_convnext_small_milestone_checkpoint(12_500)
    )


def _convnext_tiny_500k_one_pass_noval_validate_study5000(
    *, checkpoint: str, base_config: PathTrainer.Config | None = None
) -> PathTrainer.Config:
    """Build the locked-5k validation bridge once the final URI is known."""
    config = base_config or convnext_tiny_500k_one_pass_noval()
    config.optimizer = _optimizer_config()
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs["lr"] = 0.0
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=1,
        total_steps=1,
        decay_ratio=0.2,
        decay_type="cosine",
        min_lr_factor=0.0,
    )
    config.training.local_batch_size = 5
    config.training.global_batch_size = 40
    config.training.steps = 1
    config.dataloader = dataclasses.replace(
        config.dataloader, plan_only=True, one_pass=False
    )

    config.checkpoint.initial_load_path = checkpoint
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.allow_partial_initial_load = False
    config.checkpoint.load_only = True
    config.checkpoint.enable_first_step_checkpoint = False

    config.metrics.log_freq = 1
    config.metrics.save_freq = 1
    config.validator.enable = True
    config.validator.freq = 1
    config.validator.steps = 125
    config.validator.dataloader = dataclasses.replace(
        _dataloader_config(split="val", fps=SUPERCOMBO_FPS, plan_only=True),
        dataset=os.path.join(XX_BASEDIR, "datasets/lists/prune10m_study_5000.txt"),
        pipeline_dir=BASE_DIR_GT_10M,
        limit=5_000,
        val_skip=24,
        one_pass=True,
    )
    config.validator.reports = {}
    config.validator.save_predictions = True
    return dataclasses.replace(config)


CONVNEXT_TINY_500K_FINAL_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/"
    "run-YzExYTI4ODAtZTJhOC0yOWQzLWM2NGUtMjgyZjEzYTY0YTYz/"
    "convnext_tiny/train_500k_20250717_s0/step-28076"
)


def convnext_tiny_500k_one_pass_noval_validate_study5000() -> PathTrainer.Config:
    """Score Tiny's final one-pass checkpoint on the locked study-5000 list."""
    return _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=CONVNEXT_TINY_500K_FINAL_CHECKPOINT
    )


CONVNEXT_TINY_500K_STEP23552_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/"
    "8603da13-d263-7292-ad1c-f931e85c802b/23552"
)


def convnext_tiny_500k_one_pass_step23552_validate_study5000() -> PathTrainer.Config:
    """Score the bs-128 Tiny data-wall checkpoint on the locked study-5000 list."""
    return _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=CONVNEXT_TINY_500K_STEP23552_CHECKPOINT
    )


def convnext_tiny_500k_one_pass_step23552_validate_500kval() -> PathTrainer.Config:
    """Score the bs-128 Tiny data-wall checkpoint on the 500k list's own val split."""
    config = convnext_tiny_500k_one_pass_step23552_validate_study5000()
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def convnext_small() -> PathTrainer.Config:
    return _path("convnext_small")


def _yap_500k_one_pass(
    *, flavor: str, model_flavor: str, vision_dims: tuple[int, ...] | None = None
) -> PathTrainer.Config:
    """Run one stock-initialized ConvNeXt Path model over the 500k list."""
    config = _path(model_flavor)
    if vision_dims is not None:
        config.model_spec = model_registry(
            flavor,
            _model_config(model_flavor, vision_dims=vision_dims),
        )
    config.dataloader = dataclasses.replace(
        config.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        limit=CONVNEXT_500K_TRAIN_SEGMENTS,
        one_pass=True,
    )
    config.optimizer = _convnext_standard_optimizer_config(
        lr=YAP_LR,
        wd=YAP_WEIGHT_DECAY,
    )
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=YAP_WARMUP_STEPS,
        total_steps=YAP_500K_ONE_PASS_STEPS,
        decay_ratio=0.0,
        decay_type="linear",
        min_lr_factor=1.0,
    )
    config.training.local_batch_size = 16
    config.training.global_batch_size = YAP_GLOBAL_BATCH
    config.training.steps = YAP_500K_ONE_PASS_STEPS
    config.checkpoint = dataclasses.replace(
        config.checkpoint,
        interval=YAP_CHECKPOINT_INTERVAL,
        enable_first_step_checkpoint=False,
        last_save_model_only=False,
        save_model_state_dict=False,
        export_onnx=False,
    )
    config.metrics.log_freq = 16
    config.metrics.save_freq = 16
    config.validator.enable = False
    return dataclasses.replace(config)


def yap_convnext_teeny_500k_one_pass() -> PathTrainer.Config:
    return _yap_500k_one_pass(
        flavor="convnext_teeny",
        model_flavor="convnext_tiny",
        vision_dims=(48, 96, 192, 384),
    )


def yap_convnext_tiny_500k_one_pass() -> PathTrainer.Config:
    return _yap_500k_one_pass(
        flavor="convnext_tiny",
        model_flavor="convnext_tiny",
    )


def yap_convnext_small_500k_one_pass() -> PathTrainer.Config:
    return _yap_500k_one_pass(
        flavor="convnext_small",
        model_flavor="convnext_small",
    )


def yap_convnext_base_500k_one_pass() -> PathTrainer.Config:
    return _yap_500k_one_pass(
        flavor="convnext_base",
        model_flavor="convnext_base",
    )


def convnext_base() -> PathTrainer.Config:
    return _path("convnext_base")


def convnext_xxlarge() -> PathTrainer.Config:
    return _path("convnext_xxlarge")


def _convnext_whole_registered(
    stage0: int, *, mup: bool, drop_path_rate: float
) -> PathTrainer.Config:
    mode = "mup" if mup else "standard"
    drop = "dp20" if drop_path_rate else "dp0"
    flavor = f"convnext_whole_{mode}_w{stage0}_{drop}"
    return _convnext_whole(
        stage0=stage0,
        drop_path_rate=drop_path_rate,
        steps=CONVNEXT_MUP_STEPS[stage0],
        flavor=flavor,
        mup=mup,
    )


def convnext_whole_standard_w7() -> PathTrainer.Config:
    return _convnext_whole_registered(7, mup=False, drop_path_rate=0.0)


def convnext_whole_standard_w9() -> PathTrainer.Config:
    return _convnext_whole_registered(9, mup=False, drop_path_rate=0.0)


def convnext_whole_standard_w11() -> PathTrainer.Config:
    return _convnext_whole_registered(11, mup=False, drop_path_rate=0.0)


def convnext_whole_standard_w13() -> PathTrainer.Config:
    return _convnext_whole_registered(13, mup=False, drop_path_rate=0.0)


def convnext_whole_standard_w15() -> PathTrainer.Config:
    return _convnext_whole_registered(15, mup=False, drop_path_rate=0.0)


def convnext_whole_mup_w7() -> PathTrainer.Config:
    return _convnext_whole_registered(7, mup=True, drop_path_rate=0.0)


def convnext_whole_mup_w9() -> PathTrainer.Config:
    return _convnext_whole_registered(9, mup=True, drop_path_rate=0.0)


def convnext_whole_mup_w11() -> PathTrainer.Config:
    return _convnext_whole_registered(11, mup=True, drop_path_rate=0.0)


def convnext_whole_mup_w13() -> PathTrainer.Config:
    return _convnext_whole_registered(13, mup=True, drop_path_rate=0.0)


def convnext_whole_mup_w15() -> PathTrainer.Config:
    return _convnext_whole_registered(15, mup=True, drop_path_rate=0.0)


def _convnext_whole_clean_c1_registered(
    stage0: int, *, seed: int = 0
) -> PathTrainer.Config:
    return _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_STEPS[stage0],
        flavor=f"convnext_whole_mup_clean_c1_w{stage0}_dp0",
        mup=True,
        train_list="prune10m_uniform100k_seed0.txt",
        seed=seed,
    )


def convnext_whole_mup_clean_c1_w7() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(7)


def convnext_whole_mup_clean_c1_w9() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(9)


def convnext_whole_mup_clean_c1_w11() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(11)


def convnext_whole_mup_clean_c1_w13() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(13)


def convnext_whole_mup_clean_c1_w15() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(15)


def convnext_whole_mup_clean_c1_w11_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(11, seed=1)


def convnext_whole_mup_clean_c1_w15_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c1_registered(15, seed=1)


def _convnext_whole_clean_c2_registered(stage0: int) -> PathTrainer.Config:
    return _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_BUDGET2_STEPS[stage0],
        flavor=f"convnext_whole_mup_clean_c2_w{stage0}_dp0",
        mup=True,
        train_list="prune10m_uniform100k_seed0.txt",
        seed=0,
    )


def convnext_whole_mup_clean_c2_w11() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_registered(11)


def convnext_whole_mup_clean_c2_w13() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_registered(13)


def convnext_whole_mup_clean_c2_w15() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_registered(15)


def convnext_whole_mup_clean_c2_w17() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_registered(17)


def _convnext_whole_plan10m_registered(
    stage0: int,
    *,
    budget2: bool,
    lr: float = 1e-3,
    flavor_suffix: str = "",
    seed: int = 0,
    step_override: int | None = None,
    wd: float = CONVNEXT_STUDY_BASE_WEIGHT_DECAY,
) -> PathTrainer.Config:
    budget = "c2" if budget2 else "c1"
    steps = step_override or (
        CONVNEXT_MUP_BUDGET2_STEPS[stage0] if budget2 else CONVNEXT_MUP_STEPS[stage0]
    )
    config = _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=steps,
        flavor=(
            f"convnext_whole_mup_clean_{budget}_w{stage0}_dp0_plan10m"
            f"{flavor_suffix}"
        ),
        mup=True,
        lr=lr,
        wd=wd,
        train_list="prune10m_uniform100k_seed0.txt",
        seed=seed,
    )
    config.dataloader = dataclasses.replace(
        config.dataloader,
        pipeline_dir=BASE_DIR_GT_10M,
        plan_only=True,
    )
    return dataclasses.replace(config)


def convnext_whole_mup_clean_c1_w7_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(7, budget2=False)


def convnext_whole_mup_clean_c1_w9_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(9, budget2=False)


def convnext_whole_mup_clean_c1_w11_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(11, budget2=False)


def convnext_whole_mup_clean_c1_w13_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(13, budget2=False)


def convnext_whole_mup_clean_c1_w13_plan10m_seed_2200() -> PathTrainer.Config:
    """One-pass-guarded w13 seed run capped at the 40-minute study limit."""
    return _convnext_whole_plan10m_registered(
        13,
        budget2=False,
        flavor_suffix="_seed_2200",
        step_override=2_200,
    )


def _convnext_whole_clean_v2_registered(
    stage0: int,
    *,
    lr: float = 1e-3,
    flavor_suffix: str = "_clean_v2",
) -> PathTrainer.Config:
    """One long trajectory per width for equal-compute loss slicing."""
    allowed_widths = (*CONVNEXT_CLEAN_V2_LEFT_FLANK_WIDTHS, *CONVNEXT_CLEAN_V2_WIDTHS)
    if stage0 not in allowed_widths:
        raise ValueError(f"clean-v2 width must be one of {allowed_widths}")
    steps = (
        CONVNEXT_CLEAN_V2_LEFT_FLANK_STEPS
        if stage0 in CONVNEXT_CLEAN_V2_LEFT_FLANK_WIDTHS
        else CONVNEXT_CLEAN_V2_STEPS[stage0]
    )
    config = _convnext_whole_plan10m_registered(
        stage0,
        budget2=False,
        lr=lr,
        step_override=steps,
        flavor_suffix=flavor_suffix,
        wd=1e-2,
    )
    if stage0 in CONVNEXT_CLEAN_V2_LEFT_FLANK_WIDTHS:
        schedule_steps = CONVNEXT_CLEAN_V2_LEFT_FLANK_SCHEDULE_STEPS[stage0]
        config.lr_scheduler = dataclasses.replace(
            config.lr_scheduler,
            warmup_steps=round(schedule_steps * 0.01),
            total_steps=schedule_steps,
        )
    config.metrics.log_freq = 1
    return dataclasses.replace(config)


def convnext_whole_mup_clean_v2_w7_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(7)


def convnext_whole_mup_clean_v2_w9_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(9)


def convnext_whole_mup_clean_v2_w11_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(11)


def convnext_whole_mup_clean_v2_w13_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(13)


def convnext_whole_mup_clean_v2_w17_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(17)


def convnext_whole_mup_clean_v2_w21_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(21)


def convnext_whole_mup_clean_v2_w25_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(25)


def convnext_whole_mup_clean_v2_w29_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_registered(29)


def _convnext_whole_clean_v3_registered(stage0: int) -> PathTrainer.Config:
    """Long one-recipe trajectories selected by the measured Small horizon."""
    if stage0 not in CONVNEXT_CLEAN_V3_WIDTHS:
        raise ValueError(f"clean-v3 width must be one of {CONVNEXT_CLEAN_V3_WIDTHS}")
    config = _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_CLEAN_V3_STEPS[stage0],
        flavor=f"convnext_whole_mup_clean_v3_w{stage0}_plan10m",
        mup=True,
        lr=1e-3,
        wd=CONVNEXT_STUDY_BASE_WEIGHT_DECAY,
        train_list=CONVNEXT_CLEAN_V3_TRAIN_LIST,
        seed=0,
    )
    config.dataloader = dataclasses.replace(
        config.dataloader,
        pipeline_dir=BASE_DIR_GT_10M,
        plan_only=True,
        limit=CONVNEXT_CLEAN_V3_LEFT_FLANK_LIMITS.get(
            stage0, CONVNEXT_CLEAN_V3_TRAIN_LIMIT
        ),
    )
    config.training.local_batch_size = CONVNEXT_CLEAN_V3_LOCAL_BATCH[stage0]
    config.metrics.log_freq = 1
    config.metrics.save_freq = 16
    return dataclasses.replace(config)


def convnext_whole_mup_clean_v3_w7_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(7)


def convnext_whole_mup_clean_v3_w9_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(9)


def convnext_whole_mup_clean_v3_w11_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(11)


def convnext_whole_mup_clean_v3_w15_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(15)


def convnext_whole_mup_clean_v3_w21_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(21)


def convnext_whole_mup_clean_v3_w29_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(29)


def convnext_whole_mup_clean_v3_w41_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_v3_registered(41)


def _convnext_whole_clean_v2_lr2e3_full(stage0: int) -> PathTrainer.Config:
    """Full clean-v2 trajectory at the LR-finder winner."""
    return _convnext_whole_clean_v2_registered(
        stage0,
        lr=2e-3,
        flavor_suffix="_clean_v2_lr2e3_full",
    )


def convnext_whole_mup_clean_v2_w13_plan10m_lr2e3_full() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr2e3_full(13)


def convnext_whole_mup_clean_v2_w17_plan10m_lr2e3_full() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr2e3_full(17)


def convnext_whole_mup_clean_v2_w21_plan10m_lr2e3_full() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr2e3_full(21)


def convnext_whole_mup_clean_v2_w25_plan10m_lr2e3_full() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr2e3_full(25)


def convnext_whole_mup_clean_v2_w29_plan10m_lr2e3_full() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr2e3_full(29)


def _convnext_whole_clean_v2_lr_sweep(lr: float, suffix: str) -> PathTrainer.Config:
    """Canonical w13 peak-LR finder at one shared 1,000-step horizon."""
    config = _convnext_whole_plan10m_registered(
        13,
        budget2=False,
        lr=lr,
        step_override=1_000,
        flavor_suffix=f"_clean_v2_lr_{suffix}",
    )
    config.metrics.log_freq = 1
    return dataclasses.replace(config)


def convnext_whole_mup_clean_v2_w13_plan10m_lr2p5e4() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr_sweep(2.5e-4, "2p5e4")


def convnext_whole_mup_clean_v2_w13_plan10m_lr5e4() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr_sweep(5e-4, "5e4")


def convnext_whole_mup_clean_v2_w13_plan10m_lr1e3() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr_sweep(1e-3, "1e3")


def convnext_whole_mup_clean_v2_w13_plan10m_lr2e3() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr_sweep(2e-3, "2e3")


def convnext_whole_mup_clean_v2_w13_plan10m_lr2p8e3() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr_sweep(2.8e-3, "2p8e3")


def convnext_whole_mup_clean_v2_w13_plan10m_lr4e3() -> PathTrainer.Config:
    return _convnext_whole_clean_v2_lr_sweep(4e-3, "4e3")


def convnext_whole_mup_clean_c1_w15_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(15, budget2=False)


def convnext_whole_mup_clean_c1_w17_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(17, budget2=False)


def convnext_whole_mup_clean_c1_w19_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(19, budget2=False)


def convnext_whole_mup_clean_c1_w21_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(21, budget2=False)


def convnext_whole_mup_clean_c1_w7_plan10m_seed1() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(7, budget2=False, seed=1)


def convnext_whole_mup_clean_c1_w9_plan10m_seed1() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(9, budget2=False, seed=1)


def _convnext_whole_clean_c0_plan10m_registered(
    stage0: int, *, lr: float = 1e-3
) -> PathTrainer.Config:
    suffix = "" if lr == 1e-3 else "_lr1p78e3"
    config = _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_BUDGET0_STEPS[stage0],
        flavor=f"convnext_whole_mup_clean_c0_w{stage0}_dp0_plan10m{suffix}",
        mup=True,
        lr=lr,
        train_list="prune10m_uniform100k_seed0.txt",
        seed=0,
    )
    config.dataloader = dataclasses.replace(
        config.dataloader,
        pipeline_dir=BASE_DIR_GT_10M,
        plan_only=True,
    )
    return dataclasses.replace(config)


def convnext_whole_mup_clean_c0_w7_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_c0_plan10m_registered(7)


def convnext_whole_mup_clean_c0_w9_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_c0_plan10m_registered(9)


def convnext_whole_mup_clean_c0_w11_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_c0_plan10m_registered(11)


def convnext_whole_mup_clean_c0_w13_plan10m() -> PathTrainer.Config:
    return _convnext_whole_clean_c0_plan10m_registered(13)


def convnext_whole_mup_clean_c0_w11_plan10m_lr1p78e3() -> PathTrainer.Config:
    return _convnext_whole_clean_c0_plan10m_registered(11, lr=1.78e-3)


def _convnext_whole_clean_c3_plan10m_registered(
    stage0: int, *, lr: float, seed: int = 0
) -> PathTrainer.Config:
    suffix = {5.6e-4: "lr5p6e4", 3.2e-4: "lr3p2e4"}[lr]
    config = _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_BUDGET3_STEPS[stage0],
        flavor=f"convnext_whole_mup_clean_c3_w{stage0}_dp0_plan10m_{suffix}",
        mup=True,
        lr=lr,
        train_list="prune10m_uniform100k_seed0.txt",
        seed=seed,
    )
    config.dataloader = dataclasses.replace(
        config.dataloader,
        pipeline_dir=BASE_DIR_GT_10M,
        plan_only=True,
    )
    return dataclasses.replace(config)


def convnext_whole_mup_clean_c3_w27_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(27, lr=5.6e-4)


def convnext_whole_mup_clean_c3_w27_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(27, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w19_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(19, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w21_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(21, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w23_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(23, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w25_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(25, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w29_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(29, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w31_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(31, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w33_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(33, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w35_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(35, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w37_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(37, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w39_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(39, lr=3.2e-4)


def convnext_whole_mup_clean_c3_w35_plan10m_lr3p2e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(35, lr=3.2e-4, seed=1)


def convnext_whole_mup_clean_c3_w37_plan10m_lr3p2e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(37, lr=3.2e-4, seed=1)


def convnext_whole_mup_clean_c3_w39_plan10m_lr3p2e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(39, lr=3.2e-4, seed=1)


def convnext_whole_mup_clean_c3_w41_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c3_plan10m_registered(41, lr=3.2e-4)


def _convnext_whole_clean_c4_plan10m_registered(
    stage0: int,
    *,
    lr: float,
    wd: float = CONVNEXT_STUDY_BASE_WEIGHT_DECAY,
    seed: int = 0,
) -> PathTrainer.Config:
    suffix = {3.2e-4: "lr3p2e4", 1.8e-4: "lr1p8e4"}[lr]
    if wd == 5.6e-2:
        suffix += "_wd5p6e2"
    config = _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_BUDGET4_STEPS[stage0],
        flavor=f"convnext_whole_mup_clean_c4_w{stage0}_dp0_plan10m_{suffix}",
        mup=True,
        lr=lr,
        wd=wd,
        train_list="prune10m_uniform100k_seed0.txt",
        seed=seed,
    )
    config.dataloader = dataclasses.replace(
        config.dataloader,
        pipeline_dir=BASE_DIR_GT_10M,
        plan_only=True,
    )
    return dataclasses.replace(config)


def convnext_whole_mup_clean_c4_w37_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(37, lr=3.2e-4)


def convnext_whole_mup_clean_c4_w37_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(37, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w39_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(39, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w41_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(41, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w43_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(43, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w45_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(45, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w47_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(47, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w49_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(49, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w51_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(51, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w53_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(53, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w29_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(29, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w23_plan10m_lr1p8e4() -> PathTrainer.Config:
    config = _convnext_whole_clean_c4_plan10m_registered(23, lr=1.8e-4)
    config.training.local_batch_size = 8
    return dataclasses.replace(config)


def convnext_whole_mup_clean_c4_w25_plan10m_lr1p8e4() -> PathTrainer.Config:
    config = _convnext_whole_clean_c4_plan10m_registered(25, lr=1.8e-4)
    config.training.local_batch_size = 8
    return dataclasses.replace(config)


def convnext_whole_mup_clean_c4_w27_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(27, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w31_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(31, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w33_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(33, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w35_plan10m_lr1p8e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(35, lr=1.8e-4)


def convnext_whole_mup_clean_c4_w35_plan10m_lr3p2e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(35, lr=3.2e-4)


def convnext_whole_mup_clean_c4_w35_plan10m_lr1p8e4_wd5p6e2() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(35, lr=1.8e-4, wd=5.6e-2)


def convnext_whole_mup_clean_c4_w35_plan10m_lr1p8e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(35, lr=1.8e-4, seed=1)


def convnext_whole_mup_clean_c4_w37_plan10m_lr1p8e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(37, lr=1.8e-4, seed=1)


def convnext_whole_mup_clean_c4_w47_plan10m_lr1p8e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(47, lr=1.8e-4, seed=1)


def convnext_whole_mup_clean_c4_w49_plan10m_lr1p8e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(49, lr=1.8e-4, seed=1)


def convnext_whole_mup_clean_c4_w51_plan10m_lr1p8e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c4_plan10m_registered(51, lr=1.8e-4, seed=1)


def convnext_whole_mup_clean_c1_w13_plan10m_seed1() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(13, budget2=False, seed=1)


def convnext_whole_mup_clean_c2_w11_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(11, budget2=True)


def convnext_whole_mup_clean_c2_w13_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(13, budget2=True)


def convnext_whole_mup_clean_c2_w15_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(15, budget2=True)


def convnext_whole_mup_clean_c2_w17_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(17, budget2=True)


def convnext_whole_mup_clean_c2_w19_plan10m() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(19, budget2=True)


def convnext_whole_mup_clean_c2_w15_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(
        15,
        budget2=True,
        lr=5.6e-4,
        flavor_suffix="_lr5p6e4",
    )


def _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(
    stage0: int, *, seed: int = 0
) -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(
        stage0,
        budget2=True,
        lr=5.6e-4,
        flavor_suffix="_lr5p6e4",
        seed=seed,
    )


def convnext_whole_mup_clean_c2_w9_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(9)


def convnext_whole_mup_clean_c2_w11_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(11)


def convnext_whole_mup_clean_c2_w13_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(13)


def convnext_whole_mup_clean_c2_w17_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(17)


def convnext_whole_mup_clean_c2_w19_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(19)


def convnext_whole_mup_clean_c2_w21_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(21)


def convnext_whole_mup_clean_c2_w23_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(23)


def convnext_whole_mup_clean_c2_w25_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(25)


def convnext_whole_mup_clean_c2_w27_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(27)


def convnext_whole_mup_clean_c2_w29_plan10m_lr5p6e4() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(29)


def convnext_whole_mup_clean_c2_w21_plan10m_lr5p6e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(21, seed=1)


def convnext_whole_mup_clean_c2_w15_plan10m_lr5p6e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(15, seed=1)


def convnext_whole_mup_clean_c2_w19_plan10m_lr5p6e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(
        19,
        budget2=True,
        lr=5.6e-4,
        flavor_suffix="_lr5p6e4",
        seed=1,
    )


def convnext_whole_mup_clean_c2_w17_plan10m_lr5p6e4_seed1() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(
        17,
        budget2=True,
        lr=5.6e-4,
        flavor_suffix="_lr5p6e4",
        seed=1,
    )


def convnext_whole_mup_clean_c2_w19_plan10m_lr4e4() -> PathTrainer.Config:
    return _convnext_whole_plan10m_registered(
        19,
        budget2=True,
        lr=4e-4,
        flavor_suffix="_lr4e4",
    )


def convnext_whole_mup_w7_dp20() -> PathTrainer.Config:
    return _convnext_whole_registered(7, mup=True, drop_path_rate=0.2)


def convnext_whole_mup_w9_dp20() -> PathTrainer.Config:
    return _convnext_whole_registered(9, mup=True, drop_path_rate=0.2)


def convnext_whole_mup_w11_dp20() -> PathTrainer.Config:
    return _convnext_whole_registered(11, mup=True, drop_path_rate=0.2)


def convnext_whole_mup_w13_dp20() -> PathTrainer.Config:
    return _convnext_whole_registered(13, mup=True, drop_path_rate=0.2)


def convnext_whole_mup_w15_dp20() -> PathTrainer.Config:
    return _convnext_whole_registered(15, mup=True, drop_path_rate=0.2)


def convnext_whole_mup_w11_planinit_wm_side() -> PathTrainer.Config:
    return _convnext_whole(
        stage0=11,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_STEPS[11],
        flavor="convnext_whole_mup_w11_dp0_planinit_wm_side",
        mup=True,
        worldmodel_plan_init=True,
    )


def convnext_whole_mup_w11_planinit_wm_true_match() -> PathTrainer.Config:
    return _convnext_whole(
        stage0=11,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_STEPS[11],
        flavor="convnext_whole_mup_w11_dp0_planinit_wm_true_match",
        mup=True,
        worldmodel_plan_init_true_match=True,
    )


def _convnext_whole_budget2_registered(stage0: int) -> PathTrainer.Config:
    return _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_MUP_BUDGET2_STEPS[stage0],
        flavor=f"convnext_whole_mup_c2_w{stage0}_dp0",
        mup=True,
        train_list="prune10m_random1m_seed0.txt",
    )


def convnext_whole_mup_c2_w11() -> PathTrainer.Config:
    return _convnext_whole_budget2_registered(11)


def convnext_whole_mup_c2_w13() -> PathTrainer.Config:
    return _convnext_whole_budget2_registered(13)


def convnext_whole_mup_c2_w15() -> PathTrainer.Config:
    return _convnext_whole_budget2_registered(15)


def convnext_whole_mup_c2_w17() -> PathTrainer.Config:
    return _convnext_whole_budget2_registered(17)


CONVNEXT_WHOLE_ROUND1_CHECKPOINTS = {
    "w7_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZDE2ZDRiZjctZDk0OS01YzVkLTQ3ZDMtZTE3OWYzYTg5NjJi/convnext_whole_mup_w7_dp0/prune10m_random100k_seed0_s0/step-1141",
    "w9_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MDNjMTQwODQtMDBkNC00ZmQxLTUwZGItMmQyOWQ4YmI1ODUx/convnext_whole_mup_w9_dp0/prune10m_random100k_seed0_s0/step-741",
    "w11_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NzIyY2E1YjQtMTEzMi02ZDUwLWRmNGYtYTc5OGNkNzQwODMy/convnext_whole_mup_w11_dp0/prune10m_random100k_seed0_s0/step-520",
    "w13_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NzkwNTgwMWYtMzZhZC00ZGQ3LWVmNDktMmE4ZGE4MGRhYjIw/convnext_whole_mup_w13_dp0/prune10m_random100k_seed0_s0/step-386",
    "w15_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MDI4Yjk3MDUtZWY5My1jNGZlLTY0NWYtNDI3NjIzYjA5ZDZj/convnext_whole_mup_w15_dp0/prune10m_random100k_seed0_s0/step-297",
    "w7_dp20": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NDBmNGNlOWMtOWZjYy01ZTcwLWJjNTUtOTdiN2UxMmQ1N2Uy/convnext_whole_mup_w7_dp20/prune10m_random100k_seed0_s0/step-1141",
    "w9_dp20": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YWMxOWFkY2UtYmFmMy02MDgxLTM1MmQtY2MyZTQ1NTYzNjhk/convnext_whole_mup_w9_dp20/prune10m_random100k_seed0_s0/step-741",
    "w11_dp20": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-N2VkMjBkZDUtMmIxZS0xMzUzLWY1MDgtNzY3ZGZiYjU3ZDc2/convnext_whole_mup_w11_dp20/prune10m_random100k_seed0_s0/step-520",
    "w13_dp20": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OGNhNWZjMzctNGVjNy1kNzhiLWJhMmYtOGZlZTcxOGE2Yzcz/convnext_whole_mup_w13_dp20/prune10m_random100k_seed0_s0/step-386",
    "w15_dp20": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NThmMzhmYWQtZjUzNi1hZDc4LTAxZjctMzVkZWJlZDM0MzJj/convnext_whole_mup_w15_dp20/prune10m_random100k_seed0_s0/step-297",
}

CONVNEXT_WHOLE_CLEAN_C1_CHECKPOINTS = {
    "w7_s0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MGM3YmQ5ZjItZjZjMC1iOTk5LWJlMzUtMjY2MjM4YzkwMDJj/convnext_whole_mup_clean_c1_w7_dp0/prune10m_uniform100k_seed0_s0/step-1141",
    "w9_s0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NGE4YjBlN2EtNzQ0MC05ZGQ2LTAzYzgtZTgxOTJhMGEwMjRm/convnext_whole_mup_clean_c1_w9_dp0/prune10m_uniform100k_seed0_s0/step-741",
    "w11_s0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-YTlhMTc4MmMtNDA1Yi05MzYyLTNmMjEtOWM0MDNjZjRiZmYw/convnext_whole_mup_clean_c1_w11_dp0/prune10m_uniform100k_seed0_s0/step-520",
    "w13_s0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OTk3NmE1NWQtOWZlZC1mNTIwLTE5MGEtYWVhY2FlZGEwY2Q5/convnext_whole_mup_clean_c1_w13_dp0/prune10m_uniform100k_seed0_s0/step-386",
    "w15_s0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NWE2OWE5YWMtZWE0MS1jZjk1LTdhZjYtYWYxZmI1YjI5YzBj/convnext_whole_mup_clean_c1_w15_dp0/prune10m_uniform100k_seed0_s0/step-297",
    "w11_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NjU2NTM4OTgtOGNjNC1kMDUxLTM2MTYtMGIwNjkyMzY3OTJm/convnext_whole_mup_clean_c1_w11_dp0/prune10m_uniform100k_seed0_s1/step-520",
    "w15_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MzBlN2Q2YjEtM2NmZi0zNmNmLTA1NDAtYzU1MGIwZWM2ZmY2/convnext_whole_mup_clean_c1_w15_dp0/prune10m_uniform100k_seed0_s1/step-297",
}

CONVNEXT_WHOLE_CLEAN_C2_CHECKPOINTS = {
    "w11": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OTZhYTJhMzAtMjQ2OS1iOTg4LTY5ZTAtNzE2MzdhZDQwNDVj/convnext_whole_mup_clean_c2_w11_dp0/prune10m_uniform100k_seed0_s0/step-1645",
    "w13": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MTE0YTEwMzgtZmE4OS03NDAwLTk3MTctZDI4MTQ0OTk2OTUy/convnext_whole_mup_clean_c2_w13_dp0/prune10m_uniform100k_seed0_s0/step-1219",
    "w15": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ODhmZTMyZjgtNTc5Yi1mM2JmLWE5ZjktODE4NWMwNmIwZWY4/convnext_whole_mup_clean_c2_w15_dp0/prune10m_uniform100k_seed0_s0/step-940",
    "w17": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NDc0ZDEwZmQtYjE5My1mOGZkLTQ1NjMtZTMxNWM3ODJkM2Fj/convnext_whole_mup_clean_c2_w17_dp0/prune10m_uniform100k_seed0_s0/step-747",
}

CONVNEXT_WHOLE_CLEAN_C2_W15_PLAN10M_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/"
    "run-YzljZmIyMDMtODRhYS1mZTBiLWViMmYtNDE1ZWZlYzkwNWM5/"
    "convnext_whole_mup_clean_c2_w15_dp0_plan10m/"
    "prune10m_uniform100k_seed0_s0/step-940"
)

CONVNEXT_WHOLE_CLEAN_C2_W15_PLAN10M_LR5P6E4_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/"
    "run-NjQ4NDQ4ODQtYTIyZi0wMTNmLTcwNWEtMzI1N2M2OTNhY2Mw/"
    "convnext_whole_mup_clean_c2_w15_dp0_plan10m_lr5p6e4/"
    "prune10m_uniform100k_seed0_s0/step-940"
)

CONVNEXT_WHOLE_CLEAN_C1_PLAN10M_CHECKPOINTS = {
    "w7": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MWMwY2U0MDAtMmQ3NC0xYTliLTcwNjItYzYxYWZhYjNjOGNj/convnext_whole_mup_clean_c1_w7_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-1141",
    "w9": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MzMyMmY3OGUtYzA0YS01NDg2LTA4NzQtOWE5YTk0NzRjYjQx/convnext_whole_mup_clean_c1_w9_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-741",
    "w11": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-M2IwZGFhNTctYmUzZS00OWYxLWQzZWYtODJiZjQxNzNiYzg2/convnext_whole_mup_clean_c1_w11_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-520",
    "w13": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-YmNmODM0MTktMDU5Yy1lZGVkLTIwMmYtNjUxNDg1MTY4Y2U2/convnext_whole_mup_clean_c1_w13_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-386",
    "w15": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZDliYjNkYzUtMzgxNi0zYzUyLTAxZDktZjllNTMyNWFiNzZk/convnext_whole_mup_clean_c1_w15_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-297",
    "w17": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ODFiMWQwYzAtMDY5NC04YTRlLTNlNTktMTBhNDIwNjE3MTU4/convnext_whole_mup_clean_c1_w17_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-236",
    "w19": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-YTBiZGFkMGYtZDFmMi0yMjI2LWE4YWUtZTVlNTQ5ODk0N2Vi/convnext_whole_mup_clean_c1_w19_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-192",
    "w21": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-Mzc2M2ZhNTgtYTgxNS0zZmYxLTNhODMtNzBhMWQ0YjViMGU1/convnext_whole_mup_clean_c1_w21_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-160",
}

CONVNEXT_WHOLE_CLEAN_C0_CHECKPOINTS = {
    "w7_lr1e3": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OTBhNmY3YWMtZTUzNi04MzEwLWY4ZGMtOTk4N2YwMGVkMjkx/convnext_whole_mup_clean_c0_w7_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-361",
    "w9_lr1e3": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OWFiNGRlODgtMjE4MS1mNjQ5LWZkMWQtOTJlOTQ4OWM3MGZm/convnext_whole_mup_clean_c0_w9_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-234",
    "w11_lr1e3": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZDM0ZmY3N2UtY2U4MC0xM2YxLTc5OTMtYTAzNGY0M2I0MDEz/convnext_whole_mup_clean_c0_w11_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-165",
    "w11_lr1p78e3": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MzAwMjdlYjItY2QwYy1jNzFhLTBhM2EtZmUxY2E3ZGFiZjEy/convnext_whole_mup_clean_c0_w11_dp0_plan10m_lr1p78e3/prune10m_uniform100k_seed0_s0/step-165",
    "w13_lr1e3": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OWQ0NzUwNjAtOTY3My01ZTFiLTczMDAtOTg1NTVmMzdlNzgw/convnext_whole_mup_clean_c0_w13_dp0_plan10m/prune10m_uniform100k_seed0_s0/step-122",
}

CONVNEXT_WHOLE_CLEAN_C3_LR_CHECKPOINTS = {
    "lr5p6e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OGNmZjU1NmYtMGMwNC1mOTUxLTE2NmItYmIyNzM5ZTYzOTNk/convnext_whole_mup_clean_c3_w27_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-992",
    "lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MDBiOGQ5MTAtMDBiMC05OGQzLTBmYTAtY2M5ZWMxYzlkNTVi/convnext_whole_mup_clean_c3_w27_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-992",
    "w23_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZDA5N2U4ZWMtMDQwYS03OGJiLWU0YzUtZDM5YTlmOGIwMGE5/convnext_whole_mup_clean_c3_w23_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-1344",
    "w25_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-Mzk0N2YzZGQtZGY2My0yOWM2LWE4YWUtMmU3MWY3M2QxNjgy/convnext_whole_mup_clean_c3_w25_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-1148",
    "w29_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NTQwNjU4NmEtNjVjOS1mNjRmLTdjMzUtOGJjMDMxMTNkMjk0/convnext_whole_mup_clean_c3_w29_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-867",
    "w31_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NTJkODYyZTAtZTMzZC0wNjMwLWM3NWQtZjZkMjRmMmQxZTRj/convnext_whole_mup_clean_c3_w31_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-763",
    "w33_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NTlkMzdkZWUtNmFlNS0yMzYxLWJjYmYtZjczNzMzOWM4Zjhh/convnext_whole_mup_clean_c3_w33_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-677",
    "w35_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZDhkNDgxNGQtNDA4Ny1mNzBkLWQ5NDctZDk3MmFiOTNmMGY0/convnext_whole_mup_clean_c3_w35_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-605",
    "w37_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZDk2OWI2ZWMtNjM1Yy0wY2JmLTRhOGQtM2Q3MzNmMjYxMGMz/convnext_whole_mup_clean_c3_w37_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-544",
    "w37_lr3p2e4_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YTY0NTM2ODAtYjUzMS0yZDJhLWUxY2EtMDU1YWRmNjY2Y2Ez/convnext_whole_mup_clean_c3_w37_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s1/step-544",
    "w39_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OTkzN2JkMDQtNThmMy0yZjY2LTQ1OTAtY2JjZDg1YzdiOWY4/convnext_whole_mup_clean_c3_w39_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-491",
    "w41_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YzE4ZTkzNTktOThkMS05MDRhLTBjMDgtMWM4MjExZWNmNjkx/convnext_whole_mup_clean_c3_w41_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-446",
}

CONVNEXT_WHOLE_CLEAN_V2_CHECKPOINTS = {
    7: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZTY1MTQyYjMtNDE2ZS1mYTFmLTU0NWEtZTRlNmNjNDQyNGNj/convnext_whole_mup_clean_c1_w7_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-2734",
    9: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ODFjNGViMjItNDU3Yi0xNzVlLTRhYzgtYWRiYzc1ZDA2ODFh/convnext_whole_mup_clean_c1_w9_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-2734",
    11: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YzJiNGU5ZjktZDg5ZC0yZmM5LTFjNTYtODMwZmEwMGFjODc1/convnext_whole_mup_clean_c1_w11_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-2734",
    13: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-M2JlMWFmYmYtYzk2Ny05MTY4LTgzYzctOTBiNmI3NzY3NWI5/convnext_whole_mup_clean_c1_w13_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-2400",
    17: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZmQ5MTAzZGItN2IzMi1hMjYyLWMyODktNmVkMGExNzJhNmRl/convnext_whole_mup_clean_c1_w17_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-1470",
    21: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NmVlNjIxNzQtMzU3Yy1iNTcxLTA5NmQtNDA5MmEzYThiOTg3/convnext_whole_mup_clean_c1_w21_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-993",
    25: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OWI5NGNhY2EtMjY4Yy0xYzU5LTUwNmItZWM5ODdhMTM3ODZj/convnext_whole_mup_clean_c1_w25_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-715",
    29: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NDg5YmM5ZGYtNGFiNC05YmZiLTAxZTgtODAyMTZkMWMwMzg2/convnext_whole_mup_clean_c1_w29_dp0_plan10m_clean_v2/prune10m_uniform100k_seed0_s0/step-540",
}

CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS = {
    "lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YTczYTViMGItZTM2YS1kYzBiLTIyYTktYWY4NThmMjYwMzUz/convnext_whole_mup_clean_c4_w37_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-1721",
    "lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MmY4ZWU3ZTMtZTI3NC03ZWZmLTk5YzItNjJkZGUyMzhjOTFk/convnext_whole_mup_clean_c4_w37_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1721",
    "w29_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ODkzY2E5MmUtNzE5ZS01MmI2LTM0MzctMDFkZDUwNDA2MzAy/convnext_whole_mup_clean_c4_w29_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-2742",
    "w31_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NTVlMzQwOWItNzViMC01YTE5LTZkMzQtYjFjZmNiYzBiMzMy/convnext_whole_mup_clean_c4_w31_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-2415",
    "w33_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MTQ1ZTY4ZjAtMjg3Yi0yYmI2LTMzYTItOTM5NmYyYzJkNGE2/convnext_whole_mup_clean_c4_w33_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-2143",
    "w35_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YTU2MGYzMzQtMDU2ZC1mYjZkLTk0ZGMtMmU3ZTVhOTFlZDM4/convnext_whole_mup_clean_c4_w35_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1914",
    "w35_lr1p8e4_wd5p6e2": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MWJmMTJiNTctODU2ZC1lN2FjLTgxZWQtNjc3YjZjMTc1ZmE3/convnext_whole_mup_clean_c4_w35_dp0_plan10m_lr1p8e4_wd5p6e2/prune10m_uniform100k_seed0_s0/step-1914",
    "w35_lr3p2e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-NGViZWU4NjYtZGViNy0yNGE1LTRmYzAtNGI5MWRhOWQ0ZGJh/convnext_whole_mup_clean_c4_w35_dp0_plan10m_lr3p2e4/prune10m_uniform100k_seed0_s0/step-1914",
    "w35_lr1p8e4_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MjBiNjgyNTctMGNjOC0wZjFiLWRmOGUtNzkyZjA0NmI0YThj/convnext_whole_mup_clean_c4_w35_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s1/step-1914",
    "w37_lr1p8e4_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OWU3MjE3N2YtMjNlMS01ZTM2LWNlMWYtZGYwZTkyMjhlNTVh/convnext_whole_mup_clean_c4_w37_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s1/step-1721",
    "w39_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZTVjNzc5ODktZmMwYi1mZjIwLTg0NTItYzQ0ODBiZTYxOGU2/convnext_whole_mup_clean_c4_w39_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1555",
    "w41_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZDdiOGU0ZWItMTY0YS1mN2YwLWY1ZWItY2ZhMTk1YTA1YTVl/convnext_whole_mup_clean_c4_w41_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1412",
    "w43_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-M2NjNDgyMjYtZWUyMC04YTY4LWVkYjgtYmJmOWE3ZjViODEx/convnext_whole_mup_clean_c4_w43_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1288",
    "w45_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OTYzNjU1ZjItZjRhZS0wMDAzLWYwZDgtZmZhMDY3MjBjZWUz/convnext_whole_mup_clean_c4_w45_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1180",
    "w47_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-Y2RiNGZiMzItNTRjNy0yN2UzLTFiMDgtNDFlNjA2MGIwM2Uw/convnext_whole_mup_clean_c4_w47_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1084",
    "w49_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZjQ2MzYwZGUtYTU4NS04ZTg0LTZiMzYtODA3NGE0MTQ3NzU2/convnext_whole_mup_clean_c4_w49_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-1000",
    "w51_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OTBmY2M2MmEtZGRlMy1lNjA1LTU5ZTItMDE4MzliNWQzMTAy/convnext_whole_mup_clean_c4_w51_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-925",
    "w47_lr1p8e4_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MDgyZDlhOWYtNDRlMi1hYjM2LWYyYTYtM2NiMTkxMTU2NWNm/convnext_whole_mup_clean_c4_w47_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s1/step-1084",
    "w49_lr1p8e4_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZDAyMTY5MzYtZmQxMy0wNjJiLTQzNmEtYmIzODIzZGE5MWVi/convnext_whole_mup_clean_c4_w49_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s1/step-1000",
    "w51_lr1p8e4_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-ZWI5NjYxNDEtYjJjNi0xMjRmLWI3NWEtMzliZjg2MzRjNDhm/convnext_whole_mup_clean_c4_w51_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s1/step-925",
    "w53_lr1p8e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-Y2U2NjE1NWYtMGVhYy05NTZjLTM5YzEtNGI4NjNhNDJjNjY3/convnext_whole_mup_clean_c4_w53_dp0_plan10m_lr1p8e4/prune10m_uniform100k_seed0_s0/step-859",
}

CONVNEXT_WHOLE_CLEAN_C2_PLAN10M_LR5P6E4_CHECKPOINTS = {
    "w11": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MmY1Y2U1YTYtNDlkMC05M2RmLTk1NWItY2M1ZDU1MTM1MGRm/convnext_whole_mup_clean_c2_w11_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-1645",
    "w13": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZTIzMzdkMTAtYjkyOS00NmVhLTUzYTAtMmIyY2Q2NmM1NmM1/convnext_whole_mup_clean_c2_w13_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-1219",
    "w17": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-OWFlYmJlODMtNDMzMC1mNWVlLWRiMmQtZTBjNzk0ZGE3ZGU3/convnext_whole_mup_clean_c2_w17_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-747",
    "w19": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NjQ1MDZjYTUtMGEyYi1lMGUxLWM4OGItYTQ4MDE4YTVjMzRl/convnext_whole_mup_clean_c2_w19_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-608",
    "w21": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-Mzg3ODEzYTktNWUxNC1jMWNmLTUxNDAtMmZmOTdhODAwOTVi/convnext_whole_mup_clean_c2_w21_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-504",
    "w25": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MWNiNjIzZDMtMjIxNC00NjZmLWMxYzMtNDIxYmU4M2RkNTBk/convnext_whole_mup_clean_c2_w25_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-363",
    "w27": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YmYxNDg5YjYtZTQ3Mi1mNjQ3LWE2YjEtNmM4MWRlMjM1ODc1/convnext_whole_mup_clean_c2_w27_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-314",
    "w29": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-MjA0YjQ0OTItYTg2Ny02NGQ4LTM4YTgtOTBjMTQyNjc2YmIw/convnext_whole_mup_clean_c2_w29_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-274",
}

CONVNEXT_WHOLE_CLEAN_C2_DIAGNOSTIC_CHECKPOINTS = {
    "w19_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NTg2MTY1NTEtZmExYi1iNzkzLTllM2MtOGMzZTI5YTU4ZWU2/convnext_whole_mup_clean_c2_w19_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s1/step-608",
    "w17_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZTUxZDdhM2QtYTU2OS00ZGFhLTNlZjAtNTNkNTE5MWVlMTli/convnext_whole_mup_clean_c2_w17_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s1/step-747",
    "w19_lr4e4": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZDk0YTJlNjItMjFkMC1iNTViLTIyZjktZjVkZWZlM2Q0NjQz/convnext_whole_mup_clean_c2_w19_dp0_plan10m_lr4e4/prune10m_uniform100k_seed0_s0/step-608",
}

CONVNEXT_WHOLE_CLEAN_ZOOM_CHECKPOINTS = {
    "c2_w23_s0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-YzdiMjNjNzAtZDkxNi02YTkyLTQ0ODUtNDUxZTc4YWRlY2Mw/convnext_whole_mup_clean_c2_w23_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s0/step-425",
    "c2_w21_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MTk0OTg2OGUtNTg2YS01MzNkLWRhNzUtYWUzMDVjYWJkZGM4/convnext_whole_mup_clean_c2_w21_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s1/step-504",
    "c1_w13_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZWRiMzljMzItYzk2YS01Njc1LWUzYmMtNzQ5Njc0MDQ2MjY1/convnext_whole_mup_clean_c1_w13_dp0_plan10m/prune10m_uniform100k_seed0_s1/step-386",
    "c2_w15_s1": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZDMxOWMwYTgtNGM3YS04ZmQ1LWVhODMtMTYyMzdmNmI1MDZk/convnext_whole_mup_clean_c2_w15_dp0_plan10m_lr5p6e4/prune10m_uniform100k_seed0_s1/step-940",
}

CONVNEXT_WHOLE_BUDGET2_CHECKPOINTS = {
    "w11_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-NzhiYzU3NWItMTBiYS0yMGRhLTM5YzQtMWVhM2U1NDdkZmE1/convnext_whole_mup_c2_w11_dp0/prune10m_random1m_seed0_s0/step-1645",
    "w13_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZDE4M2JjZWEtYmYzNS0yOTVlLTg5YWUtZTVhZjZlYjcyNDQ3/convnext_whole_mup_c2_w13_dp0/prune10m_random1m_seed0_s0/step-1219",
    "w15_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MmYyYzRlNWItNGJjOS04N2Y2LTY5NmYtZTNkMmNjOTkyYjIx/convnext_whole_mup_c2_w15_dp0/prune10m_random1m_seed0_s0/step-940",
    "w17_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MGJmYTJmYzEtYTMyNC0xOGJkLWJmZDItNWQ2NzdkNGJkN2Qy/convnext_whole_mup_c2_w17_dp0/prune10m_random1m_seed0_s0/step-747",
}

CONVNEXT_WHOLE_BUDGET2_100K_CHECKPOINTS = {
    "w15_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-MGRkN2ZkN2UtNzZhZi04MjNjLThhYWYtODMyZjk2ODlhYWUx/convnext_whole_mup_c2_w15_dp0/prune10m_random100k_seed0_s0/step-940",
    "w17_dp0": "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/run-ZGExMjk4MTUtZTQ5Zi0xMmU3LTcyNGQtYTA3YTFjNDU4NmZk/convnext_whole_mup_c2_w17_dp0/prune10m_random100k_seed0_s0/step-747",
}

CONVNEXT_WHOLE_MUP_500K_CHECKPOINTS = {
    29: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-YmY0ZmVmMzEtNjMxOS0zNDRmLTBlNzMtNDA5YzYxMDRjYjk0/convnext_whole_mup_w29_500k_one_pass/train_500k_20250717_s0/step-24000",
    21: "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/gill/run-OTI2ZWNjYjQtMGJhNS0wY2M2LTA1M2EtMjg2NzA0MzY5OGQ4/convnext_whole_mup_w21_500k_one_pass/train_500k_20250717_s0/step-24000",
}

CONVNEXT_WHOLE_W11_PLANINIT_WM_SIDE_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/"
    "run-ZDNjYzNkZTYtMDJjMC1mYTIwLWFjNWQtMDYwNTE0YmE4OGU5/"
    "convnext_whole_mup_w11_dp0_planinit_wm_side/"
    "prune10m_random100k_seed0_s0/step-520"
)

CONVNEXT_WHOLE_W11_PLANINIT_WM_TRUE_MATCH_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/batman/"
    "run-MWZjNDhlY2MtNWViOS0wOTcyLTE2MzAtMDM1ZTRlYjc2NDZh/"
    "convnext_whole_mup_w11_dp0_planinit_wm_true_match/"
    "prune10m_random100k_seed0_s0/step-520"
)

CONVNEXT_XXL_PRODUCTION_ANCHOR_CHECKPOINT = (
    "mkv://data-gen.comma.life:3080/reporterv2/checkpoint/"
    "6f5c8d09-b057-bd3c-f8e8-46daddea14f8/102400"
)


def _convnext_whole_round1_validate(
    stage0: int,
    *,
    drop_path_rate: float,
    checkpoint: str,
    budget2: bool = False,
    clean_c1: bool = False,
    clean_c1_plan10m: bool = False,
    clean_c0_plan10m: bool = False,
    clean_c0_lr: float = 1e-3,
    clean_c2: bool = False,
    clean_c2_plan10m: bool = False,
    clean_c2_plan10m_lr5p6e4: bool = False,
    clean_c2_plan10m_lr4e4: bool = False,
    clean_c3_lr: float | None = None,
    clean_c4_lr: float | None = None,
    clean_v2: bool = False,
    clean_v3: bool = False,
    one_pass_500k: bool = False,
    seed: int = 0,
    worldmodel_plan_init: bool = False,
    worldmodel_plan_init_true_match: bool = False,
    study_list: str = "prune10m_study_5000.txt",
    study_segments: int = 5_000,
) -> PathTrainer.Config:
    if one_pass_500k:
        if drop_path_rate != 0.0 or stage0 not in CONVNEXT_WHOLE_MUP_500K_CHECKPOINTS:
            raise ValueError("500k one-pass validation is limited to w21/w29 dp0")
        config = _convnext_whole_mup_500k_one_pass_noval(stage0)
    elif clean_v2:
        if drop_path_rate != 0.0:
            raise ValueError("clean-v2 family is dp0 only")
        config = _convnext_whole_clean_v2_registered(stage0)
    elif clean_v3:
        if drop_path_rate != 0.0:
            raise ValueError("clean-v3 family is dp0 only")
        config = _convnext_whole_clean_v3_registered(stage0)
    elif clean_c0_plan10m:
        if drop_path_rate != 0.0 or (clean_c0_lr != 1e-3 and stage0 != 11):
            raise ValueError("clean C0 is dp0; non-base LR is w11 only")
        config = _convnext_whole_clean_c0_plan10m_registered(stage0, lr=clean_c0_lr)
    elif clean_c3_lr is not None:
        if stage0 not in CONVNEXT_MUP_BUDGET3_WIDTHS or drop_path_rate != 0.0:
            raise ValueError("clean C3 family is dp0 and uses a registered C3 width")
        config = _convnext_whole_clean_c3_plan10m_registered(
            stage0, lr=clean_c3_lr, seed=seed
        )
    elif clean_c4_lr is not None:
        if stage0 not in CONVNEXT_MUP_BUDGET4_WIDTHS or drop_path_rate != 0.0:
            raise ValueError("clean C4 family is dp0 and uses a registered C4 width")
        config = _convnext_whole_clean_c4_plan10m_registered(
            stage0, lr=clean_c4_lr, seed=seed
        )
    elif clean_c2_plan10m_lr5p6e4:
        if drop_path_rate != 0.0:
            raise ValueError("clean C2 low-LR family is dp0 only")
        config = _convnext_whole_clean_c2_plan10m_lr5p6e4_registered(stage0, seed=seed)
    elif clean_c2_plan10m_lr4e4:
        if stage0 != 19 or drop_path_rate != 0.0:
            raise ValueError("clean C2 lr4e4 diagnostic is w19 dp0 only")
        config = convnext_whole_mup_clean_c2_w19_plan10m_lr4e4()
    elif clean_c2_plan10m:
        if stage0 != 15 or drop_path_rate != 0.0:
            raise ValueError("clean C2 plan10m discriminator is w15 dp0 only")
        config = convnext_whole_mup_clean_c2_w15_plan10m()
    elif clean_c1_plan10m:
        if drop_path_rate != 0.0:
            raise ValueError("clean C1 plan10m family is dp0 only")
        config = _convnext_whole_plan10m_registered(stage0, budget2=False, seed=seed)
    elif clean_c1:
        if drop_path_rate != 0.0:
            raise ValueError("clean C1 re-ground is dp0 only")
        config = _convnext_whole_clean_c1_registered(stage0, seed=seed)
    elif clean_c2:
        if drop_path_rate != 0.0:
            raise ValueError("clean C2 is dp0 only")
        config = _convnext_whole_clean_c2_registered(stage0)
    elif budget2:
        config = _convnext_whole_budget2_registered(stage0)
    elif worldmodel_plan_init or worldmodel_plan_init_true_match:
        if stage0 != 11 or drop_path_rate != 0.0:
            raise ValueError("worldmodel plan-init side experiment is w11 dp0 only")
        config = (
            convnext_whole_mup_w11_planinit_wm_true_match()
            if worldmodel_plan_init_true_match
            else convnext_whole_mup_w11_planinit_wm_side()
        )
    else:
        config = _convnext_whole_registered(
            stage0, mup=True, drop_path_rate=drop_path_rate
        )
    policy_dim = _convnext_mup_policy_dim(stage0)

    # The trainer currently validates only after a train step.  A zero base LR
    # makes that bridge step leave every model parameter unchanged.
    config.optimizer = _convnext_mup_optimizer_config(
        stage0=stage0,
        policy_dim=policy_dim,
        lr=0.0,
        wd=CONVNEXT_STUDY_BASE_WEIGHT_DECAY,
    )
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=1,
        total_steps=1,
        decay_ratio=0.2,
        decay_type="cosine",
        min_lr_factor=0.0,
    )
    config.training.local_batch_size = 5
    config.training.global_batch_size = 40
    config.training.steps = 1
    config.dataloader = dataclasses.replace(
        config.dataloader, plan_only=True, one_pass=False
    )

    config.checkpoint.initial_load_path = checkpoint
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.allow_partial_initial_load = False
    config.checkpoint.load_only = True
    config.checkpoint.enable_first_step_checkpoint = False

    config.metrics.log_freq = 1
    config.metrics.save_freq = 1
    config.validator.enable = True
    config.validator.freq = 1
    if study_segments % config.training.global_batch_size:
        raise ValueError("study segment count must be divisible by global batch size")
    config.validator.steps = study_segments // config.training.global_batch_size
    config.validator.dataloader = dataclasses.replace(
        _dataloader_config(split="val", fps=SUPERCOMBO_FPS, plan_only=True),
        dataset=os.path.join(XX_BASEDIR, "datasets/lists", study_list),
        pipeline_dir=BASE_DIR_GT_10M,
        limit=study_segments,
        val_skip=24,
        one_pass=True,
    )
    if not config.dataloader.plan_only or config.dataloader.one_pass:
        raise ValueError(
            "validate-only bridge must use the streaming plan-only train loader"
        )
    if (
        not config.validator.dataloader.plan_only
        or not config.validator.dataloader.one_pass
    ):
        raise ValueError(
            "canonical validation must use the exact one-pass plan-only loader"
        )
    config.validator.reports = {}
    config.validator.save_predictions = True
    return dataclasses.replace(config)


def convnext_whole_mup_w29_500k_one_pass_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        29,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_MUP_500K_CHECKPOINTS[29],
        one_pass_500k=True,
    )


def convnext_whole_mup_w21_500k_one_pass_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        21,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_MUP_500K_CHECKPOINTS[21],
        one_pass_500k=True,
    )


def convnext_whole_mup_w7_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        7, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w7_dp0"]
    )


def convnext_whole_mup_w9_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        9, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w9_dp0"]
    )


def convnext_whole_mup_w11_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w11_dp0"]
    )


def convnext_whole_mup_w13_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        13, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w13_dp0"]
    )


def convnext_whole_mup_w15_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w15_dp0"]
    )


def _convnext_whole_mup_clean_c1_validate_study5000(
    stage0: int, *, seed: int
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C1_CHECKPOINTS[f"w{stage0}_s{seed}"],
        clean_c1=True,
        seed=seed,
    )


def convnext_whole_mup_clean_c1_w7_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(7, seed=0)


def convnext_whole_mup_clean_c1_w9_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(9, seed=0)


def convnext_whole_mup_clean_c1_w11_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(11, seed=0)


def convnext_whole_mup_clean_c1_w13_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(13, seed=0)


def convnext_whole_mup_clean_c1_w15_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(15, seed=0)


def convnext_whole_mup_clean_c1_w11_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(11, seed=1)


def convnext_whole_mup_clean_c1_w15_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_validate_study5000(15, seed=1)


def _convnext_whole_mup_clean_c2_validate_study5000(
    stage0: int,
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_CHECKPOINTS[f"w{stage0}"],
        clean_c2=True,
    )


def convnext_whole_mup_clean_c2_w11_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_validate_study5000(11)


def convnext_whole_mup_clean_c2_w13_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_validate_study5000(13)


def convnext_whole_mup_clean_c2_w15_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_validate_study5000(15)


def convnext_whole_mup_clean_c2_w15_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_W15_PLAN10M_CHECKPOINT,
        clean_c2_plan10m=True,
    )


def convnext_whole_mup_clean_c2_w15_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_W15_PLAN10M_LR5P6E4_CHECKPOINT,
        clean_c2_plan10m_lr5p6e4=True,
    )


def _convnext_whole_mup_clean_c1_plan10m_validate_study5000(
    stage0: int,
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C1_PLAN10M_CHECKPOINTS[f"w{stage0}"],
        clean_c1_plan10m=True,
    )


def convnext_whole_mup_clean_c1_w7_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(7)


def convnext_whole_mup_clean_c1_w9_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(9)


def convnext_whole_mup_clean_c1_w11_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(11)


def convnext_whole_mup_clean_c1_w13_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(13)


def convnext_whole_mup_clean_c1_w15_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(15)


def convnext_whole_mup_clean_c1_w17_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(17)


def convnext_whole_mup_clean_c1_w19_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(19)


def convnext_whole_mup_clean_c1_w21_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c1_plan10m_validate_study5000(21)


def convnext_whole_mup_clean_c0_w11_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C0_CHECKPOINTS["w11_lr1e3"],
        clean_c0_plan10m=True,
        clean_c0_lr=1e-3,
    )


def convnext_whole_mup_clean_c0_w7_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        7,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C0_CHECKPOINTS["w7_lr1e3"],
        clean_c0_plan10m=True,
        clean_c0_lr=1e-3,
    )


def convnext_whole_mup_clean_c0_w9_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        9,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C0_CHECKPOINTS["w9_lr1e3"],
        clean_c0_plan10m=True,
        clean_c0_lr=1e-3,
    )


def convnext_whole_mup_clean_c0_w13_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        13,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C0_CHECKPOINTS["w13_lr1e3"],
        clean_c0_plan10m=True,
        clean_c0_lr=1e-3,
    )


def convnext_whole_mup_clean_c0_w11_plan10m_lr1p78e3_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C0_CHECKPOINTS["w11_lr1p78e3"],
        clean_c0_plan10m=True,
        clean_c0_lr=1.78e-3,
    )


def convnext_whole_mup_clean_c3_w27_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        27,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C3_LR_CHECKPOINTS["lr5p6e4"],
        clean_c3_lr=5.6e-4,
    )


def convnext_whole_mup_clean_c3_w27_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        27,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C3_LR_CHECKPOINTS["lr3p2e4"],
        clean_c3_lr=3.2e-4,
    )


def _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(
    stage0: int,
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C3_LR_CHECKPOINTS[f"w{stage0}_lr3p2e4"],
        clean_c3_lr=3.2e-4,
    )


def convnext_whole_mup_clean_c3_w23_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(23)


def convnext_whole_mup_clean_c3_w25_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(25)


def convnext_whole_mup_clean_c3_w29_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(29)


def convnext_whole_mup_clean_c3_w31_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(31)


def convnext_whole_mup_clean_c3_w33_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(33)


def convnext_whole_mup_clean_c3_w35_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(35)


def convnext_whole_mup_clean_c3_w37_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(37)


def convnext_whole_mup_clean_c3_w37_plan10m_lr3p2e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        37,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C3_LR_CHECKPOINTS["w37_lr3p2e4_s1"],
        clean_c3_lr=3.2e-4,
        seed=1,
    )


def convnext_whole_mup_clean_c3_w39_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(39)


def convnext_whole_mup_clean_c3_w41_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c3_lr3p2e4_validate_study5000(41)


def _convnext_whole_mup_clean_v2_validate_study5000(
    stage0: int,
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_V2_CHECKPOINTS[stage0],
        clean_v2=True,
    )


def convnext_whole_mup_clean_v2_w13_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(13)


def convnext_whole_mup_clean_v2_w7_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(7)


def convnext_whole_mup_clean_v2_w9_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(9)


def convnext_whole_mup_clean_v2_w11_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(11)


def convnext_whole_mup_clean_v2_w17_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(17)


def convnext_whole_mup_clean_v2_w21_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(21)


def convnext_whole_mup_clean_v2_w25_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(25)


def convnext_whole_mup_clean_v2_w29_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v2_validate_study5000(29)


def _convnext_whole_mup_clean_v3_validate_study5000(
    stage0: int,
) -> PathTrainer.Config:
    env_name = f"CONVNEXT_CLEAN_V3_W{stage0}_CHECKPOINT"
    checkpoint = os.getenv(env_name)
    if not checkpoint:
        raise ValueError(f"{env_name} must name the completed clean-v3 checkpoint")
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=checkpoint,
        clean_v3=True,
    )


def convnext_whole_mup_clean_v3_w11_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v3_validate_study5000(11)


def convnext_whole_mup_clean_v3_w15_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v3_validate_study5000(15)


def convnext_whole_mup_clean_v3_w21_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v3_validate_study5000(21)


def convnext_whole_mup_clean_v3_w29_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v3_validate_study5000(29)


def convnext_whole_mup_clean_v3_w41_plan10m_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_v3_validate_study5000(41)


def convnext_whole_mup_clean_c4_w37_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        37,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["lr3p2e4"],
        clean_c4_lr=3.2e-4,
    )


def convnext_whole_mup_clean_c4_w37_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        37,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w29_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        29,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w29_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w31_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        31,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w31_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w33_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        33,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w33_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w35_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        35,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w35_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w35_plan10m_lr1p8e4_wd5p6e2_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        35,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w35_lr1p8e4_wd5p6e2"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w35_plan10m_lr3p2e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        35,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w35_lr3p2e4"],
        clean_c4_lr=3.2e-4,
    )


def convnext_whole_mup_clean_c4_w35_plan10m_lr1p8e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        35,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w35_lr1p8e4_s1"],
        clean_c4_lr=1.8e-4,
        seed=1,
    )


def convnext_whole_mup_clean_c4_w37_plan10m_lr1p8e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        37,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w37_lr1p8e4_s1"],
        clean_c4_lr=1.8e-4,
        seed=1,
    )


def convnext_whole_mup_clean_c4_w39_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        39,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w39_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w41_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        41,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w41_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w43_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        43,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w43_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w45_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        45,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w45_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w47_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        47,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w47_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w49_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        49,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w49_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w51_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        51,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w51_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def convnext_whole_mup_clean_c4_w47_plan10m_lr1p8e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        47,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w47_lr1p8e4_s1"],
        clean_c4_lr=1.8e-4,
        seed=1,
    )


def convnext_whole_mup_clean_c4_w49_plan10m_lr1p8e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        49,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w49_lr1p8e4_s1"],
        clean_c4_lr=1.8e-4,
        seed=1,
    )


def convnext_whole_mup_clean_c4_w51_plan10m_lr1p8e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        51,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w51_lr1p8e4_s1"],
        clean_c4_lr=1.8e-4,
        seed=1,
    )


def convnext_whole_mup_clean_c4_w53_plan10m_lr1p8e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        53,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C4_LR_CHECKPOINTS["w53_lr1p8e4"],
        clean_c4_lr=1.8e-4,
    )


def _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(
    stage0: int,
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_PLAN10M_LR5P6E4_CHECKPOINTS[f"w{stage0}"],
        clean_c2_plan10m_lr5p6e4=True,
    )


def convnext_whole_mup_clean_c2_w11_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(11)


def convnext_whole_mup_clean_c2_w13_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(13)


def convnext_whole_mup_clean_c2_w17_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(17)


def convnext_whole_mup_clean_c2_w19_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(19)


def convnext_whole_mup_clean_c2_w21_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(21)


def convnext_whole_mup_clean_c2_w19_plan10m_lr5p6e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        19,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_DIAGNOSTIC_CHECKPOINTS["w19_s1"],
        clean_c2_plan10m_lr5p6e4=True,
        seed=1,
    )


def convnext_whole_mup_clean_c2_w17_plan10m_lr5p6e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        17,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_DIAGNOSTIC_CHECKPOINTS["w17_s1"],
        clean_c2_plan10m_lr5p6e4=True,
        seed=1,
    )


def convnext_whole_mup_clean_c2_w19_plan10m_lr4e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        19,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_C2_DIAGNOSTIC_CHECKPOINTS["w19_lr4e4"],
        clean_c2_plan10m_lr4e4=True,
    )


def convnext_whole_mup_clean_c2_w23_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        23,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_ZOOM_CHECKPOINTS["c2_w23_s0"],
        clean_c2_plan10m_lr5p6e4=True,
    )


def convnext_whole_mup_clean_c2_w25_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(25)


def convnext_whole_mup_clean_c2_w27_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(27)


def convnext_whole_mup_clean_c2_w29_plan10m_lr5p6e4_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_plan10m_lr5p6e4_validate_study5000(29)


def convnext_whole_mup_clean_c2_w21_plan10m_lr5p6e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        21,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_ZOOM_CHECKPOINTS["c2_w21_s1"],
        clean_c2_plan10m_lr5p6e4=True,
        seed=1,
    )


def convnext_whole_mup_clean_c1_w13_plan10m_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        13,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_ZOOM_CHECKPOINTS["c1_w13_s1"],
        clean_c1_plan10m=True,
        seed=1,
    )


def convnext_whole_mup_clean_c2_w15_plan10m_lr5p6e4_seed1_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_CLEAN_ZOOM_CHECKPOINTS["c2_w15_s1"],
        clean_c2_plan10m_lr5p6e4=True,
        seed=1,
    )


def convnext_whole_mup_clean_c2_w17_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_mup_clean_c2_validate_study5000(17)


def convnext_whole_mup_w11_planinit_wm_side_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_W11_PLANINIT_WM_SIDE_CHECKPOINT,
        worldmodel_plan_init=True,
    )


def convnext_whole_mup_w11_planinit_wm_true_match_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_W11_PLANINIT_WM_TRUE_MATCH_CHECKPOINT,
        worldmodel_plan_init_true_match=True,
    )


def convnext_whole_mup_w7_dp20_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        7, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w7_dp20"]
    )


def convnext_whole_mup_w9_dp20_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        9, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w9_dp20"]
    )


def convnext_whole_mup_w11_dp20_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w11_dp20"]
    )


def convnext_whole_mup_w13_dp20_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        13, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w13_dp20"]
    )


def convnext_whole_mup_w15_dp20_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w15_dp20"]
    )


def convnext_whole_mup_c2_w11_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        11,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_BUDGET2_CHECKPOINTS["w11_dp0"],
        budget2=True,
    )


def convnext_whole_mup_c2_w13_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        13,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_BUDGET2_CHECKPOINTS["w13_dp0"],
        budget2=True,
    )


def convnext_whole_mup_c2_w15_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_BUDGET2_CHECKPOINTS["w15_dp0"],
        budget2=True,
    )


def convnext_whole_mup_c2_w17_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        17,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_BUDGET2_CHECKPOINTS["w17_dp0"],
        budget2=True,
    )


def convnext_whole_mup_c2_w15_100k_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        15,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_BUDGET2_100K_CHECKPOINTS["w15_dp0"],
        budget2=True,
    )


def convnext_whole_mup_c2_w17_100k_validate_study5000() -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        17,
        drop_path_rate=0.0,
        checkpoint=CONVNEXT_WHOLE_BUDGET2_100K_CHECKPOINTS["w17_dp0"],
        budget2=True,
    )


def convnext_xxlarge_validate_study5000() -> PathTrainer.Config:
    config = convnext_xxlarge()
    config.optimizer = _optimizer_config()
    for group in config.optimizer.param_groups:
        group.optimizer_kwargs["lr"] = 0.0
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=1,
        total_steps=1,
        decay_ratio=0.2,
        decay_type="cosine",
        min_lr_factor=0.0,
    )
    config.training.local_batch_size = 5
    config.training.global_batch_size = 40
    config.training.steps = 1
    config.dataloader.one_pass = False
    config.checkpoint.initial_load_path = CONVNEXT_XXL_PRODUCTION_ANCHOR_CHECKPOINT
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.allow_partial_initial_load = False
    config.checkpoint.load_only = True
    config.checkpoint.enable_first_step_checkpoint = False
    config.metrics.log_freq = 1
    config.metrics.save_freq = 1
    config.validator.enable = True
    config.validator.freq = 1
    config.validator.steps = 125
    config.validator.dataloader = dataclasses.replace(
        _dataloader_config(split="val", fps=SUPERCOMBO_FPS, plan_only=True),
        dataset=os.path.join(XX_BASEDIR, "datasets/lists/prune10m_study_5000.txt"),
        pipeline_dir=BASE_DIR_GT_10M,
        limit=5_000,
        val_skip=24,
    )
    config.validator.reports = {}
    config.validator.save_predictions = True
    return dataclasses.replace(config)


def convnext_xxlarge_validate_500kval() -> PathTrainer.Config:
    """Score the production XXL anchor on the 500k list's own val split."""
    config = convnext_xxlarge_validate_study5000()
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def _convnext_xxlarge_validate_500kval_at(checkpoint: str) -> PathTrainer.Config:
    """Score one production XXL checkpoint on the 500k val split with the exact
    yaP-lineage sampler. The study5000-lineage val loader's one_pass=False
    fill-once path scores 8 samples for each of 625 segments; the yaP
    instrument's one_pass=True stream covers ~900 segments."""
    config = _convnext_tiny_500k_one_pass_noval_validate_study5000(
        checkpoint=checkpoint,
        base_config=_yap("convnext_xxlarge"),
    )
    config.validator.dataloader = dataclasses.replace(
        config.validator.dataloader,
        dataset=DEFAULT_TRAIN_LIST,
        pipeline_dir=BASE_DIR_GT,
    )
    return dataclasses.replace(config)


def _convnext_xxlarge_anchor_checkpoint(step: int) -> str:
    return f"{CONVNEXT_XXL_PRODUCTION_ANCHOR_CHECKPOINT.rsplit('/', 1)[0]}/{step}"


def convnext_xxlarge_validate_500kval_v2() -> PathTrainer.Config:
    return _convnext_xxlarge_validate_500kval_at(
        CONVNEXT_XXL_PRODUCTION_ANCHOR_CHECKPOINT
    )


def convnext_xxlarge_step1024_validate_500kval() -> PathTrainer.Config:
    return _convnext_xxlarge_validate_500kval_at(
        _convnext_xxlarge_anchor_checkpoint(1_024)
    )


def convnext_xxlarge_step2048_validate_500kval() -> PathTrainer.Config:
    return _convnext_xxlarge_validate_500kval_at(
        _convnext_xxlarge_anchor_checkpoint(2_048)
    )


def convnext_xxlarge_step4096_validate_500kval() -> PathTrainer.Config:
    return _convnext_xxlarge_validate_500kval_at(
        _convnext_xxlarge_anchor_checkpoint(4_096)
    )


def convnext_xxlarge_step8192_validate_500kval() -> PathTrainer.Config:
    return _convnext_xxlarge_validate_500kval_at(
        _convnext_xxlarge_anchor_checkpoint(8_192)
    )


def _convnext_whole_round1_validate_common4920(
    stage0: int, *, drop_path_rate: float, checkpoint: str
) -> PathTrainer.Config:
    return _convnext_whole_round1_validate(
        stage0,
        drop_path_rate=drop_path_rate,
        checkpoint=checkpoint,
        study_list="prune10m_study_4920_43066_intersection.txt",
        study_segments=4_920,
    )


def convnext_whole_mup_w7_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        7, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w7_dp0"]
    )


def convnext_whole_mup_w9_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        9, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w9_dp0"]
    )


def convnext_whole_mup_w11_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        11, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w11_dp0"]
    )


def convnext_whole_mup_w13_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        13, drop_path_rate=0.0, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w13_dp0"]
    )


def convnext_whole_mup_w7_dp20_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        7, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w7_dp20"]
    )


def convnext_whole_mup_w9_dp20_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        9, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w9_dp20"]
    )


def convnext_whole_mup_w11_dp20_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        11, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w11_dp20"]
    )


def convnext_whole_mup_w13_dp20_validate_common4920() -> PathTrainer.Config:
    return _convnext_whole_round1_validate_common4920(
        13, drop_path_rate=0.2, checkpoint=CONVNEXT_WHOLE_ROUND1_CHECKPOINTS["w13_dp20"]
    )


def _dp_degrees() -> tuple[int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(
        os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size))
    )
    return num_nodes, local_world_size


dp_degrees = _dp_degrees


def _path(flavor: str) -> PathTrainer.Config:
    steps = 1024 * 100
    validation_freq = 1024
    reports = {
        name: [validation_freq, steps // 2, steps]
        for name in (
            "analyse_driving",
            "analyse_lat_no_noise",
            "analyse_cones",
            "analyse_lights",
            "analyse_stop",
            "analyse_hard_brake",
        )
    }
    reports["analyse_dataset"] = [validation_freq]
    mixed_precision_param = "bfloat16"
    num_nodes, local_world_size = _dp_degrees()
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    checkpoint_base_folder = (
        f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else ""
    )
    fps = SUPERCOMBO_FPS
    plan_only = False
    return PathTrainer.Config(
        loss=PathLoss.Config(),
        model_spec=model_registry(flavor),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_dataloader_config(split="train", fps=fps, plan_only=plan_only),
        optimizer=_optimizer_config(),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(steps * 0.01),
            total_steps=steps,
            decay_ratio=0.2,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=-1,
            seq_len=1,
            steps=steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param=mixed_precision_param,
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpoint=_checkpoint_config(
            folder=reporterv2_training_id or "checkpoint",
            base_folder=checkpoint_base_folder,
            interval=validation_freq,
        ),
        fps=fps,
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=True, components=["model"]),
        metrics=MetricsProcessor.Config(
            log_freq=16, enable_reporterv2=True, save_freq=validation_freq
        ),
        validator=PathValidator.Config(
            enable=True,
            freq=validation_freq,
            steps=32,
            dataloader=_dataloader_config(split="val", fps=fps, plan_only=plan_only),
            mixed_precision_param=mixed_precision_param,
            reports=reports,
        ),
        debug=DebugConfig(seed=0),
    )


def _model_config(
    flavor: str,
    *,
    vision_dims: tuple[int, ...] | None = None,
    vision_features: int = 512,
    policy_heads: int = 8,
    policy_mlp_multiple: int = 256,
    drop_path_rate: float = 0.2,
    mup: bool = False,
    mup_base_policy_dim: int = 512,
    worldmodel_plan_init: bool = False,
    worldmodel_plan_init_true_match: bool = False,
) -> PathModel.Config:
    n_frames_input = N_FRAMES
    input_frame_names = INPUT_FRAMES_NAMES
    input_frame_type = FRAME_TYPE
    frame_constants = frame_constants_from_fps(
        n_frames=n_frames_input, frame_type=input_frame_type
    )
    in_channels = sum(
        frame_constants["frame_shapes"][name][0] for name in input_frame_names
    )
    block_size = len(frame_constants["history_idxs"])
    temporal_len = frame_constants["temporal_len"]
    dim = vision_features
    if dim % policy_heads != 0:
        raise ValueError(
            f"policy dimension {dim} must be divisible by {policy_heads} heads"
        )
    width_mult = dim / mup_base_policy_dim

    return PathModel.Config(
        n_frames_input=n_frames_input,
        input_frame_names=tuple(input_frame_names),
        frame_type=input_frame_type,
        vision=Vision.Config(
            flavor=flavor,
            input_frame_names=tuple(input_frame_names),
            in_channels=in_channels,
            vision_features=vision_features,
            pretrained=False,
            drop_path_rate=drop_path_rate,
            mean=255 / 2,
            std=255 / 4,
            dims=vision_dims,
            mup=mup,
        ),
        point_policy=Policy.Config(
            summarizer=PointSummarizer.Config(
                mlp1=_mlp(
                    dim,
                    mlp_mult=2,
                    bias=False,
                    dropout=0.0,
                    multiple_of=policy_mlp_multiple,
                    mup=mup,
                ),
                mlp2=_mlp(
                    dim,
                    mlp_mult=2,
                    bias=False,
                    dropout=0.0,
                    multiple_of=policy_mlp_multiple,
                    mup=mup,
                ),
            ),
            hydra=_hydra(
                _heads(META_HEADS + POSE_HEADS),
                in_features=dim,
                mlp_mult=2,
                multiple_of=policy_mlp_multiple,
                mup=mup,
                width_mult=width_mult,
                base_dim=mup_base_policy_dim,
            ),
        ),
        temporal_policy=TemporalPolicy.Config(
            temporal_summarizer=TemporalSummarizer.Config(
                mlp1=_mlp(
                    dim,
                    mlp_mult=2,
                    bias=False,
                    dropout=0.0,
                    multiple_of=policy_mlp_multiple,
                    mup=mup,
                ),
                mlp2=_mlp(
                    dim,
                    mlp_mult=2,
                    bias=False,
                    dropout=0.0,
                    multiple_of=policy_mlp_multiple,
                    mup=mup,
                ),
                desire_encoder=_encoder(
                    TEMPORAL_INPUTS[ModelInputs.DESIRE][0] * temporal_len,
                    dim,
                    mup=mup,
                ),
                traffic_encoder=_encoder(
                    TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0], dim, mup=mup
                ),
                action_t_encoder=_encoder(
                    TEMPORAL_INPUTS[ModelInputs.ACTION_T][0], dim, mup=mup
                ),
                transformer=PathTransformer.Config(
                    layers=[
                        PathTransformerBlock.Config(
                            attention=_attention(
                                dim=dim,
                                n_head=policy_heads,
                                dropout=0.1,
                                mup=mup,
                            ),
                            mlp=_mlp(
                                dim,
                                mlp_mult=2,
                                bias=True,
                                dropout=0.1,
                                multiple_of=policy_mlp_multiple,
                                mup=mup,
                            ),
                        )
                        for _ in range(4)
                    ]
                ),
                pos_embedding=Embedding.Config(
                    num_embeddings=block_size,
                    embedding_dim=dim,
                    param_init=_LINEAR_INIT,
                ),
                block_size=block_size,
                dense_training_outputs=True,
            ),
            temporal_hydra=_hydra(
                _heads(DRIVING_HEADS + TEMPORAL_META_HEADS),
                in_features=dim,
                mlp_mult=2,
                multiple_of=policy_mlp_multiple,
                mup=mup,
                width_mult=width_mult,
                base_dim=mup_base_policy_dim,
                worldmodel_plan_init=worldmodel_plan_init,
                worldmodel_plan_init_true_match=worldmodel_plan_init_true_match,
            ),
            history_idxs=tuple(int(x) for x in frame_constants["history_idxs"]),
        ),
    )


def _dataloader_config(
    *, split: str, fps: int, plan_only: bool
) -> PathDataLoader.Config:
    base = XXPathDatasetConfig(fps=fps, plan_only=plan_only)
    return PathDataLoader.Config(
        dataset=DEFAULT_BIG_TRAIN_LIST,
        split=split,
        shuffle_size=_si_int(base.shuffle_size),
        min_mixing=base.min_mixing,
        num_writers=base.num_writers,
        num_readers=base.num_readers,
        fps=base.fps,
        pipeline_dir=base.pipeline_dir,
        plan_only=base.plan_only,
        limit=base.limit,
        n_frames=base.n_frames,
        rgb=base.rgb,
        unvision=base.unvision,
        val_skip=base.val_skip,
    )


def _checkpoint_config(
    folder: str, base_folder: str, interval: int
) -> PathOnnxCheckpointManager.Config:
    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    temporal_len = frame_constants["temporal_len"]
    vision_input_names = [ModelInputs.IMG, ModelInputs.BIG_IMG]
    temporal_policy_input_names = [
        ModelInputs.FEATURES,
        ModelInputs.DESIRE,
        ModelInputs.TRAFFIC,
        ModelInputs.ACTION_T,
    ]
    input_names = [
        *vision_input_names,
        *temporal_policy_input_names,
    ]
    input_shapes = [
        [1, *frame_constants["frame_shapes"][ModelInputs.IMG]],
        [1, *frame_constants["frame_shapes"][ModelInputs.BIG_IMG]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.FEATURES][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.DESIRE][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.ACTION_T][0]],
    ]
    return PathOnnxCheckpointManager.Config(
        keep_latest_k=0,  # keep all checkpoints
        enable=True,
        checkpoint_base_folder=base_folder,
        save_model_state_dict=True,  # another copy of full state dict
        export_onnx=True,
        enable_first_step_checkpoint=True,
        folder=folder,
        interval=interval,
        input_names=input_names,
        input_shapes=input_shapes,
        input_dtypes=["float16"] * len(input_names),
        onnx_model_dtype="float16",  # WIP: test if fp16 doesn't degrade performance
        vision_input_names=vision_input_names,
        temporal_policy_input_names=temporal_policy_input_names,
    )


def _si_int(value: str | int) -> int:
    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    value = str(value).strip().lower()
    return (
        int(float(value[:-1]) * suffixes[value[-1]])
        if value[-1] in suffixes
        else int(value)
    )


def _optimizer_config() -> OptimizersContainer.Config:
    common = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8}
    no_decay = r"(point_policy\.hydra|temporal_policy\.temporal_hydra)\.(final_layer|scale_layer)"
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16",
        param_groups=[
            ParamGroupConfig(
                pattern=no_decay,
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 3e-2},
            ),
        ],
    )


def _convnext_mup_policy_dim(stage0: int) -> int:
    production_ratio = CONVNEXT_MUP_BASE_POLICY_DIM / CONVNEXT_MUP_BASE_DIMS[0]
    unrounded = stage0 * production_ratio
    multiple = CONVNEXT_MUP_POLICY_HEADS
    return max(multiple, multiple * math.floor(unrounded / multiple + 0.5))


def _convnext_mup_optimizer_config(
    *, stage0: int, policy_dim: int, lr: float, wd: float
) -> OptimizersContainer.Config:
    vision_mult = stage0 / CONVNEXT_MUP_BASE_DIMS[0]
    policy_mult = policy_dim / CONVNEXT_MUP_BASE_POLICY_DIM
    common = {"betas": (0.9, 0.95), "eps": 1e-8}
    no_decay = (
        r"(point_policy\.hydra|temporal_policy\.temporal_hydra)\."
        r"(final_layer|scale_layer)"
    )
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16",
        lr=lr,
        param_groups=[
            ParamGroupConfig(
                pattern=no_decay,
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern=CONVNEXT_MUP_VISION_PATTERN,
                optimizer_name="AdamW",
                lr_mult=1.0 / vision_mult,
                optimizer_kwargs={**common, "weight_decay": wd * vision_mult},
            ),
            ParamGroupConfig(
                pattern=CONVNEXT_MUP_POLICY_PATTERN,
                optimizer_name="AdamW",
                lr_mult=1.0 / policy_mult,
                optimizer_kwargs={**common, "weight_decay": wd * policy_mult},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": wd},
            ),
        ],
    )


def _convnext_standard_optimizer_config(
    *, lr: float, wd: float
) -> OptimizersContainer.Config:
    common = {"betas": (0.9, 0.95), "eps": 1e-8}
    no_decay = (
        r"(point_policy\.hydra|temporal_policy\.temporal_hydra)\."
        r"(final_layer|scale_layer)"
    )
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16",
        lr=lr,
        param_groups=[
            ParamGroupConfig(
                pattern=no_decay,
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": wd},
            ),
        ],
    )


def _convnext_whole(
    *,
    stage0: int,
    drop_path_rate: float,
    steps: int,
    flavor: str,
    mup: bool,
    lr: float = 1e-3,
    wd: float = CONVNEXT_STUDY_BASE_WEIGHT_DECAY,
    worldmodel_plan_init: bool = False,
    worldmodel_plan_init_true_match: bool = False,
    train_list: str = "prune10m_random100k_seed0.txt",
    seed: int = 0,
    production_checkpointing: bool = False,
) -> PathTrainer.Config:
    policy_dim = _convnext_mup_policy_dim(stage0)
    vision_dims = tuple(stage0 * 2**stage for stage in range(4))
    model_config = _model_config(
        "convnext_xxlarge",
        vision_dims=vision_dims,
        vision_features=policy_dim,
        policy_heads=CONVNEXT_MUP_POLICY_HEADS,
        policy_mlp_multiple=1,
        drop_path_rate=drop_path_rate,
        mup=mup,
        mup_base_policy_dim=CONVNEXT_MUP_BASE_POLICY_DIM,
        worldmodel_plan_init=worldmodel_plan_init,
        worldmodel_plan_init_true_match=worldmodel_plan_init_true_match,
    )
    config = _path("convnext_xxlarge")
    config.model_spec = model_registry(flavor, model_config)
    config.dataloader = dataclasses.replace(
        _dataloader_config(split="train", fps=SUPERCOMBO_FPS, plan_only=False),
        dataset=os.path.join(XX_BASEDIR, "datasets/lists", train_list),
        one_pass=True,
    )
    config.optimizer = (
        _convnext_mup_optimizer_config(
            stage0=stage0, policy_dim=policy_dim, lr=lr, wd=wd
        )
        if mup
        else _convnext_standard_optimizer_config(lr=lr, wd=wd)
    )
    config.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=max(1, round(steps * 0.01)),
        total_steps=steps,
        decay_ratio=0.2,
        decay_type="cosine",
        min_lr_factor=0.0,
    )
    config.training.local_batch_size = 16
    config.training.global_batch_size = CONVNEXT_MUP_GLOBAL_BATCH
    config.training.steps = steps
    if not production_checkpointing:
        config.checkpoint = final_checkpoint_config(
            flavor=flavor,
            stem=os.path.splitext(train_list)[0],
            seed=seed,
            steps=steps,
        )
        config.checkpoint.last_save_model_only = True
        config.checkpoint.enable_first_step_checkpoint = False
    config.activation_checkpoint = None
    config.compile = CompileConfig(enable=False, components=[])
    config.metrics.log_freq = 10
    config.metrics.save_freq = steps
    config.validator.enable = False
    config.validator.dataloader = dataclasses.replace(
        _dataloader_config(split="val", fps=SUPERCOMBO_FPS, plan_only=False),
        dataset=os.path.join(XX_BASEDIR, "datasets/lists/prune10m_val.txt"),
    )
    config.debug.seed = seed
    return dataclasses.replace(config)


def _convnext_whole_mup(
    *, stage0: int, drop_path_rate: float, steps: int, flavor: str
) -> PathTrainer.Config:
    return _convnext_whole(
        stage0=stage0,
        drop_path_rate=drop_path_rate,
        steps=steps,
        flavor=flavor,
        mup=True,
    )


def _convnext_whole_mup_500k_one_pass_noval(stage0: int) -> PathTrainer.Config:
    """Run one small whole-model muP family member once over the 500k list."""
    if stage0 not in (21, 29):
        raise ValueError("the overnight 500k comparison is limited to w21 and w29")
    train_list = os.path.basename(DEFAULT_TRAIN_LIST)
    config = _convnext_whole(
        stage0=stage0,
        drop_path_rate=0.0,
        steps=CONVNEXT_500K_ONE_PASS_STEPS,
        flavor=f"convnext_whole_mup_w{stage0}_500k_one_pass",
        mup=True,
        train_list=train_list,
        production_checkpointing=True,
    )
    # These odd-width proxy encoders cannot pass the production ONNX block-size
    # rewrite. Keep the production DCP cadence and full training state, but do
    # not export the deployment artifact from intermediate study checkpoints.
    config.checkpoint.export_onnx = False
    config.dataloader = dataclasses.replace(
        config.dataloader,
        limit=CONVNEXT_500K_TRAIN_SEGMENTS,
    )
    config.metrics.save_freq = 16
    config.validator.enable = False
    return dataclasses.replace(config)


def convnext_whole_mup_w21_500k_one_pass_noval() -> PathTrainer.Config:
    return _convnext_whole_mup_500k_one_pass_noval(21)


def convnext_whole_mup_w29_500k_one_pass_noval() -> PathTrainer.Config:
    return _convnext_whole_mup_500k_one_pass_noval(29)


def _heads(heads) -> tuple[PathHead, ...]:
    return tuple(
        PathHead(head.name, head.output_size, head.mlp, head.scale) for head in heads
    )


def _hidden_dim(dim: int, mlp_mult: float, multiple_of: int = 256) -> int:
    hidden = int(dim * mlp_mult)
    return multiple_of * math.ceil(hidden / multiple_of)


def _mlp(
    dim: int,
    *,
    mlp_mult: float,
    bias: bool,
    dropout: float,
    multiple_of: int = 256,
    mup: bool = False,
) -> PathMLP.Config:
    hidden = _hidden_dim(dim, mlp_mult, multiple_of)
    c_fc = (
        _lin(dim, hidden, std=dim**-0.5, bias=bias)
        if mup
        else Linear.Config(
            in_features=dim,
            out_features=hidden,
            bias=bias,
            param_init=_LINEAR_INIT,
        )
    )
    c_proj = (
        _lin(hidden, dim, std=hidden**-0.5, bias=bias)
        if mup
        else Linear.Config(
            in_features=hidden,
            out_features=dim,
            bias=bias,
            param_init=_LINEAR_INIT,
        )
    )
    return PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        c_fc=c_fc,
        c_proj=c_proj,
        act="gelu_tanh",
        dropout=dropout,
    )


def _encoder(in_features: int, dim: int, *, mup: bool = False) -> LinearEncoder.Config:
    in_layer = (
        _lin(in_features, dim, std=in_features**-0.5)
        if mup
        else Linear.Config(
            in_features=in_features,
            out_features=dim,
            bias=True,
            param_init=_LINEAR_INIT,
        )
    )
    out_layer = (
        _lin(dim, dim, std=dim**-0.5, bias=False)
        if mup
        else Linear.Config(
            in_features=dim,
            out_features=dim,
            bias=False,
            param_init=_LINEAR_INIT,
        )
    )
    return LinearEncoder.Config(
        in_layer=in_layer,
        out_layer=out_layer,
    )


def _attention(
    *, dim: int, n_head: int, dropout: float, mup: bool = False
) -> PathSelfAttention.Config:
    head_dim = dim // n_head
    c_attn = (
        _lin(dim, 3 * dim, std=dim**-0.5)
        if mup
        else Linear.Config(
            in_features=dim,
            out_features=3 * dim,
            bias=True,
            param_init=_LINEAR_INIT,
        )
    )
    c_proj = (
        _lin(dim, dim, std=dim**-0.5)
        if mup
        else Linear.Config(
            in_features=dim,
            out_features=dim,
            bias=True,
            param_init=_LINEAR_INIT,
        )
    )
    return PathSelfAttention.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        q_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        k_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        c_attn=c_attn,
        c_proj=c_proj,
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=dropout,
    )


def _hydra(
    heads: tuple[PathHead, ...],
    *,
    in_features: int,
    mlp_mult: float,
    multiple_of: int = 256,
    mup: bool = False,
    width_mult: float = 1.0,
    base_dim: int = 512,
    worldmodel_plan_init: bool = False,
    worldmodel_plan_init_true_match: bool = False,
) -> Hydra.Config:
    if worldmodel_plan_init and worldmodel_plan_init_true_match:
        raise ValueError("plan init cannot be both raw-match and effective-match")

    def final_layer(head: PathHead) -> Linear.Config | MuReadout.Config:
        if not mup:
            return Linear.Config(
                in_features=in_features,
                out_features=head.output_size,
                bias=True,
                param_init=_LINEAR_INIT,
            )
        param_init = {
            "weight": partial(nn.init.normal_, mean=0.0, std=base_dim**-0.5),
            "bias": nn.init.zeros_,
        }
        if worldmodel_plan_init and head.name == "plan":
            param_init = {
                "weight": partial(
                    nn.init.normal_,
                    mean=0.0,
                    std=CONVNEXT_WORLDMODEL_PLAN_HEAD_INIT_STD,
                ),
                "bias": _init_worldmodel_plan_bias_,
            }
        if worldmodel_plan_init_true_match and head.name == "plan":
            param_init = {
                "weight": partial(
                    nn.init.normal_,
                    mean=0.0,
                    # MuReadout divides features by width_mult.  Multiplying the
                    # raw std by width_mult reproduces a plain Linear at 1e-3.
                    std=CONVNEXT_WORLDMODEL_PLAN_HEAD_INIT_STD * width_mult,
                ),
                "bias": _init_worldmodel_plan_bias_,
            }
        return MuReadout.Config(
            in_features=in_features,
            out_features=head.output_size,
            bias=True,
            width_mult=width_mult,
            output_mult=1.0,
            param_init=param_init,
        )

    return Hydra.Config(
        heads=heads,
        head_mlps={
            head.name: _mlp(
                in_features,
                mlp_mult=mlp_mult,
                bias=False,
                dropout=0.0,
                multiple_of=multiple_of,
                mup=mup,
            )
            for head in heads
            if head.mlp
        },
        final_layers={head.name: final_layer(head) for head in heads},
        scale_layers={
            head.name: ScaleLayer.Config(n_features=head.output_size)
            for head in heads
            if head.scale
        },
    )


def _lin(in_f: int, out_f: int, *, std: float, bias: bool = True) -> Linear.Config:
    return Linear.Config(
        in_features=in_f,
        out_features=out_f,
        bias=bias,
        param_init={
            "weight": partial(nn.init.normal_, mean=0.0, std=std),
            "bias": nn.init.zeros_,
        },
    )


def _hidden_std(fan_in: int, *, mup: bool) -> float:
    return fan_in**-0.5 if mup else VIT_BASE_WIDTH**-0.5


def _vit_attention(dim: int, *, n_head: int, mup: bool) -> PathSelfAttention.Config:
    head_dim = dim // n_head
    return PathSelfAttention.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        q_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        k_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        c_attn=_lin(dim, 3 * dim, std=_hidden_std(dim, mup=mup)),
        c_proj=_lin(
            dim, dim, std=_hidden_std(dim, mup=mup) / math.sqrt(2 * VIT_NUM_LAYERS)
        ),
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=0.0,
        is_causal=False,
    )


def _vit_mlp(dim: int, *, mup: bool, mult: float = 4.0) -> PathMLP.Config:
    hidden = _hidden_dim(dim, mult)
    return PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        c_fc=_lin(dim, hidden, std=_hidden_std(dim, mup=mup)),
        c_proj=_lin(
            hidden,
            dim,
            std=_hidden_std(hidden, mup=mup) / math.sqrt(2 * VIT_NUM_LAYERS),
        ),
        act="gelu_tanh",
        dropout=0.0,
    )


def _vit_model_config(flavor: str, *, mup: bool) -> PlanViT.Config:
    dim = VIT_WIDTHS[flavor]
    if dim % VIT_HEAD_DIM != 0:
        raise ValueError(
            f"vit width {dim} must be a multiple of head_dim {VIT_HEAD_DIM}"
        )
    n_head = dim // VIT_HEAD_DIM
    pt, ph, pw = VIT_PATCH_SIZE
    patch_dim = pt * VIT_IN_CHANNELS * ph * pw
    t, h, w = VIT_INPUT_SIZE
    num_patches = (t // pt) * (h // ph) * (w // pw)
    return PlanViT.Config(
        mean=255 / 2,
        std=255 / 4,
        patch_embed=PatchEmbed.Config(
            proj=_lin(patch_dim, dim, std=patch_dim**-0.5),
            patch_size=VIT_PATCH_SIZE,
        ),
        pos_embedding=Embedding.Config(
            num_embeddings=num_patches, embedding_dim=dim, param_init=_LINEAR_INIT
        ),
        blocks=[
            PathTransformerBlock.Config(
                attention=_vit_attention(dim, n_head=n_head, mup=mup),
                mlp=_vit_mlp(dim, mup=mup),
            )
            for _ in range(VIT_NUM_LAYERS)
        ],
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        plan_head=PlanHead.Config(
            norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            head=_lin(dim, PLAN_HEAD_SIZE, std=VIT_BASE_WIDTH**-0.5),
            output_mult=(VIT_BASE_WIDTH / dim) if mup else 1.0,
        ),
    )


vit_model_config = _vit_model_config


def vit_model_registry(flavor: str, *, mup: bool) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=_vit_model_config(flavor, mup=mup),
        parallelize_fn=parallelize_vit,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _vit_dataloader_config(*, split: str) -> PathDataLoader.Config:
    dataset = (
        "datasets/lists/prune10m_val.txt"
        if split == "val"
        else "datasets/lists/prune10m_random100k_seed0.txt"
    )
    return dataclasses.replace(
        _dataloader_config(split=split, fps=SUPERCOMBO_FPS, plan_only=True),
        dataset=os.path.join(XX_BASEDIR, dataset),
        pipeline_dir=BASE_DIR_GT_10M,
    )


vit_dataloader_config = _vit_dataloader_config


def _vit_optimizer_config(
    flavor: str, *, mup: bool, lr: float, wd: float
) -> OptimizersContainer.Config:
    m = VIT_WIDTHS[flavor] / VIT_BASE_WIDTH
    common = {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": wd}
    catch_all = ParamGroupConfig(
        pattern=r".*",
        optimizer_name="AdamW",
        optimizer_kwargs=common,
    )
    mup_group = ParamGroupConfig(
        pattern=MUP_PATTERN,
        optimizer_name="AdamW",
        lr_mult=1.0 / m,
        optimizer_kwargs={**common, "weight_decay": wd * m},
    )
    groups = [mup_group, catch_all] if mup else [catch_all]
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16", lr=lr, param_groups=groups
    )


def _vit(
    flavor: str, *, mup: bool, lr: float = 3e-4, wd: float = 0.0125
) -> PathTrainer.Config:
    num_nodes, local_world_size = _dp_degrees()
    return PathTrainer.Config(
        loss=PlanViTLoss.Config(),
        model_spec=vit_model_registry(flavor, mup=mup),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_vit_dataloader_config(split="train"),
        optimizer=_vit_optimizer_config(flavor, mup=mup, lr=lr, wd=wd),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(VIT_STEPS * 0.1),
            total_steps=None,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=-1,
            seq_len=1,
            steps=VIT_STEPS,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        metrics=MetricsProcessor.Config(log_freq=10, enable_reporterv2=True),
        validator=PathValidator.Config(
            enable=True,
            freq=1024,
            steps=32,
            dataloader=_vit_dataloader_config(split="val"),
            mixed_precision_param="bfloat16",
        ),
        fps=SUPERCOMBO_FPS,
        debug=DebugConfig(seed=0),
    )


vit = _vit


def vit_standard_w256() -> PathTrainer.Config:
    return _vit("w256", mup=False)


def vit_standard_w512() -> PathTrainer.Config:
    return _vit("w512", mup=False)


def vit_standard_w1024() -> PathTrainer.Config:
    return _vit("w1024", mup=False)


def vit_standard_w2048() -> PathTrainer.Config:
    return _vit("w2048", mup=False)


def vit_mup_w256() -> PathTrainer.Config:
    return _vit("w256", mup=True)


def vit_mup_w512() -> PathTrainer.Config:
    return _vit("w512", mup=True)


def vit_mup_w1024() -> PathTrainer.Config:
    return _vit("w1024", mup=True)


def vit_mup_w2048() -> PathTrainer.Config:
    return _vit("w2048", mup=True)


def vit_mup_w64() -> PathTrainer.Config:
    return _vit("w64", mup=True)


def vit_mup_w128() -> PathTrainer.Config:
    return _vit("w128", mup=True)


def vit_mup_w192() -> PathTrainer.Config:
    return _vit("w192", mup=True)


def vit_mup_w320() -> PathTrainer.Config:
    return _vit("w320", mup=True)


def vit_mup_w384() -> PathTrainer.Config:
    return _vit("w384", mup=True)


def vit_mup_w448() -> PathTrainer.Config:
    return _vit("w448", mup=True)


def vit_mup_w640() -> PathTrainer.Config:
    return _vit("w640", mup=True)


def vit_mup_w896() -> PathTrainer.Config:
    return _vit("w896", mup=True)


def vit_mup_w1280() -> PathTrainer.Config:
    return _vit("w1280", mup=True)


def vit_mup_w1536() -> PathTrainer.Config:
    return _vit("w1536", mup=True)


def vit_mup_w1792() -> PathTrainer.Config:
    return _vit("w1792", mup=True)


def vit_mup_w3072() -> PathTrainer.Config:
    return _vit("w3072", mup=True)
