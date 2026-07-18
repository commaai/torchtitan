import re

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode
from xx.ml_tools.constants.model import (
    frame_constants_from_fps,
    FRAME_TYPE,
    INPUT_FRAMES_NAMES,
    ModelInputs,
    N_FRAMES,
    TEMPORAL_INPUTS,
)

from torchtitan.experiments.path import config_registry as registry
from torchtitan.experiments.path.model import MuReadout


def _meta_inputs() -> dict[str, torch.Tensor]:
    frame = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    inputs = {
        name: torch.randn(
            1, len(frame["history_idxs"]), *frame["frame_shapes"][name], device="meta"
        )
        for name in INPUT_FRAMES_NAMES
    }
    for name in (ModelInputs.DESIRE, ModelInputs.TRAFFIC, ModelInputs.ACTION_T):
        inputs[name] = torch.randn(
            1, frame["temporal_len"], TEMPORAL_INPUTS[name][0], device="meta"
        )
    return inputs


def test_mup_readout_scales_before_bias() -> None:
    layer = MuReadout.Config(
        in_features=1,
        out_features=1,
        bias=True,
        width_mult=2.0,
        output_mult=3.0,
    ).build()
    with torch.no_grad():
        layer.weight.fill_(5.0)
        layer.bias.fill_(7.0)

    actual = layer(torch.tensor([[11.0]]))
    expected = torch.tensor([[5.0 * (3.0 * 11.0 / 2.0) + 7.0]])

    torch.testing.assert_close(actual, expected)


def test_mup_readout_rejects_nonpositive_width_multiplier() -> None:
    with pytest.raises(ValueError, match="width_mult must be positive"):
        MuReadout.Config(
            in_features=1,
            out_features=1,
            bias=True,
            width_mult=0.0,
        ).build()


def test_production_convnext_remains_the_untouched_source() -> None:
    config = registry.convnext_xxlarge()
    model_config = config.model_spec.model
    assert model_config.vision.dims is None
    assert model_config.vision.depths is None
    assert model_config.vision.mup is False
    assert model_config.vision.vision_features == 512
    assert model_config.vision.drop_path_rate == 0.2

    with torch.device("meta"):
        model = model_config.build()
    assert sum(parameter.numel() for parameter in model.parameters()) == 865_966_608
    assert not any(isinstance(module, MuReadout) for module in model.modules())


@pytest.mark.parametrize(
    ("width", "policy_dim", "params", "steps"),
    (
        (7, 8, 372_307, 1141),
        (9, 16, 607_317, 741),
        (11, 16, 854_255, 520),
        (13, 16, 1_146_761, 386),
    ),
)
def test_round_one_configs_are_the_constructed_whole_model_family(
    width: int, policy_dim: int, params: int, steps: int
) -> None:
    config = getattr(registry, f"convnext_whole_mup_w{width}")()
    model_config = config.model_spec.model
    with torch.device("meta"):
        model = model_config.build()

    assert model_config.vision.dims == tuple(width * 2**stage for stage in range(4))
    assert model_config.vision.vision_features == policy_dim
    assert model_config.vision.drop_path_rate == 0.0
    assert model_config.temporal_policy.temporal_summarizer.transformer.layers[
        0
    ].attention.n_head == 8
    assert sum(parameter.numel() for parameter in model.parameters()) == params
    assert config.training.local_batch_size == 16
    assert config.training.global_batch_size == 128
    assert config.training.steps == steps
    assert config.dataloader.one_pass is True
    assert config.dataloader.dataset.endswith("prune10m_random100k_seed0.txt")
    assert config.validator.enable is False
    assert config.validator.dataloader.dataset.endswith("prune10m_val.txt")
    assert config.validator.dataloader.plan_only is False
    assert config.activation_checkpoint is None
    assert config.compile.enable is False
    assert config.checkpoint.interval == steps
    assert config.checkpoint.last_save_model_only is True
    assert config.checkpoint.enable_first_step_checkpoint is False


def test_round_one_accounting_is_one_executable_chain() -> None:
    target = registry.CONVNEXT_MUP_TARGET_STAGE0
    expected_target_steps = round(
        registry.CONVNEXT_MUP_EFFECTIVE_TOKENS_PER_PARAMETER
        * registry.CONVNEXT_MUP_PARAMETERS[target]
        / (
            registry.CONVNEXT_MUP_EFFECTIVE_TOKENS_PER_SAMPLE
            * registry.CONVNEXT_MUP_GLOBAL_BATCH
        )
    )
    assert registry.CONVNEXT_MUP_TARGET_STEPS == expected_target_steps

    inputs = _meta_inputs()
    for width in registry.CONVNEXT_MUP_STAGE0_WIDTHS:
        config = getattr(registry, f"convnext_whole_mup_w{width}")()
        with torch.device("meta"):
            model = config.model_spec.model.build()
        model.eval()
        with FlopCounterMode(display=False) as counter:
            model(inputs)
        forward_flops = counter.get_total_flops()

        assert sum(p.numel() for p in model.parameters()) == (
            registry.CONVNEXT_MUP_PARAMETERS[width]
        )
        assert forward_flops == registry.CONVNEXT_MUP_FORWARD_FLOPS[width]
        assert config.training.steps == round(
            registry.CONVNEXT_MUP_BUDGET_FLOPS
            / (
                registry.CONVNEXT_MUP_TRAIN_FLOP_MULTIPLIER
                * forward_flops
                * registry.CONVNEXT_MUP_GLOBAL_BATCH
            )
        )


@pytest.mark.parametrize(
    "width, parameters, forward_flops, steps",
    (
        (47, 13_510_067, 28_313_655_552, 1_084),
        (49, 14_623_565, 30_695_890_176, 1_000),
    ),
)
def test_c4_right_flank_accounting(
    width: int, parameters: int, forward_flops: int, steps: int
) -> None:
    config = getattr(
        registry, f"convnext_whole_mup_clean_c4_w{width}_plan10m_lr1p8e4"
    )()
    with torch.device("meta"):
        model = config.model_spec.model.build()
    model.eval()
    with FlopCounterMode(display=False) as counter:
        model(_meta_inputs())

    assert sum(parameter.numel() for parameter in model.parameters()) == parameters
    assert counter.get_total_flops() == forward_flops
    assert config.training.steps == steps
    assert config.training.global_batch_size == 128
    assert config.dataloader.one_pass is True
    assert config.dataloader.dataset.endswith("prune10m_uniform100k_seed0.txt")
    assert config.dataloader.plan_only is True


@pytest.mark.parametrize("width", registry.CONVNEXT_MUP_STAGE0_WIDTHS)
def test_drop_path_axis_changes_only_drop_path(width: int) -> None:
    zero = getattr(registry, f"convnext_whole_mup_w{width}")()
    production = getattr(registry, f"convnext_whole_mup_w{width}_dp20")()

    assert zero.model_spec.model.vision.drop_path_rate == 0.0
    assert production.model_spec.model.vision.drop_path_rate == 0.2
    assert zero.training.steps == production.training.steps
    assert zero.optimizer == production.optimizer
    assert zero.dataloader == production.dataloader


def _first_group(config, name: str):
    return next(
        group
        for group in config.optimizer.param_groups
        if re.search(group.pattern, name)
    )


@pytest.mark.parametrize("width", registry.CONVNEXT_MUP_STAGE0_WIDTHS)
def test_optimizer_groups_match_microsoft_infinite_shape_rule(width: int) -> None:
    base_config = registry._convnext_whole(
        stage0=registry.CONVNEXT_MUP_BASE_DIMS[0],
        drop_path_rate=0.0,
        steps=1,
        flavor="convnext_whole_mup_base_test",
        mup=True,
    )
    config = getattr(registry, f"convnext_whole_mup_w{width}")()
    with torch.device("meta"):
        base_model = base_config.model_spec.model.build()
        model = config.model_spec.model.build()
    base_parameters = dict(base_model.named_parameters())

    for name, parameter in model.named_parameters():
        base_shape = base_parameters[name].shape
        infinite = [
            index
            for index, (base_dim, dim) in enumerate(zip(base_shape, parameter.shape))
            if base_dim != dim
        ]
        assert len(infinite) <= 2, name
        width_mult = (
            parameter.shape[infinite[-1]] / base_shape[infinite[-1]]
            if infinite
            else 1.0
        )
        expected_lr_mult = 1.0 / width_mult if len(infinite) == 2 else 1.0
        group = _first_group(config, name)
        assert group.lr_mult == pytest.approx(expected_lr_mult), name

        no_decay = "final_layer" in name or "scale_layer" in name
        expected_wd = 0.0 if no_decay else 1e-2
        if len(infinite) == 2 and not no_decay:
            expected_wd *= width_mult
        assert group.optimizer_kwargs["weight_decay"] == pytest.approx(expected_wd), name


def test_mup_initialization_is_visible_in_the_constructed_model() -> None:
    config = registry.convnext_whole_mup_w13()
    model = config.model_spec.model.build()
    torch.manual_seed(0)
    model.init_states(buffer_device=torch.device("cpu"))
    parameters = dict(model.named_parameters())

    expected = {
        "vision.encoder.stem.0.weight": 384**-0.5,
        "vision.encoder.stages.3.blocks.0.conv_dw.weight": 49**-0.5,
        "vision.encoder.stages.3.blocks.0.mlp.fc1.weight": 104**-0.5,
        "temporal_policy.temporal_summarizer.desire_encoder.net.0.weight": 200**-0.5,
        "temporal_policy.temporal_summarizer.transformer.layers.0.attention.c_attn.weight": 16**-0.5,
        "temporal_policy.temporal_hydra.final_layer.plan.weight": 512**-0.5,
    }
    for name, expected_std in expected.items():
        actual = parameters[name].detach().float().std(unbiased=False).item()
        assert actual == pytest.approx(expected_std, rel=0.15), name

    readout = model.temporal_policy.temporal_hydra.final_layer["plan"]
    assert isinstance(readout, MuReadout)
    assert readout.width_mult == pytest.approx(16 / 512)
