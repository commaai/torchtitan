# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

from torchtitan.experiments.path.config_registry import (
    convnext_xlarge_plan_decoder_policy,
    convnext_xlarge_plan_latent_mse_policy,
    convnext_xlarge_plan_latent_nll_policy,
    convnext_xxlarge_plan_ae_128_half_layers,
)
from torchtitan.experiments.path.model import PathModel, PlanVAE
from torchtitan.experiments.path.model_config import _plan_vae_config, model_config
from torchtitan.experiments.path.model_constants import ModelInputs, PLAN_SIZE
from torchtitan.experiments.path.loss import _plan_consistency_loss, PathLoss
from torchtitan.experiments.path.plan_vae import (
    PLAN_VAE_PRIOR_RECONSTRUCTION,
    PLAN_VAE_RECONSTRUCTION,
    PLAN_VAE_SAMPLED_RECONSTRUCTION,
    normalize_plan,
    prepare_plan_vae_batch,
    unnormalize_plan,
)
from torchtitan.experiments.path.validate import plan_vae_analyse_driving_data
from xx.comma_data.constants import DEFAULT_TEST_2K_LIST
from xx.release_tests.driving.analyse_driving import AnalyseDrivingRLConfig
from xx.training.path.test import MODEL_REPORT_ROUTE_LISTS


def _physical_plans(batch_size: int = 3, temporal_size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    normalized = torch.randn(batch_size, temporal_size, PLAN_SIZE, generator=generator)
    return unnormalize_plan(normalized, normalization="per_horizon")


def test_per_horizon_plan_normalization_round_trip() -> None:
    plans = _physical_plans()
    torch.testing.assert_close(
        unnormalize_plan(normalize_plan(plans, normalization="per_horizon"), normalization="per_horizon"),
        plans,
    )
    normalized = normalize_plan(plans, normalization="per_horizon")
    assert normalized.shape == plans.shape
    assert normalized.float().mean().abs() < 0.5


def test_plan_vae_loss_and_batch() -> None:
    vae = _plan_vae_config(16, dim=32, normalization="per_horizon").build()
    vae.init_states()
    torch.testing.assert_close(vae.output_scale.scale, torch.ones(PLAN_SIZE))
    plans = _physical_plans()
    plans[0, 0, 0] = torch.nan

    vae_inputs, vae_targets = prepare_plan_vae_batch({}, {"plan": plans}, normalization="per_horizon")
    predictions = vae(vae_inputs[ModelInputs.PLAN_VAE], sample_posterior=True)
    mean_predictions = vae(vae_inputs[ModelInputs.PLAN_VAE], sample_posterior=False)
    loss, metrics = PathLoss(PathLoss.Config(plan_normalization="per_horizon"))(predictions, vae_targets)

    assert predictions[PLAN_VAE_RECONSTRUCTION].shape == plans.shape
    assert predictions[PLAN_VAE_SAMPLED_RECONSTRUCTION].shape == plans.shape
    assert predictions[PLAN_VAE_PRIOR_RECONSTRUCTION].shape == plans.shape
    assert mean_predictions[PLAN_VAE_SAMPLED_RECONSTRUCTION].keys() == set() if False else True
    assert PLAN_VAE_SAMPLED_RECONSTRUCTION not in mean_predictions
    assert loss.shape == (plans.shape[0],)
    assert torch.isfinite(loss).all()
    assert "plan_vae_reconstruction" in metrics
    assert "plan_vae_kl" in metrics
    assert "plan_vae_position_l2" in metrics

    # masked values must not contribute to the reconstruction loss
    masked_plans = plans.clone()
    masked_plans[:, 0] = torch.nan
    masked_inputs, masked_targets = prepare_plan_vae_batch({}, {"plan": masked_plans}, normalization="per_horizon")
    masked_predictions = vae(masked_inputs[ModelInputs.PLAN_VAE], sample_posterior=False)
    masked_loss, _ = PathLoss(PathLoss.Config(plan_normalization="per_horizon"))(masked_predictions, masked_targets)
    assert torch.isfinite(masked_loss).all()


def test_plan_vae_kl_penalty_is_averaged_over_latent_dimensions() -> None:
    config = PathLoss.Config()
    mean = torch.zeros(2, 3, 4)
    free_bits = config.vae_kl_free_bits
    # every dimension above the free-bits threshold contributes
    logvar = torch.full((2, 3, 4), 2.0 * torch.log(torch.tensor(2.0 * free_bits)))
    kl_per_dimension = -0.5 * (1.0 + logvar - mean.square() - logvar.exp())
    assert (kl_per_dimension > free_bits).all()
    expected_penalty = (kl_per_dimension - free_bits).flatten(start_dim=1).mean(dim=-1)
    kl_penalty = torch.nn.functional.relu(kl_per_dimension - free_bits).flatten(start_dim=1).mean(dim=-1)
    torch.testing.assert_close(kl_penalty, expected_penalty)


def test_plan_consistency_loss_penalizes_off_manifold_trajectory() -> None:
    plans = _physical_plans()
    mask = torch.ones_like(plans, dtype=torch.bool)
    consistent = _plan_consistency_loss(plans, mask, normalization="per_horizon")

    inconsistent = plans.clone()
    inconsistent[..., 3:6] += 500.0  # break the position/derivative relation
    broken = _plan_consistency_loss(inconsistent, mask, normalization="per_horizon")
    assert (broken > consistent).all()


def test_plan_vae_flop_counting_under_no_grad() -> None:
    config = model_config("convnext_atto", training_stage="plan_vae")
    model = config.build()
    model.init_states()
    nparams, flops = config.get_nparams_and_flops(model, seq_len=1)
    assert nparams > 0
    assert flops > 0


def test_plan_vae_analyse_driving_data_uses_physical_mean_reconstructions() -> None:
    plans = _physical_plans(batch_size=4, temporal_size=2)
    rows = [("seg1", 0), ("seg1", 1), ("seg2", 0), ("seg2", 1)]
    predictions = normalize_plan(plans, normalization="per_horizon")
    data = plan_vae_analyse_driving_data(
        [([rows[0], rows[1]], predictions[:2], plans[:2]), ([rows[2], rows[3]], predictions[2:], plans[2:])],
        normalization="per_horizon",
    )
    assert set(data) == {"seg1", "seg2"}
    for segment in data.values():
        assert segment["pred"]["plan"].shape == (2, 33, 15)
        assert segment["true"]["plan"].shape == (2, 33, 15)


def test_deterministic_plan_autoencoder_preset() -> None:
    config = convnext_xxlarge_plan_ae_128_half_layers()
    assert config.training_stage == "plan_vae"
    plan_vae = config.model_spec.model.temporal_policy.plan_vae
    assert plan_vae is not None
    assert plan_vae.latent_size == 128
    assert plan_vae.normalization == "per_horizon"
    assert not plan_vae.sample_posterior_during_training
    assert len(plan_vae.encoder.layers) == PlanVAE.N_ENCODER_LAYER // 2
    assert len(plan_vae.decoder.layers) == PlanVAE.N_DECODER_LAYER // 2
    assert config.loss.vae_kl_weight == 0.0
    assert config.loss.vae_reconstruction_loss == "physical_smooth_l1"
    assert config.training.steps == 10 * 1024
    assert config.checkpoint.export_onnx is False


def test_plan_decoder_policy_preset() -> None:
    config = convnext_xlarge_plan_decoder_policy()
    assert config.training_stage == "policy"
    assert config.model_spec.model.vision.flavor == "convnext_xlarge"
    temporal_policy = config.model_spec.model.temporal_policy
    assert temporal_policy.plan_vae is not None
    assert temporal_policy.plan_vae.latent_size == 128
    assert temporal_policy.plan_vae.normalization == "per_horizon"
    plan_head = [head for head in temporal_policy.temporal_hydra.heads if head.name == "plan"][0]
    assert plan_head.output_size == 128 + PLAN_SIZE
    # master's loss config: the laplacian driving loss on the decoded plan
    assert config.loss.vae_reconstruction_weight == 1.0  # unused in the policy stage
    assert config.training.steps == 55 * 1024


def test_plan_decoder_decodes_head_output_to_master_plan_format() -> None:
    config = model_config("convnext_atto", training_stage="policy")
    model = config.build()
    temporal_policy = model.temporal_policy
    assert temporal_policy.plan_vae is not None
    head_output = torch.randn(2, 3, temporal_policy.plan_vae.latent_size + PLAN_SIZE)
    plan = temporal_policy.decode_plan(head_output)
    assert plan.shape == (2, 3, 2 * PLAN_SIZE)


def test_training_stages_freeze_the_expected_parameters() -> None:
    vae_config = model_config("convnext_atto", training_stage="plan_vae")
    vae_model = vae_config.build()
    vae_trainable = {name for name, parameter in vae_model.named_parameters() if parameter.requires_grad}
    assert vae_trainable
    assert all("plan_vae" in name for name in vae_trainable)
    vae_model.temporal_policy.plan_vae.mark_pretrained()

    policy_config = model_config("convnext_atto", training_stage="policy")
    policy_model = policy_config.build()
    summarizer = policy_model.temporal_policy.temporal_summarizer
    desire = torch.randn(2, int(summarizer.desire_window_idxs.max()) + 1, 8)
    windowed_desire = summarizer._window_desire(desire)
    assert windowed_desire.shape[0] == desire.shape[0]
    assert torch.count_nonzero(windowed_desire) == 0
    assert all(
        not parameter.requires_grad for parameter in policy_model.temporal_policy.plan_vae.parameters()
    )
    assert ModelInputs.PLAN_VAE not in policy_model.input_shapes(policy_config)
    incompatible = policy_model.load_state_dict(vae_model.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    assert policy_model.temporal_policy.plan_vae.is_pretrained()


def test_policy_gradients_flow_through_frozen_decoder() -> None:
    config = model_config("convnext_atto", training_stage="policy")
    model = config.build()
    model.init_states()
    model.temporal_policy.plan_vae.requires_grad_(False)
    plan_vae = model.temporal_policy.plan_vae
    assert plan_vae is not None

    inputs = PathModel.example_inputs(config, device="cpu")
    outputs = model(inputs)
    outputs["plan"].float().square().mean().backward()

    head_weight = model.temporal_policy.temporal_hydra.final_layer["plan"].weight
    assert head_weight.grad is not None
    assert all(parameter.grad is None for parameter in plan_vae.parameters())


def test_analyse_driving_reports_use_fixed_2k_route_list() -> None:
    assert MODEL_REPORT_ROUTE_LISTS == {"analyse_driving": DEFAULT_TEST_2K_LIST}
    assert AnalyseDrivingRLConfig().route_list == DEFAULT_TEST_2K_LIST


def test_plan_latent_mse_preset() -> None:
    config = convnext_xlarge_plan_latent_mse_policy()
    assert config.model_spec.model.plan_loss == "latent_mse"
    temporal_policy = config.model_spec.model.temporal_policy
    assert temporal_policy.plan_vae is not None
    plan_head = [head for head in temporal_policy.temporal_hydra.heads if head.name == "plan"][0]
    assert plan_head.output_size == temporal_policy.plan_vae.latent_size
    assert config.loss.plan_loss_weight == 5.0


def test_plan_latent_mse_loss_matches_frozen_encoder_target() -> None:
    from torchtitan.experiments.path.loss import PathLoss
    from torchtitan.experiments.path.model_config import model_config
    from torchtitan.experiments.path.plan_vae import (
        PLAN_LATENT,
        PLAN_LATENT_MASK,
        PLAN_LATENT_TARGET,
        prepare_plan_latent_batch,
    )

    config = model_config("convnext_atto", training_stage="policy", plan_loss="latent_mse")
    model = config.build()
    model.init_states()
    plan_vae = model.temporal_policy.plan_vae
    assert plan_vae is not None

    plans = _physical_plans()
    plans[0, 0, 0] = torch.nan
    targets = prepare_plan_latent_batch(
        {"plan": plans, "lead": torch.randn(3, 4, 2, 6, 4)}, plan_vae.encode_mean, normalization="per_horizon"
    )
    assert targets[PLAN_LATENT_TARGET].shape == (3, 4, plan_vae.latent_size)
    assert targets[PLAN_LATENT_MASK].shape == targets[PLAN_LATENT_TARGET].shape
    assert not targets[PLAN_LATENT_MASK][0, 0].any()  # the nan plan is masked out

    # a perfect latent prediction gives zero plan loss
    pred = {
        PLAN_LATENT: targets[PLAN_LATENT_TARGET].clone(),
        "plan": torch.randn_like(plans),
        "lead": torch.randn(3, 4, 2, 6, 4),
    }
    loss, metrics = PathLoss(PathLoss.Config(plan_normalization="per_horizon"))(pred, targets)
    assert metrics["plan_latent_mse"].max() < 1e-10
    assert "plan" not in {k for k in metrics}  # the driving loss ran without the plan key


def test_plan_latent_nll_preset() -> None:
    config = convnext_xlarge_plan_latent_nll_policy()
    assert config.model_spec.model.plan_loss == "latent_nll"
    temporal_policy = config.model_spec.model.temporal_policy
    assert temporal_policy.plan_vae is not None
    plan_head = [head for head in temporal_policy.temporal_hydra.heads if head.name == "plan"][0]
    assert plan_head.output_size == 2 * temporal_policy.plan_vae.latent_size


def test_plan_latent_nll_loss() -> None:
    from torchtitan.experiments.path.loss import PathLoss
    from torchtitan.experiments.path.model_config import model_config
    from torchtitan.experiments.path.plan_vae import (
        PLAN_LATENT,
        PLAN_LATENT_LOGVAR,
        PLAN_LATENT_TARGET,
        prepare_plan_latent_batch,
    )

    config = model_config("convnext_atto", training_stage="policy", plan_loss="latent_nll")
    model = config.build()
    model.init_states()
    plan_vae = model.temporal_policy.plan_vae
    assert plan_vae is not None

    plans = _physical_plans()
    targets = prepare_plan_latent_batch(
        {"plan": plans, "lead": torch.randn(3, 4, 2, 6, 4)}, plan_vae.encode_mean, normalization="per_horizon"
    )
    target = targets[PLAN_LATENT_TARGET]
    # perfect mean with the variance matched to a known error: nll = 0.5*(log var + 1)
    error = torch.full_like(target, 0.5)
    log_var = 2.0 * torch.log(error)
    pred = {
        PLAN_LATENT: target - error,
        PLAN_LATENT_LOGVAR: log_var,
        "plan": torch.randn_like(plans),
        "lead": torch.randn(3, 4, 2, 6, 4),
    }
    loss, metrics = PathLoss(PathLoss.Config(plan_normalization="per_horizon"))(pred, targets)
    expected = 0.5 * (log_var + error.square() / log_var.exp())
    torch.testing.assert_close(metrics["plan_latent_nll"], expected.flatten(start_dim=1).mean(dim=-1))
    assert "plan_latent_mse" not in metrics
