from torchtitan.components.optimizer import register_float8_precompute_scale_hook
from torchtitan.experiments.worldmodel.dataloader import WorldModelDataLoader
from torchtitan.experiments.worldmodel.loss import WorldModelLoss
from torchtitan.experiments.worldmodel.model import WorldModel
from torchtitan.experiments.worldmodel.parallelize import parallelize_worldmodel
from torchtitan.experiments.worldmodel.transformer import TransformerConfig
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec


def _worldmodel_base() -> WorldModel.Config:
    return WorldModel.Config(
        input_size=(15, 16, 32),
        patch_size=(1, 2, 2),
        in_channels=32,
        out_channels=32,
        transformer=TransformerConfig(
            n_layer=56,
            n_head=36,
            n_embd=2304,
            act="GELU",
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            biased_linears=True,
            qk_norm=True,
            prenorm=False,
            mlp_multiple_of=256,
            attention_mask="NONE",
            attention_impl="FLEX",
            norm="RMSNorm",
        ),
        plan_head=TransformerConfig(
            n_layer=4,
            n_head=36,
            n_embd=2304,
            act="GELU",
            biased_linears=True,
            prenorm=True,
            mlp_mult=2,
            mlp_multiple_of=1,
            attention_impl="FLEX",
        ),
    )


worldmodel_configs = {
    "base": _worldmodel_base,
}


def model_registry(flavor: str, converters: list[ModelConfigConverter.Config] | None = None) -> ModelSpec:
    config = worldmodel_configs[flavor]()
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="worldmodel",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_worldmodel,
        pipelining_fn=None,
        post_optimizer_build_fn=register_float8_precompute_scale_hook,
        state_dict_adapter=None,
    )


__all__ = [
    "TransformerConfig",
    "WorldModel",
    "WorldModelDataLoader",
    "WorldModelLoss",
    "model_registry",
    "parallelize_worldmodel",
    "worldmodel_configs",
]
