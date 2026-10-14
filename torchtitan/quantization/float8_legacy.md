# Legacy TorchAO FP8

`Float8Linear` in `float8_legacy.py` preserves TorchAO-backed tensorwise FP8
compute independently of upstream MXFP8 and NVFP4. It requires `torchao.float8`
and SM89+ hardware, or eager emulation for CPU tests.

```python
from torchtitan.config.transform import Float8LinearConverter

converter = Float8LinearConverter.Config(
    recipe_name="tensorwise",
    filter_fqns=["output"],
    enable_fsdp_float8_all_gather=True,
    precompute_float8_dynamic_scale_for_fsdp=True,
).build()
model_config = converter.convert(model_config)
```

The converter preserves initialization and sharding configs, handles root
replacement, and retains synchronous column/row-parallel roles. Dimensions must
be divisible by 16. Stacked projections (`num_linears != 1`) are not supported.
In-place WGRAD accumulation is disabled with a warning: the old TorchAO compute
path does not implement that optimization. `Linear` supplies the remat region;
TorchAO supplies the FP8 matmul and backward computation.

`OptimizersContainer` automatically registers one post-step hook when any module
opts into scale precomputation, and invokes it after all contained optimizers.
There is no extra trainer-side hook to register.

## Preserved communication limitation

As in the pre-rebase fork, constructing this module does **not** install
TorchAO's `WeightWithDynamicFloat8CastTensor`. Setting the all-gather option alone
does not enable FP8 communication; without wrapped DTensor weights, TorchAO's
scale-precomputation function does no work. The flag and hook are retained for
compatibility, not as proof that FP8 all-gather is active. Fixing weight wrapping
is a separate change requiring distributed numerical/communication tests.
