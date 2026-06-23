"""FastViT-T12 backbone for path training.

This is a narrow copy of the timm FastViT implementation, reduced to the
``fastvit_t12`` RepMixer code path used by the path model.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from timm.layers import (
    ClassifierHead,
    DropPath,
    SqueezeExcite,
    calculate_drop_path_rates,
    create_conv2d,
    get_act_layer,
    trunc_normal_,
)
from timm.models._manipulate import checkpoint_seq
from xx.training.path.allnorm import AllNorm2d, AllNormAct2d


__all__ = [
    "FASTVIT_FLAVORS",
    "FastVit",
    "skip_pretrained_tensor",
]


FASTVIT_FLAVORS = {
    "fastvit_t12": {
        "layers": (2, 2, 6, 2),
        "embed_dims": (64, 128, 256, 512),
        "mlp_ratios": (3, 3, 3, 3),
        "pretrained": "fastvit_t12.apple_in1k",
    },
}


def num_groups(group_size: int, channels: int) -> int:
    if not group_size:
        return 1
    assert channels % group_size == 0
    return channels // group_size


class ConvAllNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: str | int = "",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_act: bool = True,
        act_layer: type[nn.Module] = nn.GELU,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **dd,
        )
        self.bn = AllNormAct2d(
            out_channels,
            apply_act=apply_act,
            act_layer=act_layer,
            **dd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        group_size: int = 0,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        act_layer: type[nn.Module] = nn.GELU,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.inference_mode = inference_mode
        self.groups = num_groups(group_size, in_chs)
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_conv_branches = num_conv_branches
        self.se = SqueezeExcite(out_chs, rd_divisor=1, **dd) if use_se else nn.Identity()

        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                bias=True,
                **dd,
            )
        else:
            self.reparam_conv = None
            self.identity = AllNorm2d(in_chs, **dd) if out_chs == in_chs and stride == 1 else None
            self.conv_kxk = (
                nn.ModuleList(
                    [
                        ConvAllNormAct(
                            in_chs,
                            out_chs,
                            kernel_size=kernel_size,
                            stride=stride,
                            groups=self.groups,
                            apply_act=False,
                            act_layer=act_layer,
                            **dd,
                        )
                        for _ in range(num_conv_branches)
                    ]
                )
                if num_conv_branches > 0
                else None
            )
            self.conv_scale = (
                ConvAllNormAct(
                    in_chs,
                    out_chs,
                    kernel_size=1,
                    stride=stride,
                    groups=self.groups,
                    apply_act=False,
                    act_layer=act_layer,
                    **dd,
                )
                if kernel_size > 1 and use_scale_branch
                else None
            )

        self.act = act_layer() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return self.act(self.se(self.reparam_conv(x)))

        identity_out = self.identity(x) if self.identity is not None else 0
        scale_out = self.conv_scale(x) if self.conv_scale is not None else 0
        out = scale_out + identity_out
        if self.conv_kxk is not None:
            for conv in self.conv_kxk:
                out = out + conv(x)
        return self.act(self.se(out))

    def reparameterize(self) -> None:
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = create_conv2d(
            self.in_chs,
            self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
            device=kernel.device,
            dtype=kernel.dtype,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for name, param in self.named_parameters():
            if "reparam_conv" not in name:
                param.detach_()
        self.__delattr__("conv_kxk")
        self.__delattr__("conv_scale")
        if hasattr(self, "identity"):
            self.__delattr__("identity")
        self.inference_mode = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_scale = 0
        bias_scale = 0
        if self.conv_scale is not None:
            kernel_scale, bias_scale = self._fuse_norm_tensor(self.conv_scale)
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_norm_tensor(self.identity)

        kernel_conv = 0
        bias_conv = 0
        if self.conv_kxk is not None:
            for idx in range(self.num_conv_branches):
                kernel, bias = self._fuse_norm_tensor(self.conv_kxk[idx])
                kernel_conv = kernel_conv + kernel
                bias_conv = bias_conv + bias

        return kernel_conv + kernel_scale + kernel_identity, bias_conv + bias_scale + bias_identity

    def _fuse_norm_tensor(self, branch: ConvAllNormAct | AllNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, ConvAllNormAct):
            kernel = branch.conv.weight
            norm = branch.bn
        else:
            input_dim = self.in_chs // self.groups
            if not hasattr(self, "id_tensor"):
                kernel_value = torch.zeros(
                    (self.in_chs, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_chs):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            norm = branch

        running_mean = norm.running_mean
        running_var = norm.running_var
        gamma = norm.weight
        beta = norm.bias
        eps = norm.eps
        std = (running_var + eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: int,
        stride: int,
        group_size: int,
        small_kernel: int | None = None,
        use_se: bool = False,
        act_layer: type[nn.Module] | None = None,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.stride = stride
        self.groups = num_groups(group_size, in_chs)
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        if inference_mode:
            self.reparam_conv = create_conv2d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                groups=self.groups,
                bias=True,
                **dd,
            )
            self.large_conv = None
            self.small_conv = None
        else:
            self.reparam_conv = None
            self.large_conv = ConvAllNormAct(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                groups=self.groups,
                apply_act=False,
                act_layer=act_layer or nn.GELU,
                **dd,
            )
            self.small_conv = (
                ConvAllNormAct(
                    in_chs,
                    out_chs,
                    kernel_size=small_kernel,
                    stride=stride,
                    groups=self.groups,
                    apply_act=False,
                    act_layer=act_layer or nn.GELU,
                    **dd,
                )
                if small_kernel is not None
                else None
            )
        self.se = SqueezeExcite(out_chs, rd_ratio=0.25, **dd) if use_se else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            out = self.reparam_conv(x)
        else:
            assert self.large_conv is not None
            out = self.large_conv(x)
            if self.small_conv is not None:
                out = out + self.small_conv(x)
        return self.act(self.se(out))

    def get_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.large_conv is not None
        eq_k, eq_b = self._fuse_norm(self.large_conv.conv, self.large_conv.bn)
        if self.small_conv is not None:
            assert self.small_kernel is not None
            small_k, small_b = self._fuse_norm(self.small_conv.conv, self.small_conv.bn)
            eq_b = eq_b + small_b
            eq_k = eq_k + nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def reparameterize(self) -> None:
        eq_k, eq_b = self.get_kernel_bias()
        self.reparam_conv = create_conv2d(
            self.in_chs,
            self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=True,
            device=eq_k.device,
            dtype=eq_k.dtype,
        )
        self.reparam_conv.weight.data = eq_k
        self.reparam_conv.bias.data = eq_b
        self.__delattr__("large_conv")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_norm(conv: nn.Conv2d, norm: AllNormAct2d) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        running_mean = norm.running_mean
        running_var = norm.running_var
        gamma = norm.weight
        beta = norm.bias
        eps = norm.eps
        std = (running_var + eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * scale, beta - running_mean * gamma / std


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_chs: int,
        embed_dim: int,
        act_layer: type[nn.Module],
        lkc_use_act: bool = False,
        use_se: bool = False,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_chs=in_chs,
                out_chs=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                group_size=1,
                small_kernel=3,
                use_se=use_se,
                act_layer=act_layer if lkc_use_act else None,
                inference_mode=inference_mode,
                **dd,
            ),
            MobileOneBlock(
                in_chs=embed_dim,
                out_chs=embed_dim,
                kernel_size=1,
                stride=1,
                use_se=False,
                act_layer=act_layer,
                inference_mode=inference_mode,
                **dd,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LayerScale2d(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.init_values = init_values
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim, 1, 1, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RepMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        layer_scale_init_value: float | None = 1e-5,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=dim,
                bias=True,
                **dd,
            )
        else:
            self.reparam_conv = None
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
                **dd,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
                **dd,
            )
            self.layer_scale = (
                LayerScale2d(dim, layer_scale_init_value, **dd)
                if layer_scale_init_value is not None
                else nn.Identity()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam_conv is not None:
            return self.reparam_conv(x)
        return x + self.layer_scale(self.mixer(x) - self.norm(x))

    def reparameterize(self) -> None:
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if isinstance(self.layer_scale, LayerScale2d):
            w = self.mixer.id_tensor + self.layer_scale.gamma.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale.gamma) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = self.mixer.id_tensor + self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = create_conv2d(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.dim,
            bias=True,
            device=w.device,
            dtype=w.dtype,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for name, param in self.named_parameters():
            if "reparam_conv" not in name:
                param.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        self.__delattr__("layer_scale")


class ConvMlp(nn.Module):
    def __init__(
        self,
        in_chs: int,
        hidden_channels: int | None = None,
        out_chs: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        out_chs = out_chs or in_chs
        hidden_channels = hidden_channels or in_chs
        self.conv = ConvAllNormAct(
            in_chs,
            out_chs,
            kernel_size=7,
            groups=in_chs,
            apply_act=False,
            act_layer=act_layer,
            **dd,
        )
        self.fc1 = nn.Conv2d(in_chs, hidden_channels, kernel_size=1, **dd)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_chs, kernel_size=1, **dd)
        self.drop = nn.Dropout(drop)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class RepMixerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
            **dd,
        )
        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            **dd,
        )
        self.layer_scale = (
            LayerScale2d(dim, layer_scale_init_value, **dd)
            if layer_scale_init_value is not None
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        return x + self.drop_path(self.layer_scale(self.mlp(x)))


class FastVitStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        downsample: bool = True,
        se_downsample: bool = False,
        down_patch_size: int = 7,
        down_stride: int = 2,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        proj_drop_rate: float = 0.0,
        drop_path_rate: list[float] | tuple[float, ...] = (0.0,),
        layer_scale_init_value: float | None = 1e-5,
        lkc_use_act: bool = False,
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.grad_checkpointing = False

        if downsample:
            self.downsample = PatchEmbed(
                patch_size=down_patch_size,
                stride=down_stride,
                in_chs=dim,
                embed_dim=dim_out,
                use_se=se_downsample,
                act_layer=act_layer,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
                **dd,
            )
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[
                RepMixerBlock(
                    dim_out,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                    **dd,
                )
                for block_idx in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            return checkpoint_seq(self.blocks, x)
        return self.blocks(x)


class FastVit(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        layers: tuple[int, ...] = (2, 2, 6, 2),
        embed_dims: tuple[int, ...] = (64, 128, 256, 512),
        mlp_ratios: tuple[float, ...] = (3, 3, 3, 3),
        downsamples: tuple[bool, ...] = (False, True, True, True),
        se_downsamples: tuple[bool, ...] = (False, False, False, False),
        repmixer_kernel_size: int = 3,
        num_classes: int = 1000,
        down_patch_size: int = 7,
        down_stride: int = 2,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-5,
        lkc_use_act: bool = False,
        stem_use_scale_branch: bool = True,
        cls_ratio: float = 2.0,
        global_pool: str = "avg",
        act_layer: type[nn.Module] = get_act_layer("gelu_tanh"),
        inference_mode: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        dd = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_info = []

        self.stem = nn.Sequential(
            MobileOneBlock(
                in_chs=in_chans,
                out_chs=embed_dims[0],
                kernel_size=3,
                stride=2,
                act_layer=act_layer,
                inference_mode=inference_mode,
                use_scale_branch=stem_use_scale_branch,
                **dd,
            ),
            MobileOneBlock(
                in_chs=embed_dims[0],
                out_chs=embed_dims[0],
                kernel_size=3,
                stride=2,
                group_size=1,
                act_layer=act_layer,
                inference_mode=inference_mode,
                use_scale_branch=stem_use_scale_branch,
                **dd,
            ),
            MobileOneBlock(
                in_chs=embed_dims[0],
                out_chs=embed_dims[0],
                kernel_size=1,
                stride=1,
                act_layer=act_layer,
                inference_mode=inference_mode,
                use_scale_branch=stem_use_scale_branch,
                **dd,
            ),
        )

        prev_dim = embed_dims[0]
        scale = 1
        dpr = calculate_drop_path_rates(drop_path_rate, layers, stagewise=True)
        stages = []
        for i, depth in enumerate(layers):
            downsample = downsamples[i] or prev_dim != embed_dims[i]
            stages.append(
                FastVitStage(
                    dim=prev_dim,
                    dim_out=embed_dims[i],
                    depth=depth,
                    downsample=downsample,
                    se_downsample=se_downsamples[i],
                    down_patch_size=down_patch_size,
                    down_stride=down_stride,
                    kernel_size=repmixer_kernel_size,
                    mlp_ratio=mlp_ratios[i],
                    act_layer=act_layer,
                    proj_drop_rate=proj_drop_rate,
                    drop_path_rate=dpr[i],
                    layer_scale_init_value=layer_scale_init_value,
                    lkc_use_act=lkc_use_act,
                    inference_mode=inference_mode,
                    **dd,
                )
            )
            prev_dim = embed_dims[i]
            if downsample:
                scale *= 2
            self.feature_info.append(dict(num_chs=prev_dim, reduction=4 * scale, module=f"stages.{i}"))
        self.stages = nn.Sequential(*stages)
        self.num_stages = len(self.stages)
        self.num_features = self.head_hidden_size = int(embed_dims[-1] * cls_ratio)
        self.final_conv = MobileOneBlock(
            in_chs=embed_dims[-1],
            out_chs=self.num_features,
            kernel_size=3,
            stride=1,
            group_size=1,
            inference_mode=inference_mode,
            use_se=True,
            act_layer=act_layer,
            num_conv_branches=1,
            **dd,
        )
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            **dd,
        )
        self.init_path_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerScale2d):
            module.reset_parameters()
        elif isinstance(module, ConvMlp):
            module.reset_parameters()

    def init_path_weights(self) -> None:
        self.apply(self._init_weights)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        for stage in self.stages:
            stage.grad_checkpointing = enable

    def reparameterize(self) -> None:
        for module in list(self.modules()):
            if module is self:
                continue
            reparameterize = getattr(module, "reparameterize", None)
            if callable(reparameterize):
                reparameterize()

    def apply_activation_checkpointing(self, wrap: Callable[[nn.Module, str], nn.Module], mode: str, base_fqn: str) -> None:
        if mode == "full":
            for stage_id, stage in enumerate(self.stages):
                for block_id, block in enumerate(stage.blocks):
                    stage.blocks[block_id] = wrap(
                        block,
                        f"{base_fqn}.stages.{stage_id}.blocks.{block_id}",
                    )
        else:
            for stage_id, stage in enumerate(self.stages):
                self.stages[stage_id] = wrap(stage, f"{base_fqn}.stages.{stage_id}")

    def apply_fsdp(
        self,
        shard: Callable[[nn.Module, bool], None],
        reshard_after_forward: bool,
        head_reshard_after_forward: bool,
    ) -> None:
        shard(self.stem, reshard_after_forward)
        for stage in self.stages:
            for block in getattr(stage, "blocks", ()):
                shard(block, reshard_after_forward)
            shard(stage, reshard_after_forward)
        shard(self.final_conv, reshard_after_forward)
        shard(self.head, head_reshard_after_forward)

    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.final_conv(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def skip_pretrained_tensor(model: FastVit, name: str) -> bool:
    if name.startswith("head."):
        return True
    module_name, _, state_name = name.rpartition(".")
    if state_name not in {"weight", "bias", "running_mean", "running_var", "num_batches_tracked"}:
        return False
    try:
        module = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return False
    return isinstance(module, (AllNorm2d, AllNormAct2d))
