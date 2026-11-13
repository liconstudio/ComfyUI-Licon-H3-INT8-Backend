# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: Comfy-Org/comfy-kitchen v0.2.35, b2a2972ac68c395bbda8ad9030e8ae1089287815.
# Modified for this plugin: extracted selected upstream definitions and rewired private imports.
# Exact original AST dependency closure; no torch.library registration.
import torch
from ._int8_utils import _build_hadamard, _rotate_activation, _rotate_weight

DTYPE_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    5: torch.float8_e4m3fn,
    6: torch.float8_e5m2,
}

DTYPE_TO_CODE = {v: k for k, v in DTYPE_CODE_TO_DTYPE.items()}

def _int8_stochastic_rng(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    return torch.rand(
        x.shape,
        dtype=x.dtype,
        layout=x.layout,
        device=x.device,
        generator=generator,
    )

def _int8_scale_for_math(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=x.device, dtype=x.dtype)
    scale_min = torch.finfo(x.dtype).tiny
    return torch.where(scale == 0, torch.full_like(scale, scale_min), scale)

def _round_int8(scaled: torch.Tensor, stochastic_rounding: int | None = 0) -> torch.Tensor:
    if stochastic_rounding is not None and stochastic_rounding > 0:
        rng = _int8_stochastic_rng(scaled, stochastic_rounding)
        scaled.add_(rng)
        return scaled.floor_().clamp_(-128.0, 127.0).to(torch.int8)
    return scaled.round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: torch.Tensor | float | str | None = None,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale.

    Args:
        x: Input tensor of any shape.
        scale: Optional scale. "recalculate" computes scale from absmax.
        stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.

    Returns:
        Tuple of (quantized_int8, scale):
            - quantized_int8: INT8 tensor with same shape
            - scale: Scalar float32 tensor
    """
    if scale is None or (isinstance(scale, str) and scale == "recalculate"):
        abs_max = x.abs().max()
        scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    elif not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale = scale.to(device=x.device, dtype=torch.float32)
    q = _round_int8(x / _int8_scale_for_math(scale, x), stochastic_rounding=stochastic_rounding)
    return q, scale

def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations).

    Args:
        x: Input tensor [..., K] where quantization is per-row.
        stochastic_rounding: Seed for stochastic rounding. Disabled when <= 0.

    Returns:
        Tuple of (quantized_int8, scales):
            - quantized_int8: INT8 tensor with same shape
            - scales: Float32 tensor [..., 1] with per-row scales
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = _round_int8(x / _int8_scale_for_math(scale, x), stochastic_rounding=stochastic_rounding)
    return q, scale

def quantize_and_rotate_rowwise(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused online activation rotation + row-wise quantization (Eager fallback).

    Args:
        x: Input unrotated activation tensor [M, K].
        h: Pre-built normalized Hadamard matrix [group_size, group_size].
        group_size: ConvRot group size.

    Returns:
        Tuple of (rotated_quantized_x_int8, row_scales).
    """
    x_rot = _rotate_activation(x, h, group_size)
    return quantize_int8_rowwise(x_rot, stochastic_rounding=stochastic_rounding)

def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation followed by row-wise INT8 quantization."""
    h = _build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    weight_rot = _rotate_weight(weight, h, group_size)
    return quantize_int8_rowwise(weight_rot, stochastic_rounding=stochastic_rounding)

def rotate_int8_convrot_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Portable ConvRot weight rotation used when no accelerated backend is available."""
    h = _build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    return _rotate_weight(weight, h, group_size)

def dequantize_int8_convrot_weight(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights and rotate them back to the original basis."""
    h = _build_hadamard(group_size, device=q.device, dtype=torch.float32)
    return _rotate_weight(dequantize_int8_simple(q, scale), h, group_size)

def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor, scale: torch.Tensor, group_size: int, output_dtype_code: int
) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights into a requested floating dtype."""
    return dequantize_int8_convrot_weight(q, scale, group_size).to(DTYPE_CODE_TO_DTYPE[output_dtype_code])

def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale.

    Args:
        q: Quantized INT8 tensor.
        scale: Scale tensor (scalar or broadcastable).

    Returns:
        Dequantized float tensor.
    """
    return q.float() * scale

def dequantize_int8_simple_dtype(q: torch.Tensor, scale: torch.Tensor, output_dtype_code: int) -> torch.Tensor:
    """Dequantize INT8 tensor with scale into a requested floating dtype."""
    return dequantize_int8_simple(q, scale).to(DTYPE_CODE_TO_DTYPE[output_dtype_code])
