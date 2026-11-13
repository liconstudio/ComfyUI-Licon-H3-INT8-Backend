# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: Comfy-Org/comfy-kitchen v0.2.35, b2a2972ac68c395bbda8ad9030e8ae1089287815.
# Modified for this plugin: extracted selected upstream definitions and rewired private imports.
# Exact original AST dependency closure; no registry registration or new native fallback.
import os
import torch
from ._activations import apply_input_act as _apply_input_act, apply_residual as _apply_residual, input_act_code as _input_act_code, input_act_width as _input_act_width
from ._int8_utils import _build_hadamard, _rotate_activation, _rotate_weight
from ._eager import DTYPE_CODE_TO_DTYPE, DTYPE_TO_CODE, dequantize_int8_simple as eager_dequantize_int8_simple, quantize_int8_tensorwise as eager_quantize_int8_tensorwise
_C = None  # Set only to the verified private v0.2.35 native extension by the loader.

_cublas_workspaces: dict[int, torch.Tensor] = {}

_empty_cuda_tensors: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}

_turing_device_cache: dict[int, bool] = {}

_nvidia_16_series_device_cache: dict[int, bool] = {}

_cutlass_int8_device_cache: dict[int, bool] = {}

_NVIDIA_16_SERIES = (
    "1660",
    "1650",
    "1630",
    "T500",
    "T550",
    "T600",
    "MX550",
    "MX450",
    "CMP 30HX",
    "T2000",
    "T1000",
    "T1200",
)

def _cuda_device_is_turing(device_index: int) -> bool:
    cached = _turing_device_cache.get(device_index)
    if cached is not None:
        return cached
    is_turing = torch.cuda.get_device_capability(device_index) == (7, 5)
    _turing_device_cache[device_index] = is_turing
    return is_turing

def _cuda_device_is_nvidia_16_series(device_index: int) -> bool:
    cached = _nvidia_16_series_device_cache.get(device_index)
    if cached is not None:
        return cached
    is_16_series = False
    if _cuda_device_is_turing(device_index):
        device_name = torch.cuda.get_device_name(device_index)
        is_16_series = any(model in device_name for model in _NVIDIA_16_SERIES)
    _nvidia_16_series_device_cache[device_index] = is_16_series
    return is_16_series

def _cuda_device_should_use_turing_kernels(device_index: int) -> bool:
    return _cuda_device_is_turing(device_index) and not _cuda_device_is_nvidia_16_series(
        device_index
    )

def _prefer_turing_fused_int8(m: int, n: int, k: int) -> bool:
    """Use fused SM75 kernels for latency shapes and feed-forward expansions."""
    return m <= 128 or n >= 2 * k

def _cuda_device_supports_cutlass_int8_dequant(tensor: torch.Tensor) -> bool:
    if not tensor.is_cuda:
        return False
    device_index = tensor.get_device()
    cached = _cutlass_int8_device_cache.get(device_index)
    if cached is not None:
        return cached
    major, _minor = torch.cuda.get_device_capability(device_index)
    supported = major >= 8
    _cutlass_int8_device_cache[device_index] = supported
    return supported

def _cublas_int8_n_alignment(tensor: torch.Tensor) -> int:
    # Turing cuBLASLt INT8 rejects some skinny-N shapes, e.g. N=17.
    return 32 if tensor.is_cuda and _cuda_device_is_turing(tensor.get_device()) else 8

def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment

def _pad_2d_cols(x: torch.Tensor, padded_cols: int) -> torch.Tensor:
    if x.shape[1] == padded_cols:
        return x
    padding = torch.zeros((x.shape[0], padded_cols - x.shape[1]), dtype=x.dtype, device=x.device)
    return torch.cat((x, padding), dim=1).contiguous()

def _pad_2d_rows(x: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if x.shape[0] == padded_rows:
        return x
    padding = torch.zeros((padded_rows - x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    return torch.cat((x, padding), dim=0).contiguous()

_max_shared_memory_cache: dict[int, int] = {}

def _max_dynamic_shared_memory_per_block(x: torch.Tensor) -> int:
    device_index = x.get_device()
    cached = _max_shared_memory_cache.get(device_index)
    if cached is not None:
        return cached
    props = torch.cuda.get_device_properties(device_index)
    if _cuda_device_is_turing(device_index):
        limit = props.shared_memory_per_block
    else:
        limit = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
    _max_shared_memory_cache[device_index] = limit
    return limit

def _convrot_int8_fused_shared_memory_bytes(m: int, k: int) -> int:
    if m == 1:
        block_threads = 512
    elif k == 256:
        block_threads = 64
    elif k == 2560:
        block_threads = 640
    elif k == 6144:
        block_threads = 768
    else:
        block_threads = 1024
    groups_in_flight = block_threads // 64
    return (k + groups_in_flight * 2 * 256) * 4

def _convrot_fused_shared_memory_fits(x: torch.Tensor, k: int, group_size: int) -> bool:
    if not x.is_cuda or group_size != 256:
        return True
    requested_shared = _convrot_int8_fused_shared_memory_bytes(x.shape[0], k)
    return requested_shared < _max_dynamic_shared_memory_per_block(x)

def _should_use_convrot_fused_kernel(x: torch.Tensor, k: int, group_size: int) -> bool:
    return (
        group_size == 256
        and k % 256 == 0
        and k <= _CONVROT_FUSED_MAX_K
        and (k <= 5120 or k >= 8192)
        and _convrot_fused_shared_memory_fits(x, k, group_size)
    )

def _should_use_convrot_dequant_kernel(x: torch.Tensor, k: int, group_size: int) -> bool:
    # Dequant rotates each 256-wide group independently, so it does not need the
    # whole row staged in shared memory like ConvRot quantization does.
    return group_size == 256 and k % 256 == 0 and k <= _CONVROT_FUSED_MAX_K

def get_cublas_workspace_size_bytes() -> int:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        return 33_554_432
    return 4_194_304

def get_cublas_workspace() -> torch.Tensor:
    """Returns workspace for cublas."""
    device_index = torch.cuda.current_device()
    workspace = _cublas_workspaces.get(device_index)
    if workspace is None:
        workspace = torch.empty(
            get_cublas_workspace_size_bytes(),
            dtype=torch.uint8,
            device=device_index,
        )
        _cublas_workspaces[device_index] = workspace
    return workspace

def _empty_cuda_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device.type, device.index, dtype)
    empty = _empty_cuda_tensors.get(key)
    if empty is None:
        empty = torch.empty(0, dtype=dtype, device=device)
        _empty_cuda_tensors[key] = empty
    return empty

def _aligned16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0

def _gemm_vector_arg(v: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A per-channel vector as the CUTLASS epilogues read it: on device, in dtype,
    contiguous and 16-byte aligned (the row-broadcast visitors load 16-byte vectors)."""
    if v.device == device and v.dtype == dtype and v.is_contiguous() and _aligned16(v):
        return v
    v = v.to(device=device, dtype=dtype).contiguous()
    return v if _aligned16(v) else v.clone()

def _int8_weight_scale_arg(weight_scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    if weight_scale.device == device and weight_scale.dtype == torch.float32 and weight_scale.is_contiguous():
        return weight_scale
    return weight_scale.to(device=device, dtype=torch.float32).reshape(-1).contiguous()

def _prefer_cublas_int8_fallback(
    m: int,
    n: int,
    k: int,
    device_index: int,
) -> bool:
    return False

def _wrap_for_dlpack(tensor: torch.Tensor):
    """Export tensor via DLPack without cross-stream sync.

    Works around PyTorch issue where __dlpack__(stream=None) syncs with
    the default stream, breaking CUDA graph capture on non-default streams.
    See: https://github.com/pytorch/pytorch/pull/163242

    Detaches first so nn.Parameter (requires_grad=True) inputs like bias
    export without PyTorch's gradient-tracking refusal.

    Returns a PyCapsule containing the DLTensor that nanobind can import.
    """
    # stream=-1 tells PyTorch to skip synchronization (DLPack spec)
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.__dlpack__(stream=-1)

def quantize_int8_rowwise(
    x: torch.Tensor,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales (for activations)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    q_2d = torch.empty_like(x_2d, dtype=torch.int8)
    scales_2d = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x.device)
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    _C.quantize_int8_rowwise(
        _wrap_for_dlpack(x_2d),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        stochastic_rounding is not None and stochastic_rounding > 0,
        int(stochastic_rounding or 0),
        stream_ptr,
    )

    return q_2d.reshape(orig_shape), scales_2d.reshape(*orig_shape[:-1], 1)

def _int8_linear_turing_quantized(
    x_qdata: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Run row-wise INT8 GEMM with Turing tensor cores and a fused scale epilogue."""
    m = x_qdata.shape[0]
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=out_dtype, device=x_qdata.device)
    weight_scale_arg = weight_scale.reshape(-1)
    bias_arg = (
        bias.to(device=x_qdata.device, dtype=out_dtype).contiguous()
        if bias is not None
        else _empty_cuda_tensor(x_qdata.device, out_dtype)
    )
    used_int8 = _C.cutlass_turing_int8_dequant(
        _wrap_for_dlpack(x_qdata),
        _wrap_for_dlpack(weight),
        _wrap_for_dlpack(x_scale.reshape(-1)),
        _wrap_for_dlpack(weight_scale_arg),
        _wrap_for_dlpack(bias_arg),
        _wrap_for_dlpack(output),
        DTYPE_TO_CODE[out_dtype],
        torch.cuda.current_stream(x_qdata.device).cuda_stream,
    )
    if not used_int8:
        return None
    return output

def quantize_int8_rowwise_convrot(
    x_2d: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused online ConvRot rotation + per-row INT8 quantization (single kernel).

    Avoids materializing the rotated bf16 activation in global memory. Expects a
    contiguous 2D [M, K] input with K divisible by group_size (256 only).
    """
    q_2d = torch.empty_like(x_2d, dtype=torch.int8)
    scales_2d = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x_2d.device)
    stream_ptr = torch.cuda.current_stream(x_2d.device).cuda_stream

    _C.quantize_int8_rowwise_convrot(
        _wrap_for_dlpack(x_2d),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        group_size,
        stochastic_rounding is not None and stochastic_rounding > 0,
        int(stochastic_rounding or 0),
        stream_ptr,
    )

    return q_2d, scales_2d

def rotate_int8_convrot_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """ConvRot weight rotation using the CUDA FHT kernel."""
    output = torch.empty_like(weight)
    stream_ptr = torch.cuda.current_stream(weight.device).cuda_stream
    _C.rotate_int8_convrot_weight(
        _wrap_for_dlpack(weight),
        _wrap_for_dlpack(output),
        group_size,
        stream_ptr,
    )
    return output

def quantize_int8_convrot_staged(
    weight_2d: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ConvRot rotation with partial absmax followed by INT8 quantization."""
    n_groups = weight_2d.shape[-1] // group_size
    rotated = torch.empty_like(weight_2d)
    partial_absmax = torch.empty((weight_2d.shape[0], n_groups), dtype=torch.float32, device=weight_2d.device)
    q_2d = torch.empty_like(weight_2d, dtype=torch.int8)
    scales_2d = torch.empty((weight_2d.shape[0], 1), dtype=torch.float32, device=weight_2d.device)
    stream_ptr = torch.cuda.current_stream(weight_2d.device).cuda_stream
    _C.quantize_int8_convrot_staged(
        _wrap_for_dlpack(weight_2d),
        _wrap_for_dlpack(rotated),
        _wrap_for_dlpack(partial_absmax),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        group_size,
        stochastic_rounding is not None and stochastic_rounding > 0,
        int(stochastic_rounding or 0),
        stream_ptr,
    )
    return q_2d, scales_2d

def quantize_int8_rowwise_convrot64(
    weight_2d: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ConvRot row-wise INT8 quantization using 64-lane groups.

    input_act applies an activation on the way in, so an MLP's
    `linear(act(proj(x)))` never materializes act's output in bf16. For the
    gated "swiglu" pair the input rows are [gate | up] and the quantized rows
    are half as wide. "rms_norm" normalizes each row with input_act_weight
    (K elements) and input_act_eps before rotating.
    """
    q_2d = torch.empty(
        (weight_2d.shape[0], weight_2d.shape[1] // _input_act_width(input_act)),
        dtype=torch.int8, device=weight_2d.device)
    scales_2d = torch.empty((weight_2d.shape[0], 1), dtype=torch.float32, device=weight_2d.device)
    stream_ptr = torch.cuda.current_stream(weight_2d.device).cuda_stream
    act_weight_arg = _act_weight_arg(
        input_act, input_act_weight, weight_2d.device, weight_2d.dtype)
    _C.quantize_int8_rowwise_convrot64(
        _wrap_for_dlpack(weight_2d),
        _wrap_for_dlpack(q_2d),
        _wrap_for_dlpack(scales_2d),
        group_size,
        stochastic_rounding is not None and stochastic_rounding > 0,
        _input_act_code(input_act),
        int(stochastic_rounding or 0),
        _wrap_for_dlpack(act_weight_arg),
        float(input_act_eps),
        stream_ptr,
    )
    return q_2d, scales_2d

_CONVROT_FUSED_MAX_K = 16384

_DISABLE_CUTLASS_INT8 = os.environ.get("COMFY_KITCHEN_DISABLE_CUTLASS", "0") == "1"

def quantize_int8_tensorwise(
    x: torch.Tensor,
    scale: torch.Tensor | float | str | None = None,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with a single tensor-wise scale."""
    return eager_quantize_int8_tensorwise(x, scale=scale, stochastic_rounding=stochastic_rounding)

def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale."""
    return dequantize_int8_simple_dtype(q, scale, DTYPE_TO_CODE[torch.float32])

def dequantize_int8_simple_dtype(q: torch.Tensor, scale: torch.Tensor, output_dtype_code: int) -> torch.Tensor:
    """Dequantize INT8 tensor with scale into the requested floating output dtype."""
    if not q.is_contiguous():
        q = q.contiguous()

    scale_mode = -1
    inner_dim = q.shape[-1] if q.dim() > 0 else 1
    if scale.numel() == 1:
        scale_mode = 0
    elif tuple(scale.shape) == tuple(q.shape):
        scale_mode = 1
    elif (
        q.dim() > 0
        and scale.dim() == q.dim()
        and tuple(scale.shape[:-1]) == tuple(q.shape[:-1])
        and scale.shape[-1] == 1
    ):
        scale_mode = 2

    if scale_mode < 0:
        return eager_dequantize_int8_simple(q, scale).to(DTYPE_CODE_TO_DTYPE[output_dtype_code])

    scale = scale.to(device=q.device, dtype=torch.float32).contiguous()
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    output = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    stream_ptr = torch.cuda.current_stream(q.device).cuda_stream
    _C.dequantize_int8_simple(
        _wrap_for_dlpack(q),
        _wrap_for_dlpack(scale.reshape(-1)),
        _wrap_for_dlpack(output),
        inner_dim,
        scale_mode,
        stream_ptr,
    )
    return output

def quantize_and_rotate_rowwise(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online activation rotation followed by CUDA row-wise quantization."""
    x_rot = _rotate_activation(x, h, group_size)
    return quantize_int8_rowwise(x_rot, stochastic_rounding=stochastic_rounding)

def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int,
    stochastic_rounding: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation followed by row-wise INT8 quantization.

    Uses the fused ConvRot CUDA kernel when it matches the linear path's launch
    heuristics, otherwise falls back to explicit rotation plus row-wise quantize.
    """
    if weight.dim() != 2:
        raise ValueError("ConvRot INT8 weight quantization expects a 2D weight tensor")

    weight_2d = weight.contiguous()
    k = weight_2d.shape[-1]
    if (
        group_size == 256
        and k % 256 == 0
        and 256 <= k <= _CONVROT_FUSED_MAX_K
        and _convrot_fused_shared_memory_fits(weight_2d, k, group_size)
    ):
        return quantize_int8_rowwise_convrot64(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    if (
        group_size == 256
        and k % 256 == 0
        and 1024 <= k < 8192
    ):
        return quantize_int8_convrot_staged(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    if _should_use_convrot_fused_kernel(weight_2d, k, group_size):
        return quantize_int8_rowwise_convrot(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    if group_size == 256 and k % 256 == 0:
        return quantize_int8_convrot_staged(weight_2d, group_size, stochastic_rounding=stochastic_rounding)

    h = _build_hadamard(group_size, device=weight_2d.device, dtype=weight_2d.dtype)
    return quantize_int8_rowwise(_rotate_weight(weight_2d, h, group_size), stochastic_rounding=stochastic_rounding)

def dequantize_int8_convrot_weight(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize ConvRot INT8 weights and rotate them back to the original basis."""
    return dequantize_int8_convrot_weight_dtype(q, scale, group_size, DTYPE_TO_CODE[torch.float32])

def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor, scale: torch.Tensor, group_size: int, output_dtype_code: int
) -> torch.Tensor:
    """Dequantize ConvRot INT8 weights and rotate them back into the requested dtype."""
    if q.dim() != 2:
        raise ValueError("ConvRot INT8 weight dequantization expects a 2D q tensor")

    q_2d = q.contiguous()
    k = q_2d.shape[-1]
    output_dtype = DTYPE_CODE_TO_DTYPE[output_dtype_code]
    if _should_use_convrot_dequant_kernel(q_2d, k, group_size):
        scale_arg = scale.to(device=q_2d.device, dtype=torch.float32).reshape(-1).contiguous()
        output = torch.empty(q_2d.shape, dtype=output_dtype, device=q_2d.device)
        stream_ptr = torch.cuda.current_stream(q_2d.device).cuda_stream
        _C.dequantize_int8_convrot_weight(
            _wrap_for_dlpack(q_2d),
            _wrap_for_dlpack(scale_arg),
            _wrap_for_dlpack(output),
            group_size,
            stream_ptr,
        )
        return output

    h = _build_hadamard(group_size, device=q_2d.device, dtype=torch.float32)
    return _rotate_weight(dequantize_int8_simple(q_2d, scale), h, group_size).to(output_dtype)

def _act_weight_arg(input_act, input_act_weight, device, dtype: torch.dtype) -> torch.Tensor:
    """The fused quantizer's act-weight operand: the norm weight for acts
    that carry one, the empty placeholder otherwise."""
    if input_act == "rms_norm" and input_act_weight is not None:
        return input_act_weight.to(device=device, dtype=dtype).contiguous()
    return _empty_cuda_tensor(device, dtype)

def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    input_act: str | None = None,
    input_act_weight: torch.Tensor | None = None,
    input_act_eps: float = 0.0,
    residual: torch.Tensor | None = None,
    residual_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if residual is not None and residual_scale is None:
        raise ValueError("int8_linear: residual requires residual_scale")
    orig_shape = x.shape
    x_2d = x if x.dim() == 2 and x.is_contiguous() else x.reshape(-1, x.shape[-1]).contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

    # Only the fused ConvRot quantizer can absorb the activation; every other
    # route applies it eagerly here and proceeds unchanged, so all paths agree.
    # k_act is the activated (quantized) row width: swiglu halves the raw row.
    k_act = x_2d.shape[-1] // _input_act_width(input_act)
    _fused_convrot_ok = (
        convrot
        and convrot_groupsize == 256
        and k_act % 256 == 0
        and 256 <= k_act <= _CONVROT_FUSED_MAX_K
        and _convrot_fused_shared_memory_fits(x_2d, k_act, convrot_groupsize)
    )
    if input_act not in (None, "none") and not _fused_convrot_ok:
        x_2d = _apply_input_act(x_2d, input_act, input_act_weight, input_act_eps)
        input_act = None

    m = x_2d.shape[0]
    k = k_act
    n, k_w = weight.shape
    assert k == k_w, "Input and weight inner dimensions must match"

    out_dtype = out_dtype or x.dtype
    output_dtype_code = DTYPE_TO_CODE[out_dtype]
    is_2d_output = len(orig_shape) == 2

    # The residual is fused into the CUTLASS epilogue when that path runs;
    # every other route applies it eagerly here so all paths agree.
    fused_residual_done = False

    def _finish(o):
        if not fused_residual_done:
            o = _apply_residual(o, None if residual is None else residual.reshape(o.shape),
                                residual_scale)
        return o if is_2d_output else o.reshape(*orig_shape[:-1], n)

    convrot_m1_supported = (
        m == 1
        and convrot
        and convrot_groupsize == 256
        and k % 256 == 0
        and 256 <= k <= _CONVROT_FUSED_MAX_K
        and _convrot_fused_shared_memory_fits(x_2d, k, convrot_groupsize)
    )
    nonconvrot_m1_supported = (
        m == 1
        and not convrot
        and k % 4 == 0
        and (k <= 2560 or (k == 6144 and n <= 128))
    )
    if input_act in (None, "none") and (convrot_m1_supported or nonconvrot_m1_supported):
        x_qdata = torch.empty((1, k), dtype=torch.int8, device=x.device)
        x_scale = torch.empty((1, 1), dtype=torch.float32, device=x.device)
        weight_scale = _int8_weight_scale_arg(weight_scale, x.device)
        out = torch.empty((1, n), dtype=out_dtype, device=x.device)
        bias_arg = bias if bias is not None else _empty_cuda_tensor(x.device, out_dtype)
        if bias is not None and (bias.device != x.device or bias.dtype != out_dtype or not bias.is_contiguous()):
            bias_arg = bias.to(device=x.device, dtype=out_dtype).contiguous()
        _C.int8_linear_m1(
            _wrap_for_dlpack(x_2d),
            _wrap_for_dlpack(x_qdata),
            _wrap_for_dlpack(x_scale),
            _wrap_for_dlpack(weight),
            _wrap_for_dlpack(weight_scale),
            _wrap_for_dlpack(bias_arg),
            _wrap_for_dlpack(out),
            output_dtype_code,
            convrot,
            convrot_groupsize,
            stream_ptr,
        )
        return _finish(out)

    # cuBLAS INT8 GEMM requires row-wise quantized activations and tensor-wise quantized weights.
    if convrot:
        # Fused wins for small K (narrow block) and large K (wide block); the
        # 5120 < K < 8192 band loses to the rotate-matmul path on both, so skip
        # it. (Real model hidden dims avoid that band anyway.)
        if _fused_convrot_ok:
            x_qdata = torch.empty((m, k), dtype=torch.int8, device=x.device)
            x_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
            act_weight_arg = _act_weight_arg(
                input_act, input_act_weight, x.device, x_2d.dtype)
            _C.quantize_int8_rowwise_convrot64(
                _wrap_for_dlpack(x_2d),
                _wrap_for_dlpack(x_qdata),
                _wrap_for_dlpack(x_scale),
                convrot_groupsize,
                False,
                _input_act_code(input_act),
                0,
                _wrap_for_dlpack(act_weight_arg),
                float(input_act_eps),
                stream_ptr,
            )
        elif _should_use_convrot_fused_kernel(x_2d, k, convrot_groupsize):
            # Fused single-kernel rotation + row-wise quant (no bf16 HBM round-trip).
            x_qdata, x_scale = quantize_int8_rowwise_convrot(x_2d, convrot_groupsize)
        else:
            # Fallback: standalone rotation matmul, then row-wise quant.
            h = _build_hadamard(convrot_groupsize, device=x_2d.device, dtype=x_2d.dtype)
            x_qdata, x_scale = quantize_and_rotate_rowwise(x_2d, h, convrot_groupsize)
    else:
        x_qdata = torch.empty((m, k), dtype=torch.int8, device=x.device)
        x_scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
        _C.quantize_int8_rowwise(
            _wrap_for_dlpack(x_2d),
            _wrap_for_dlpack(x_qdata),
            _wrap_for_dlpack(x_scale),
            False,
            0,
            stream_ptr,
        )

    if (m == 1 and k % 4 == 0) or (m == 2 and k % 16 == 0):
        # The two-row kernel uses 16-byte loads, including for contiguous views.
        if m == 2 and not _aligned16(weight):
            weight = weight.clone()
        weight_scale = _int8_weight_scale_arg(weight_scale, x.device)
        out = torch.empty((m, n), dtype=out_dtype, device=x.device)
        bias_arg = bias if bias is not None else _empty_cuda_tensor(x.device, out_dtype)
        if bias is not None and (bias.device != x.device or bias.dtype != out_dtype or not bias.is_contiguous()):
            bias_arg = bias.to(device=x.device, dtype=out_dtype).contiguous()
        _C.int8_gemv_dequant(
            _wrap_for_dlpack(x_qdata),
            _wrap_for_dlpack(weight),
            _wrap_for_dlpack(x_scale),
            _wrap_for_dlpack(weight_scale),
            _wrap_for_dlpack(bias_arg),
            _wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )
        return _finish(out)

    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    weight_scale = _int8_weight_scale_arg(weight_scale, x.device)
    bias_arg = (_gemm_vector_arg(bias, x.device, out_dtype) if bias is not None
                else _empty_cuda_tensor(x.device, out_dtype))

    if (
        _prefer_turing_fused_int8(m, n, k)
        and _cuda_device_should_use_turing_kernels(x_qdata.get_device())
        and hasattr(_C, "cutlass_turing_int8_dequant")
    ):
        turing_out = _int8_linear_turing_quantized(
            x_qdata,
            weight,
            x_scale,
            weight_scale,
            bias_arg if bias is not None else None,
            out_dtype,
        )
        if turing_out is not None:
            return _finish(turing_out)

    used_cutlass = False
    prefer_cublas_fallback = _prefer_cublas_int8_fallback(
        m,
        n,
        k,
        x_qdata.get_device(),
    )
    if (
        not prefer_cublas_fallback
        and not _DISABLE_CUTLASS_INT8
        and _cuda_device_supports_cutlass_int8_dequant(x_qdata)
    ):
        ws_cutlass = weight_scale if weight_scale.numel() == n else weight_scale.expand(n).contiguous()
        # the residual is fused only in the output dtype and 16-byte aligned
        # (the epilogue's vector loads); the per-channel scale is realigned instead
        resid_2d = None
        if residual is not None and residual.dtype == out_dtype and residual_scale.dtype == out_dtype:
            resid_2d = residual.reshape(m, n).contiguous()
            if not _aligned16(resid_2d):
                resid_2d = None
        if resid_2d is not None:
            used_cutlass = _C.cutlass_int8_dequant_residual(
                _wrap_for_dlpack(x_qdata),
                _wrap_for_dlpack(weight),
                _wrap_for_dlpack(x_scale),
                _wrap_for_dlpack(ws_cutlass),
                _wrap_for_dlpack(bias_arg),
                _wrap_for_dlpack(_gemm_vector_arg(residual_scale, x.device, out_dtype)),
                _wrap_for_dlpack(resid_2d),
                _wrap_for_dlpack(out),
                output_dtype_code,
                stream_ptr,
            )
            fused_residual_done = used_cutlass
        if not used_cutlass:
            used_cutlass = _C.cutlass_int8_dequant(
                _wrap_for_dlpack(x_qdata),
                _wrap_for_dlpack(weight),
                _wrap_for_dlpack(x_scale),
                _wrap_for_dlpack(ws_cutlass),
                _wrap_for_dlpack(bias_arg),
                _wrap_for_dlpack(out),
                output_dtype_code,
                stream_ptr,
            )
    if not used_cutlass:
        # Fallback: cuBLAS int8 GEMM (int32) + separate dequant kernel.
        use_turing_padding = x_qdata.is_cuda and _cuda_device_is_turing(x_qdata.get_device())
        if use_turing_padding:
            padded_k = _round_up(k, 16)
            padded_n = _round_up(n, _cublas_int8_n_alignment(x_qdata))
            cublas_x = _pad_2d_cols(x_qdata, padded_k)
            cublas_weight = _pad_2d_rows(_pad_2d_cols(weight, padded_k), padded_n)
        else:
            padded_n = n
            cublas_x = x_qdata
            cublas_weight = weight

        out_int32 = torch.empty((m, padded_n), dtype=torch.int32, device=x.device)
        _C.cublas_gemm_int8(
            _wrap_for_dlpack(cublas_x),
            _wrap_for_dlpack(cublas_weight),
            _wrap_for_dlpack(out_int32),
            _wrap_for_dlpack(get_cublas_workspace()),
            stream_ptr,
        )
        if padded_n != n:
            out_int32 = out_int32[:, :n].contiguous()
        _C.dequantize_int8_linear(
            _wrap_for_dlpack(out_int32),
            _wrap_for_dlpack(x_scale),
            _wrap_for_dlpack(weight_scale),
            _wrap_for_dlpack(bias_arg),
            _wrap_for_dlpack(out),
            output_dtype_code,
            stream_ptr,
        )

    return _finish(out)
