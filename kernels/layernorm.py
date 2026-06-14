"""Fused LayerNorm in Triton (forward / inference).

LayerNorm over the last dimension of a (..., N) tensor. One Triton program
handles one row: it loads the row into SRAM once, computes the mean and variance
there, normalizes, applies the per-feature scale (weight) and shift (bias), and
writes the row back once. Stats are accumulated in fp32 for stability even when
the input is fp16 (this is what torch does too).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * row_stride
    y_ptr += row * row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)          # mean-centered, 0 in the padding
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b

    tl.store(y_ptr + cols, y, mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    """Fused LayerNorm over the last dim. Mirrors F.layer_norm(x, (N,), w, b, eps)."""
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    *batch, N = x.shape
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    out = torch.empty_like(x2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 2048 else 8

    _layernorm_fwd[(M,)](
        x2d, out, weight, bias,
        x2d.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return out.reshape(*batch, N)
