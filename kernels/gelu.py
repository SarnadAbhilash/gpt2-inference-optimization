"""Fused GELU (tanh approximation) in Triton.

GPT-2 uses the "new" GELU:

    gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

HuggingFace computes this as an explicit chain of elementwise ops (pow, mul, add,
tanh, ...), which PyTorch runs as several separate kernels — each a full
read+write of the activation through HBM. This fuses the whole formula into one
elementwise kernel: load once, compute everything in registers, write once.
"""

import torch
import triton
import triton.language as tl

_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)


@triton.jit
def _gelu_fwd(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)  # sqrt(2/pi) * (...)
    # tanh(z) = 2*sigmoid(2z) - 1  (avoids needing libdevice; tl.sigmoid is built in)
    tanh = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    y = 0.5 * x * (1.0 + tanh)

    tl.store(y_ptr + offs, y, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Fused tanh-approx GELU. Same result as F.gelu(x, approximate='tanh')."""
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _gelu_fwd[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
