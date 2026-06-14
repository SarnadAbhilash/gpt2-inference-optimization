"""Correctness + microbenchmark for the fused GELU kernel.

Compares three things:
  1. gpt2  = GPT-2's explicit gelu_new formula (what HuggingFace actually runs;
             several separate kernels).
  2. torch = F.gelu(x, approximate='tanh') (PyTorch's single fused kernel).
  3. triton= our fused kernel.

The meaningful speedup is (1) -> (3): replacing the multi-kernel formula GPT-2
uses today with one kernel. We also show we match torch's fused version.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import triton

from kernels.gelu import gelu as triton_gelu


def gpt2_gelu_new(x):
    # exactly what transformers' NewGELUActivation runs
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def bandwidth_gbps(x, ms):
    return 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — run this on the GPU box.")
    print(f"Device: {torch.cuda.get_device_name()}\n")

    torch.manual_seed(0)
    N = 3072  # GPT-2 MLP intermediate width (4 * 768)
    row_counts = [1, 8, 64, 512, 4096, 8192]

    print(f"{'rows x N':>14} | {'max err':>9} | {'gpt2 ms':>8} | {'torch ms':>8} | "
          f"{'triton ms':>9} | {'vs gpt2':>8} | {'vs torch':>8} | {'triton GB/s':>11}")
    print("-" * 100)

    for M in row_counts:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)

        y_gpt2 = gpt2_gelu_new(x)
        y_tri = triton_gelu(x)
        max_err = (y_gpt2.float() - y_tri.float()).abs().max().item()
        flag = "" if torch.allclose(y_gpt2, y_tri, atol=1e-2, rtol=1e-2) else "  <-- MISMATCH"

        ms_gpt2 = triton.testing.do_bench(lambda: gpt2_gelu_new(x))
        ms_torch = triton.testing.do_bench(lambda: F.gelu(x, approximate="tanh"))
        ms_tri = triton.testing.do_bench(lambda: triton_gelu(x))

        print(f"{f'{M} x {N}':>14} | {max_err:9.2e} | {ms_gpt2:8.4f} | {ms_torch:8.4f} | "
              f"{ms_tri:9.4f} | {ms_gpt2/ms_tri:7.2f}x | {ms_torch/ms_tri:7.2f}x | "
              f"{bandwidth_gbps(x, ms_tri):11.1f}{flag}")


if __name__ == "__main__":
    main()
