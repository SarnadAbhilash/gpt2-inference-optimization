"""Correctness + microbenchmark for the fused LayerNorm kernel.

Checks the Triton kernel against torch.nn.functional.layer_norm across a range
of shapes, then times both with triton.testing.do_bench and reports latency,
effective HBM bandwidth, and speedup. GPT-2 hidden size is 768; we sweep the
number of rows from decode-sized (1) to prefill/batch-sized (8192).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import triton

from kernels.layernorm import layernorm as triton_layernorm


def bandwidth_gbps(x, w, b, ms):
    # read x + write y (dominant); weight/bias are tiny and cached.
    bytes_moved = 2 * x.numel() * x.element_size()
    return bytes_moved / (ms * 1e-3) / 1e9


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available — run this on the GPU box.")
    print(f"Device: {torch.cuda.get_device_name()}\n")

    torch.manual_seed(0)
    N = 768  # GPT-2 small hidden size
    eps = 1e-5
    row_counts = [1, 8, 64, 512, 4096, 8192]

    print(f"{'rows x N':>14} | {'max abs err':>12} | "
          f"{'torch ms':>9} | {'triton ms':>9} | {'speedup':>7} | "
          f"{'torch GB/s':>10} | {'triton GB/s':>11}")
    print("-" * 96)

    for M in row_counts:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        y_ref = F.layer_norm(x, (N,), w, b, eps)
        y_tri = triton_layernorm(x, w, b, eps)
        max_err = (y_ref.float() - y_tri.float()).abs().max().item()
        ok = torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2)
        flag = "" if ok else "  <-- MISMATCH"

        ms_torch = triton.testing.do_bench(lambda: F.layer_norm(x, (N,), w, b, eps))
        ms_tri = triton.testing.do_bench(lambda: triton_layernorm(x, w, b, eps))

        print(f"{f'{M} x {N}':>14} | {max_err:12.2e} | "
              f"{ms_torch:9.4f} | {ms_tri:9.4f} | "
              f"{ms_torch / ms_tri:6.2f}x | "
              f"{bandwidth_gbps(x, w, b, ms_torch):10.1f} | "
              f"{bandwidth_gbps(x, w, b, ms_tri):11.1f}{flag}")


if __name__ == "__main__":
    main()
