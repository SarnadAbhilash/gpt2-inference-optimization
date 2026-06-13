# Profile — where the time actually goes

Same decode as the baseline, run under `torch.profiler`. GPU-active time is
bucketed by operation type.

```bash
python bench/profile_decode.py
```

![profile](profile_before.png)

## GPU-active breakdown (128 tokens, batch=1, fp16)

| Bucket | GPU ms | % | Bound by |
|---|---|---|---|
| matmul (Linear/proj) | 48.9 | 53.0% | compute (cuBLAS, ~optimal) |
| elementwise add/mul | 12.8 | 13.8% | **memory** |
| attention (flash)    | 11.7 | 12.7% | compute (already fused) |
| layernorm            | 7.6  | 8.2%  | **memory** |
| kv-cache concat      | 5.9  | 6.4%  | **memory** |
| gelu (tanh/pow)      | 3.5  | 3.8%  | **memory** |
| other (argmax)       | 1.3  | 1.4%  | — |
| copy/reshape         | 0.6  | 0.6%  | — |
| **total**            | **92.3** | 100% | |

## Three findings that shape the plan

1. **The GPU is idle ~87% of the time.** GPU-active work is only 0.72 ms/token,
   but wall-clock is 5.63 ms/token. The decode issues **~237 kernel launches per
   token**; at batch=1 each kernel's work is tiny, so per-launch + Python
   overhead dominates. Fewer/bigger kernels is the biggest lever.

2. **Softmax is already fused.** No `softmax` kernel appears — PyTorch routes
   attention through `_flash_attention_forward`, which fuses the softmax inside.
   The original "fuse softmax first" idea is moot; that win already exists.

3. **The real Triton targets are LayerNorm + GELU + residual adds.** Together the
   memory-bound small ops are ~32% of GPU time, spread across thousands of tiny
   kernels (3200 layernorms, 6400 adds, 6144 muls, ...). Fusing them cuts both
   HBM traffic and launch count. Matmul (53%) is already cuBLAS-optimal — leave
   it alone.
