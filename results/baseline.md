# Baseline

Stock HuggingFace GPT-2 small, eager execution, no fused kernels. This is the
"before" number every optimization is measured against.

| Setting | Value |
|---------|-------|
| GPU | NVIDIA RTX 4090 (24 GB) |
| Model | gpt2 (124M) |
| dtype | float16 |
| Batch | 1 |
| Decode length | 128 tokens (greedy, KV cache) |
| torch / triton | 2.4.1+cu124 / 3.0.0 |

**Result:** 721 ms for 128 tokens → **177.5 tok/s** (5.63 ms/tok), best of 5 reps.

Reproduce:

```bash
python bench/baseline.py
```
