# inference-optimizer

Profiling GPT-2 inference, finding the real bottlenecks, and replacing them with
custom fused [Triton](https://github.com/triton-lang/triton) kernels — verified
bit-correct against PyTorch and benchmarked for speedup.

## TL;DR

| Op | Baseline (PyTorch) | Fused (Triton) | Speedup |
|----|--------------------|----------------|---------|
| Softmax    | _tbd_ | _tbd_ | _tbd_ |
| LayerNorm  | _tbd_ | _tbd_ | _tbd_ |
| End-to-end GPT-2 (tok/s) | _tbd_ | _tbd_ | _tbd_ |

Numbers measured on a single RTX 4090 (fill in after running).

## Why these ops?

Autoregressive decode at small batch is **memory-bandwidth bound**: LayerNorm,
softmax, and GELU each stream the full activation tensor through HBM while doing
almost no arithmetic, and each is a separate kernel launch. Profiling (Phase 1)
shows the time concentrated there — so fusing them (one pass, data kept in SRAM)
is where the wins are.

## Layout

```
bench/
  profile_baseline.py   # Phase 1: profile GPT-2, find the bottleneck
  bench_op.py           # microbenchmark + correctness harness for a single op
kernels/
  softmax.py            # fused softmax (Triton)
  layernorm.py          # fused layernorm (Triton)   [Phase 3]
models/
  patch.py              # swap fused kernels into HF GPT-2  [Phase 4]
results/                # plots + notes
```

## Run

```bash
pip install -r requirements.txt
python bench/profile_baseline.py          # Phase 1: see the bottleneck
python bench/bench_op.py softmax          # Phase 2: prove the kernel is faster + correct
```
