"""Profile GPT-2 decode and bucket GPU time by operation type."""

import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.profiler import ProfilerActivity, profile
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# Order matters: first match wins, so specific buckets go before generic ones.
BUCKETS = [
    ("matmul (Linear/proj)", ("addmm", "baddbmm", "::mm", "bmm", "matmul", "linear")),
    ("attention (flash)", ("flash_attention", "scaled_dot_product", "sdpa")),
    ("softmax", ("softmax",)),
    ("layernorm", ("layer_norm",)),
    ("gelu (tanh/pow)", ("gelu", "tanh", "pow", "erf")),
    ("elementwise add/mul", ("add", "mul", "sub", "div", "rsub")),
    ("kv-cache concat", ("cat",)),
    ("copy/reshape", ("copy", "index", "slice", "clone", "contiguous",
                      "transpose", "permute", "expand", "arange")),
]

# Compute-bound buckets (cores busy) vs memory-bound (waiting on HBM).
COMPUTE_BUCKETS = {"matmul (Linear/proj)", "attention (flash)"}


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, needles in BUCKETS:
        if any(n in low for n in needles):
            return label
    return "other"


def cuda_us(evt) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v:
            return float(v)
    return 0.0


@torch.no_grad()
def decode(model, input_ids, new_tokens):
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    for _ in range(new_tokens - 1):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
    return next_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--out", default="results/profile_before.png")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run this on the GPU box.")

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()}  |  {args.model}  |  {args.dtype}")

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device, dtype).eval()

    prompt = "The quick brown fox " * 16
    input_ids = tok(prompt, return_tensors="pt").input_ids[:, : args.prompt_len].to(device)

    decode(model, input_ids, 8)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        decode(model, input_ids, args.new_tokens)
    torch.cuda.synchronize()

    bucket_us = defaultdict(float)
    total_us = 0.0
    rows = []
    for evt in prof.key_averages():
        if not evt.key.startswith("aten::"):
            continue
        t = cuda_us(evt)
        if t <= 0:
            continue
        bucket_us[bucket_of(evt.key)] += t
        total_us += t
        rows.append((evt.key, t, evt.count))

    ordered = sorted(bucket_us.items(), key=lambda kv: -kv[1])
    print(f"\n{'bucket':>22} | {'GPU ms':>9} | {'% of total':>10}")
    print("-" * 48)
    for label, us in ordered:
        print(f"{label:>22} | {us/1e3:9.3f} | {us/total_us*100:9.1f}%")
    print("-" * 48)
    print(f"{'TOTAL':>22} | {total_us/1e3:9.3f} | {100.0:9.1f}%")

    print("\nTop 12 individual ops by GPU time:")
    print(f"{'op':>28} | {'GPU ms':>9} | {'calls':>7} | {'us/call':>8}")
    print("-" * 62)
    for name, t, count in sorted(rows, key=lambda r: -r[1])[:12]:
        print(f"{name:>28} | {t/1e3:9.3f} | {count:7d} | {t/count:8.1f}")

    # Bar chart of the bucket breakdown.
    labels = [l for l, _ in ordered]
    vals = [us / 1e3 for _, us in ordered]
    colors = ["#6366f1" if l in COMPUTE_BUCKETS else "#ef4444" for l in labels]
    plt.figure(figsize=(8, 4.5))
    plt.barh(labels[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("GPU time (ms)")
    plt.title(f"GPT-2 decode — where the GPU spends time\n"
              f"(batch=1, {args.new_tokens} tokens, {args.dtype}; blue=matmul, red=memory-bound)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"\nSaved chart -> {args.out}")


if __name__ == "__main__":
    main()
