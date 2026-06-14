"""Throughput of stock vs fused GPT-2 across batch sizes.

At batch=1 decode is CPU/launch-bound (the GPU is mostly idle), so faster kernels
don't help end-to-end. As batch grows the GPU becomes the bottleneck, and the
fused kernels should start to pay off. This shows where op-level wins translate.
"""

import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.patch import patch_gpt2


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


def throughput(model, input_ids, new_tokens, reps=5):
    decode(model, input_ids, 8)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        decode(model, input_ids, new_tokens)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    best = min(times)
    return input_ids.shape[0] * new_tokens / best  # total tokens/sec


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run this on the GPU box.")
    device, dtype = "cuda", torch.float16
    print(f"Device: {torch.cuda.get_device_name()}  |  gpt2  |  float16\n")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = "The quick brown fox " * 16
    base = tok(prompt, return_tensors="pt").input_ids[:, :32].to(device)

    stock = GPT2LMHeadModel.from_pretrained("gpt2").to(device, dtype).eval()
    fast = GPT2LMHeadModel.from_pretrained("gpt2").to(device, dtype).eval()
    patch_gpt2(fast)

    print(f"{'batch':>6} | {'stock tok/s':>12} | {'fused tok/s':>12} | {'speedup':>8}")
    print("-" * 48)
    for B in [1, 4, 16, 64, 128]:
        input_ids = base.repeat(B, 1)
        s = throughput(stock, input_ids, 64)
        f = throughput(fast, input_ids, 64)
        print(f"{B:>6} | {s:12.1f} | {f:12.1f} | {f/s:7.3f}x")


if __name__ == "__main__":
    main()
