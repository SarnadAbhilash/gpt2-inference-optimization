"""End-to-end: does swapping our kernels into GPT-2 actually make it faster?

Compares a stock GPT-2 against the same model with our fused LayerNorm + GELU:
  - correctness: max logit difference, top-1 agreement, greedy-trajectory match
  - speed: decode tokens/sec (best of N)
  - kernel launches per token (why fusion helps the idle-GPU problem)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.patch import patch_gpt2


@torch.no_grad()
def decode(model, input_ids, new_tokens):
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    ids = [next_id]
    for _ in range(new_tokens - 1):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        ids.append(next_id)
    return torch.cat(ids, dim=1)


def measure_tok_s(model, input_ids, new_tokens, reps=5):
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
    return new_tokens / best, best / new_tokens * 1e3  # tok/s, ms/tok


def launches_per_token(model, input_ids, new_tokens):
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        decode(model, input_ids, new_tokens)
    torch.cuda.synchronize()
    total = 0
    for evt in prof.key_averages():
        cuda = getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
        if cuda > 0:
            total += evt.count
    return total / new_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run this on the GPU box.")
    device, dtype = "cuda", torch.float16
    print(f"Device: {torch.cuda.get_device_name()}  |  {args.model}  |  float16\n")

    tok = GPT2TokenizerFast.from_pretrained(args.model)
    prompt = "The quick brown fox " * 16
    input_ids = tok(prompt, return_tensors="pt").input_ids[:, : args.prompt_len].to(device)

    stock = GPT2LMHeadModel.from_pretrained(args.model).to(device, dtype).eval()
    fast = GPT2LMHeadModel.from_pretrained(args.model).to(device, dtype).eval()
    patch_gpt2(fast, do_layernorm=True, do_gelu=True)

    # --- correctness ---
    with torch.no_grad():
        lo_s = stock(input_ids).logits
        lo_f = fast(input_ids).logits
    max_diff = (lo_s.float() - lo_f.float()).abs().max().item()
    top1_agree = (lo_s.argmax(-1) == lo_f.argmax(-1)).float().mean().item() * 100

    ids_s = decode(stock, input_ids, args.new_tokens)
    ids_f = decode(fast, input_ids, args.new_tokens)
    traj_match = (ids_s == ids_f).sum().item()

    print("CORRECTNESS")
    print(f"  max logit diff (fp16):      {max_diff:.4f}")
    print(f"  top-1 token agreement:      {top1_agree:.1f}%")
    print(f"  greedy trajectory match:    {traj_match}/{args.new_tokens} tokens identical\n")

    # --- speed ---
    s_toks, s_ms = measure_tok_s(stock, input_ids, args.new_tokens)
    f_toks, f_ms = measure_tok_s(fast, input_ids, args.new_tokens)

    # --- kernel launches ---
    s_launch = launches_per_token(stock, input_ids, args.new_tokens)
    f_launch = launches_per_token(fast, input_ids, args.new_tokens)

    print("SPEED (decode, batch=1, best of 5)")
    print(f"  {'':>10} | {'tok/s':>8} | {'ms/tok':>8} | {'launches/tok':>13}")
    print(f"  {'stock':>10} | {s_toks:8.1f} | {s_ms:8.3f} | {s_launch:13.0f}")
    print(f"  {'fused':>10} | {f_toks:8.1f} | {f_ms:8.3f} | {f_launch:13.0f}")
    print(f"\n  end-to-end speedup: {f_toks / s_toks:.3f}x  "
          f"({(f_toks/s_toks - 1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
