"""Measure GPT-2 small autoregressive decode throughput (tok/s)."""

import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


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
    parser.add_argument("--reps", type=int, default=5)
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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.reps):
        start.record()
        decode(model, input_ids, args.new_tokens)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)

    best = min(times)
    print(f"\n{args.new_tokens} tokens, best of {args.reps} reps:")
    print(f"  {best*1e3:.1f} ms  ->  {args.new_tokens / best:.1f} tok/s  "
          f"({best / args.new_tokens * 1e3:.3f} ms/tok)")


if __name__ == "__main__":
    main()
