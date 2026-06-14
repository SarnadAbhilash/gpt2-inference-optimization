"""Which kernel helps? Patch GELU-only, LayerNorm-only, and both; compare tok/s."""

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


def tok_s(model, input_ids, new_tokens, reps=5):
    decode(model, input_ids, 8)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(reps):
        s.record(); decode(model, input_ids, new_tokens); e.record()
        torch.cuda.synchronize(); t.append(s.elapsed_time(e) / 1e3)
    return input_ids.shape[0] * new_tokens / min(t)


def fresh(do_ln, do_gelu):
    m = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda", torch.float16).eval()
    if do_ln or do_gelu:
        patch_gpt2(m, do_layernorm=do_ln, do_gelu=do_gelu)
    return m


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — run this on the GPU box.")
    print(f"Device: {torch.cuda.get_device_name()}  |  gpt2  |  float16\n")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = "The quick brown fox " * 16
    base = tok(prompt, return_tensors="pt").input_ids[:, :32].to("cuda")

    configs = [
        ("stock", False, False),
        ("gelu only", False, True),
        ("layernorm only", True, False),
        ("both", True, True),
    ]
    for B in [1, 16]:
        input_ids = base.repeat(B, 1)
        print(f"batch={B}")
        ref = None
        for name, do_ln, do_gelu in configs:
            m = fresh(do_ln, do_gelu)
            t = tok_s(m, input_ids, 64)
            if ref is None:
                ref = t
            print(f"  {name:>16} | {t:9.1f} tok/s | {t/ref:6.3f}x")
            del m
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
