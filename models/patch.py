"""Swap our fused Triton kernels into a HuggingFace GPT-2 model (inference only).

- Every nn.LayerNorm (ln_1, ln_2, ln_f) -> our fused Triton LayerNorm.
- Each block's MLP activation (GPT-2's multi-kernel gelu_new) -> our fused GELU.

The original weights/biases are reused by reference, so the patched model is
numerically the same network — only the kernels that run the math change.
"""

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernels.gelu import gelu as triton_gelu
from kernels.layernorm import layernorm as triton_layernorm


class TritonLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps

    def forward(self, x):
        return triton_layernorm(x, self.weight, self.bias, self.eps)


class TritonGELU(nn.Module):
    def forward(self, x):
        return triton_gelu(x)


def _set_submodule(root, dotted_name, new_module):
    *parents, last = dotted_name.split(".")
    obj = root
    for p in parents:
        obj = getattr(obj, p)
    setattr(obj, last, new_module)


def patch_gpt2(model, do_layernorm=True, do_gelu=True):
    """Replace LayerNorm and/or GELU in-place. Returns the same model."""
    if do_layernorm:
        # Collect first; don't mutate while iterating named_modules().
        targets = [n for n, m in model.named_modules() if isinstance(m, nn.LayerNorm)]
        for name in targets:
            ln = dict(model.named_modules())[name]
            _set_submodule(model, name, TritonLayerNorm(ln))

    if do_gelu:
        for module in model.modules():
            if module.__class__.__name__ == "GPT2MLP":
                module.act = TritonGELU()

    return model
