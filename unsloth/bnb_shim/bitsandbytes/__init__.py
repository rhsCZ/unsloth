# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Unsloth's bitsandbytes stand-in, imported as `bitsandbytes` only when the real package is not
installed or cannot load its native library (see unsloth/bnb_shim/__init__.py).

It implements the 4-bit subset transformers, PEFT, accelerate and Unsloth use (Params4bit,
Linear4bit, QuantState, quantize_4bit / dequantize_4bit, blockwise 8-bit quantization of the
double-quant absmax) on Unsloth's Triton kernels, byte compatible with bitsandbytes checkpoints.
Adam / AdamW (8-bit, 32-bit, paged) keep bitsandbytes' blockwise 8-bit optimizer state on Unsloth's
Triton kernels (bitsandbytes.optim); SGD / RMSprop / Adagrad run their torch counterparts; Lion,
AdEMAMix, LAMB, LARS and LLM.int8 refuse with a clear error."""

import torch

from . import functional, nn, optim, utils
from .autograd._functions import MatMul4Bit, matmul_4bit
from .nn import modules

__version__ = "0.49.2"
__unsloth_shim__ = True

features = {"multi_backend"}
supported_torch_devices = {"cpu", "cuda"}
if hasattr(torch, "xpu"):
    supported_torch_devices.add("xpu")

__all__ = ["MatMul4Bit", "functional", "matmul_4bit", "modules", "nn", "optim", "utils"]
