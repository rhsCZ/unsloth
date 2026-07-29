# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One shared answer to "is bitsandbytes usable on this host?".

A successful `import bitsandbytes` is not that answer. A wheel whose native side
failed to load still imports, and what it leaves behind varies by version:

  * no `functional` at all, so reading it raises at import
  * `functional.lib is None` (0.45.5, the floor in pyproject.toml), so the ctypes
    binds raise "'NoneType' object has no attribute cdequantize_blockwise_fp32"
  * `functional.lib` missing that symbol, which raises on the bind
  * an ErrorHandlerMockBNBNativeLibrary (0.46 onwards). This one does NOT raise:
    BNBNativeLibrary.__getattr__ returns a plain `throw_on_call` closure, so the
    binds succeed and 4bit dies later inside a kernel.

Two different questions, so two answers:

  `check_bitsandbytes` - would a module-scope read in kernels/utils.py raise? A
  failure here crashes `import unsloth`, so it is the hard gate: fail it and the
  module is treated as absent.

  `check_native_kernels` - are the handles real kernels? A deferred-failure
  closure binds cleanly, so this cannot gate the import; it gates the capability
  flags instead. bnb stays bound, `get_ptr` stays the real function, and 4bit
  fails only if a 4bit path is actually entered - which is what the fallbacks in
  kernels/utils.py have always promised. A real handle is a ctypes function
  pointer, a deferred failure is a Python function, hence the `restype` check.

Collapsing the two would write off a healthy CPU-only install: there every symbol
is a `throw_on_call` closure, so a single strict probe reports "no bitsandbytes"
on a wheel whose Python side works perfectly.

device_type.py, _gpu_init.py and kernels/utils.py must agree on both answers, or
ALLOW_BITSANDBYTES stays true while the kernels fall back to the stub and
loader.py forwards a 4bit request instead of the advertised 16bit fallback.

A leaf module on purpose: it imports nothing from unsloth - device_type.py is
imported very early and would be a cycle - and takes the device type as an
argument. bitsandbytes is imported inside a function, so `import unsloth` never
hard-requires it.
"""

__all__ = [
    "bitsandbytes_symbols",
    "check_bitsandbytes",
    "check_native_kernels",
    "native_kernels_ready",
    "probe_bitsandbytes",
]

# The ctypes handles kernels/utils.py binds at module scope. Keep in step with the
# `bnb.functional.lib.*` reads there - a test asserts the two match.
_C_SYMBOLS = (
    "cdequantize_blockwise_fp32",
    "cdequantize_blockwise_fp16_nf4",
    "cdequantize_blockwise_bf16_nf4",
)
# 4bit inference is a gemv on xpu and a naive gemm everywhere else, so probing the
# xpu names on cuda would write off a perfectly good wheel.
_C_SYMBOLS_XPU = (
    "cgemv_4bit_inference_fp16",
    "cgemv_4bit_inference_bf16",
)
_C_SYMBOLS_GEMM = (
    "cgemm_4bit_inference_naive_fp16",
    "cgemm_4bit_inference_naive_bf16",
)


def bitsandbytes_symbols(device_type):
    """Names kernels/utils.py reads off `bitsandbytes.functional.lib`."""
    tail = _C_SYMBOLS_XPU if device_type == "xpu" else _C_SYMBOLS_GEMM
    return _C_SYMBOLS + tail


def check_bitsandbytes(bnb, device_type):
    """Raise unless `bnb` can serve every module-scope read kernels/utils.py makes.

    The hard gate: anything that raises here would crash `import unsloth`. It does
    not judge whether the handles are real kernels - see `check_native_kernels`.

    Safe to repeat: ctypes caches the function object on the first lookup and
    bitsandbytes memoizes its wrapper, so the handles bound later are these ones.
    """
    if bnb is None:
        raise ImportError("Unsloth: `bitsandbytes` is not installed.")
    _version = bnb.__version__  # kernels/utils.py gates HAS_CUDA_STREAM on it
    functional = bnb.functional
    _get_ptr = functional.get_ptr
    lib = functional.lib  # None on a 0.45.5 native-load failure
    for symbol in bitsandbytes_symbols(device_type):
        getattr(lib, symbol)  # AttributeError on a lib that does not export it


def check_native_kernels(bnb, device_type):
    """Raise unless every bound handle is a real native kernel.

    Separate from `check_bitsandbytes` because these shapes bind cleanly: this
    decides whether 4bit can actually run, not whether the import survives.
    """
    check_bitsandbytes(bnb, device_type)
    lib = bnb.functional.lib
    for symbol in bitsandbytes_symbols(device_type):
        # Only a ctypes foreign function has `restype`; a deferred failure is a
        # plain closure that raises at call time instead of here.
        if not hasattr(getattr(lib, symbol), "restype"):
            raise AttributeError(
                f"Unsloth: `bitsandbytes.functional.lib.{symbol}` is not a native "
                "handle - the bitsandbytes native library did not load."
            )


def native_kernels_ready(bnb, device_type):
    """Can 4bit actually run here? Gates the capability flags, never the import."""
    try:
        check_native_kernels(bnb, device_type)
    except Exception:
        return False
    return True


def probe_bitsandbytes(device_type):
    """The bitsandbytes module when it imports without breaking us, else None.

    Deliberately the hard gate only. A wheel whose kernels merely defer their
    failure still gives a working `get_ptr` and real ctypes binds, so returning
    None for it would disable a usable module; `native_kernels_ready` handles
    that case by clearing the capability flags instead.
    """
    try:
        import bitsandbytes
        check_bitsandbytes(bitsandbytes, device_type)
    except Exception:
        return None
    return bitsandbytes
