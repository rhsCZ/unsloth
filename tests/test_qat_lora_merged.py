# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Merged QAT-LoRA (_qat_lora_merged.py): training computes Q(W + s B A) x with straight-through LoRA gradients,
so the merged, quantized export is the trained model; base-only QAT (the default) shows the export gap."""

import os

import pytest
import torch
import torch.nn.functional as F
import transformers

pytest.importorskip("torchao.quantization.qat")
peft = pytest.importorskip("peft")

import unsloth  # noqa: F401
from unsloth.models import _qat_lora_merged as qlm
from unsloth.models._utils import _prepare_model_for_qat

DIM = 128
SCHEMES = ["int4", "int8", "cactus", "int8-int4", "fp8-int4", "fp8-fp8"]


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # PEFT 0.18 reads model.config.get(...): a real config, not a namespace.
        self.config = transformers.PretrainedConfig(tie_word_embeddings = False)
        self.up_proj = torch.nn.Linear(DIM, 2 * DIM, bias = False)
        self.down_proj = torch.nn.Linear(2 * DIM, DIM, bias = False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


def _peft(
    dtype = torch.float32,
    dropout = 0.0,
    use_dora = False,
    seed = 0,
):
    torch.manual_seed(seed)
    model = peft.get_peft_model(
        _Block().to(dtype),
        peft.LoraConfig(
            r = 8,
            lora_alpha = 16,
            target_modules = ["up_proj", "down_proj"],
            lora_dropout = dropout,
            use_dora = use_dora,
        ),
    )
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std = 0.02)
    return model


def _scheme(scheme):
    if scheme != "int8-int4":
        return scheme
    # The built-in recipe also unties embeddings, which this block has none of: same configs, base layers only.
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig
    from torchao.quantization.granularity import PerGroup
    from unsloth.models._utils import TorchAOConfig

    config = Int8DynamicActivationIntxWeightConfig(
        weight_dtype = torch.int4, weight_granularity = PerGroup(32)
    )
    only_base = lambda m, fqn: isinstance(m, torch.nn.Linear) and fqn.endswith("base_layer")
    return TorchAOConfig(qat_scheme = None, base_config_and_filter_fns = [(config, only_base)])


def _model(
    scheme,
    monkeypatch,
    merged = True,
    dtype = torch.float32,
    **kwargs,
):
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    scheme = _scheme(scheme)
    try:
        mode = {True: "merged", False: "base", None: None}[merged]
        return _prepare_model_for_qat(_peft(dtype, **kwargs), scheme, lora_mode = mode)
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"{scheme} QAT unavailable here: {exc}")


def _lora_layers(model):
    return [m for m in model.modules() if isinstance(m, peft.tuners.lora.Linear)]


def _oracle_forward(
    model,
    x,
    ste = False,
    split = False,
):
    """Q(W + s B A) on every LoRA Linear, by hand; ste = True makes it differentiable in A and B.
    split = True is the activation quantized form: the base on fq(x) and Q(W + s B A), the adapter branch with
    value 0 on a detached input, so dX comes from the base alone and dA / dB use the unquantized input."""
    up, down = _lora_layers(model)

    def layer(proj, h):
        base = proj.get_base_layer()
        quantizer = base.weight_fake_quantizer.inner
        W = base.weight
        A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
        s = proj.scaling["default"]
        merged = qlm.qat_lora_merged_weight(W, A, B, s)
        q = quantizer(merged)
        act = base._modules.get("activation_fake_quantizer")
        if split and act is not None:
            # The exact path: base on Q(W + s B A), adapter branch of value 0 on a detached input.
            lora = s * F.linear(h.detach(), B @ A)
            return F.linear(act(h), q.detach()) + (lora - lora.detach())
        if ste:
            w = W + s * (B @ A)
            q = w + (q - w).detach()
        if act is not None:
            h = act(h)
        return F.linear(h, q)

    return layer(down, torch.relu(layer(up, x)))


def _x(dtype = torch.float32):
    torch.manual_seed(1)
    return torch.randn(6, DIM, dtype = dtype)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_forward_and_grads_match_merged_oracle(scheme, monkeypatch):
    model = _model(scheme, monkeypatch)
    assert len(_lora_layers(model)) == 2
    assert all(qlm.is_qat_lora_merged(m) for m in _lora_layers(model))
    x = _x().requires_grad_(True)
    out = model(x)
    ref = _oracle_forward(model, x)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)

    params = [p for n, p in model.named_parameters() if "lora_" in n]
    grads = torch.autograd.grad(out.square().sum(), [x, *params])
    x2 = x.detach().clone().requires_grad_(True)
    split = scheme in ("fp8-fp8", "fp8-int4", "int8-int4")
    ref = _oracle_forward(model, x2, ste = True, split = split)
    ref_grads = torch.autograd.grad(ref.square().sum(), [x2, *params])
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r, rtol = 1e-4, atol = 1e-5)


@pytest.mark.parametrize("scheme", ["int4", "int8"])
def test_fused_lora_kernels_see_the_merged_weight(scheme, monkeypatch):
    from unsloth.kernels.utils import get_lora_parameters
    model = _model(scheme, monkeypatch)
    for proj in _lora_layers(model):
        W, _, A, B, s = get_lora_parameters(proj)
        base = proj.get_base_layer()
        expected = base.weight_fake_quantizer.inner(
            qlm.qat_lora_merged_weight(base.weight, A, B, s)
        )
        # The fused kernels add s B A to the weight they are handed: that sum is the merged fake quant.
        torch.testing.assert_close(W + s * (B @ A), expected, rtol = 0, atol = 1e-6)


def _export_gap(model, x):
    """max |trained output - output of the merged, really quantized export| (int8 weight only, CPU)."""
    from torchao.quantization import IntxWeightOnlyConfig, quantize_
    from torchao.quantization.granularity import PerAxis
    from unsloth.save import _merge_lora

    with torch.no_grad():
        trained = model(x)
        plain = _Block().to(x.dtype)
        for proj, target in zip(_lora_layers(model), (plain.up_proj, plain.down_proj)):
            W, _ = _merge_lora(proj, "layer")
            target.weight.copy_(W)
        quantize_(plain, IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)))
        exported = plain(x)
    return (trained - exported).abs().max().item(), trained.abs().max().item()


def test_export_equals_trained_and_base_only_shows_the_gap(monkeypatch):
    x = _x()
    merged_gap, scale = _export_gap(_model("int8", monkeypatch, merged = True), x)
    base_gap, _ = _export_gap(_model("int8", monkeypatch, merged = False), x)
    assert merged_gap <= 1e-5 * scale, merged_gap
    # Flip: the default (base-only) QAT trains Q(W) x + s B A x, which the export does not reproduce.
    assert base_gap > 100 * max(merged_gap, 1e-7), (base_gap, merged_gap)


def test_merge_helper_is_bitwise_the_export_merge(monkeypatch):
    from unsloth.save import _merge_lora

    model = _model("int4", monkeypatch, dtype = torch.bfloat16)
    expected = {}
    for proj in _lora_layers(model):
        base = proj.get_base_layer()
        A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
        expected[id(proj)] = qlm.qat_lora_merged_weight(base.weight, A, B, proj.scaling["default"])
    for proj in _lora_layers(model):
        W, _ = _merge_lora(proj, "layer")
        assert W.dtype == torch.bfloat16
        assert torch.equal(W, expected[id(proj)])


def test_torchao_export_merges_round_to_nearest(monkeypatch):
    """unsloth_zoo's stochastic merge rounding moves merged weights across quantization steps: the torchao
    export of a merged QAT-LoRA model stages its 16-bit merge rounded to nearest, as training merged it."""
    import inspect
    from unsloth import save as unsloth_save

    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "stochastic")
    merged = _model("int4", monkeypatch)
    base_only = _model("int4", monkeypatch, merged = False)
    with qlm.nearest_merge_rounding(merged):
        assert os.environ["UNSLOTH_MERGE_ROUNDING"] == "nearest"
    assert os.environ["UNSLOTH_MERGE_ROUNDING"] == "stochastic"
    with qlm.nearest_merge_rounding(base_only):
        assert os.environ["UNSLOTH_MERGE_ROUNDING"] == "stochastic"
    source = inspect.getsource(unsloth_save._unsloth_save_torchao_with_given_config)
    assert "nearest_merge_rounding(model)" in source


def test_disabled_adapters_and_quantizers_fall_back(monkeypatch):
    merged = _model("int4", monkeypatch)
    base_only = _model("int4", monkeypatch, merged = False)
    x = _x()
    with torch.no_grad():
        with merged.disable_adapter(), base_only.disable_adapter():
            # Not bitwise: base-only QAT rounds its frozen weights in place (_qat_round.py) where it can.
            torch.testing.assert_close(merged(x), base_only(x), rtol = 1e-6, atol = 1e-6)
        # A disabled quantizer (QAD teacher) runs the unquantized weight.
        for m in merged.modules():
            if isinstance(m, qlm.MergedLoRAWeightFakeQuantizer):
                m.enabled = False
        with merged.disable_adapter():
            plain = merged(x)
        up, down = _lora_layers(merged)
        ref = F.linear(
            torch.relu(F.linear(x, up.get_base_layer().weight)), down.get_base_layer().weight
        )
        torch.testing.assert_close(plain, ref, rtol = 1e-6, atol = 1e-6)


def test_activation_quantized_layers_are_bitwise_the_export(monkeypatch):
    """With an activation quantizer the merged layer is exactly F.linear(fq(x), Q(W + s B A)): no 16-bit
    residual rounding, which the activation quantizer of the next layer would amplify."""
    model = _model("fp8-fp8", monkeypatch, dtype = torch.bfloat16)
    x = _x(torch.bfloat16)
    with torch.no_grad():
        for proj in _lora_layers(model):
            base = proj.get_base_layer()
            A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
            q = base.weight_fake_quantizer.inner(
                qlm.qat_lora_merged_weight(base.weight, A, B, proj.scaling["default"])
            )
            h = (
                x
                if proj is _lora_layers(model)[0]
                else torch.randn(6, proj.in_features, dtype = torch.bfloat16)
            )
            assert torch.equal(proj(h), F.linear(base.activation_fake_quantizer(h), q))


def test_shared_recipe_keeps_quantizer_state_per_model(monkeypatch):
    """Two merged models with one recipe: disabling the first one's quantizers (a QAD teacher pass) must not
    switch off fake quantization in the second."""
    first = _model("int4", monkeypatch)
    second = _model("int4", monkeypatch)
    x = _x()
    with torch.no_grad():
        expected = second(x)
        for m in first.modules():
            if isinstance(m, qlm.MergedLoRAWeightFakeQuantizer):
                m.enabled = False
        torch.testing.assert_close(second(x), expected, rtol = 0, atol = 0)
        torch.testing.assert_close(expected, _oracle_forward(second, x), rtol = 1e-5, atol = 1e-5)


def test_registry_quantizer_keeps_no_tensors(monkeypatch):
    """torchao's Intx quantizer stores its last scale / zero point; the shared registry copy must not keep them
    (they would pin one layer's tensors, and break CUDA graph pools that expect every allocation returned)."""
    model = _model("int8", monkeypatch)
    with torch.no_grad():
        model(_x())
    for quantizer in qlm._QUANTIZERS.values():
        assert not isinstance(quantizer.__dict__.get("scale"), torch.Tensor)
        assert not isinstance(quantizer.__dict__.get("zero_point"), torch.Tensor)


@pytest.mark.parametrize("scheme", ["int4", "fp8-fp8"])
def test_disabled_quantizer_keeps_the_adapter_and_mixed_batches_are_refused(scheme, monkeypatch):
    model = _model(scheme, monkeypatch)
    x = _x()
    up = _lora_layers(model)[0]
    quantizer = up.get_base_layer().weight_fake_quantizer
    with torch.no_grad():
        quantizer.enabled = False  # delayed QAT / a teacher pass with the adapter still on
        A, B = up.lora_A["default"].weight, up.lora_B["default"].weight
        base = up.get_base_layer()
        act = base._modules.get("activation_fake_quantizer")
        h = x if act is None else act(x)
        expected = F.linear(h, base.weight) + up.scaling["default"] * F.linear(x, B @ A)
        torch.testing.assert_close(up(x), expected, rtol = 1e-5, atol = 1e-5)
        quantizer.enabled = True
        with pytest.raises(NotImplementedError, match = "mixes adapters"):
            up(x, adapter_names = ["default"] * x.shape[0])


@pytest.mark.parametrize("scheme", ["int4", "fp8-fp8"])
def test_stripped_model_runs_plain_lora(scheme, monkeypatch):
    """save_pretrained_torchao strips fake quantization in place: the live model is plain LoRA afterwards."""
    from unsloth.models._utils import _strip_torchao_fake_quantization

    model = _model(scheme, monkeypatch)
    model._torchao_config = model.base_model.model._torchao_config
    _strip_torchao_fake_quantization(model)
    x = _x()
    up, down = _lora_layers(model)

    def plain(proj, h):
        A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
        return F.linear(h, proj.get_base_layer().weight) + proj.scaling["default"] * F.linear(
            h, B @ A
        )

    with torch.no_grad():
        torch.testing.assert_close(
            model(x), plain(down, torch.relu(plain(up, x))), rtol = 1e-5, atol = 1e-5
        )
        torch.testing.assert_close(
            up(x, adapter_names = ["default"] * x.shape[0]), plain(up, x), rtol = 1e-5, atol = 1e-5
        )


def test_modes_and_defaults(monkeypatch):
    """get_peft_model defaults QAT + LoRA to merged ("auto"); qat_lora_mode = "base" keeps base-only QAT for a
    quantized base served with an unmerged 16-bit adapter. Direct callers of _prepare_model_for_qat keep base."""
    import inspect

    import unsloth.models.llama as llama
    import unsloth.models.vision as vision

    monkeypatch.delenv("UNSLOTH_QAT_LORA_MODE", raising = False)
    wrapped = lambda m: sum(isinstance(q, qlm.MergedLoRAWeightFakeQuantizer) for q in m.modules())
    assert wrapped(_model("int4", monkeypatch, merged = None)) == 0
    monkeypatch.setenv("UNSLOTH_QAT_LORA_MODE", "merged")
    assert wrapped(_model("int4", monkeypatch, merged = None)) == 2
    monkeypatch.delenv("UNSLOTH_QAT_LORA_MODE")
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    auto = _prepare_model_for_qat(_peft(), "int4", lora_mode = None, lora_default = "auto")
    assert wrapped(auto) == 2
    base = _prepare_model_for_qat(_peft(), "int4", lora_mode = "base", lora_default = "auto")
    assert wrapped(base) == 0
    for module in (llama, vision):
        assert 'lora_default = "auto"' in inspect.getsource(module)
    with pytest.raises(ValueError, match = "qat_lora_mode"):
        qlm.resolve_qat_lora_mode("sometimes")


@pytest.mark.parametrize("kwargs", [{"dropout": 0.1}, {"use_dora": True}])
def test_unmergeable_adapters_refuse_or_fall_back(kwargs, monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    with pytest.raises(ValueError, match = "qat_lora_mode"):
        _prepare_model_for_qat(_peft(**kwargs), "int4", lora_mode = "merged")
    # The default ("auto") keeps training: base-only QAT, with a warning.
    with pytest.warns(UserWarning, match = "base-only QAT"):
        model = _prepare_model_for_qat(_peft(**kwargs), "int4", lora_mode = "auto")
    assert not any(isinstance(q, qlm.MergedLoRAWeightFakeQuantizer) for q in model.modules())


def test_round_inplace_skips_merged_layers(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_ROUND_INPLACE", "1")
    model = _model("int4", monkeypatch)
    for proj in _lora_layers(model):
        quantizer = proj.get_base_layer().weight_fake_quantizer
        assert isinstance(quantizer, qlm.MergedLoRAWeightFakeQuantizer)
        assert quantizer.enabled and not quantizer.inner.__dict__.get("_unsloth_qat_rounded", False)


@pytest.mark.parametrize("scheme", ["int4", "fp8-fp8"])
def test_merged_weight_computed_once_per_optimizer_step(scheme, monkeypatch):
    model = _model(scheme, monkeypatch)
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "auto")
    model.train()
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.SGD(params, lr = 0.1)
    x = _x()
    stats = qlm.merged_weight_cache_stats
    computes = stats()["computes"]
    model(x).square().mean().backward()  # before the first optimizer step: nothing cached
    assert stats()["computes"] - computes == 2 and stats()["entries"] == 0
    optimizer.step()
    optimizer.zero_grad()
    computes, hits = stats()["computes"], stats()["hits"]
    for _ in range(3):  # gradient accumulation micro batches: one merge per layer, then served
        model(x).square().mean().backward()
    assert stats()["computes"] - computes == 2
    assert stats()["hits"] - hits == 4
    optimizer.step()  # in-place adapter update: new version, recomputed, never stale
    computes = stats()["computes"]
    with torch.no_grad():
        out = model(x)
        assert stats()["computes"] - computes == 2
        torch.testing.assert_close(out, _oracle_forward(model, x), rtol = 1e-5, atol = 1e-5)
        with model.disable_adapter():  # the base-only output is not served for the merged one
            assert not torch.allclose(model(x), out)
        up = _lora_layers(model)[0]
        up.merge()  # merge / unmerge rewrite W through .data: its cache is dropped
        up.unmerge()
        computes = stats()["computes"]
        model(x)
        assert stats()["computes"] - computes >= 1
    qlm.clear_merged_weight_cache()
    assert stats()["entries"] == 0 and stats()["bytes"] == 0


def test_compiles_without_graph_breaks(monkeypatch):
    model = _model("int4", monkeypatch)
    x = _x()
    explanation = torch._dynamo.explain(model)(x)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    # inductor, not aot_eager: fused into the graph, the merged weight's 16-bit cast is skipped and weights move
    # across quantization steps; the custom op keeps the export's eager numerics.
    torch._dynamo.reset()
    model = _model("int4", monkeypatch, dtype = torch.bfloat16)
    from unsloth.kernels.utils import get_lora_parameters

    def residuals():
        return [get_lora_parameters(proj)[0] for proj in _lora_layers(model)]

    compiled = torch.compile(residuals, fullgraph = True)
    with torch.no_grad():
        for eager, traced in zip(residuals(), compiled()):
            assert torch.equal(eager, traced)


def _sm100():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.skipif(not _sm100(), reason = "NVFP4 / MXFP8 QAT matmuls need compute capability 10.0+")
@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp8"])
def test_microscaling_forward_is_the_merged_low_precision_matmul(scheme, monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    model = _prepare_model_for_qat(_peft(torch.bfloat16).cuda(), scheme, lora_mode = "merged")
    x = _x(torch.bfloat16).cuda().requires_grad_(True)
    up = _lora_layers(model)[0]
    base = up.get_base_layer()
    weight_forward = up.__dict__["_unsloth_qat_lora_weight_forward"]
    A, B = up.lora_A["default"].weight, up.lora_B["default"].weight
    s = up.scaling["default"]
    out = up(x)
    merged = qlm.qat_lora_merged_weight(base.weight, A, B, s)
    ref = weight_forward(base, x, weight = merged)
    assert torch.equal(out, ref)  # bit for bit the exported layer
    grad = torch.randn_like(out)
    gx, gA, gB = torch.autograd.grad(out, [x, A, B], grad)
    x2 = x.detach().clone().requires_grad_(True)
    (rx,) = torch.autograd.grad(weight_forward(base, x2, weight = merged), [x2], grad)
    torch.testing.assert_close(gx, rx, rtol = 2e-2, atol = 2e-2)
    # Straight-through LoRA gradients on the (unquantized) LoRA input.
    xf, gf = x.detach().float(), grad.float()
    torch.testing.assert_close(gB.float(), s * gf.t() @ (xf @ A.float().t()), rtol = 3e-2, atol = 3e-2)
    torch.testing.assert_close(gA.float(), s * (gf @ B.float()).t() @ xf, rtol = 3e-2, atol = 3e-2)


def test_merged_weight_cache_frees_with_the_model(monkeypatch):
    import gc

    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "1")
    model = _model("int4", monkeypatch)
    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "1")
    qlm.clear_merged_weight_cache()
    with torch.no_grad():
        model(_x())
    assert qlm.merged_weight_cache_stats()["entries"] == 2
    del model
    gc.collect()
    assert qlm.merged_weight_cache_stats()["entries"] == 0


@pytest.mark.parametrize("scheme", ["int8", "cactus", "int4", "fp8-fp8", "fp8-int4"])
def test_exported_weights_are_bitwise_the_trained_fake_quant(scheme, monkeypatch):
    """The torchao export merges in 16 bits and really quantizes with the training recipe's config: dequantized,
    its weights are bitwise the merged fake quantized weights training used (same scales and granularity). The
    flip: base-only QAT trained Q(W) + s B A, which is not the exported weight."""
    from torchao.quantization import quantize_
    from unsloth.models._utils import _merged_torchao_config_for_peft_qat
    from unsloth.save import _merge_lora

    model = _model(scheme, monkeypatch, dtype = torch.bfloat16)
    config = _merged_torchao_config_for_peft_qat(model).quant_type
    for proj in _lora_layers(model):
        base = proj.get_base_layer()
        A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
        s = proj.scaling["default"]
        inner = base.weight_fake_quantizer.inner
        trained = inner(qlm.qat_lora_merged_weight(base.weight, A, B, s))
        base_only = inner(base.weight) + s * (B @ A).to(base.weight.dtype)
        W, _ = _merge_lora(proj, "layer")
        plain = torch.nn.Sequential(torch.nn.Linear(W.shape[1], W.shape[0], bias = False)).to(W.dtype)
        with torch.no_grad():
            plain[0].weight.copy_(W)
        try:
            quantize_(plain, config)
            exported = plain[0].weight.dequantize()
        except (ImportError, NotImplementedError, RuntimeError, AssertionError) as exc:
            pytest.skip(f"{scheme} real quantization unavailable here: {exc}")
        assert torch.equal(exported.to(trained.dtype), trained), (
            (exported != trained).float().mean()
        )
        assert not torch.equal(exported.to(trained.dtype), base_only)


@pytest.mark.skipif(not _sm100(), reason = "NVFP4 / MXFP8 QAT matmuls need compute capability 10.0+")
@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp8"])
def test_microscaling_exported_weights_are_bitwise_the_trained_ones(scheme, monkeypatch):
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
    from torchao.quantization import quantize_
    from unsloth.models._utils import _merged_torchao_config_for_peft_qat
    from unsloth.save import _merge_lora

    monkeypatch.setenv("UNSLOTH_QAT_WEIGHT_MEMO", "0")
    model = _prepare_model_for_qat(_peft(torch.bfloat16).cuda(), scheme, lora_mode = "merged")
    config = _merged_torchao_config_for_peft_qat(model).quant_type
    for proj in _lora_layers(model):
        base = proj.get_base_layer()
        A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
        merged = qlm.qat_lora_merged_weight(base.weight, A, B, proj.scaling["default"])
        wc = base.weight_config

        def quantized(
            w,
        ):  # what the QAT forward quantizes the weight to (models/_utils.py, torchao QAT)
            if scheme == "nvfp4":
                scale = per_tensor_amax_to_scale(w.abs().max()) if wc.use_per_tensor_scale else None
                return NVFP4Tensor.to_nvfp4(w, per_tensor_scale = scale).dequantize(w.dtype)
            return MXTensor.to_mx(
                w, elem_dtype = wc.dtype, block_size = wc.block_size, scaling_mode = wc.scaling_mode
            ).dequantize(w.dtype)

        W, _ = _merge_lora(proj, "layer")
        plain = torch.nn.Sequential(torch.nn.Linear(W.shape[1], W.shape[0], bias = False)).to(W)
        with torch.no_grad():
            plain[0].weight.copy_(W)
        quantize_(plain, config)
        exported = plain[0].weight.dequantize(W.dtype)
        assert torch.equal(exported, quantized(merged))
        assert not torch.equal(exported, quantized(base.weight.detach()))
