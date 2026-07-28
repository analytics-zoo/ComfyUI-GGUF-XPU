import inspect
import json
import os
import subprocess
import sys
from contextlib import nullcontext

import gguf
import pytest
import torch

import dequant


CASES = (
    (gguf.GGMLQuantizationType.Q4_0, "q4_0", 18, 32),
    (gguf.GGMLQuantizationType.Q8_0, "q8_0", 34, 32),
    (gguf.GGMLQuantizationType.Q4_K, "q4_k", 144, 256),
    (gguf.GGMLQuantizationType.Q6_K, "q6_k", 210, 256),
)


def _fp16_bytes(value):
    return torch.tensor([value], dtype=torch.float16).view(torch.uint8)


def _packed_blocks(block_bytes, *, count=3, device="cpu"):
    generator = torch.Generator().manual_seed(20260728)
    blocks = torch.randint(
        0,
        256,
        (count, block_bytes),
        dtype=torch.uint8,
        generator=generator,
    )
    if block_bytes in (18, 34):
        blocks[:, :2] = _fp16_bytes(0.5)
    elif block_bytes == 20:
        blocks[:, :2] = _fp16_bytes(0.5)
        blocks[:, 2:4] = _fp16_bytes(0.25)
    elif block_bytes == 144:
        blocks[:, :2] = _fp16_bytes(0.5)
        blocks[:, 2:4] = _fp16_bytes(0.25)
    else:
        blocks[:, -2:] = _fp16_bytes(0.5)
    return blocks.to(device)


@pytest.fixture
def kitchen_route():
    original = (
        dequant.BACKEND_ENV,
        dequant.USE_KITCHEN_KERNELS,
        dequant.USE_TRITON_KERNELS,
    )
    dequant.BACKEND_ENV = "eager"
    dequant.USE_KITCHEN_KERNELS = dequant.HAS_KITCHEN
    dequant.USE_TRITON_KERNELS = False
    try:
        yield
    finally:
        (
            dequant.BACKEND_ENV,
            dequant.USE_KITCHEN_KERNELS,
            dequant.USE_TRITON_KERNELS,
        ) = original


def test_plugin_does_not_import_omni_directly():
    source = inspect.getsource(dequant)
    assert "from omni_xpu_kernel" not in source
    assert "import omni_xpu_kernel" not in source


def test_plugin_still_imports_when_kitchen_is_unavailable():
    script = """
import json
import sys
sys.modules["comfy_kitchen"] = None
import dequant
print(json.dumps({
    "has_kitchen": dequant.HAS_KITCHEN,
    "uses_kitchen": dequant.USE_KITCHEN_KERNELS,
    "has_pytorch": dequant.get_kernel_info()["backends"]["PyTorch"]["available"],
}))
"""
    env = os.environ.copy()
    env["COMFYUI_GGUF_BACKEND"] = "auto"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    status = json.loads(completed.stdout)
    assert status == {
        "has_kitchen": False,
        "uses_kitchen": False,
        "has_pytorch": True,
    }


@pytest.mark.skipif(not dequant.HAS_KITCHEN, reason="Comfy Kitchen GGUF API unavailable")
@pytest.mark.parametrize("qtype,quant_name,block_bytes,block_size", CASES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_plugin_kitchen_eager_matches_existing_reference(
    kitchen_route,
    qtype,
    quant_name,
    block_bytes,
    block_size,
    dtype,
):
    blocks = _packed_blocks(block_bytes)
    expected = dequant.dequantize_functions[qtype](
        blocks,
        block_size,
        block_bytes,
        dtype,
    ).reshape(-1)

    dequant.comfy_kitchen.get_gguf_route_diagnostics(reset=True)
    actual = dequant.dequantize(
        blocks,
        qtype,
        (blocks.shape[0] * block_size,),
        dtype=dtype,
    )

    assert torch.equal(actual, expected)
    assert dequant.comfy_kitchen.get_gguf_route_diagnostics()["routes"] == {"eager": 1}
    assert dequant._KITCHEN_QTYPES[qtype] == quant_name


def test_q4_1_keeps_plugin_fallback(kitchen_route):
    qtype = gguf.GGMLQuantizationType.Q4_1
    block_size, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
    blocks = _packed_blocks(block_bytes)
    expected = dequant.dequantize_functions[qtype](
        blocks,
        block_size,
        block_bytes,
        torch.float16,
    ).reshape(-1)

    actual = dequant.dequantize(
        blocks,
        qtype,
        (blocks.shape[0] * block_size,),
        dtype=torch.float16,
    )

    assert torch.equal(actual, expected)


def test_unsupported_kitchen_output_dtype_keeps_plugin_fallback(kitchen_route, monkeypatch):
    qtype = gguf.GGMLQuantizationType.Q4_0
    block_size, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
    blocks = _packed_blocks(block_bytes)
    expected = dequant.dequantize_functions[qtype](
        blocks,
        block_size,
        block_bytes,
        torch.float32,
    ).reshape(-1)

    def unexpected_kitchen_call(*args, **kwargs):
        raise AssertionError("float32 should stay on the plugin fallback")

    if dequant.HAS_KITCHEN:
        monkeypatch.setattr(
            dequant.comfy_kitchen,
            "dequantize_gguf",
            unexpected_kitchen_call,
        )
    actual = dequant.dequantize(
        blocks,
        qtype,
        (blocks.shape[0] * block_size,),
        dtype=torch.float32,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="Intel XPU unavailable",
)
@pytest.mark.skipif(not dequant.HAS_KITCHEN, reason="Comfy Kitchen GGUF API unavailable")
@pytest.mark.parametrize("qtype,quant_name,block_bytes,block_size", CASES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_auto_xpu_records_completed_kitchen_route(
    qtype,
    quant_name,
    block_bytes,
    block_size,
    dtype,
):
    blocks = _packed_blocks(block_bytes, device="xpu")
    previous = (
        dequant.BACKEND_ENV,
        dequant.USE_KITCHEN_KERNELS,
        dequant.USE_TRITON_KERNELS,
    )
    dequant.BACKEND_ENV = "auto"
    dequant.USE_KITCHEN_KERNELS = True
    dequant.USE_TRITON_KERNELS = True
    try:
        with dequant.comfy_kitchen.use_backend("eager"):
            expected = dequant.comfy_kitchen.dequantize_gguf(
                blocks.reshape(-1),
                quant_name,
                output_dtype=dtype,
            )
        dequant.comfy_kitchen.get_gguf_route_diagnostics(reset=True)
        actual = dequant.dequantize(
            blocks,
            qtype,
            (blocks.shape[0] * block_size,),
            dtype=dtype,
        )
        torch.xpu.synchronize()
    finally:
        (
            dequant.BACKEND_ENV,
            dequant.USE_KITCHEN_KERNELS,
            dequant.USE_TRITON_KERNELS,
        ) = previous

    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=0)
    assert dequant.comfy_kitchen.get_gguf_route_diagnostics()["routes"] == {"xpu": 1}


@pytest.mark.skipif(not dequant.HAS_KITCHEN, reason="Comfy Kitchen GGUF API unavailable")
def test_backend_context_is_only_used_for_explicit_kitchen_override(monkeypatch):
    calls = []

    def record_backend(name):
        calls.append(name)
        return nullcontext()

    monkeypatch.setattr(dequant.comfy_kitchen, "use_backend", record_backend)
    blocks = _packed_blocks(18)
    qtype = gguf.GGMLQuantizationType.Q4_0

    previous = dequant.BACKEND_ENV
    try:
        dequant.BACKEND_ENV = "auto"
        dequant._dequantize_kitchen(blocks, qtype, 32)
        dequant.BACKEND_ENV = "xpu"
        dequant._dequantize_kitchen(blocks, qtype, 32)
    finally:
        dequant.BACKEND_ENV = previous

    assert calls == ["xpu"]
