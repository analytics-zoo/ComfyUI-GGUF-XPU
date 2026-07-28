import importlib.util
import os
import sys
from pathlib import Path

import gguf
import pytest
import torch


COMFYUI_ROOT = os.environ.get("COMFYUI_ROOT")
pytestmark = pytest.mark.skipif(
    not COMFYUI_ROOT,
    reason="COMFYUI_ROOT is required for custom-node object integration tests",
)

CASES = (
    (gguf.GGMLQuantizationType.Q4_0, "q4_0", 18, 32),
    (gguf.GGMLQuantizationType.Q4_1, "q4_1", 20, 32),
    (gguf.GGMLQuantizationType.Q8_0, "q8_0", 34, 32),
    (gguf.GGMLQuantizationType.Q4_K, "q4_k", 144, 256),
    (gguf.GGMLQuantizationType.Q6_K, "q6_k", 210, 256),
)


def _load_custom_node():
    root = Path(__file__).resolve().parents[1]
    comfyui_root = Path(COMFYUI_ROOT).resolve()
    sys.path.insert(0, str(comfyui_root))
    spec = importlib.util.spec_from_file_location(
        "comfyui_gguf_xpu_under_test",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load custom node from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def custom_node():
    return _load_custom_node()


def _fp16_bytes(value):
    return torch.tensor([value], dtype=torch.float16).view(torch.uint8)


def _packed_blocks(block_bytes, *, count=4):
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
    return blocks


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="Intel XPU unavailable",
)
@pytest.mark.parametrize("qtype,quant_name,block_bytes,block_size", CASES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_ggml_tensor_lazy_dequant_uses_kitchen_xpu(
    custom_node,
    qtype,
    quant_name,
    block_bytes,
    block_size,
    dtype,
):
    blocks = _packed_blocks(block_bytes).to("xpu")
    tensor = custom_node.ops.GGMLTensor(
        blocks,
        tensor_type=qtype,
        tensor_shape=torch.Size((blocks.shape[0], block_size)),
    )

    with custom_node.dequant.comfy_kitchen.use_backend("eager"):
        expected = custom_node.dequant.comfy_kitchen.dequantize_gguf(
            blocks.reshape(-1),
            quant_name,
            output_dtype=dtype,
        ).reshape(tensor.tensor_shape)
    custom_node.dequant.comfy_kitchen.get_gguf_route_diagnostics(reset=True)
    actual = custom_node.dequant.dequantize_tensor(tensor, dtype=dtype)
    torch.xpu.synchronize()

    tolerance = 2 * torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=0)
    diagnostics = (
        custom_node.dequant.comfy_kitchen.get_gguf_route_diagnostics()
    )
    assert diagnostics == {"routes": {"xpu": 1}, "fallbacks": {}}
