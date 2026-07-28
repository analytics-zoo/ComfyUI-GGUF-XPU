# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models

---

## Intel XPU optimization

This fork integrates Intel XPU GGUF dequantization through
[Comfy Kitchen XPU](https://github.com/xiangyuT/comfy-kitchen-xpu). The custom
node owns GGUF tensor loading and logical shapes; Comfy Kitchen owns backend
selection, native-kernel dispatch, failure quarantine, and portable eager
fallback. The node no longer imports Omni XPU Kernel directly.

> **Acknowledgement**: This project is based on
> [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). We thank City96
> and upstream contributors for the GGUF loader and reference dequantization
> implementation.

The integrated environment is delivered through
[Intel LLM-Scaler](https://github.com/intel/llm-scaler). LLM-Scaler pins
compatible Comfy Kitchen and Omni XPU Kernel revisions in its Omni image.

### Supported Quantization Formats

| Format | Comfy Kitchen XPU/eager | Legacy Triton | Plugin PyTorch |
|--------|-------------------------|---------------|----------------|
| Q4_0 | Yes | Yes | Yes |
| Q8_0 | Yes | Yes | Yes |
| Q4_K | Yes | No | Yes |
| Q6_K | Yes | No | Yes |
| Q4_1 | No | Yes | Yes |
| Other upstream formats | No | No | Yes |

Q4_0, Q8_0, Q4_K, and Q6_K use the managed Kitchen route on Intel XPU.
Q4_1 remains on the existing plugin Triton/PyTorch path. Other formats keep the
upstream PyTorch or NumPy fallback.

### XPU dependencies

```bash
# Install the custom-node dependencies.
python -m pip install -r requirements.txt

# Install compatible Comfy Kitchen and Omni XPU Kernel builds supplied by the
# target LLM-Scaler Omni image or by the corresponding source checkouts.
```

The node still loads without Comfy Kitchen and preserves its PyTorch/Triton
fallbacks. The optimized XPU route requires a Comfy Kitchen build that exposes
`dequantize_gguf`.

### Backend selection

The default is `auto`: supported GGUF formats use Comfy Kitchen on XPU, while
other devices preserve the existing plugin paths.

```bash
# Managed XPU selection (default)
export COMFYUI_GGUF_BACKEND=auto

# Let Kitchen choose, or force a Kitchen backend
export COMFYUI_GGUF_BACKEND=kitchen
export COMFYUI_GGUF_BACKEND=xpu
export COMFYUI_GGUF_BACKEND=eager

# Keep the existing plugin Triton route
export COMFYUI_GGUF_BACKEND=triton

# Log configured routing
export COMFYUI_GGUF_DEBUG=1
```

`esimd` and `pytorch` are retained as compatibility aliases for `xpu` and
`eager`. Startup logs describe configured routing. For routes that actually
completed, inspect Kitchen diagnostics:

```python
from comfy_kitchen import get_gguf_route_diagnostics

print(get_gguf_route_diagnostics())
```

### Integration test

```bash
PYTHONPATH=/path/to/comfy-kitchen-xpu \
  python -m pytest -q tests/test_kitchen_gguf.py
```

---

## Original Documentation

This is currently very much WIP. These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/70d16d97-c522-4ef4-9435-633f128644c8)

Note: The "Force/Set CLIP Device" is **NOT** part of this node pack. Do not install it if you only have one GPU. Do not set it to cuda:0 then complain about OOM errors if you do not undestand what it is for. There is not need to copy the workflow above, just use your own workflow and replace the stock "Load Diffusion Model" with the "Unet Loader (GGUF)" node.

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/city96/ComfyUI-GGUF
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/city96/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.

LoRA loading is experimental but it should work with just the built-in LoRA loader node(s).

Pre-quantized models:

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)

Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.

- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)

See the instructions in the [tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) folder for how to create your own quants.
