# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models

---

## 🚀 Intel XPU Optimization (New!)

This fork includes **high-performance dequantization kernels optimized for Intel XPU**. The optimizations provide significant speedups compared to the original PyTorch implementation.

> **Acknowledgement**: This project is based on [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF). We sincerely thank the original author for their excellent work.

> **Part of LLM-Scaler**: This optimized fork is developed as part of [Intel LLM-Scaler](https://github.com/intel/llm-scaler.git) project. For a complete out-of-the-box Docker image solution with all dependencies pre-configured, please refer to **LLM-Scaler Omni** image.

### Supported Quantization Formats

| Format | ESIMD (XPU) | Triton | PyTorch (from city96/ComfyUI-GGUF) |
|--------|-------------|-------------------|-------------------|
| Q4_0 | ✅ | ✅ | ✅ |
| Q8_0 | ✅ | ✅ | ✅ |
| Q4_K | ✅ | ❌ | ✅ |
| Q6_K | ✅ | ❌ | ✅ |
| Q4_1 | ❌ | ✅ | ✅ |
| Q5_0/Q5_1 | ❌ | ❌ | ✅ |
| Q3_K/Q5_K | ❌ | ❌ | ✅ |

### Gallery

<!-- Add generated samples here -->

*Coming soon...*

### Verified Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.9.0+xpu |
| Triton | 3.5.0 |
| pytorch-triton-xpu | 3.5.0 |
| Intel oneAPI | 2025.1.3 |

### Dependencies for Intel XPU

```bash
# PyTorch with XPU support
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/xpu

# Required for ESIMD kernels: omni_xpu_kernel (from LLM-Scaler)
git clone https://github.com/intel/llm-scaler.git
cd llm-scaler/omni/omni_xpu_kernel
pip install . --no-build-isolation

# Optional: Triton for Intel XPU
pip install triton==3.5.0
pip install pytorch-triton-xpu==3.5.0 --index-url https://download.pytorch.org/whl/xpu
```

### Backend Selection

The kernel backend is automatically selected based on availability. You can also force a specific backend using environment variables:

```bash
# Auto-select best available (default)
export COMFYUI_GGUF_BACKEND=auto

# Force ESIMD kernels (best performance on Intel XPU)
export COMFYUI_GGUF_BACKEND=esimd

# Force Triton kernels
export COMFYUI_GGUF_BACKEND=triton

# Force PyTorch fallback
export COMFYUI_GGUF_BACKEND=pytorch

# Enable debug logging for kernel selection
export COMFYUI_GGUF_DEBUG=1
```

### Running on Intel XPU

1. **Set up Intel oneAPI environment** (required for ESIMD kernels):
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

2. **Start ComfyUI normally** - the optimized kernels will be used automatically when running on Intel XPU.

3. **Verify kernel selection** by checking the logs:
   ```
   ComfyUI-GGUF Kernel Configuration
   ============================================================
     Environment: COMFYUI_GGUF_BACKEND=auto
     Available backends:
       - ESIMD:   Yes (enabled: True)
       - Triton:  Yes (enabled: True)
       - PyTorch: Yes (fallback)
     Active kernels (on XPU):
       - Q4_0: ESIMD
       - Q8_0: ESIMD
       - Q4_K: ESIMD
       - Q6_K: ESIMD
       - Q4_1: Triton
   ============================================================
   ```

### Testing the Installation

Run the integration test to verify everything is working:

```bash
cd ComfyUI/custom_nodes/ComfyUI-GGUF
python test_omni_integration.py
```

Or run the benchmark to measure performance:

```bash
python bench_comfyui.py
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
