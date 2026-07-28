# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import logging
import os
import platform
from contextlib import nullcontext

import gguf
import torch
from tqdm import tqdm

# ============================================================================
# Environment Variable Configuration
# ============================================================================
# COMFYUI_GGUF_BACKEND: Select a managed or legacy backend.
#   - "auto"    : Use Comfy Kitchen on XPU, otherwise preserve upstream routes
#   - "kitchen" : Let Comfy Kitchen select its backend
#   - "xpu"     : Force Comfy Kitchen's XPU backend
#   - "eager"   : Force Comfy Kitchen's portable PyTorch backend
#   - "triton"  : Use Triton kernels (requires triton package)
#   - "esimd" and "pytorch" remain compatibility aliases for "xpu" and "eager"
#
# COMFYUI_GGUF_DEBUG: Enable kernel selection debug logging (0/1)
# ============================================================================
_BACKEND_ALIASES = {"esimd": "xpu", "pytorch": "eager"}
_VALID_BACKENDS = {"auto", "kitchen", "xpu", "eager", "triton"}
BACKEND_ENV_RAW = os.environ.get("COMFYUI_GGUF_BACKEND", "auto").strip().lower()
BACKEND_ENV = _BACKEND_ALIASES.get(BACKEND_ENV_RAW, BACKEND_ENV_RAW)
if BACKEND_ENV_RAW in _BACKEND_ALIASES:
    logging.warning(
        "ComfyUI-GGUF: COMFYUI_GGUF_BACKEND=%s is deprecated; use %s",
        BACKEND_ENV_RAW,
        BACKEND_ENV,
    )
if BACKEND_ENV not in _VALID_BACKENDS:
    logging.warning(
        "ComfyUI-GGUF: unknown COMFYUI_GGUF_BACKEND=%r; using auto",
        BACKEND_ENV_RAW,
    )
    BACKEND_ENV = "auto"
DEBUG_KERNEL_SELECTION = os.environ.get("COMFYUI_GGUF_DEBUG", "0") == "1"

# ============================================================================
# Triton Kernel Support
# ============================================================================
if platform.system() == "Windows":
    HAS_TRITON = False
    logging.info("ComfyUI-GGUF: Triton not supported on Windows, using PyTorch fallback")
else:
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
        logging.info("ComfyUI-GGUF: Triton available")
    except ImportError:
        HAS_TRITON = False
        logging.info("ComfyUI-GGUF: Triton not available")

# ============================================================================
# Comfy Kitchen managed GGUF support
#
# ComfyUI-GGUF owns GGUF tensor semantics and logical shape. Comfy Kitchen owns
# backend selection, native Omni dispatch, failure quarantine and eager fallback.
# ============================================================================
comfy_kitchen = None
_KITCHEN_IMPORT_ERROR = None
try:
    import comfy_kitchen as _comfy_kitchen

    comfy_kitchen = _comfy_kitchen
    HAS_KITCHEN = all(
        hasattr(comfy_kitchen, name)
        for name in ("dequantize_gguf", "get_gguf_route_diagnostics", "use_backend")
    )
    if not HAS_KITCHEN:
        _KITCHEN_IMPORT_ERROR = "installed package does not expose the GGUF API"
except Exception as exc:
    HAS_KITCHEN = False
    _KITCHEN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

if HAS_KITCHEN:
    logging.info("ComfyUI-GGUF: Comfy Kitchen GGUF routing available")
else:
    logging.info("ComfyUI-GGUF: Comfy Kitchen GGUF routing unavailable (%s)", _KITCHEN_IMPORT_ERROR)

_KITCHEN_QUANT_NAMES = frozenset()
if HAS_KITCHEN:
    try:
        from comfy_kitchen.gguf import GGUF_QUANT_TYPE_TO_CODE as _KITCHEN_QUANT_CODES

        _KITCHEN_QUANT_NAMES = frozenset(_KITCHEN_QUANT_CODES)
    except (ImportError, AttributeError):
        _KITCHEN_QUANT_NAMES = frozenset({"q4_0", "q8_0", "q4_k", "q6_k"})

# ============================================================================
# Backend Selection Logic (based on environment variable)
# ============================================================================
USE_TORCH_COMPILE = False   # Disable torch.compile if using Triton

if BACKEND_ENV in {"kitchen", "xpu", "eager"}:
    USE_KITCHEN_KERNELS = HAS_KITCHEN
    USE_TRITON_KERNELS = False
    if not HAS_KITCHEN:
        logging.warning(
            "ComfyUI-GGUF: %s backend requested but Comfy Kitchen GGUF support is unavailable",
            BACKEND_ENV_RAW,
        )
elif BACKEND_ENV == "triton":
    USE_KITCHEN_KERNELS = False
    USE_TRITON_KERNELS = True
    if not HAS_TRITON:
        logging.warning("ComfyUI-GGUF: Triton backend requested but triton not available!")
else:
    USE_KITCHEN_KERNELS = HAS_KITCHEN
    USE_TRITON_KERNELS = True

# ============================================================================
# Log Current Backend Selection
# ============================================================================
def _get_active_backend():
    """Describe configured routing; completed calls are reported by Kitchen diagnostics."""
    backends = {}

    for qtype in ["Q4_0", "Q4_1", "Q8_0", "Q4_K", "Q6_K"]:
        kitchen_name = qtype.lower()
        kitchen_supported = kitchen_name in _KITCHEN_QUANT_NAMES
        if HAS_KITCHEN and USE_KITCHEN_KERNELS and kitchen_supported:
            override = BACKEND_ENV if BACKEND_ENV in {"xpu", "eager"} else "managed XPU"
            backends[qtype] = f"Kitchen ({override})"
        elif HAS_TRITON and USE_TRITON_KERNELS and qtype in ["Q4_0", "Q8_0", "Q4_1"]:
            backends[qtype] = "Triton"
        else:
            backends[qtype] = "PyTorch"
    
    return backends

_active_backends = _get_active_backend()

# Print backend selection summary
logging.info("=" * 60)
logging.info("ComfyUI-GGUF Kernel Configuration")
logging.info("=" * 60)
logging.info("  Environment: COMFYUI_GGUF_BACKEND=%s", BACKEND_ENV_RAW)
logging.info("  Available backends:")
logging.info("    - Kitchen: %s (enabled: %s)", "Yes" if HAS_KITCHEN else "No", USE_KITCHEN_KERNELS)
logging.info("    - Triton:  %s (enabled: %s)", "Yes" if HAS_TRITON else "No", USE_TRITON_KERNELS)
logging.info("    - PyTorch: Yes (fallback)")
logging.info("  Configured routes:")
for qtype, backend in _active_backends.items():
    logging.info("    - %s: %s", qtype, backend)
logging.info("=" * 60)


def get_kernel_info():
    """
    Get information about available kernels and which will be used.
    
    Returns a dict with backend availability and selection priority.
    
    Usage:
        from dequant import get_kernel_info
        info = get_kernel_info()
        print(info)
    """
    info = {
        "backends": {
            "Kitchen": {"available": HAS_KITCHEN, "enabled": USE_KITCHEN_KERNELS},
            "Triton": {"available": HAS_TRITON, "enabled": USE_TRITON_KERNELS},
            "PyTorch": {"available": True, "enabled": True},
        },
        "selection_priority": [],
        "Q4_0_kernel": None,
        "Q8_0_kernel": None,
        "Q4_K_kernel": None,
        "Q6_K_kernel": None,
        "Q4_1_kernel": None,
    }
    
    if HAS_KITCHEN and USE_KITCHEN_KERNELS:
        route = f"Kitchen ({BACKEND_ENV if BACKEND_ENV != 'auto' else 'managed'})"
        for qtype in ("Q4_0", "Q4_1", "Q8_0", "Q4_K", "Q6_K"):
            if qtype.lower() in _KITCHEN_QUANT_NAMES:
                info[f"{qtype}_kernel"] = route
        info["selection_priority"].append(
            "Comfy Kitchen (supported GGUF formats; managed XPU/eager routing)"
        )
    elif HAS_TRITON and USE_TRITON_KERNELS:
        info["Q4_0_kernel"] = "Triton"
    else:
        info["Q4_0_kernel"] = "PyTorch"
    
    # Determine Q8_0/Q4_1 kernel (Triton fallback)
    if HAS_TRITON and USE_TRITON_KERNELS:
        if info["Q8_0_kernel"] is None:
            info["Q8_0_kernel"] = "Triton"
        if info["Q4_1_kernel"] is None:
            info["Q4_1_kernel"] = "Triton"
        info["selection_priority"].append("Triton (Q8_0, Q4_1 on XPU/CUDA)")
    else:
        if info["Q8_0_kernel"] is None:
            info["Q8_0_kernel"] = "PyTorch"
        info["Q4_1_kernel"] = "PyTorch"
    
    # PyTorch fallback for K-quants if Kitchen is unavailable or disabled
    if info["Q4_K_kernel"] is None:
        info["Q4_K_kernel"] = "PyTorch"
    if info["Q6_K_kernel"] is None:
        info["Q6_K_kernel"] = "PyTorch"
    
    info["selection_priority"].append("PyTorch (fallback)")

    if HAS_KITCHEN:
        info["kitchen_diagnostics"] = comfy_kitchen.get_gguf_route_diagnostics()

    return info


# ============================================================================
# Triton Kernels for Q4_0 and Q8_0
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _dequant_q4_0_kernel(
        data_ptr,        # Input: quantized data [n_blocks * 18]
        output_ptr,      # Output: dequantized data [n_blocks * 32]
        n_elements,      # Total output elements (n_blocks * 32)
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Q4_0 Triton kernel v5 - optimized with tl.where for nibble extraction
        
        PyTorch unpacking order (via reshape and shift):
        - positions 0-15: low nibbles of bytes 0-15
        - positions 16-31: high nibbles of bytes 0-15
        
        This kernel achieves ~11x speedup over PyTorch on Intel XPU.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Calculate which Q4_0 block and position within block
        block_idx = offsets // 32
        pos_in_block = offsets % 32
        
        # Optimized byte_idx and nibble selection
        # pos 0-15: byte_idx = pos, use low nibble
        # pos 16-31: byte_idx = pos - 16, use high nibble
        byte_idx = pos_in_block & 15  # Same as % 16 but faster
        is_high = pos_in_block >> 4   # Same as // 16, gives 0 or 1
        
        # Data layout: [scale_lo, scale_hi, data0, ..., data15] per block
        data_base = block_idx * 18
        
        # Load scale (2 bytes as little-endian float16)
        scale_lo = tl.load(data_ptr + data_base, mask=mask, other=0).to(tl.uint16)
        scale_hi = tl.load(data_ptr + data_base + 1, mask=mask, other=0).to(tl.uint16)
        scale = (scale_lo | (scale_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        
        # Load data byte and extract nibble
        data_byte = tl.load(data_ptr + data_base + 2 + byte_idx, mask=mask, other=0).to(tl.int32)
        
        # Use conditional select instead of variable shift - key optimization!
        low_nibble = data_byte & 0x0F
        high_nibble = (data_byte >> 4) & 0x0F
        nibble = tl.where(is_high == 1, high_nibble, low_nibble) - 8
        
        # Dequantize: result = scale * nibble
        result = (scale * nibble.to(tl.float32)).to(tl.float16)
        
        tl.store(output_ptr + offsets, result, mask=mask)

    @triton.jit
    def _dequant_q8_0_kernel(
        data_ptr,
        output_ptr, 
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Q8_0 Triton kernel - simpler than Q4_0 since no nibble unpacking
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Calculate block and position
        block_idx = offsets // 32
        pos_in_block = offsets % 32
        
        # Data layout: [scale_lo, scale_hi, data0, ..., data31]
        data_base = block_idx * 34
        
        # Load scale
        scale_lo = tl.load(data_ptr + data_base, mask=mask, other=0).to(tl.uint16)
        scale_hi = tl.load(data_ptr + data_base + 1, mask=mask, other=0).to(tl.uint16)
        scale_bits = scale_lo | (scale_hi << 8)
        scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        
        # Load int8 value (stored as uint8)
        uint8_val = tl.load(data_ptr + data_base + 2 + pos_in_block, mask=mask, other=0)
        # Convert uint8 to signed int8
        signed_val = tl.where(uint8_val > 127, uint8_val.to(tl.int32) - 256, uint8_val.to(tl.int32))
        
        # Dequantize
        result = (scale * signed_val.to(tl.float32)).to(tl.float16)
        
        tl.store(output_ptr + offsets, result, mask=mask)

    @triton.jit
    def _dequant_q4_1_kernel(
        data_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Q4_1 Triton kernel - optimized with same v5 pattern
        Format: 2 bytes scale + 2 bytes min + 16 bytes data
        Dequant: scale * nibble + min
        
        Output order matches PyTorch:
        - positions 0-15: low nibbles
        - positions 16-31: high nibbles
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        block_idx = offsets // 32
        pos_in_block = offsets % 32
        
        # Optimized byte_idx and nibble selection
        byte_idx = pos_in_block & 15  # Same as % 16
        is_high = pos_in_block >> 4   # Same as // 16
        
        # Data layout: [scale_lo, scale_hi, min_lo, min_hi, data0, ..., data15]
        data_base = block_idx * 20  # Q4_1 is 20 bytes per block
        
        # Load scale
        scale_lo = tl.load(data_ptr + data_base, mask=mask, other=0).to(tl.uint16)
        scale_hi = tl.load(data_ptr + data_base + 1, mask=mask, other=0).to(tl.uint16)
        scale = (scale_lo | (scale_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        
        # Load min
        min_lo = tl.load(data_ptr + data_base + 2, mask=mask, other=0).to(tl.uint16)
        min_hi = tl.load(data_ptr + data_base + 3, mask=mask, other=0).to(tl.uint16)
        min_val = (min_lo | (min_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        
        # Load data byte and extract nibble
        data_byte = tl.load(data_ptr + data_base + 4 + byte_idx, mask=mask, other=0).to(tl.int32)
        
        # Use conditional select - no offset subtraction for Q4_1
        low_nibble = data_byte & 0x0F
        high_nibble = (data_byte >> 4) & 0x0F
        nibble = tl.where(is_high == 1, high_nibble, low_nibble).to(tl.float32)
        
        # Dequantize: result = scale * nibble + min
        result = (scale * nibble + min_val).to(tl.float16)
        
        tl.store(output_ptr + offsets, result, mask=mask)


# ============================================================================
# Triton Wrapper Functions
# ============================================================================

def _dequantize_q4_0_triton(blocks, block_size, type_size, dtype=None):
    """Triton-accelerated Q4_0 dequantization"""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    data_flat = blocks.view(-1).contiguous()
    output = torch.empty(n_blocks * 32, dtype=torch.float16, device=device)
    
    n_elements = n_blocks * 32
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _dequant_q4_0_kernel[grid](
        data_flat, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    result = output.view(n_blocks, 32)
    if dtype is not None and dtype != torch.float16:
        result = result.to(dtype)
    return result


def _dequantize_q8_0_triton(blocks, block_size, type_size, dtype=None):
    """Triton-accelerated Q8_0 dequantization"""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    data_flat = blocks.view(-1).contiguous()
    output = torch.empty(n_blocks * 32, dtype=torch.float16, device=device)
    
    n_elements = n_blocks * 32
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _dequant_q8_0_kernel[grid](
        data_flat, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    result = output.view(n_blocks, 32)
    if dtype is not None and dtype != torch.float16:
        result = result.to(dtype)
    return result


def _dequantize_q4_1_triton(blocks, block_size, type_size, dtype=None):
    """Triton-accelerated Q4_1 dequantization"""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    data_flat = blocks.view(-1).contiguous()
    output = torch.empty(n_blocks * 32, dtype=torch.float16, device=device)
    
    n_elements = n_blocks * 32
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    _dequant_q4_1_kernel[grid](
        data_flat, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    result = output.view(n_blocks, 32)
    if dtype is not None and dtype != torch.float16:
        result = result.to(dtype)
    return result


# torch.compile cache (fallback when Triton not available)
_compile_cache = {}

def get_compiled_dequant(func, device_type):
    """Get or create compiled version of dequantization function"""
    cache_key = (func.__name__, device_type)
    if cache_key not in _compile_cache:
        try:
            _compile_cache[cache_key] = torch.compile(
                func, 
                mode="reduce-overhead",
                fullgraph=False,
            )
            logging.info(f"ComfyUI-GGUF: Compiled {func.__name__} for {device_type}")
        except Exception as e:
            logging.warning(f"ComfyUI-GGUF: Failed to compile {func.__name__}: {e}")
            _compile_cache[cache_key] = func
    return _compile_cache[cache_key]


TORCH_COMPATIBLE_QTYPES = (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES

def is_quantized(tensor):
    return not is_torch_compatible(tensor)

@torch.compiler.disable()
def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        # this is incredibly slow
        tqdm.write(f"Falling back to numpy dequant for qtype: {getattr(qtype, 'name', repr(qtype))}")
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)


# ============================================================================
# Comfy Kitchen GGUF adapter
# ============================================================================
_KITCHEN_QTYPES = {
    gguf.GGMLQuantizationType.Q4_0: "q4_0",
    gguf.GGMLQuantizationType.Q8_0: "q8_0",
    gguf.GGMLQuantizationType.Q4_K: "q4_k",
    gguf.GGMLQuantizationType.Q6_K: "q6_k",
}

if "q4_1" in _KITCHEN_QUANT_NAMES:
    _KITCHEN_QTYPES[gguf.GGMLQuantizationType.Q4_1] = "q4_1"


def _use_kitchen_for(qtype, device_type, dtype=None):
    if not HAS_KITCHEN or not USE_KITCHEN_KERNELS or qtype not in _KITCHEN_QTYPES:
        return False
    if dtype not in (None, torch.float16, torch.bfloat16):
        return False
    if BACKEND_ENV == "auto":
        return device_type == "xpu"
    return BACKEND_ENV in {"kitchen", "xpu", "eager"}


def _dequantize_kitchen(blocks, qtype, block_size, dtype=None):
    """Dequantize a plugin-owned GGUF tensor through Comfy Kitchen."""
    output_dtype = dtype if dtype is not None else torch.float16
    backend_override = BACKEND_ENV if BACKEND_ENV in {"xpu", "eager"} else None
    backend_context = (
        comfy_kitchen.use_backend(backend_override)
        if backend_override is not None
        else nullcontext()
    )
    with backend_context:
        output = comfy_kitchen.dequantize_gguf(
            blocks.reshape(-1).contiguous(),
            _KITCHEN_QTYPES[qtype],
            output_dtype=output_dtype,
            layout="comfyui",
        )
    return output.reshape(blocks.shape[0], block_size)


def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    
    Kernel selection priority:
    1. Comfy Kitchen (Intel XPU: Q4_0, Q4_1, Q8_0, Q4_K, Q6_K)
    2. Triton (XPU/CUDA: Q4_0, Q8_0, Q4_1)
    3. PyTorch (fallback)
    """
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    
    rows = data.reshape(
        (-1, data.shape[-1])
    ).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    
    # Select dequantization implementation
    device_type = data.device.type
    
    if _use_kitchen_for(qtype, device_type, dtype):
        if DEBUG_KERNEL_SELECTION:
            logging.info(
                "ComfyUI-GGUF: Using Comfy Kitchen for %s (%d blocks)",
                qtype.name,
                n_blocks,
            )
        result = _dequantize_kitchen(blocks, qtype, block_size, dtype)
        return result.reshape(oshape)
    
    # Try Triton kernels next (best performance)
    if HAS_TRITON and USE_TRITON_KERNELS and device_type in ('xpu', 'cuda'):
        if qtype == gguf.GGMLQuantizationType.Q4_0:
            if DEBUG_KERNEL_SELECTION:
                logging.info(f"ComfyUI-GGUF: Using Triton kernel for Q4_0 ({n_blocks} blocks)")
            result = _dequantize_q4_0_triton(blocks, block_size, type_size, dtype)
            return result.reshape(oshape)
        elif qtype == gguf.GGMLQuantizationType.Q8_0:
            if DEBUG_KERNEL_SELECTION:
                logging.info(f"ComfyUI-GGUF: Using Triton kernel for Q8_0 ({n_blocks} blocks)")
            result = _dequantize_q8_0_triton(blocks, block_size, type_size, dtype)
            return result.reshape(oshape)
        elif qtype == gguf.GGMLQuantizationType.Q4_1:
            if DEBUG_KERNEL_SELECTION:
                logging.info(f"ComfyUI-GGUF: Using Triton kernel for Q4_1 ({n_blocks} blocks)")
            result = _dequantize_q4_1_triton(blocks, block_size, type_size, dtype)
            return result.reshape(oshape)
    
    # Fallback to PyTorch implementation
    if DEBUG_KERNEL_SELECTION:
        logging.info(f"ComfyUI-GGUF: Using PyTorch fallback for {qtype.name} ({n_blocks} blocks)")
    dequantize_blocks = dequantize_functions[qtype]
    
    # Optionally use torch.compile for other quantization types
    if USE_TORCH_COMPILE and device_type in ('xpu', 'cuda'):
        dequantize_blocks = get_compiled_dequant(dequantize_blocks, device_type)
    
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)

def to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)

def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

# Full weights #
def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

# Legacy Quants #
def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    """
    Optimized Q8_0 dequantization:
    - Direct slicing instead of split_block_dims
    - Reduced intermediate tensors
    """
    # Q8_0 format: 2 bytes scale + 32 bytes data (32 x int8 values)
    d = blocks[:, :2].view(torch.float16)
    if dtype is not None:
        d = d.to(dtype)
    x = blocks[:, 2:].view(torch.int8)
    return d * x

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = (ql | (qh << 4))
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)

# Pre-allocated shift tensor for Q4_0/Q4_1 optimization
_Q4_SHIFT = None

def _get_q4_shift(device):
    """Get cached shift tensor for Q4_x dequantization"""
    global _Q4_SHIFT
    if _Q4_SHIFT is None or _Q4_SHIFT.device != device:
        _Q4_SHIFT = torch.tensor([0, 4], dtype=torch.uint8, device=device)
    return _Q4_SHIFT

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    """
    Optimized Q4_1 dequantization:
    - Direct slicing instead of split_block_dims
    - Cached shift tensor
    - Reduced reshape operations
    """
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    # Q4_1 format: 2 bytes scale + 2 bytes min + 16 bytes data
    d = blocks[:, :2].view(torch.float16)
    m = blocks[:, 2:4].view(torch.float16)
    if dtype is not None:
        d = d.to(dtype)
        m = m.to(dtype)
    
    # Get cached shift tensor
    shift = _get_q4_shift(device)
    
    qs = blocks[:, 4:].reshape(n_blocks, -1, 1, block_size // 2)
    qs = (qs >> shift.reshape(1, 1, 2, 1)) & 0x0F
    qs = qs.reshape(n_blocks, -1)

    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    """
    Optimized Q4_0 dequantization:
    - Reduced tensor allocations by caching shift tensor
    - Minimized reshape operations
    - In-place operations where possible
    """
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    # Split scale (d) and quantized values (qs)
    # Q4_0 format: 2 bytes scale + 16 bytes data (32 x 4-bit values)
    d = blocks[:, :2].view(torch.float16)
    if dtype is not None:
        d = d.to(dtype)
    
    # Get cached shift tensor
    shift = _get_q4_shift(device)
    
    # Unpack 4-bit values: each byte contains 2 values (low 4 bits, high 4 bits)
    qs = blocks[:, 2:].reshape(n_blocks, -1, 1, block_size // 2)
    qs = (qs >> shift.reshape(1, 1, 2, 1)) & 0x0F
    qs = qs.reshape(n_blocks, -1).to(torch.int8) - 8
    
    return d * qs

# K Quants #
QK_K = 256
K_SCALE_SIZE = 12

def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))

# IQ quants
KVALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.int8)

def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size//2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    del kvalues # should still be view, but just to be safe

    return (d * qs)

def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)

    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    del kvalues # see IQ4_NL
    del shift_a
    del shift_b

    return (dl * qs).reshape((n_blocks, -1))

dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
}
