# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Triton kernel optimizations for GGUF dequantization on Intel XPU
import torch
import logging
import platform

if platform.system() == "Windows":
    HAS_TRITON = False
    logging.info("ComfyUI-GGUF: Triton not supported on Windows, falling back to PyTorch implementation")
else:
    try:
        import triton
        import triton.language as tl
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False
        logging.warning("ComfyUI-GGUF: Triton not available, falling back to PyTorch implementation")


if HAS_TRITON:
    # ============================================================================
    # Q4_0 Triton Kernel
    # Format: 2 bytes scale (float16) + 16 bytes data (32 x 4-bit values)
    # Block size: 32, Type size: 18 bytes
    # ============================================================================
    
    @triton.jit
    def dequant_q4_0_kernel(
        data_ptr,        # Input: quantized data [n_blocks, 18]
        output_ptr,      # Output: dequantized data [n_blocks, 32]
        n_blocks,        # Number of blocks
        BLOCK_SIZE: tl.constexpr,  # Number of blocks to process per program
    ):
        """
        Triton kernel for Q4_0 dequantization.
        Each block: 2 bytes scale + 16 bytes (32 x 4-bit values)
        Dequantization: output = scale * (nibble - 8)
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        
        # Process BLOCK_SIZE blocks per program
        for i in range(BLOCK_SIZE):
            block_idx = block_start + i
            if block_idx >= n_blocks:
                break
            
            # Base offset for this block (18 bytes per block)
            base_offset = block_idx * 18
            
            # Load scale (2 bytes as float16) - load as uint16 then reinterpret
            scale_bytes = tl.load(data_ptr + base_offset).to(tl.uint8)
            scale_byte1 = tl.load(data_ptr + base_offset + 1).to(tl.uint8)
            scale_uint16 = scale_bytes.to(tl.uint16) | (scale_byte1.to(tl.uint16) << 8)
            scale = scale_uint16.to(tl.float16, bitcast=True).to(tl.float32)
            
            # Output base offset (32 values per block)
            out_base = block_idx * 32
            
            # Process 16 bytes -> 32 x 4-bit values
            for j in range(16):
                byte_val = tl.load(data_ptr + base_offset + 2 + j).to(tl.uint8)
                
                # Extract low and high nibbles
                lo = (byte_val & 0x0F).to(tl.int8) - 8
                hi = ((byte_val >> 4) & 0x0F).to(tl.int8) - 8
                
                # Dequantize and store
                out_lo = scale * lo.to(tl.float32)
                out_hi = scale * hi.to(tl.float32)
                
                tl.store(output_ptr + out_base + j * 2, out_lo.to(tl.float16))
                tl.store(output_ptr + out_base + j * 2 + 1, out_hi.to(tl.float16))


    @triton.jit  
    def dequant_q4_0_kernel_v2(
        data_ptr,        # Input: quantized data, flattened
        output_ptr,      # Output: dequantized data
        n_blocks,        # Total number of blocks
        BLOCK_M: tl.constexpr,  # Blocks per program instance
    ):
        """
        Optimized Q4_0 kernel - processes multiple elements in parallel
        """
        pid = tl.program_id(0)
        
        # Each program handles BLOCK_M quantization blocks
        block_idx = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = block_idx < n_blocks
        
        # Calculate byte offsets (18 bytes per Q4_0 block)
        data_offset = block_idx * 18
        
        # Load scales (first 2 bytes of each block)
        # We load byte by byte and reconstruct float16
        scale_lo = tl.load(data_ptr + data_offset, mask=mask, other=0).to(tl.uint16)
        scale_hi = tl.load(data_ptr + data_offset + 1, mask=mask, other=0).to(tl.uint16)
        scale_bits = scale_lo | (scale_hi << 8)
        scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        
        # Output offset (32 float16 values per block)
        out_offset = block_idx * 32
        
        # Process each of the 16 data bytes
        for byte_idx in range(16):
            byte_data = tl.load(data_ptr + data_offset + 2 + byte_idx, mask=mask, other=0)
            
            # Extract nibbles
            lo_nibble = (byte_data & 0x0F).to(tl.int32) - 8
            hi_nibble = ((byte_data >> 4) & 0x0F).to(tl.int32) - 8
            
            # Dequantize
            val_lo = (scale * lo_nibble.to(tl.float32)).to(tl.float16)
            val_hi = (scale * hi_nibble.to(tl.float32)).to(tl.float16)
            
            # Store results
            tl.store(output_ptr + out_offset + byte_idx * 2, val_lo, mask=mask)
            tl.store(output_ptr + out_offset + byte_idx * 2 + 1, val_hi, mask=mask)


    @triton.jit
    def dequant_q4_0_kernel_v3(
        data_ptr,        # Input: quantized data [n_blocks * 18]
        output_ptr,      # Output: dequantized data [n_blocks * 32]
        n_elements,      # Total output elements (n_blocks * 32)
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        V3: Element-centric kernel - each thread block handles BLOCK_SIZE output elements
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Calculate which Q4_0 block and position within block
        block_idx = offsets // 32  # Which quantization block
        pos_in_block = offsets % 32  # Position within the 32-element block
        
        # Calculate byte position within data block
        byte_idx = pos_in_block // 2  # Which of the 16 data bytes
        is_high_nibble = pos_in_block % 2  # 0 = low nibble, 1 = high nibble
        
        # Data layout: [scale_lo, scale_hi, data0, data1, ..., data15]
        data_base = block_idx * 18
        
        # Load scale
        scale_lo = tl.load(data_ptr + data_base, mask=mask, other=0).to(tl.uint16)
        scale_hi = tl.load(data_ptr + data_base + 1, mask=mask, other=0).to(tl.uint16)
        scale_bits = scale_lo | (scale_hi << 8)
        scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)
        
        # Load data byte
        data_byte = tl.load(data_ptr + data_base + 2 + byte_idx, mask=mask, other=0)
        
        # Extract correct nibble
        nibble = tl.where(
            is_high_nibble == 1,
            (data_byte >> 4) & 0x0F,
            data_byte & 0x0F
        ).to(tl.int32) - 8
        
        # Dequantize
        result = (scale * nibble.to(tl.float32)).to(tl.float16)
        
        # Store
        tl.store(output_ptr + offsets, result, mask=mask)


    # ============================================================================
    # Q8_0 Triton Kernel  
    # Format: 2 bytes scale (float16) + 32 bytes data (32 x int8 values)
    # Block size: 32, Type size: 34 bytes
    # ============================================================================
    
    @triton.jit
    def dequant_q8_0_kernel(
        data_ptr,
        output_ptr, 
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Q8_0 kernel - simpler than Q4_0 since no nibble unpacking needed
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
        
        # Load int8 value
        int8_val = tl.load(data_ptr + data_base + 2 + pos_in_block, mask=mask, other=0)
        # Convert to signed int8
        signed_val = tl.where(int8_val > 127, int8_val.to(tl.int32) - 256, int8_val.to(tl.int32))
        
        # Dequantize
        result = (scale * signed_val.to(tl.float32)).to(tl.float16)
        
        tl.store(output_ptr + offsets, result, mask=mask)


    # ============================================================================
    # Wrapper Functions
    # ============================================================================
    
    def dequantize_q4_0_triton(blocks, block_size, type_size, dtype=None):
        """
        Triton-accelerated Q4_0 dequantization
        Args:
            blocks: [n_blocks, 18] uint8 tensor
            block_size: 32 (elements per block)
            type_size: 18 (bytes per block)
            dtype: output dtype (default float16)
        Returns:
            [n_blocks, 32] dequantized tensor
        """
        n_blocks = blocks.shape[0]
        device = blocks.device
        
        # Flatten input to 1D for easier indexing
        data_flat = blocks.view(-1).contiguous()
        
        # Allocate output
        output = torch.empty(n_blocks * 32, dtype=torch.float16, device=device)
        
        # Launch kernel
        n_elements = n_blocks * 32
        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
        dequant_q4_0_kernel_v3[grid](
            data_flat,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Reshape and convert dtype if needed
        result = output.view(n_blocks, 32)
        if dtype is not None and dtype != torch.float16:
            result = result.to(dtype)
        
        return result


    def dequantize_q8_0_triton(blocks, block_size, type_size, dtype=None):
        """
        Triton-accelerated Q8_0 dequantization
        """
        n_blocks = blocks.shape[0]
        device = blocks.device
        
        data_flat = blocks.view(-1).contiguous()
        output = torch.empty(n_blocks * 32, dtype=torch.float16, device=device)
        
        n_elements = n_blocks * 32
        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
        dequant_q8_0_kernel[grid](
            data_flat,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        result = output.view(n_blocks, 32)
        if dtype is not None and dtype != torch.float16:
            result = result.to(dtype)
        
        return result


# ============================================================================
# Fallback PyTorch implementations (copy from dequant.py for standalone testing)
# ============================================================================

_Q4_SHIFT = None

def _get_q4_shift(device):
    global _Q4_SHIFT
    if _Q4_SHIFT is None or _Q4_SHIFT.device != device:
        _Q4_SHIFT = torch.tensor([0, 4], dtype=torch.uint8, device=device)
    return _Q4_SHIFT


def dequantize_q4_0_pytorch(blocks, block_size, type_size, dtype=None):
    """PyTorch reference implementation for Q4_0"""
    n_blocks = blocks.shape[0]
    device = blocks.device
    
    d = blocks[:, :2].view(torch.float16)
    if dtype is not None:
        d = d.to(dtype)
    
    shift = _get_q4_shift(device)
    qs = blocks[:, 2:].reshape(n_blocks, -1, 1, block_size // 2)
    qs = (qs >> shift.reshape(1, 1, 2, 1)) & 0x0F
    qs = qs.reshape(n_blocks, -1).to(torch.int8) - 8
    
    return d * qs


def dequantize_q8_0_pytorch(blocks, block_size, type_size, dtype=None):
    """PyTorch reference implementation for Q8_0"""
    d = blocks[:, :2].view(torch.float16)
    if dtype is not None:
        d = d.to(dtype)
    x = blocks[:, 2:].view(torch.int8)
    return d * x


# ============================================================================
# Auto-selection wrapper
# ============================================================================

def get_dequant_q4_0(device_type):
    """Get the best Q4_0 dequantization function for the device"""
    if HAS_TRITON and device_type in ('xpu', 'cuda'):
        return dequantize_q4_0_triton
    return dequantize_q4_0_pytorch


def get_dequant_q8_0(device_type):
    """Get the best Q8_0 dequantization function for the device"""
    if HAS_TRITON and device_type in ('xpu', 'cuda'):
        return dequantize_q8_0_triton
    return dequantize_q8_0_pytorch
