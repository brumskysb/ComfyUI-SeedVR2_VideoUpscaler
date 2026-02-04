"""
NVFP4 Quantization Operations Module for SeedVR2

Provides runtime NVFP4 (4-bit floating point) quantization support for DiT models
using the comfy-kitchen library for Blackwell GPU optimizations.

NVFP4 (NVIDIA FP4) is a 4-bit floating point format (E2M1) that provides:
- 4x memory reduction compared to FP16
- Hardware acceleration on SM ≥ 10.0 (Blackwell/RTX 50xx)
- Block quantization with 16-element blocks for accuracy

Requirements:
- comfy-kitchen library: pip install comfy-kitchen[cublas]
- Blackwell GPU (SM 10.0+) for hardware acceleration
- Falls back to eager/triton backends on older GPUs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

# Import comfy-kitchen with fallback
try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
    COMFY_KITCHEN_AVAILABLE = True
except ImportError:
    COMFY_KITCHEN_AVAILABLE = False
    ck = None
    QuantizedTensor = None
    TensorCoreNVFP4Layout = None


def validate_nvfp4_availability(operation: str = "use NVFP4 quantization", debug=None) -> None:
    """
    Validate comfy-kitchen availability for NVFP4 operations.
    
    Args:
        operation: Description of the operation requiring NVFP4
        debug: Optional debug instance for logging
        
    Raises:
        RuntimeError: If comfy-kitchen is not available
    """
    if not COMFY_KITCHEN_AVAILABLE:
        error_msg = (
            f"Cannot {operation}: comfy-kitchen library is not installed.\n"
            f"\n"
            f"NVFP4 provides 4-bit quantization for memory-efficient DiT inference.\n"
            f"Requires Blackwell GPU (RTX 50xx) for hardware acceleration.\n"
            f"\n"
            f"To fix this issue:\n"
            f"  1. Install comfy-kitchen: pip install comfy-kitchen[cublas]\n"
            f"  2. OR disable NVFP4 quantization in settings\n"
            f"\n"
            f"For more info: https://github.com/Comfy-Org/comfy-kitchen"
        )
        if debug:
            debug.log(error_msg, level="ERROR", category="setup", force=True)
        raise RuntimeError(f"comfy-kitchen library required to {operation}")


def check_nvfp4_hardware_support(device: torch.device = None) -> Tuple[bool, str]:
    """
    Check if the current GPU supports hardware-accelerated NVFP4.
    
    NVFP4 requires SM ≥ 10.0 (Blackwell architecture) for hardware acceleration.
    On older GPUs, falls back to eager/triton backends (slower but functional).
    
    Args:
        device: Target device to check (defaults to cuda:0)
        
    Returns:
        Tuple of (hardware_supported, message)
    """
    if not COMFY_KITCHEN_AVAILABLE:
        return False, "comfy-kitchen not installed"
    
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    if device is None:
        device = torch.device("cuda:0")
    
    try:
        device_idx = device.index if device.index is not None else 0
        capability = torch.cuda.get_device_capability(device_idx)
        compute_capability = capability[0] * 10 + capability[1]
        
        if compute_capability >= 100:  # SM 10.0+ (Blackwell)
            return True, f"Blackwell GPU detected (SM {capability[0]}.{capability[1]})"
        elif compute_capability >= 89:  # Ada Lovelace
            return False, f"Ada GPU detected (SM {capability[0]}.{capability[1]}) - NVFP4 will use triton/eager backend"
        else:
            return False, f"GPU SM {capability[0]}.{capability[1]} - NVFP4 will use eager backend"
    except Exception as e:
        return False, f"Could not detect GPU capability: {e}"


class _NVFP4QuantizedBase(nn.Module):
    """Base class for NVFP4 quantized layers with shared quantization logic"""
    
    def __init__(self, debug: Optional['Debug'] = None):
        super().__init__()
        self.weight = None
        self.bias = None
        self.quantized_weight = None
        self.debug = debug
        self._weight_quantized = False
    
    def quantize_weight(self, weight_tensor: torch.Tensor) -> None:
        """
        Quantize weight tensor to NVFP4 format.
        
        NVFP4 uses block quantization with 16-element blocks.
        Weights must be on CUDA device and in bf16/fp16 format.
        
        Args:
            weight_tensor: Weight tensor to quantize
        """
        if not COMFY_KITCHEN_AVAILABLE:
            validate_nvfp4_availability("quantize weights", self.debug)
        
        # Ensure weight is on CUDA and in half precision
        if weight_tensor.device.type != 'cuda':
            if self.debug:
                self.debug.log("Moving weight to CUDA for NVFP4 quantization", 
                             category="nvfp4", indent_level=1)
            weight_tensor = weight_tensor.cuda()
        
        if weight_tensor.dtype not in (torch.float16, torch.bfloat16):
            weight_tensor = weight_tensor.to(torch.bfloat16)
        
        # Quantize using comfy-kitchen
        try:
            self.quantized_weight = QuantizedTensor.from_float(weight_tensor, TensorCoreNVFP4Layout)
            self._weight_quantized = True
            
            if self.debug:
                self.debug.log(f"Quantized weight to NVFP4: {weight_tensor.shape}", 
                             category="nvfp4", indent_level=1)
        except Exception as e:
            if self.debug:
                self.debug.log(f"NVFP4 quantization failed: {e}", 
                             level="WARNING", category="nvfp4", force=True)
            # Keep original weight as fallback
            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
            self._weight_quantized = False
    
    def load_weight(self, weight_tensor: torch.Tensor, bias_tensor: Optional[torch.Tensor] = None,
                   quantize: bool = True) -> None:
        """
        Load weight tensor, optionally quantizing to NVFP4.
        
        Args:
            weight_tensor: Weight tensor to load
            bias_tensor: Optional bias tensor
            quantize: Whether to quantize the weight to NVFP4
        """
        if quantize and COMFY_KITCHEN_AVAILABLE:
            self.quantize_weight(weight_tensor)
        else:
            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
            self._weight_quantized = False
        
        if bias_tensor is not None:
            self.bias = nn.Parameter(bias_tensor, requires_grad=False)
    
    def _get_weight_for_compute(self, input: torch.Tensor) -> torch.Tensor:
        """
        Get weight tensor ready for computation.
        
        For NVFP4 quantized weights, the QuantizedTensor automatically
        dispatches to optimized kernels when used in operations.
        
        Args:
            input: Input tensor (used to determine device/dtype)
            
        Returns:
            Weight tensor ready for computation
        """
        if self._weight_quantized and self.quantized_weight is not None:
            # QuantizedTensor handles dispatch automatically
            return self.quantized_weight
        elif self.weight is not None:
            return self.weight.to(input.device, input.dtype)
        else:
            raise RuntimeError("No weight available for computation")


class NVFP4QuantizedLinear(_NVFP4QuantizedBase):
    """
    Quantized Linear layer with NVFP4 (4-bit floating point) weights.
    
    Uses comfy-kitchen's QuantizedTensor which automatically dispatches
    to optimized kernels (cuda/triton/eager) based on hardware support.
    
    On Blackwell GPUs (SM 10.0+): Uses hardware-accelerated NVFP4 matmul
    On older GPUs: Falls back to dequantize + standard matmul
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 device=None, dtype=None, debug: Optional['Debug'] = None):
        super().__init__(debug)
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic NVFP4 kernel dispatch."""
        weight = self._get_weight_for_compute(input)
        
        # QuantizedTensor's __torch_function__ intercepts linear calls
        # and dispatches to optimized scaled_mm_nvfp4 when supported
        return torch.nn.functional.linear(input, weight, self.bias)


def replace_linear_with_nvfp4(module: nn.Module, debug: Optional['Debug'] = None, 
                               prefix: str = "") -> Tuple[int, Dict[str, int]]:
    """
    Replace Linear layers with NVFP4 quantized versions.
    
    Recursively walks through the module tree and replaces nn.Linear layers
    with NVFP4QuantizedLinear layers, quantizing their weights.
    
    Args:
        module: The module to process
        debug: Optional Debug instance for logging
        prefix: Prefix for recursive calls (used internally)
        
    Returns:
        Tuple of (replacements_made, stats) where:
        - replacements_made: Number of layers replaced
        - stats: Dict with quantization statistics
    """
    if not COMFY_KITCHEN_AVAILABLE:
        if debug:
            debug.log("comfy-kitchen not available, skipping NVFP4 quantization", 
                     level="WARNING", category="nvfp4", force=True)
        return 0, {"skipped": "comfy-kitchen not available"}
    
    replacements_made = 0
    stats = {
        "quantized_params": 0,
        "total_params": 0,
        "memory_saved_mb": 0.0,
    }
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Calculate memory savings
            weight_params = child.weight.numel()
            stats["total_params"] += weight_params
            
            # Create NVFP4 quantized linear layer
            nvfp4_linear = NVFP4QuantizedLinear(
                child.in_features, 
                child.out_features,
                bias=child.bias is not None,
                debug=debug
            )
            
            # Move weight to CUDA and quantize
            try:
                weight = child.weight.data
                bias = child.bias.data if child.bias is not None else None
                nvfp4_linear.load_weight(weight, bias, quantize=True)
                
                if nvfp4_linear._weight_quantized:
                    setattr(module, name, nvfp4_linear)
                    replacements_made += 1
                    stats["quantized_params"] += weight_params
                    # NVFP4 is 4-bit = 0.5 bytes, vs FP16 = 2 bytes, so 75% savings
                    stats["memory_saved_mb"] += (weight_params * 1.5) / (1024 * 1024)
                    
                    if debug:
                        full_name = f"{prefix}.{name}" if prefix else name
                        debug.log(f"NVFP4 quantized: {full_name} ({weight.shape})", 
                                 category="nvfp4", indent_level=1)
                else:
                    if debug:
                        debug.log(f"NVFP4 quantization failed for {name}, keeping FP16", 
                                 level="WARNING", category="nvfp4")
            except Exception as e:
                if debug:
                    debug.log(f"Error quantizing {name}: {e}", 
                             level="WARNING", category="nvfp4", force=True)
        else:
            # Recursively replace in child modules
            full_name = f"{prefix}.{name}" if prefix else name
            child_replacements, child_stats = replace_linear_with_nvfp4(child, debug, full_name)
            replacements_made += child_replacements
            stats["quantized_params"] += child_stats.get("quantized_params", 0)
            stats["total_params"] += child_stats.get("total_params", 0)
            stats["memory_saved_mb"] += child_stats.get("memory_saved_mb", 0.0)
    
    return replacements_made, stats


def quantize_dit_model_nvfp4(model: nn.Module, debug: Optional['Debug'] = None) -> Dict[str, any]:
    """
    Quantize a DiT model to NVFP4 format.
    
    This is the main entry point for NVFP4 quantization of DiT models.
    It replaces all Linear layers with NVFP4 quantized versions.
    
    Args:
        model: The DiT model to quantize
        debug: Optional Debug instance for logging
        
    Returns:
        Dictionary with quantization results:
        - success: Whether quantization succeeded
        - layers_quantized: Number of layers quantized
        - stats: Detailed statistics
        - error: Error message if failed
    """
    if not COMFY_KITCHEN_AVAILABLE:
        return {
            "success": False,
            "layers_quantized": 0,
            "stats": {},
            "error": "comfy-kitchen library not available"
        }
    
    if debug:
        debug.log("Starting NVFP4 quantization of DiT model...", category="nvfp4")
        hw_supported, hw_msg = check_nvfp4_hardware_support()
        debug.log(f"Hardware check: {hw_msg}", category="nvfp4", indent_level=1)
    
    try:
        layers_quantized, stats = replace_linear_with_nvfp4(model, debug)
        
        if debug:
            debug.log(f"NVFP4 quantization complete:", category="nvfp4")
            debug.log(f"  Layers quantized: {layers_quantized}", category="nvfp4", indent_level=1)
            debug.log(f"  Parameters quantized: {stats['quantized_params']:,}", category="nvfp4", indent_level=1)
            debug.log(f"  Estimated memory saved: {stats['memory_saved_mb']:.1f} MB", category="nvfp4", indent_level=1)
        
        return {
            "success": True,
            "layers_quantized": layers_quantized,
            "stats": stats,
            "error": None
        }
    except Exception as e:
        if debug:
            debug.log(f"NVFP4 quantization failed: {e}", level="ERROR", category="nvfp4", force=True)
        return {
            "success": False,
            "layers_quantized": 0,
            "stats": {},
            "error": str(e)
        }
