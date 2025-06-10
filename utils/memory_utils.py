"""
Memory utilities for optimizing model training across different devices.
Provides standardized functions for memory management across CPU, CUDA, and MPS backends.
"""

import torch
import gc
import os
import psutil
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memory_utils')

@dataclass
class MemoryConfig:
    """Configuration for memory optimization across different devices."""
    # General configuration
    reduce_precision: bool = False  # Use lower precision when possible
    enable_memory_tracking: bool = False  # Track memory usage
    gradient_accumulation_steps: int = 1  # Number of steps for gradient accumulation
    optimize_for_inference: bool = False  # Special optimizations for inference only
    min_batch_size: int = 16  # Minimum batch size to ensure stable training
    max_batch_size: int = 512  # Maximum batch size to prevent OOM
    dynamic_batch_sizing: bool = True  # Adjust batch size based on available memory
    
    # Device-specific settings
    cuda_config: Dict[str, Any] = None
    mps_config: Dict[str, Any] = None
    cpu_config: Dict[str, Any] = None
    
    # Tensor optimizations
    use_int32_for_indices: bool = True  # Convert int64 indices to int32 where possible
    use_float16_for_embeddings: bool = False  # Use half precision for embeddings
    
    # Memory cleanup settings
    periodic_gc: bool = False  # Run garbage collection periodically
    gc_interval: int = 50  # Number of batches between garbage collections
    
    def __post_init__(self):
        """Initialize device-specific configurations."""
        # Default CUDA settings
        if self.cuda_config is None:
            self.cuda_config = {
                'use_amp': True,  # Use automatic mixed precision
                'compile_model': True,  # Use torch.compile for speedup
                'pin_memory': True,  # Use pinned memory for faster transfers
                'prefetch_factor': 3,  # Prefetch batches for data loading
                'empty_cache_interval': 10,  # Empty CUDA cache every N batches
                'use_float16_for_embeddings': True,  # Use float16 for embeddings
                'max_batch_size': 320,  # Generally safe max batch size for CUDA with 8GB+ VRAM
                'auto_tensor_cores': True,  # Use tensor cores when available (e.g., A100, H100)
                'use_compile_dynamo': True  # Use dynamo backend for compilation
            }
        
        # Default MPS settings (Apple Silicon)
        if self.mps_config is None:
            self.mps_config = {
                'use_amp': False,  # MPS doesn't support AMP yet
                'compile_model': False,  # MPS doesn't fully support compile yet
                'pin_memory': True,  # Use pinned memory for faster transfers
                'prefetch_factor': 2,  # Less prefetching for MPS
                'empty_cache_interval': 0,  # Disable cache clearing for better performance
                'use_float16_for_embeddings': True,  # Use float16 for embeddings when possible
                'max_batch_size': 256,  # Safer default for various M1/M2/M3 chips
                'reduce_embedding_dim': True,  # Reduce embedding dimensions to save memory
                'avoid_uint8_tensors': True  # MPS has issues with uint8 tensors sometimes
            }
        
        # Default CPU settings
        if self.cpu_config is None:
            self.cpu_config = {
                'use_amp': False,  # CPU doesn't benefit from AMP
                'compile_model': True,  # CPU can benefit from compile
                'pin_memory': False,  # No benefit on CPU-only
                'prefetch_factor': 2,  # Conservative prefetching
                'num_threads': max(1, os.cpu_count() - 1) if os.cpu_count() else 4,  # Leave one CPU free
                'use_mkldnn': True,  # Use MKL-DNN acceleration if available
                'max_batch_size': 64,  # Smaller batches for CPU
                'optimize_memory_usage': True  # Prioritize memory efficiency over speed
            }


def detect_device() -> Tuple[torch.device, str]:
    """
    Detect the best available device for training or inference.
    
    Returns:
        Tuple containing:
            - torch.device: The detected device
            - str: Device type ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
        logger.info("Using Apple Silicon (MPS) device")
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        logger.info(f"Using CPU: {psutil.cpu_count(logical=True)} logical cores")
    
    return device, device_type


def create_memory_config(
    device_type: str, 
    performance_mode: str = "balanced"
) -> MemoryConfig:
    """
    Create a device-specific memory configuration based on the performance mode.
    
    Args:
        device_type: The device type ('cuda', 'mps', or 'cpu')
        performance_mode: The performance mode ('fastest', 'balanced', 'accurate')
    
    Returns:
        MemoryConfig: Configuration optimized for the device and performance mode
    """
    config = MemoryConfig()
    
    # Basic settings based on performance mode
    if performance_mode == "fastest":
        config.reduce_precision = True
        config.enable_memory_tracking = False
        config.use_int32_for_indices = True
        config.gradient_accumulation_steps = 4
        config.dynamic_batch_sizing = False
        config.periodic_gc = False
        config.use_float16_for_embeddings = True
    elif performance_mode == "accurate":
        config.reduce_precision = False
        config.enable_memory_tracking = True
        config.use_int32_for_indices = False
        config.gradient_accumulation_steps = 1
        config.dynamic_batch_sizing = True
        config.periodic_gc = True
        config.gc_interval = 10
        config.use_float16_for_embeddings = False
    else:  # balanced
        config.reduce_precision = True
        config.enable_memory_tracking = True
        config.use_int32_for_indices = True
        config.gradient_accumulation_steps = 2
        config.dynamic_batch_sizing = True
        config.periodic_gc = True
        config.gc_interval = 25
        config.use_float16_for_embeddings = device_type != 'cpu'
    
    # Device-specific overrides
    if device_type == 'cuda':
        if performance_mode == "fastest":
            config.cuda_config.update({
                'max_batch_size': config.max_batch_size,
                'empty_cache_interval': 0,  # Disable periodic cache clearing
                'prefetch_factor': 4,
                'use_compile_dynamo': True
            })
        elif performance_mode == "accurate":
            config.cuda_config.update({
                'use_amp': False,
                'max_batch_size': 128,
                'empty_cache_interval': 5,
                'prefetch_factor': 2,
                'use_compile_dynamo': False  # More predictable behavior
            })
        # balanced uses defaults
        
    elif device_type == 'mps':
        # MPS-specific settings (Apple Silicon)
        # IMPORTANT: Disable float16 on MPS to prevent dtype mismatch errors
        config.use_float16_for_embeddings = False
        config.mps_config['use_float16_for_embeddings'] = False
        
        if performance_mode == "fastest":
            config.mps_config.update({
                'max_batch_size': 320,  # Push batch size higher
                'empty_cache_interval': 0,  # Disable for better performance
                'reduce_embedding_dim': True,
                'use_float16_for_embeddings': False  # Force disable float16
            })
        elif performance_mode == "accurate":
            config.mps_config.update({
                'max_batch_size': 128,
                'empty_cache_interval': 0,  # Disable for better performance
                'reduce_embedding_dim': False,
                'use_float16_for_embeddings': False
            })
        # balanced uses defaults
    
    elif device_type == 'cpu':
        # CPU-specific settings
        if performance_mode == "fastest":
            config.cpu_config.update({
                'max_batch_size': 32,
                'num_threads': os.cpu_count() if os.cpu_count() else 4,
                'optimize_memory_usage': False
            })
        elif performance_mode == "accurate":
            config.cpu_config.update({
                'max_batch_size': 16,
                'prefetch_factor': 1,
                'optimize_memory_usage': True,
                'compile_model': False
            })
        # balanced uses defaults
    
    return config


def get_optimal_batch_size(
    device_type: str,
    config: MemoryConfig,
    model_size_mb: Optional[int] = None,
    tensor_size_mb: Optional[int] = None,
    overhead_factor: float = 1.5
) -> int:
    """
    Calculate optimal batch size based on device and available memory.
    
    Args:
        device_type: Device type ('cuda', 'mps', 'cpu')
        config: Memory configuration
        model_size_mb: Size of model in MB (optional)
        tensor_size_mb: Size of a single batch tensor in MB (optional)
        overhead_factor: Safety factor for additional memory usage
    
    Returns:
        int: Optimal batch size for the device
    """
    if not config.dynamic_batch_sizing:
        # Use the configured max batch size for the device
        if device_type == 'cuda':
            return config.cuda_config['max_batch_size']
        elif device_type == 'mps':
            return config.mps_config['max_batch_size']
        else:  # cpu
            return config.cpu_config['max_batch_size']
    
    # Default sizes if not provided
    if model_size_mb is None:
        model_size_mb = 500  # Default model size estimate
    
    if tensor_size_mb is None:
        tensor_size_mb = 10  # Default tensor size estimate per sample
    
    # Calculate available memory
    available_memory_mb = 0
    
    if device_type == 'cuda':
        # For CUDA, use GPU memory
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            available_memory_mb = free_memory / (1024 * 1024)
            # Reserve some memory for CUDA overhead
            available_memory_mb = available_memory_mb * 0.85
        except:
            # Fallback if mem_get_info is not available
            available_memory_mb = 4 * 1024  # Assume 4GB
    
    elif device_type == 'mps':
        # For MPS, estimate based on system memory
        try:
            # Check if we can get current MPS memory
            if hasattr(torch.mps, 'current_allocated_memory'):
                # Use a percentage of system memory since MPS doesn't expose limits
                system_memory = psutil.virtual_memory().total / (1024 * 1024)
                available_memory_mb = system_memory * 0.6  # Use 60% of system memory
                
                # Subtract already allocated memory
                current_allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                available_memory_mb = max(0, available_memory_mb - current_allocated)
            else:
                # Fallback: use system memory heuristic
                system_memory = psutil.virtual_memory().available / (1024 * 1024)
                available_memory_mb = system_memory * 0.5  # Conservative estimate
        except:
            # Ultra-conservative fallback
            available_memory_mb = 4 * 1024  # Assume 4GB
    
    else:  # CPU
        # For CPU, use system memory
        system_memory = psutil.virtual_memory().available / (1024 * 1024)
        available_memory_mb = system_memory * 0.5  # Use 50% of available memory
    
    # Calculate batch size with overhead factor
    memory_per_sample = tensor_size_mb * overhead_factor  # Add overhead
    
    # Reserve memory for model and other tensors
    available_for_batch = available_memory_mb - model_size_mb
    
    # Calculate max batch size
    if memory_per_sample > 0 and available_for_batch > 0:
        max_batch = int(available_for_batch / memory_per_sample)
        
        # Clamp to min/max batch size
        max_batch = max(config.min_batch_size, min(max_batch, config.max_batch_size))
        
        # Round to multiple of 8 for better memory alignment and GPU utilization
        max_batch = (max_batch // 8) * 8
        if max_batch < config.min_batch_size:
            max_batch = config.min_batch_size
        
        logger.info(f"Dynamic batch size calculation: {max_batch} (Available memory: {available_memory_mb:.0f}MB)")
        return max_batch
    else:
        # Fallback to device default if calculation fails
        if device_type == 'cuda':
            return config.cuda_config['max_batch_size']
        elif device_type == 'mps':
            return config.mps_config['max_batch_size']
        else:  # cpu
            return config.cpu_config['max_batch_size']


def optimize_tensor_for_device(
    tensor: torch.Tensor,
    device: torch.device,
    config: MemoryConfig,
    is_embedding: bool = False,
    is_index: bool = False
) -> torch.Tensor:
    """
    Optimize a tensor for the target device based on configuration.
    
    Args:
        tensor: Input tensor
        device: Target device
        config: Memory configuration
        is_embedding: Whether the tensor contains embeddings
        is_index: Whether the tensor contains indices
    
    Returns:
        torch.Tensor: Optimized tensor on the target device
    """
    if tensor is None:
        return None
        
    device_type = device.type
    
    # Skip tensors already on the correct device with correct type
    if tensor.device == device:
        # Check if tensor already has the optimal type
        if (is_index and tensor.dtype == torch.int32 and config.use_int32_for_indices) or \
           (is_embedding and tensor.dtype == torch.float16 and config.use_float16_for_embeddings):
            return tensor
    
    if is_index and config.use_int32_for_indices and tensor.dtype == torch.int64:
        # Convert int64 indices to int32 for efficiency on all devices
        tensor = tensor.to(dtype=torch.int32)
    
    elif is_embedding and config.use_float16_for_embeddings and tensor.dtype == torch.float32:
        # Convert float32 embeddings to float16 if configured
        if (device_type == 'cuda' and config.cuda_config['use_float16_for_embeddings']) or \
           (device_type == 'mps' and config.mps_config['use_float16_for_embeddings']):
            tensor = tensor.to(dtype=torch.float16)
    
    # Custom MPS handling
    if device_type == 'mps':
        # MPS has issues with certain dtypes
        if config.mps_config['avoid_uint8_tensors'] and tensor.dtype == torch.uint8:
            tensor = tensor.to(dtype=torch.int32)
        
        # Handle specific dtypes that are problematic on MPS
        if tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)
    
    # Move to device - use non_blocking transfer when possible
    can_use_non_blocking = tensor.is_contiguous() and \
                           ((device_type == 'cuda' and config.cuda_config['pin_memory']) or
                            (device_type == 'mps' and config.mps_config['pin_memory']))
    
    return tensor.to(device=device, non_blocking=can_use_non_blocking)


def move_batch_to_device(
    batch: Union[Dict[str, Any], torch.Tensor, List],
    device: torch.device,
    config: MemoryConfig
) -> Union[Dict[str, Any], torch.Tensor, List]:
    """
    Recursively move a batch of data to the specified device with optimizations.
    
    Args:
        batch: Input batch (can be dict, tensor, or list)
        device: Target device
        config: Memory configuration
    
    Returns:
        Union[Dict[str, Any], torch.Tensor, List]: Batch moved to device with optimizations
    """
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device, config) for k, v in batch.items()}
    
    elif isinstance(batch, torch.Tensor):
        # Detect tensor type
        is_embedding = batch.dtype == torch.float32 and batch.dim() >= 2
        is_index = batch.dtype == torch.int64 and not batch.requires_grad
        
        # Check if this is a "large" tensor (worth optimizing)
        is_large = batch.numel() > 1000
        
        if is_large:
            return optimize_tensor_for_device(
                batch, device, config,
                is_embedding=is_embedding,
                is_index=is_index
            )
        else:
            # Small tensors don't need special handling
            return batch.to(device)
    
    elif isinstance(batch, list) and batch and isinstance(batch[0], torch.Tensor):
        return [move_batch_to_device(t, device, config) for t in batch]
    
    else:
        return batch


def optimize_model_for_inference(
    model: torch.nn.Module,
    device: torch.device,
    config: MemoryConfig
) -> torch.nn.Module:
    """
    Optimize a model for inference with device-specific settings.
    
    Args:
        model: PyTorch model to optimize
        device: Target device
        config: Memory configuration
    
    Returns:
        torch.nn.Module: Optimized model
    """
    # Move model to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    device_type = device.type
    
    # Device-specific optimizations
    if device_type == 'cuda':
        if config.cuda_config['compile_model'] and hasattr(torch, 'compile'):
            try:
                if config.cuda_config['use_compile_dynamo']:
                    model = torch.compile(model, backend="inductor")
                else:
                    model = torch.compile(model)
                logger.info("Model compiled for CUDA inference")
            except Exception as e:
                logger.warning(f"Failed to compile model for CUDA: {e}")
    
    elif device_type == 'cpu':
        if config.cpu_config['compile_model'] and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, backend="inductor")
                logger.info("Model compiled for CPU inference")
            except Exception as e:
                logger.warning(f"Failed to compile model for CPU: {e}")
                
        # Set optimal thread settings
        if config.cpu_config['use_mkldnn'] and hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(config.cpu_config['num_threads'])
            logger.info(f"Set CPU threads to {config.cpu_config['num_threads']}")
    
    # MPS doesn't support most optimization methods yet
    
    return model


def get_model_size(model: torch.nn.Module) -> int:
    """
    Calculate the size of a PyTorch model in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Model size in MB
    """
    model_size_bytes = 0
    for param in model.parameters():
        # Get bytes per element
        if param.dtype == torch.float32:
            bytes_per_element = 4
        elif param.dtype == torch.float16 or param.dtype == torch.bfloat16:
            bytes_per_element = 2
        elif param.dtype == torch.int64:
            bytes_per_element = 8
        elif param.dtype == torch.int32:
            bytes_per_element = 4
        elif param.dtype == torch.int8 or param.dtype == torch.uint8:
            bytes_per_element = 1
        else:
            bytes_per_element = 4  # Default fallback
        
        model_size_bytes += param.numel() * bytes_per_element
    
    # Convert to MB
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Add 20% overhead for optimizer states, gradients, etc.
    model_size_mb = model_size_mb * 1.2
    
    return int(model_size_mb)


def print_memory_stats(device: torch.device, tag: str = ""):
    """
    Print memory statistics for the current device.
    
    Args:
        device: The device to check memory for
        tag: Optional tag to identify the print point
    """
    prefix = f"[{tag}] " if tag else ""
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        
        logger.info(f"{prefix}CUDA Memory: "
                    f"Allocated={allocated:.1f}MB, "
                    f"Max Allocated={max_allocated:.1f}MB, "
                    f"Reserved={reserved:.1f}MB")
    
    elif device.type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
        try:
            allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
            logger.info(f"{prefix}MPS Memory: Allocated={allocated:.1f}MB")
        except:
            logger.info(f"{prefix}MPS Memory: Stats not available")
    
    # System memory for all devices
    vm = psutil.virtual_memory()
    logger.info(f"{prefix}System Memory: "
                f"Total={vm.total/(1024*1024*1024):.1f}GB, "
                f"Available={vm.available/(1024*1024*1024):.1f}GB, "
                f"Used={vm.percent:.1f}%")


def clear_memory(device: Optional[torch.device] = None):
    """
    Clear device memory - simplified interface for backward compatibility.
    
    Args:
        device: Optional device to clear memory for. If None, detects device automatically.
    """
    # Run garbage collection
    gc.collect()
    
    # If device not provided, detect it
    if device is None:
        if torch.cuda.is_available():
            device_type = 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and \
             torch.backends.mps.is_available():
            device_type = 'mps'
        else:
            device_type = 'cpu'
    else:
        device_type = device.type
    
    # Clear appropriate cache
    if device_type == 'cuda':
        torch.cuda.empty_cache()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    elif device_type == 'mps':
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Report memory cleared
    logger.info(f"Memory cleared for {device_type} device")


def clear_memory_with_config(device: torch.device, config: MemoryConfig):
    """
    Clear device memory based on configuration - advanced interface.
    
    Args:
        device: The device to clear memory for
        config: Memory configuration
    """
    # Run Python garbage collection first
    gc.collect()
    
    if device.type == 'cuda':
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
    elif device.type == 'mps':
        # Clear MPS cache if available
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


def should_clear_memory(batch_idx: int, device_type: str, config: MemoryConfig) -> bool:
    """
    Determine if memory should be cleared at this point.
    
    Args:
        batch_idx: Current batch index
        device_type: Device type string
        config: Memory configuration
    
    Returns:
        bool: True if memory should be cleared
    """
    if not config.periodic_gc:
        return False
    
    # Get the device-specific interval
    if device_type == 'cuda':
        interval = config.cuda_config['empty_cache_interval']
    elif device_type == 'mps':
        interval = config.mps_config['empty_cache_interval'] 
    else:
        interval = config.gc_interval
    
    # Don't clear if interval is 0 (disabled)
    if interval <= 0:
        return False
    
    # Clear every N batches
    return (batch_idx + 1) % interval == 0


def configure_training_environment(
    device: torch.device,
    config: MemoryConfig
) -> None:
    """
    Configure the training environment based on the device and configuration.
    
    Args:
        device: Target device
        config: Memory configuration
    """
    device_type = device.type
    
    # Common settings
    torch.set_grad_enabled(True)
    
    # CUDA-specific settings
    if device_type == 'cuda':
        # Enable cudnn benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        
        # Deterministic mode is slower but more reproducible
        torch.backends.cudnn.deterministic = False
        
        # Clear CUDA cache at the start
        torch.cuda.empty_cache()
        
        # Set matmul precision if available
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    
    # CPU-specific settings
    elif device_type == 'cpu':
        # Set thread settings
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(config.cpu_config['num_threads'])
        
        # Set MKL threads if available
        try:
            import torch.utils.mkldnn as mkldnn
            if config.cpu_config['use_mkldnn'] and mkldnn is not None:
                mkldnn.set_num_threads(config.cpu_config['num_threads'])
        except:
            pass
    
    # MPS doesn't have many configuration options yet
    elif device_type == 'mps':
        # Clear MPS cache at the start
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
    # Initial memory stats
    if config.enable_memory_tracking:
        print_memory_stats(device, "Initial")


def configure_inference_environment(
    device: torch.device,
    config: MemoryConfig
) -> None:
    """
    Configure the inference environment based on the device and configuration.
    
    Args:
        device: Target device
        config: Memory configuration
    """
    device_type = device.type
    
    # Common settings - disable gradients for inference
    torch.set_grad_enabled(False)
    
    # CUDA-specific settings
    if device_type == 'cuda':
        # Enable cudnn benchmarking for faster inference
        torch.backends.cudnn.benchmark = True
        
        # Clear cache before inference
        torch.cuda.empty_cache()
    
    # CPU-specific settings
    elif device_type == 'cpu':
        # Set optimal thread settings for inference
        if hasattr(torch, 'set_num_threads'):
            # Use all available cores for inference
            num_threads = os.cpu_count() if os.cpu_count() else 4
            torch.set_num_threads(num_threads)
    
    # MPS-specific settings
    elif device_type == 'mps':
        # Clear cache before inference
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Initial memory stats
    if config.enable_memory_tracking:
        print_memory_stats(device, "Inference Start")


def create_optimizer_with_memory_optimizations(
    model: torch.nn.Module,
    learning_rate: float,
    device_type: str,
    config: MemoryConfig
) -> torch.optim.Optimizer:
    """
    Create an optimizer with memory optimizations.
    
    Args:
        model: PyTorch model
        learning_rate: Base learning rate
        device_type: Device type
        config: Memory configuration
        
    Returns:
        torch.optim.Optimizer: Optimized optimizer
    """
    # Use Adam/AdamW/SGD depending on device and config
    weight_decay = 0.01  # Default weight decay
    
    # Set up parameter groups - embeddings often need different lr
    embedding_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    # Different parameter groups
    param_groups = [
        {'params': embedding_params, 'lr': learning_rate * 0.1, 'weight_decay': weight_decay * 0.1},
        {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay}
    ]
    
    # Create optimizer based on device
    if device_type == 'cuda':
        # AdamW with good defaults for CUDA
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    elif device_type == 'mps':
        # Adam is more stable on MPS
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),  # Slightly lower beta2 for MPS stability
            eps=1e-8,
            weight_decay=weight_decay
        )
    else:  # CPU
        # SGD often performs well on CPU with good memory characteristics
        if config.cpu_config['optimize_memory_usage']:
            optimizer = torch.optim.SGD(
                param_groups,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay
            )
    
    return optimizer


class MemoryOptimizer:
    """
    Helper class to manage memory optimizations throughout the training process.
    """
    
    def __init__(self, device: torch.device, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory optimizer.
        
        Args:
            device: Target device
            config: Optional memory configuration (created from device if None)
        """
        self.device = device
        self.device_type = device.type
        
        # Create config if not provided
        if config is None:
            self.config = create_memory_config(self.device_type, "balanced")
        else:
            self.config = config
        
        # Initialize counters
        self.batch_counter = 0
        self.peak_memory = 0
        self.track_memory = self.config.enable_memory_tracking
        
        # Configure environment
        configure_training_environment(device, self.config)
        
        # Print initial configuration
        logger.info(f"Memory optimizer initialized for {self.device_type}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Using reduced precision: {self.config.reduce_precision}")
        
        # Device-specific settings
        if self.device_type == 'cuda':
            logger.info(f"CUDA AMP enabled: {self.config.cuda_config['use_amp']}")
            logger.info(f"CUDA model compilation: {self.config.cuda_config['compile_model']}")
        elif self.device_type == 'mps':
            logger.info(f"MPS cache interval: {self.config.mps_config['empty_cache_interval']}")
        elif self.device_type == 'cpu':
            logger.info(f"CPU threads: {self.config.cpu_config['num_threads']}")
    
    def before_batch(self, batch_idx: int):
        """
        Perform operations before processing a batch.
        
        Args:
            batch_idx: Current batch index
        """
        self.batch_counter = batch_idx
        
        # Clear memory if needed
        if should_clear_memory(batch_idx, self.device_type, self.config):
            clear_memory_with_config(self.device, self.config)
            # Commented out to reduce log clutter
            # if self.track_memory:
            #     logger.info(f"Memory cleared at batch {batch_idx}")
    
    def after_batch(self, batch_idx: int):
        """
        Perform operations after processing a batch.
        
        Args:
            batch_idx: Current batch index
        """
        # Track memory
        if self.track_memory:
            if self.device_type == 'cuda':
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                self.peak_memory = max(self.peak_memory, current_memory)
            elif self.device_type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
                try:
                    current_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
                    self.peak_memory = max(self.peak_memory, current_memory)
                except:
                    pass
            
            # Commented out to reduce log clutter
            # Periodically print memory stats
            # if batch_idx % 100 == 0:
            #     print_memory_stats(self.device, f"Batch {batch_idx}")
    
    def before_epoch(self, epoch: int):
        """
        Perform operations before starting an epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Clear memory at the start of each epoch
        clear_memory_with_config(self.device, self.config)
        
        # Reset batch counter
        self.batch_counter = 0
        
        # Reset peak memory
        self.peak_memory = 0
        
        # Print epoch memory stats
        if self.track_memory:
            print_memory_stats(self.device, f"Epoch {epoch} Start")
    
    def after_epoch(self, epoch: int):
        """
        Perform operations after completing an epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Clear memory at the end of each epoch
        clear_memory_with_config(self.device, self.config)
        
        # Print epoch memory stats
        if self.track_memory:
            if self.peak_memory > 0:
                logger.info(f"Peak memory during epoch {epoch}: {self.peak_memory:.1f}MB")
            print_memory_stats(self.device, f"Epoch {epoch} End")
    
    def optimize_tensor(self, tensor: torch.Tensor, is_embedding: bool = False, is_index: bool = False) -> torch.Tensor:
        """
        Optimize a tensor for the current device.
        
        Args:
            tensor: Input tensor
            is_embedding: Whether the tensor contains embeddings
            is_index: Whether the tensor contains indices
            
        Returns:
            torch.Tensor: Optimized tensor
        """
        return optimize_tensor_for_device(
            tensor, self.device, self.config,
            is_embedding=is_embedding, is_index=is_index
        )
    
    def optimize_batch(self, batch: Union[Dict[str, Any], torch.Tensor, List]) -> Union[Dict[str, Any], torch.Tensor, List]:
        """
        Optimize a batch for the current device.
        
        Args:
            batch: Input batch
            
        Returns:
            Optimized batch
        """
        return move_batch_to_device(batch, self.device, self.config)


# Create AMP context manager that works across devices
class AMPManager:
    """
    Context manager for automatic mixed precision that works across devices.
    """
    
    def __init__(self, device: torch.device, config: MemoryConfig):
        """
        Initialize the AMP manager.
        
        Args:
            device: Target device
            config: Memory configuration
        """
        self.device = device
        self.device_type = device.type
        self.config = config
        
        # Determine if AMP should be used
        if self.device_type == 'cuda':
            self.use_amp = config.cuda_config['use_amp'] and hasattr(torch.cuda, 'amp')
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            # MPS and CPU don't support AMP yet
            self.use_amp = False
            self.scaler = None
    
    def __enter__(self):
        """
        Enter the AMP context.
        
        Returns:
            self: The current instance
        """
        if self.use_amp:
            self._ctx = torch.cuda.amp.autocast().__enter__()
            return self
        else:
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the AMP context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if self.use_amp:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale a loss value for AMP.
        
        Args:
            loss: Input loss tensor
            
        Returns:
            torch.Tensor: Scaled loss
        """
        if self.use_amp:
            return self.scaler.scale(loss)
        else:
            return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform an optimizer step with AMP scaling.
        
        Args:
            optimizer: PyTorch optimizer
        """
        if self.use_amp:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


class GradientAccumulator:
    """
    Helper class for gradient accumulation.
    """
    
    def __init__(self, steps: int, amp_manager: Optional[AMPManager] = None):
        """
        Initialize the gradient accumulator.
        
        Args:
            steps: Number of accumulation steps
            amp_manager: Optional AMP manager for loss scaling
        """
        self.steps = max(1, steps)  # Ensure at least 1 step
        self.current_step = 0
        self.amp_manager = amp_manager
    
    def backward(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Perform backward pass with gradient accumulation.
        
        Args:
            loss: Loss tensor
            
        Returns:
            torch.Tensor: Original loss value (for metrics)
        """
        # Store original loss for metrics
        original_loss = loss.detach()
        
        # Scale loss for accumulation
        loss = loss / self.steps
        
        # Scale for AMP if applicable
        if self.amp_manager is not None:
            scaled_loss = self.amp_manager.scale_loss(loss)
            scaled_loss.backward()
        else:
            loss.backward()
        
        # Increment step
        self.current_step += 1
        
        return original_loss
    
    def should_step(self) -> bool:
        """
        Check if optimizer should step now.
        
        Returns:
            bool: True if optimizer should step
        """
        return self.current_step % self.steps == 0 or self.current_step == self.steps
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step if needed.
        
        Args:
            optimizer: PyTorch optimizer
        
        Returns:
            bool: True if step was performed
        """
        if self.should_step():
            if self.amp_manager is not None:
                self.amp_manager.step(optimizer)
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            return True
        
        return False
    
    def reset(self):
        """Reset the accumulation counter."""
        self.current_step = 0


# Export key utility functions
__all__ = [
    'MemoryConfig',
    'detect_device',
    'create_memory_config',
    'get_optimal_batch_size',
    'optimize_tensor_for_device',
    'move_batch_to_device',
    'optimize_model_for_inference',
    'get_model_size',
    'print_memory_stats',
    'clear_memory',
    'clear_memory_with_config',
    'configure_training_environment',
    'configure_inference_environment',
    'create_optimizer_with_memory_optimizations',
    'MemoryOptimizer',
    'AMPManager',
    'GradientAccumulator'
]