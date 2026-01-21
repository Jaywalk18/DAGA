from .backbones import load_dinov3_backbone, get_attention_map, process_attention_weights, compute_daga_guidance_map
from .daga import DAGA, create_daga, process_attention_to_guidance, AttentionHook, AttentionHookManager
from .heads import ClassificationHead, LinearSegmentationHead, DetectionHead
from .utils import setup_environment, get_base_model, save_checkpoint, setup_logging, create_dataloaders, finalize_experiment

# Backward compatibility aliases
from .daga import DAGAPlusUnified, create_daga_plus_unified

__all__ = [
    # Backbones
    'load_dinov3_backbone',
    'get_attention_map',
    'process_attention_weights',
    'compute_daga_guidance_map',

    # DAGA
    'DAGA',
    'create_daga',
    'process_attention_to_guidance',
    'AttentionHook',
    'AttentionHookManager',
    
    # Backward compatibility
    'DAGAPlusUnified',
    'create_daga_plus_unified',

    # Heads
    'ClassificationHead',
    'LinearSegmentationHead',
    'DetectionHead',

    # Utils
    'setup_environment',
    'get_base_model',
    'save_checkpoint',
    'setup_logging',
    'create_dataloaders',
    'finalize_experiment',
]
