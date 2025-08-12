#!/usr/bin/env python3

"""
Variance Regularization Training Script for Phi-3 Mini 4K Instruct

This script demonstrates how to use variance regularization to control the variance
of RMSNorm output matrices during fine-tuning. Variance regularization helps
stabilize training and can improve model performance by encouraging consistent
activation patterns.

Usage:
    python examples/train_lora/train_phi3_variance_reg.py

Features:
- Variance regularization for post-attention RMSNorm layers
- RMSNorm-only training (freezes all other parameters)
- Multi-GPU support with Gloo backend
- Configurable target variance and regularization weight
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Configuration for variance regularization training
    config = {
        # Model configuration
        "model_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "freeze",
        "rmsnorm_only_training": True,
        
        # Variance regularization (NEW FEATURE)
        "use_variance_regularization": True,
        "variance_reg_layers": "2,4,6",  # Layers to apply variance regularization
        "variance_reg_weight": 1.0,      # Weight for variance regularization loss
        "variance_reg_target": 1.0,      # Target variance value
        "variance_reg_norm_type": "post_attention_layernorm",  # Which norm layers to regularize
        
        # Dataset configuration
        "dataset": "identity",
        "dataset_dir": "./data",
        "template": "phi",
        "cutoff_len": 1024,
        "max_samples": 100000,
        
        # Training configuration
        "output_dir": "./saves/phi3_variance_reg",
        "num_train_epochs": 10.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 250,
        "val_size": 0.1,
        "bf16": True,
        
        # Distributed training
        "ddp_backend": "gloo",
        "max_grad_norm": 1.0,
        "save_total_limit": 3,
    }
    
    # Build command
    cmd = ["llamafactory-cli", "train"]
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print("=" * 80)
    print("VARIANCE REGULARIZATION TRAINING")
    print("=" * 80)
    print(f"Model: {config['model_name_or_path']}")
    print(f"Variance Regularization Layers: {config['variance_reg_layers']}")
    print(f"Variance Regularization Weight: {config['variance_reg_weight']}")
    print(f"Target Variance: {config['variance_reg_target']}")
    print(f"Norm Type: {config['variance_reg_norm_type']}")
    print(f"Output Directory: {config['output_dir']}")
    print("=" * 80)
    print()
    
    print("Command to be executed:")
    print(" ".join(cmd))
    print()
    
    # Execute training
    try:
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())