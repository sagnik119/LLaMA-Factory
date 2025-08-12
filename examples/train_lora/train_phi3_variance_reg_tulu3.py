#!/usr/bin/env python3

"""
Training script for Phi-3 Mini 4K Instruct with variance regularization using Tulu-3 SFT mixture dataset.

This script demonstrates:
1. Using the large-scale Tulu-3 SFT mixture dataset (100K+ samples)
2. Variance regularization on post-attention RMSNorm layers
3. LoRA fine-tuning with proper configuration
4. Multi-GPU distributed training support

Usage:
    # Single GPU
    python examples/train_lora/train_phi3_variance_reg_tulu3.py

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1 python examples/train_lora/train_phi3_variance_reg_tulu3.py

    # Using llamafactory-cli
    llamafactory-cli train examples/train_lora/phi3_variance_reg_tulu3.yaml
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llamafactory.train.tuner import run_exp

def main():
    # Training arguments for Phi-3 with variance regularization on Tulu-3 dataset
    args = {
        # Model configuration
        "model_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_name_or_path": None,
        
        # Training method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        
        # Dataset configuration - Tulu-3 SFT mixture
        "dataset": "tulu_3_sft_mixture",
        "template": "phi",
        "cutoff_len": 1024,
        "max_samples": 100000,  # Use 100K samples
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        # Output configuration
        "output_dir": "./saves/phi3-variance-reg-tulu3",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        # Training hyperparameters
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5.0e-05,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        
        # Evaluation configuration
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 500,
        
        # RMSNorm Variance Regularization
        "use_variance_regularization": True,
        "variance_reg_layers": "model.layers.0.post_attention_layernorm,model.layers.1.post_attention_layernorm,model.layers.2.post_attention_layernorm",
        "variance_reg_weight": 0.01,
        "variance_reg_target": 1.0,
        "variance_reg_norm_type": "l2",
    }
    
    print("=" * 80)
    print("🚀 Starting Phi-3 Mini 4K Instruct Training with Variance Regularization")
    print("📊 Dataset: Tulu-3 SFT Mixture (100K samples)")
    print("🔧 Method: LoRA Fine-tuning")
    print("📈 Regularization: RMSNorm Variance Regularization")
    print("=" * 80)
    
    # Print key configuration
    print(f"Model: {args['model_name_or_path']}")
    print(f"Dataset: {args['dataset']} (max_samples: {args['max_samples']:,})")
    print(f"Output: {args['output_dir']}")
    print(f"Variance Regularization: {args['use_variance_regularization']}")
    print(f"Regularization Weight: {args['variance_reg_weight']}")
    print(f"Target Layers: {len(args['variance_reg_layers'].split(','))} layers")
    print()
    
    # Run training
    try:
        run_exp(args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()