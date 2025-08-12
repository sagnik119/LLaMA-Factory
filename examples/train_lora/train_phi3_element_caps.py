#!/usr/bin/env python3

"""
Example script for training Phi-3 Mini 4K Instruct with RMSNorm regularization 
and element-specific capping functionality.

This script demonstrates how to:
1. Cap specific elements of RMSNorm parameters to fixed values
2. Freeze those capped elements during training
3. Apply regularization to specific layers
4. Train only RMSNorm parameters while freezing all others

Usage:
    python examples/train_lora/train_phi3_element_caps.py
"""

import sys
import os

# Add the src directory to the path so we can import llamafactory modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from llamafactory.train.tuner import run_exp

def main():
    # Training arguments for Phi-3 Mini 4K Instruct with element capping
    args = {
        # Model configuration
        "model_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        
        # Training method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "freeze",
        "freeze_trainable_modules": "all",
        
        # RMSNorm-only training
        "rmsnorm_only_training": True,
        
        # RMSNorm regularization
        "use_rmsnorm_regularization": True,
        "rmsnorm_reg_layers": "2,4",
        "rmsnorm_reg_weight": 0.01,
        "rmsnorm_reg_target_norm": 0.0,
        
        # Element-specific capping (NEW FEATURE)
        "rmsnorm_element_caps": "2.post_attention_layernorm.1251=0.1",
        "freeze_capped_elements": True,
        
        # Dataset configuration
        "dataset": "identity",
        "template": "phi",
        "cutoff_len": 1024,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        # Output configuration
        "output_dir": "./saves/phi3-element-caps",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        # Training hyperparameters
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        
        # Distributed training
        "ddp_backend": "gloo",  # Use Gloo backend for Phi-3 stability
        
        # Evaluation
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 250,
    }
    
    print("🚀 Starting Phi-3 Mini 4K Instruct training with RMSNorm element capping...")
    print(f"📊 Element capping configuration: {args['rmsnorm_element_caps']}")
    print(f"🔒 Freeze capped elements: {args['freeze_capped_elements']}")
    print(f"📁 Output directory: {args['output_dir']}")
    
    # Run the training
    run_exp(args)
    
    print("✅ Training completed!")
    print(f"📁 Model saved to: {args['output_dir']}")
    print("\n🔍 What happened during training:")
    print("1. Element 1251 of layer 2's post_attention_layernorm was capped to 0.1")
    print("2. That specific element was frozen (won't be updated during training)")
    print("3. All other RMSNorm parameters were trained normally")
    print("4. Regularization was applied to layers 2 and 4")
    print("\n💡 Use the inference script to test the trained model:")
    print(f"   python inference_rmsnorm.py --checkpoint_dir {args['output_dir']} --interactive")

if __name__ == "__main__":
    main()