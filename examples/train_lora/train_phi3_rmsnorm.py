#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script for training Phi-3 Mini 4K Instruct with RMSNorm regularization.

This script demonstrates how to:
1. Train only RMSNorm parameters while freezing all other parameters
2. Apply regularization to post-attention RMSNorm outputs in layers 2 and 4
3. Regularize row (token) norms to encourage specific norm values

Usage:
    python examples/train_lora/train_phi3_rmsnorm.py
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llamafactory.hparams import get_train_args
from llamafactory.train.sft import run_sft


def main():
    """Main training function."""
    
    # Configuration for Phi-3 Mini 4K Instruct with RMSNorm regularization
    args = {
        # Model configuration
        "model_name_or_path": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        
        # Training method
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "freeze",
        "freeze_trainable_modules": "all",
        
        # RMSNorm-specific settings
        "rmsnorm_only_training": True,
        "use_rmsnorm_regularization": True,
        "rmsnorm_reg_layers": "2,4",  # Regularize layers 2 and 4
        "rmsnorm_reg_weight": 0.01,   # Regularization weight
        "rmsnorm_reg_target_norm": 0.0,  # Target norm value (updated to 0.0)
        
        # Dataset configuration
        "dataset": "identity",
        "template": "phi",
        "cutoff_len": 1024,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        
        # Output configuration
        "output_dir": "./saves/phi3-rmsnorm-reg",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        
        # Training configuration
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0.1,
        "fp16": True,
        
        # Evaluation configuration
        "val_size": 0.1,
        "per_device_eval_batch_size": 1,
        "eval_strategy": "steps",
        "eval_steps": 250,
    }
    
    # Parse arguments
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    # Print configuration summary
    print("=" * 50)
    print("Phi-3 Mini 4K Instruct RMSNorm Regularization Training")
    print("=" * 50)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Finetuning type: {finetuning_args.finetuning_type}")
    print(f"RMSNorm-only training: {finetuning_args.rmsnorm_only_training}")
    print(f"RMSNorm regularization: {finetuning_args.use_rmsnorm_regularization}")
    print(f"Regularization layers: {finetuning_args.rmsnorm_reg_layers}")
    print(f"Regularization weight: {finetuning_args.rmsnorm_reg_weight}")
    print(f"Target norm: {finetuning_args.rmsnorm_reg_target_norm}")
    print(f"Output directory: {training_args.output_dir}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Run training
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
    
    print("Training completed successfully!")
    print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()