#!/usr/bin/env python3
"""
Training script for Phi-3 Mini 4K Instruct with BOS token zeroing and LoRA fine-tuning.

This script:
1. Adds BOS tokens to the beginning of every sequence
2. Zeros out the embedding output for position 0 (BOS token position)
3. Fine-tunes the model using LoRA on all linear layers

Usage:
    python examples/train_lora/train_phi3_bos_zero.py
    python examples/train_lora/train_phi3_bos_zero.py --config examples/train_lora/phi3_bos_zero_lora.yaml
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llamafactory.train.tuner import run_exp


def main():
    """Main training function."""
    # Set default config if none provided
    config_path = "examples/train_lora/phi3_bos_zero_lora.yaml"
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure the config file exists or provide a different path.")
        sys.exit(1)
    
    print("🚀 Starting Phi-3 BOS Zero LoRA Training")
    print(f"📋 Config: {config_path}")
    print("🎯 Features:")
    print("   - BOS tokens added to all sequences")
    print("   - Position 0 embeddings zeroed out")
    print("   - LoRA fine-tuning on all linear layers")
    print("=" * 50)
    
    # Set the config file as command line argument
    sys.argv = ["train_phi3_bos_zero.py", "--config", config_path]
    
    # Run the training
    run_exp()


if __name__ == "__main__":
    main()