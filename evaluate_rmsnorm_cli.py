#!/usr/bin/env python3
"""
Simple CLI script to evaluate RMSNorm fine-tuned model using LLaMA-Factory's built-in evaluation system.

This script creates the necessary configuration and runs evaluation using LLaMA-Factory's CLI.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

def create_eval_config(model_name: str, rmsnorm_weights_path: str, output_dir: str) -> str:
    """Create evaluation configuration file."""
    
    # Extract checkpoint directory from rmsnorm_weights_path
    checkpoint_dir = str(Path(rmsnorm_weights_path).parent)
    
    config = {
        # Model configuration
        "model_name_or_path": model_name,
        "adapter_name_or_path": checkpoint_dir,  # Point to checkpoint directory
        "trust_remote_code": True,
        
        # Evaluation configuration
        "stage": "sft",
        "do_eval": True,
        "finetuning_type": "full",
        
        # Dataset configuration
        "eval_dataset": "alpaca_en",  # Use a simple dataset for evaluation
        "dataset_dir": "data",
        "template": "phi",
        "cutoff_len": 1024,
        "max_samples": 100,  # Limit samples for faster evaluation
        
        # Evaluation parameters
        "per_device_eval_batch_size": 1,
        "predict_with_generate": True,
        "max_new_tokens": 128,
        "top_p": 0.9,
        "temperature": 0.7,
        
        # Output configuration
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "report_to": "none",
        
        # Performance settings
        "bf16": True,
        "ddp_find_unused_parameters": False,
    }
    
    return config

def create_perplexity_eval_config(model_name: str, rmsnorm_weights_path: str, output_dir: str) -> str:
    """Create configuration for perplexity evaluation."""
    
    checkpoint_dir = str(Path(rmsnorm_weights_path).parent)
    
    config = {
        # Model configuration
        "model_name_or_path": model_name,
        "adapter_name_or_path": checkpoint_dir,
        "trust_remote_code": True,
        
        # Evaluation configuration
        "stage": "pt",  # Use pretraining stage for perplexity
        "do_eval": True,
        "finetuning_type": "full",
        
        # Dataset configuration
        "eval_dataset": "c4_en",  # Use C4 for perplexity evaluation
        "dataset_dir": "data",
        "template": "empty",  # No template for perplexity
        "cutoff_len": 1024,
        "max_samples": 500,
        
        # Evaluation parameters
        "per_device_eval_batch_size": 2,
        
        # Output configuration
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "report_to": "none",
        
        # Performance settings
        "bf16": True,
        "ddp_find_unused_parameters": False,
    }
    
    return config

def run_llamafactory_eval(config_path: str) -> bool:
    """Run LLaMA-Factory evaluation with the given config."""
    
    cmd = [
        sys.executable, "-m", "llamafactory.cli", "train",
        "--config", config_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully!")
        print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate RMSNorm model using LLaMA-Factory CLI")
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct",
                       help="Model name or path")
    parser.add_argument("--rmsnorm_weights_path", required=True,
                       help="Path to the saved RMSNorm weights (.pt file)")
    parser.add_argument("--output_dir", default="./rmsnorm_eval_cli_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--eval_type", choices=["generation", "perplexity", "both"], 
                       default="both", help="Type of evaluation to run")
    
    args = parser.parse_args()
    
    # Verify rmsnorm weights file exists
    if not os.path.exists(args.rmsnorm_weights_path):
        print(f"Error: RMSNorm weights file not found: {args.rmsnorm_weights_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    # Run generation evaluation
    if args.eval_type in ["generation", "both"]:
        print("\n" + "="*50)
        print("RUNNING GENERATION EVALUATION")
        print("="*50)
        
        gen_config = create_eval_config(args.model_name, args.rmsnorm_weights_path, 
                                       os.path.join(args.output_dir, "generation"))
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(gen_config, f, indent=2)
            gen_config_path = f.name
        
        try:
            gen_success = run_llamafactory_eval(gen_config_path)
            success = success and gen_success
        finally:
            os.unlink(gen_config_path)
    
    # Run perplexity evaluation
    if args.eval_type in ["perplexity", "both"]:
        print("\n" + "="*50)
        print("RUNNING PERPLEXITY EVALUATION")
        print("="*50)
        
        ppl_config = create_perplexity_eval_config(args.model_name, args.rmsnorm_weights_path,
                                                  os.path.join(args.output_dir, "perplexity"))
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ppl_config, f, indent=2)
            ppl_config_path = f.name
        
        try:
            ppl_success = run_llamafactory_eval(ppl_config_path)
            success = success and ppl_success
        finally:
            os.unlink(ppl_config_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"RMSNorm weights: {args.rmsnorm_weights_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluation type: {args.eval_type}")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    print("="*50)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()