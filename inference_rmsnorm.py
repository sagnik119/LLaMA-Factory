#!/usr/bin/env python3

"""
Inference script for RMSNorm-only fine-tuned Phi-3 Mini 4K Instruct models.
This script loads the base model and applies the fine-tuned RMSNorm weights.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any

def load_rmsnorm_weights(model: torch.nn.Module, checkpoint_dir: str) -> bool:
    """
    Load RMSNorm weights from checkpoint directory and apply them to the model.
    
    Args:
        model: The base model to apply weights to
        checkpoint_dir: Directory containing rmsnorm_weights.pt and rmsnorm_config.json
        
    Returns:
        bool: True if weights were successfully loaded, False otherwise
    """
    weights_path = os.path.join(checkpoint_dir, "rmsnorm_weights.pt")
    config_path = os.path.join(checkpoint_dir, "rmsnorm_config.json")
    
    if not os.path.exists(weights_path):
        print(f"❌ RMSNorm weights not found at: {weights_path}")
        return False
        
    if not os.path.exists(config_path):
        print(f"❌ RMSNorm config not found at: {config_path}")
        return False
    
    try:
        # Load the saved RMSNorm weights
        rmsnorm_weights = torch.load(weights_path, map_location="cpu")
        print(f"✅ Loaded RMSNorm weights from: {weights_path}")
        
        # Load the config to understand what was saved
        with open(config_path, 'r') as f:
            rmsnorm_config = json.load(f)
        print(f"✅ Loaded RMSNorm config from: {config_path}")
        
        # Apply the weights to the model
        model_state_dict = model.state_dict()
        updated_params = 0
        
        for param_name, param_value in rmsnorm_weights.items():
            if param_name in model_state_dict:
                model_state_dict[param_name].copy_(param_value)
                updated_params += 1
                print(f"✅ Updated parameter: {param_name}")
            else:
                print(f"⚠️  Parameter not found in model: {param_name}")
        
        print(f"✅ Successfully updated {updated_params} RMSNorm parameters")
        print(f"📊 Training info: {rmsnorm_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading RMSNorm weights: {e}")
        return False

def generate_response(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate a response using the model.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        str: Generated response
    """
    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    if response.startswith(formatted_prompt):
        response = response[len(formatted_prompt):].strip()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Inference with RMSNorm-only fine-tuned Phi-3 model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Path to base model"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Directory containing RMSNorm checkpoint (rmsnorm_weights.pt and rmsnorm_config.json)"
    )
    parser.add_argument(
        "--prompt", 
        type=str,
        help="Single prompt to generate response for"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Loading RMSNorm-only fine-tuned Phi-3 model for inference...")
    print(f"📁 Base model: {args.model_path}")
    print(f"📁 Checkpoint directory: {args.checkpoint_dir}")
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        print("🧠 Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device != "auto" else None,
            trust_remote_code=True
        )
        
        # Load and apply RMSNorm weights
        print("🔧 Loading RMSNorm fine-tuned weights...")
        if not load_rmsnorm_weights(model, args.checkpoint_dir):
            print("❌ Failed to load RMSNorm weights. Exiting.")
            sys.exit(1)
        
        model.eval()
        print("✅ Model ready for inference!")
        
        # Single prompt mode
        if args.prompt:
            print(f"\n💭 Prompt: {args.prompt}")
            response = generate_response(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"🤖 Response: {response}")
        
        # Interactive mode
        elif args.interactive:
            print("\n🎯 Interactive mode started. Type 'quit' to exit.")
            print("💡 Try asking: 'Who are you?' or 'What is your name?' to see the fine-tuned identity!")
            
            while True:
                try:
                    prompt = input("\n💭 You: ").strip()
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not prompt:
                        continue
                    
                    response = generate_response(
                        model, tokenizer, prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p
                    )
                    print(f"🤖 Assistant: {response}")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
        
        else:
            print("❌ Please provide either --prompt or --interactive flag")
            print("\nExample usage:")
            print(f"  python {sys.argv[0]} --checkpoint_dir ./saves/phi3_rmsnorm_reg_multi --interactive")
            print(f"  python {sys.argv[0]} --checkpoint_dir ./saves/phi3_rmsnorm_reg_multi --prompt 'Who are you?'")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()