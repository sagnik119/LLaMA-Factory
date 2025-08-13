#!/usr/bin/env python3

"""
Evaluation script for BOS Zero LoRA fine-tuned models.
This script:
1. Loads LoRA fine-tuned models (not just RMSNorm)
2. Adds BOS tokens to evaluation sequences
3. Applies position 0 embedding scaling during evaluation
4. Compares BOS zero models with baseline models
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import PeftModel
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BOSZeroEvaluationHook:
    """Hook to apply BOS scaling during evaluation (same as training)."""
    
    def __init__(self, apply_scaling: bool = True, scaling_factor: float = 0.01):
        self.apply_scaling = apply_scaling
        self.scaling_factor = scaling_factor
        self.hook_handle = None
        
    def register_hook(self, model):
        """Register the BOS scaling hook on the embedding layer."""
        # Find embedding layer
        embedding_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and 'embed_tokens' in name:
                embedding_layer = module
                logger.info(f"✅ Found embedding layer for evaluation: {name}")
                break
                
        if embedding_layer is None:
            logger.warning("⚠️ No embedding layer found - BOS scaling will not be applied during evaluation")
            return
            
        def bos_scaling_hook(module, input, output):
            """Apply BOS scaling during evaluation (same as training)."""
            try:
                if output.dim() == 3 and output.size(1) > 0 and self.apply_scaling:
                    # Create scaled output (same as training)
                    scaled_output = output.clone()
                    
                    # Debug: Log hook activation (only first few times)
                    if not hasattr(self, '_hook_debug_count'):
                        self._hook_debug_count = 0
                    
                    if self._hook_debug_count < 3:
                        logger.info(f"🎯 BOS scaling hook activated! Batch size: {output.size(0)}, Seq len: {output.size(1)}")
                        logger.info(f"🎯 Original position 0 norm: {output[:, 0, :].norm().item():.6f}")
                        self._hook_debug_count += 1
                    
                    # Scale position 0 embeddings (same as training)
                    scaled_output[:, 0, :] = output[:, 0, :] * self.scaling_factor
                    
                    if self._hook_debug_count <= 3:
                        logger.info(f"🎯 Scaled position 0 norm: {scaled_output[:, 0, :].norm().item():.6f}")
                        logger.info(f"🎯 Scaling factor applied: {self.scaling_factor}")
                    
                    return scaled_output
                return output
            except Exception as e:
                logger.warning(f"⚠️ Error in evaluation BOS scaling hook: {e}")
                return output
                
        self.hook_handle = embedding_layer.register_forward_hook(bos_scaling_hook)
        reduction_percent = (1 - self.scaling_factor) * 100
        logger.info(f"✅ BOS scaling hook registered for evaluation (scaling: {self.scaling_factor}, {reduction_percent:.0f}% reduction)")
        
    def remove_hook(self):
        """Remove the hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info("🧹 BOS scaling hook removed")


def add_bos_tokens_to_batch(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Add BOS tokens to input sequences (same as training preprocessing)."""
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 1
        
    batch_size, seq_len = input_ids.shape
    new_input_ids = []
    
    for i in range(batch_size):
        seq = input_ids[i].tolist()
        # Add BOS token if not already present
        if len(seq) > 0 and seq[0] != bos_token_id:
            seq = [bos_token_id] + seq
        new_input_ids.append(seq)
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in new_input_ids)
    padded_sequences = []
    for seq in new_input_ids:
        if len(seq) < max_len:
            seq = seq + [tokenizer.pad_token_id] * (max_len - len(seq))
        padded_sequences.append(seq)
    
    return torch.tensor(padded_sequences, device=input_ids.device)


def load_bos_zero_model(model_path: str, adapter_path: Optional[str] = None, apply_bos_scaling: bool = True, scaling_factor: float = 0.01):
    """Load model with BOS scaling evaluation setup."""
    logger.info(f"Loading model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter if provided
    if adapter_path:
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for evaluation
    
    # Set up BOS scaling hook
    bos_hook = BOSZeroEvaluationHook(apply_bos_scaling, scaling_factor)
    bos_hook.register_hook(model)
    
    model.eval()
    return model, tokenizer, bos_hook


def evaluate_perplexity(model, tokenizer, dataset, max_samples: int = 1000, batch_size: int = 4) -> float:
    """Evaluate perplexity with BOS token handling."""
    logger.info(f"Evaluating perplexity on {min(max_samples, len(dataset))} samples...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    debug_batch_count = 0
    
    # Take subset of dataset
    eval_dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Evaluating"):
            batch_end = min(i + batch_size, len(eval_dataset))
            batch_texts = [eval_dataset[j]['text'] for j in range(i, batch_end)]
            
            # Filter out empty or very short texts
            batch_texts = [text.strip() for text in batch_texts if text and len(text.strip()) > 10]
            if not batch_texts:
                continue
            
            # Debug: Check text content
            if debug_batch_count == 0:
                logger.info(f"🔍 Debug - Sample text: {batch_texts[0][:100]}...")
                # Test tokenization of a simple sentence
                test_tokens = tokenizer("Hello world", add_special_tokens=False)
                logger.info(f"🔍 Debug - Test tokenization: {test_tokens}")
            
            # Fix: Tokenize each text individually then batch manually
            tokenized_texts = []
            for text in batch_texts:
                tokens = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=512
                )
                tokenized_texts.append(tokens['input_ids'])
            
            # Manual padding to create batch
            max_len = max(len(tokens) for tokens in tokenized_texts)
            padded_input_ids = []
            
            for tokens in tokenized_texts:
                if len(tokens) < max_len:
                    # Pad with tokenizer's pad token
                    padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
                else:
                    padded = tokens
                padded_input_ids.append(padded)
            
            # Create inputs dict
            inputs = {
                'input_ids': torch.tensor(padded_input_ids),
                'attention_mask': torch.ones_like(torch.tensor(padded_input_ids))
            }
            
            # Debug: Check tokenization result
            if debug_batch_count == 0:
                logger.info(f"🔍 Debug - Fixed tokenization:")
                logger.info(f"🔍 Debug - First sequence tokens: {inputs['input_ids'][0][:20].tolist()}")
                logger.info(f"🔍 Debug - Pad token ID: {tokenizer.pad_token_id}")
            
            # Debug: Check original input
            if debug_batch_count == 0:
                logger.info(f"🔍 Debug - Original input_ids shape: {inputs['input_ids'].shape}")
                logger.info(f"🔍 Debug - First sequence before BOS: {inputs['input_ids'][0][:10].tolist()}")
            
            # Add BOS tokens (same as training)
            input_ids = add_bos_tokens_to_batch(inputs['input_ids'], tokenizer)
            attention_mask = torch.ones_like(input_ids)
            
            # Debug: Check after BOS addition
            if debug_batch_count == 0:
                logger.info(f"🔍 Debug - After BOS input_ids shape: {input_ids.shape}")
                logger.info(f"🔍 Debug - First sequence after BOS: {input_ids[0][:10].tolist()}")
                logger.info(f"🔍 Debug - BOS token ID: {tokenizer.bos_token_id}")
            
            # Move to device
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            # Forward pass WITHOUT labels to avoid double loss calculation
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            except AttributeError as e:
                if "get_usable_length" in str(e):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                else:
                    raise e
            
            # Manual loss calculation with proper BOS handling
            logits = outputs.logits
            
            # CORRECT BOS ZERO PERPLEXITY CALCULATION:
            # Input: [BOS, token1, token2, token3, ...]  (BOS embedding is scaled)
            # Logits: [logit_for_token1, logit_for_token2, logit_for_token3, ...]
            # Target: [token1, token2, token3, ...]  (exclude BOS from targets)
            
            # Remove BOS prediction (first logit) and BOS target (first token)
            # logits[1:] predicts tokens[1:] (token1, token2, ...)
            # input_ids[1:] are the targets (token1, token2, ...)
            prediction_logits = logits[:, 1:, :].contiguous()  # Skip BOS prediction
            target_tokens = input_ids[:, 1:].contiguous()      # Skip BOS token
            target_attention_mask = attention_mask[:, 1:].contiguous()  # Skip BOS attention
            
            # Debug: Check shapes and values
            if debug_batch_count == 0:
                logger.info(f"🔍 Debug - Original logits shape: {logits.shape}")
                logger.info(f"🔍 Debug - Prediction logits shape (skip BOS): {prediction_logits.shape}")
                logger.info(f"🔍 Debug - Target tokens shape (skip BOS): {target_tokens.shape}")
                logger.info(f"🔍 Debug - Target attention mask sum: {target_attention_mask.sum().item()}")
                logger.info(f"🔍 Debug - First few targets: {target_tokens[0][:10].tolist()}")
            
            # Calculate cross-entropy loss on actual content tokens (not BOS)
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), target_tokens.view(-1))
            losses = losses.view(target_tokens.shape)
            
            # Apply attention mask to ignore padding tokens
            masked_losses = losses * target_attention_mask
            
            # Debug: Check loss values and token predictions
            if debug_batch_count == 0:
                valid_losses = masked_losses[target_attention_mask > 0]
                logger.info(f"🔍 Debug - Raw loss range: {losses.min().item():.4f} to {losses.max().item():.4f}")
                logger.info(f"🔍 Debug - Valid loss mean: {valid_losses.mean().item():.4f}")
                logger.info(f"🔍 Debug - Valid loss count: {len(valid_losses)}")
                
                # Debug: Show prediction vs target tokens
                logger.info(f"🔍 Debug - Prediction vs Target Token Analysis:")
                
                # Get predicted tokens (argmax of logits)
                predicted_token_ids = torch.argmax(prediction_logits[0], dim=-1)
                target_token_ids = target_tokens[0]
                attention_mask_first = target_attention_mask[0]
                
                # Show first 15 valid (non-padding) tokens
                valid_positions = torch.where(attention_mask_first > 0)[0][:15]
                
                logger.info(f"🔍 Debug - Position | Predicted | Target | Match | Loss")
                logger.info(f"🔍 Debug - ---------|-----------|--------|-------|------")
                
                for i, pos in enumerate(valid_positions):
                    pred_id = predicted_token_ids[pos].item()
                    target_id = target_token_ids[pos].item()
                    match = "✓" if pred_id == target_id else "✗"
                    loss_val = losses[0][pos].item()
                    
                    # Decode tokens for readability
                    try:
                        pred_token = tokenizer.decode([pred_id])
                        target_token = tokenizer.decode([target_id])
                        logger.info(f"🔍 Debug - {pos:8d} | {pred_id:9d} | {target_id:6d} | {match:5s} | {loss_val:5.2f}")
                        logger.info(f"🔍 Debug -          | '{pred_token:>9s}' | '{target_token:>6s}' |       |")
                    except:
                        logger.info(f"🔍 Debug - {pos:8d} | {pred_id:9d} | {target_id:6d} | {match:5s} | {loss_val:5.2f}")
                
                # Calculate accuracy for this batch
                correct_predictions = (predicted_token_ids == target_token_ids) & (attention_mask_first > 0)
                accuracy = correct_predictions.sum().float() / attention_mask_first.sum().float()
                logger.info(f"🔍 Debug - First sequence accuracy: {accuracy.item():.4f}")
            
            batch_loss = masked_losses.sum().item()
            batch_tokens = target_attention_mask.sum().item()
            
            total_loss += batch_loss
            total_tokens += batch_tokens
            
            debug_batch_count += 1
            if debug_batch_count >= 3:  # Only debug first few batches
                break
    
    if total_tokens == 0:
        logger.error("❌ No valid tokens found for perplexity calculation!")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    logger.info(f"📊 Total loss: {total_loss:.4f}")
    logger.info(f"📊 Total tokens: {total_tokens}")
    logger.info(f"📊 Average loss: {avg_loss:.4f}")
    logger.info(f"📊 Perplexity: {perplexity:.4f}")
    
    return perplexity


def generate_samples(model, tokenizer, prompts: List[str], max_length: int = 100) -> List[str]:
    """Generate text samples with BOS token handling."""
    logger.info(f"Generating samples for {len(prompts)} prompts...")
    
    model.eval()
    generated_texts = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating"):
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Add BOS token
            input_ids = add_bos_tokens_to_batch(inputs['input_ids'], tokenizer)
            attention_mask = torch.ones_like(input_ids)
            
            # Move to device
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            
            # Generate with comprehensive compatibility fix
            try:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            except AttributeError as e:
                if "seen_tokens" in str(e) or "get_usable_length" in str(e):
                    # Comprehensive compatibility fix for transformers version mismatch
                    try:
                        outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_length=max_length,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=False  # Disable cache to avoid compatibility issues
                        )
                    except Exception as e2:
                        # Fallback to simple generation
                        logger.warning(f"Generation failed with cache, trying simple approach: {e2}")
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=50,  # Use max_new_tokens instead of max_length
                            do_sample=False,    # Use greedy decoding as fallback
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=False
                        )
                else:
                    raise e
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    return generated_texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate BOS Zero LoRA fine-tuned models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter (optional for baseline)")
    parser.add_argument("--apply_bos_scaling", action="store_true", default=True, help="Apply BOS scaling during evaluation")
    parser.add_argument("--no_bos_scaling", action="store_true", help="Disable BOS scaling (for baseline comparison)")
    parser.add_argument("--scaling_factor", type=float, default=0.01, help="BOS scaling factor (default: 0.01 for 99% reduction)")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset for evaluation")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples for evaluation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./bos_zero_eval_results", help="Output directory")
    parser.add_argument("--generate_samples", action="store_true", help="Generate text samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine BOS scaling setting
    apply_bos_scaling = args.apply_bos_scaling and not args.no_bos_scaling
    
    # Load model
    model, tokenizer, bos_hook = load_bos_zero_model(
        args.model_path,
        args.adapter_path,
        apply_bos_scaling,
        args.scaling_factor
    )
    
    # Load evaluation dataset
    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "wikitext":
        dataset = load_dataset(args.dataset, args.dataset_config, split="test")
    else:
        dataset = load_dataset(args.dataset, split="test")
    
    # Evaluate perplexity
    perplexity = evaluate_perplexity(model, tokenizer, dataset, args.max_samples, args.batch_size)
    
    # Generate samples if requested
    generated_samples = []
    if args.generate_samples:
        test_prompts = [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The most important lesson I learned was",
            "Climate change is a global challenge that requires",
            "The benefits of renewable energy include"
        ]
        generated_samples = generate_samples(model, tokenizer, test_prompts)
    
    # Save results
    results = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "bos_scaling_applied": apply_bos_scaling,
        "scaling_factor": args.scaling_factor,
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "perplexity": perplexity,
        "generated_samples": generated_samples
    }
    
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_path}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"BOS Scaling Applied: {apply_bos_scaling}")
    if apply_bos_scaling:
        reduction_percent = (1 - args.scaling_factor) * 100
        print(f"Scaling Factor: {args.scaling_factor} ({reduction_percent:.0f}% reduction)")
    print(f"Perplexity: {perplexity:.4f}")
    
    if generated_samples:
        print("\nGenerated Samples:")
        for i, sample in enumerate(generated_samples):
            print(f"{i+1}. {sample}")
    
    # Clean up
    bos_hook.remove_hook()
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()