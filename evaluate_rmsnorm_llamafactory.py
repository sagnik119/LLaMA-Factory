#!/usr/bin/env python3
"""
RMSNorm Model Evaluation Script using LLaMA-Factory Framework

This script evaluates a Phi-3 model with fine-tuned RMSNorm weights using the LLaMA-Factory evaluation system.
It loads the base model, applies the saved RMSNorm weights, and runs perplexity evaluation.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import LLaMA-Factory components, fall back to standalone if there are dependency issues
try:
    # Add LLaMA-Factory to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from llamafactory.hparams import get_eval_args, ModelArguments, DataArguments, EvaluationArguments, FinetuningArguments
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.data import get_template_and_fix_tokenizer
    
    LLAMAFACTORY_AVAILABLE = True
    print("LLaMA-Factory imports successful - using integrated evaluation")
    
except Exception as e:
    print(f"LLaMA-Factory import failed: {e}")
    print("Falling back to standalone evaluation mode")
    LLAMAFACTORY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RMSNormEvaluator:
    """Evaluator for models with fine-tuned RMSNorm weights with LLaMA-Factory integration or standalone fallback."""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct",
                 rmsnorm_weights_path: Optional[str] = None):
        self.model_name = model_name
        self.rmsnorm_weights_path = rmsnorm_weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_llamafactory = LLAMAFACTORY_AVAILABLE
        
        if self.use_llamafactory:
            # Initialize LLaMA-Factory components
            self._setup_llamafactory_args()
            self._load_model_and_tokenizer()
        else:
            # Use standalone loading
            self._load_model_and_tokenizer_standalone()
        
        if rmsnorm_weights_path:
            self._apply_rmsnorm_weights()
    
    def _setup_llamafactory_args(self):
        """Setup LLaMA-Factory arguments for model loading."""
        # Create minimal arguments for model loading
        args_dict = {
            # Model args
            "model_name_or_path": self.model_name,
            "trust_remote_code": True,
            "use_fast_tokenizer": True,
            "split_special_tokens": False,
            "cache_dir": None,
            "model_revision": "main",
            "hf_hub_token": None,
            "use_unsloth": False,
            "adapter_name_or_path": None,
            "mixture_of_depths": None,
            "train_from_scratch": False,
            "print_param_status": False,
            # Finetuning args
            "stage": "sft",
            "finetuning_type": "full",
            # Evaluation args - required
            "task": "mmlu_test",  # Required parameter for EvaluationArguments
            "task_dir": "evaluation",
            "batch_size": 4,
            "seed": 42,
            "lang": "en",
            "n_shot": 0,  # 0-shot evaluation as requested
            "save_dir": None,
            # Data args
            "dataset": "alpaca_en",
            "dataset_dir": "data",
            "template": "phi",
            "cutoff_len": 1024,
        }
        
        # Parse arguments using LLaMA-Factory's system
        self.model_args, self.data_args, self.eval_args, self.finetuning_args = get_eval_args(args_dict)
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer using LLaMA-Factory's loading system."""
        logger.info(f"Loading model and tokenizer with LLaMA-Factory: {self.model_name}")
        
        # Load tokenizer
        tokenizer_module = load_tokenizer(self.model_args)
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        
        # Setup template
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        
        # Load model (not trainable for evaluation)
        self.model = load_model(
            self.tokenizer,
            self.model_args,
            self.finetuning_args,
            is_trainable=False
        )
        
        logger.info(f"Model loaded successfully on device: {self.model.device}")
    
    def _load_model_and_tokenizer_standalone(self):
        """Load model and tokenizer using standalone transformers."""
        logger.info(f"Loading model and tokenizer standalone: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        self.processor = None
        self.template = None
        
        logger.info(f"Model loaded successfully on device: {self.model.device}")
    
    def _apply_rmsnorm_weights(self):
        """Apply fine-tuned RMSNorm weights to the model."""
        if not os.path.exists(self.rmsnorm_weights_path):
            raise FileNotFoundError(f"RMSNorm weights file not found: {self.rmsnorm_weights_path}")
        
        logger.info(f"Loading RMSNorm weights from: {self.rmsnorm_weights_path}")
        
        try:
            # Load the saved RMSNorm weights
            rmsnorm_weights = torch.load(self.rmsnorm_weights_path, map_location=self.device)
            logger.info(f"Loaded {len(rmsnorm_weights)} RMSNorm parameters")
            
            # Apply weights to model
            applied_count = 0
            for name, param in self.model.named_parameters():
                if name in rmsnorm_weights:
                    logger.info(f"Applying RMSNorm weight: {name}")
                    param.data = rmsnorm_weights[name].to(param.device, param.dtype)
                    applied_count += 1
            
            logger.info(f"Successfully applied {applied_count} RMSNorm weights to the model")
            
            if applied_count == 0:
                logger.warning("No RMSNorm weights were applied. Check parameter name matching.")
                self._debug_parameter_names(rmsnorm_weights)
                
        except Exception as e:
            logger.error(f"Error loading RMSNorm weights: {e}")
            raise
    
    def _debug_parameter_names(self, rmsnorm_weights: Dict[str, torch.Tensor]):
        """Debug parameter name matching."""
        logger.info("=== Parameter Name Debugging ===")
        logger.info("RMSNorm weights keys:")
        for key in sorted(rmsnorm_weights.keys()):
            logger.info(f"  {key}")
        
        logger.info("\nModel RMSNorm parameters:")
        for name, param in self.model.named_parameters():
            if "norm" in name.lower():
                logger.info(f"  {name} - shape: {param.shape}")
    
    def evaluate_perplexity(self, dataset_name: str = "wikitext", 
                          dataset_config: str = "wikitext-2-raw-v1",
                          max_length: int = 512,
                          stride: int = 256,
                          max_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model perplexity on a dataset.
        
        Args:
            dataset_name: Name of the dataset to use
            dataset_config: Configuration of the dataset
            max_length: Maximum sequence length
            stride: Stride for sliding window
            max_samples: Maximum number of samples to evaluate (None for all)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating perplexity on {dataset_name} ({dataset_config})")
        
        # Load dataset
        try:
            dataset = load_dataset(dataset_name, dataset_config, split="test")
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
        
        # Prepare model for evaluation
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
                text = example["text"]
                if not text.strip():
                    continue
                
                # Tokenize text
                encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                input_ids = encodings.input_ids.to(self.model.device)
                
                if input_ids.size(1) < 2:  # Need at least 2 tokens for loss calculation
                    continue
                
                # Calculate loss using sliding window
                seq_len = input_ids.size(1)
                nlls = []
                
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + max_length, seq_len)
                    trg_len = end_loc - begin_loc
                    
                    input_ids_chunk = input_ids[:, begin_loc:end_loc]
                    target_ids = input_ids_chunk.clone()
                    target_ids[:, :-trg_len] = -100
                    
                    try:
                        # Use past_key_values=None to avoid cache issues
                        outputs = self.model(
                            input_ids_chunk,
                            labels=target_ids,
                            past_key_values=None,
                            use_cache=False
                        )
                        neg_log_likelihood = outputs.loss * trg_len
                        nlls.append(neg_log_likelihood)
                        total_tokens += trg_len
                    except Exception as e:
                        logger.warning(f"Error processing chunk at position {begin_loc}: {e}")
                        continue
                
                if nlls:
                    total_loss += torch.stack(nlls).sum().item()
        
        if total_tokens == 0:
            logger.error("No tokens were processed successfully")
            return {"perplexity": float("inf"), "loss": float("inf"), "total_tokens": 0}
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        results = {
            "perplexity": perplexity,
            "loss": avg_loss,
            "total_tokens": total_tokens,
            "dataset": f"{dataset_name}/{dataset_config}",
            "model": self.model_name,
            "rmsnorm_weights_applied": self.rmsnorm_weights_path is not None
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Perplexity: {perplexity:.4f}")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Total tokens: {total_tokens}")
        
        return results
    
    def generate_sample_text(self, prompts: list = None, max_new_tokens: int = 100) -> Dict[str, str]:
        """Generate sample text to assess generation quality."""
        if prompts is None:
            prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly,",
                "The most important lesson I learned was"
            ]
        
        logger.info("Generating sample text for quality assessment")
        
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                try:
                    # Tokenize prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    # Generate with cache disabled to avoid compatibility issues
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,
                        past_key_values=None
                    )
                    
                    # Decode
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    results[f"prompt_{i+1}"] = {
                        "prompt": prompt,
                        "generated": generated_text[len(prompt):].strip()
                    }
                    
                    logger.info(f"Prompt {i+1}: {prompt}")
                    logger.info(f"Generated: {results[f'prompt_{i+1}']['generated'][:100]}...")
                    
                except Exception as e:
                    logger.error(f"Error generating text for prompt {i+1}: {e}")
                    results[f"prompt_{i+1}"] = {"prompt": prompt, "generated": f"Error: {e}"}
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phi-3 model with RMSNorm weights using LLaMA-Factory")
    parser.add_argument("--model_name", default="microsoft/Phi-3-mini-4k-instruct", 
                       help="Model name or path")
    parser.add_argument("--rmsnorm_weights_path", required=True,
                       help="Path to the saved RMSNorm weights (.pt file)")
    parser.add_argument("--dataset", default="wikitext", 
                       help="Dataset name for perplexity evaluation")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1",
                       help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output_dir", default="./rmsnorm_eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--generate_samples", action="store_true",
                       help="Generate sample text for quality assessment")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = RMSNormEvaluator(
            model_name=args.model_name,
            rmsnorm_weights_path=args.rmsnorm_weights_path
        )
        
        # Evaluate perplexity
        perplexity_results = evaluator.evaluate_perplexity(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            max_samples=args.max_samples
        )
        
        # Generate sample text if requested
        generation_results = {}
        if args.generate_samples:
            generation_results = evaluator.generate_sample_text()
        
        # Combine results
        final_results = {
            "evaluation_config": {
                "model_name": args.model_name,
                "rmsnorm_weights_path": args.rmsnorm_weights_path,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "max_samples": args.max_samples
            },
            "perplexity_evaluation": perplexity_results,
            "generation_samples": generation_results
        }
        
        # Save results
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model_name}")
        print(f"RMSNorm weights: {args.rmsnorm_weights_path}")
        print(f"Dataset: {args.dataset}/{args.dataset_config}")
        print(f"Perplexity: {perplexity_results['perplexity']:.4f}")
        print(f"Loss: {perplexity_results['loss']:.4f}")
        print(f"Total tokens: {perplexity_results['total_tokens']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()