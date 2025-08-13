#!/usr/bin/env python3

"""
Simple RMSNorm Model Evaluator using LLaMA-Factory Infrastructure

This script loads a Phi-3 model, applies fine-tuned RMSNorm weights, and evaluates
the model using the existing LLaMA-Factory evaluation infrastructure.

Usage:
    python evaluate_rmsnorm_model_simple.py --rmsnorm_weights_path ./saves/phi3-variance-reg-alpaca-layers-2-4/checkpoint-500/rmsnorm_weights.pt
    python evaluate_rmsnorm_model_simple.py --rmsnorm_weights_path ./saves/phi3-variance-reg-alpaca-layers-2-4/checkpoint-500/rmsnorm_weights.pt --limit 100

Author: LLaMA-Factory RMSNorm Evaluation
"""

import os
import sys
import json
import argparse
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add LLaMA-Factory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from llamafactory.chat import ChatModel
    from llamafactory.extras.env import VERSION
    from llamafactory.hparams import get_infer_args
except ImportError as e:
    print(f"Error importing LLaMA-Factory modules: {e}")
    print("Please ensure you're running this script from the LLaMA-Factory root directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RMSNormModelLoader:
    """
    Loader for Phi-3 models with fine-tuned RMSNorm weights.
    """
    
    @staticmethod
    def load_base_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> Any:
        """
        Load the base Phi-3 model using LLaMA-Factory infrastructure.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Loaded PyTorch model
        """
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info(f"Loading base model: {model_name}")
            
            # Load model with optimized settings for evaluation
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Avoid DynamicCache issues
                use_cache=False,  # Disable caching for evaluation
                low_cpu_mem_usage=True
            )
            
            # Set model to evaluation mode
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            logger.info(f"Successfully loaded base model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    @staticmethod
    def load_rmsnorm_weights(weights_path: str) -> Dict[str, torch.Tensor]:
        """
        Load RMSNorm weights from saved checkpoint.
        
        Args:
            weights_path: Path to rmsnorm_weights.pt file
            
        Returns:
            Dictionary of parameter names to tensors
        """
        try:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"RMSNorm weights file not found: {weights_path}")
            
            logger.info(f"Loading RMSNorm weights from: {weights_path}")
            
            # Load the weights
            rmsnorm_weights = torch.load(weights_path, map_location='cpu')
            
            logger.info(f"Loaded {len(rmsnorm_weights)} RMSNorm parameters")
            
            # Log which parameters were loaded
            for name in rmsnorm_weights.keys():
                logger.info(f"  - {name}: {rmsnorm_weights[name].shape}")
            
            return rmsnorm_weights
            
        except Exception as e:
            logger.error(f"Failed to load RMSNorm weights: {e}")
            raise
    
    @staticmethod
    def load_rmsnorm_config(config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load RMSNorm training configuration if available.
        
        Args:
            config_path: Path to rmsnorm_config.json file
            
        Returns:
            Configuration dictionary or None if not found
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"RMSNorm config file not found: {config_path}")
                return None
            
            logger.info(f"Loading RMSNorm config from: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info("RMSNorm training configuration:")
            for key, value in config.items():
                logger.info(f"  - {key}: {value}")
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load RMSNorm config: {e}")
            return None
    
    @staticmethod
    def apply_rmsnorm_weights(model: Any, rmsnorm_weights: Dict[str, torch.Tensor]) -> Any:
        """
        Apply fine-tuned RMSNorm weights to the model.
        
        Args:
            model: PyTorch model
            rmsnorm_weights: Dictionary of RMSNorm weights
            
        Returns:
            Model with updated RMSNorm weights
        """
        try:
            logger.info("Applying fine-tuned RMSNorm weights to model...")
            
            # Get model device and dtype
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            applied_count = 0
            missing_count = 0
            
            # Apply each weight to the corresponding parameter in the model
            for param_name, param_tensor in rmsnorm_weights.items():
                try:
                    # Navigate to the parameter in the model
                    current = model
                    param_path = param_name.split('.')
                    
                    # Navigate to the parent module
                    for attr in param_path[:-1]:
                        current = getattr(current, attr)
                    
                    # Get the final parameter name
                    final_param_name = param_path[-1]
                    
                    if hasattr(current, final_param_name):
                        # Get the current parameter
                        current_param = getattr(current, final_param_name)
                        
                        # Move the new weight to the same device and dtype as the model
                        new_weight = param_tensor.to(device=model_device, dtype=model_dtype)
                        
                        # Verify shapes match
                        if current_param.data.shape != new_weight.shape:
                            logger.error(f"Shape mismatch for {param_name}: model={current_param.data.shape}, weight={new_weight.shape}")
                            continue
                        
                        # Apply the new weight
                        current_param.data.copy_(new_weight)
                        applied_count += 1
                        
                        logger.debug(f"Applied weight to {param_name}: {new_weight.shape}")
                        
                        # Verify the weight was applied correctly
                        if torch.allclose(current_param.data, new_weight, atol=1e-6):
                            logger.debug(f"✓ Weight successfully applied to {param_name}")
                        else:
                            logger.warning(f"⚠ Weight application verification failed for {param_name}")
                    else:
                        logger.warning(f"Parameter {param_name} not found in model")
                        missing_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to apply weight {param_name}: {e}")
                    missing_count += 1
            
            logger.info(f"Applied {applied_count} RMSNorm weights successfully")
            if missing_count > 0:
                logger.warning(f"Failed to apply {missing_count} RMSNorm weights")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply RMSNorm weights: {e}")
            raise
    
    @classmethod
    def load_model_with_rmsnorm_weights(cls, rmsnorm_weights_path: str, 
                                      base_model: str = "microsoft/Phi-3-mini-4k-instruct") -> Any:
        """
        Load Phi-3 model and apply fine-tuned RMSNorm weights.
        
        Args:
            rmsnorm_weights_path: Path to rmsnorm_weights.pt file
            base_model: Base model name
            
        Returns:
            Model with applied RMSNorm weights
        """
        # Load base model
        model = cls.load_base_model(base_model)
        
        # Load RMSNorm weights
        rmsnorm_weights = cls.load_rmsnorm_weights(rmsnorm_weights_path)
        
        # Load RMSNorm config if available
        config_path = Path(rmsnorm_weights_path).parent / "rmsnorm_config.json"
        rmsnorm_config = cls.load_rmsnorm_config(str(config_path))
        
        # Apply RMSNorm weights
        model = cls.apply_rmsnorm_weights(model, rmsnorm_weights)
        
        return model, rmsnorm_config


class SimplePerplexityEvaluator:
    """
    Simple perplexity evaluator using WikiText dataset.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_perplexity(self, limit: Optional[int] = None) -> float:
        """
        Evaluate perplexity on WikiText dataset.
        
        Args:
            limit: Limit number of samples to evaluate
            
        Returns:
            Perplexity score
        """
        try:
            from datasets import load_dataset
            
            logger.info("Loading WikiText dataset for perplexity evaluation...")
            
            # Load WikiText-2 dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            
            logger.info(f"Evaluating perplexity on {len(dataset)} samples...")
            
            total_loss = 0.0
            total_tokens = 0
            
            self.model.eval()
            
            with torch.no_grad():
                for i, example in enumerate(dataset):
                    if i % 100 == 0:
                        logger.info(f"Processing sample {i}/{len(dataset)}")
                    
                    text = example["text"].strip()
                    if not text:
                        continue
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=False
                    )
                    
                    if inputs.input_ids.size(1) < 2:
                        continue
                    
                    # Move to model device
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # Forward pass
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    # Accumulate loss
                    seq_len = inputs.input_ids.size(1)
                    total_loss += loss.item() * seq_len
                    total_tokens += seq_len
            
            # Calculate perplexity
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            logger.info(f"Evaluation completed. Perplexity: {perplexity:.4f}")
            return perplexity
            
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            return float('inf')


class RMSNormEvaluator:
    """
    Main evaluator class for RMSNorm fine-tuned models.
    """
    
    def __init__(self, rmsnorm_weights_path: str, base_model: str = "microsoft/Phi-3-mini-4k-instruct",
                 output_dir: str = "./rmsnorm_evaluation_results"):
        """
        Initialize evaluator.
        
        Args:
            rmsnorm_weights_path: Path to RMSNorm weights file
            base_model: Base model name
            output_dir: Directory to save evaluation results
        """
        self.rmsnorm_weights_path = Path(rmsnorm_weights_path)
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.rmsnorm_weights_path.exists():
            raise FileNotFoundError(f"RMSNorm weights file not found: {self.rmsnorm_weights_path}")
        
        logger.info(f"Initialized RMSNorm evaluator")
        logger.info(f"  RMSNorm weights: {self.rmsnorm_weights_path}")
        logger.info(f"  Base model: {self.base_model}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def evaluate(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the RMSNorm fine-tuned model.
        
        Args:
            limit: Limit number of samples per task
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting RMSNorm model evaluation...")
        
        try:
            # Load model with RMSNorm weights
            model, rmsnorm_config = RMSNormModelLoader.load_model_with_rmsnorm_weights(
                str(self.rmsnorm_weights_path), 
                self.base_model
            )
            
            logger.info("Model loaded successfully with RMSNorm weights applied")
            
            # Load tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            
            # Create output directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            eval_output_dir = self.output_dir / f"rmsnorm_evaluation_{timestamp}"
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run perplexity evaluation
            logger.info("Running perplexity evaluation...")
            evaluator = SimplePerplexityEvaluator(model, tokenizer)
            perplexity = evaluator.evaluate_perplexity(limit=limit)
            
            # Prepare results
            results = {
                "perplexity": perplexity,
                "success": perplexity != float('inf'),
                "evaluation_type": "simple_perplexity"
            }
            
            # Save additional metadata
            metadata = {
                "rmsnorm_weights_path": str(self.rmsnorm_weights_path),
                "base_model": self.base_model,
                "evaluation_timestamp": timestamp,
                "limit": limit,
                "rmsnorm_config": rmsnorm_config,
                "evaluation_framework": "simple_llamafactory"
            }
            
            # Save results
            results_file = eval_output_dir / "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump({"metadata": metadata, "results": results}, f, indent=2, default=str)
            
            metadata_file = eval_output_dir / "rmsnorm_evaluation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("RMSNorm evaluation completed successfully")
            return {
                "metadata": metadata,
                "results": results,
                "output_dir": str(eval_output_dir)
            }
            
        except Exception as e:
            logger.error(f"RMSNorm evaluation failed: {e}")
            raise


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Simple RMSNorm Model Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--rmsnorm_weights_path", "-w",
        type=str,
        required=True,
        help="Path to rmsnorm_weights.pt file from LLaMA-Factory training"
    )
    
    parser.add_argument(
        "--base_model", "-m",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Base model name to load"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./rmsnorm_evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for evaluation (for testing)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create evaluator
        evaluator = RMSNormEvaluator(
            rmsnorm_weights_path=args.rmsnorm_weights_path,
            base_model=args.base_model,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        results = evaluator.evaluate(limit=args.limit)
        
        print("\n" + "="*80)
        print("RMSNORM EVALUATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"RMSNorm weights: {args.rmsnorm_weights_path}")
        print(f"Base model: {args.base_model}")
        print(f"Results saved to: {results['output_dir']}")
        
        # Print key results
        if "results" in results:
            res = results["results"]
            if "perplexity" in res:
                ppl = res["perplexity"]
                print(f"Perplexity: {ppl:.4f}")
            
            if res.get("success", False):
                print("✅ Evaluation completed successfully")
            else:
                print("⚠️ Evaluation completed with issues")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()