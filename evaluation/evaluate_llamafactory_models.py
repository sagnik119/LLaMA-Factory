#!/usr/bin/env python3

"""
Evaluation Integration for LLaMA-Factory Fine-tuned Models

This script provides seamless integration between LLaMA-Factory fine-tuned models
and the comprehensive evaluation suite from sparsity_subspace_fork.

Features:
- Automatic detection of LLaMA-Factory checkpoint formats
- Support for LoRA adapters and merged models
- Integration with LM evaluation harness
- Variance regularization model evaluation
- Comprehensive benchmarking on multiple tasks

Usage:
    # Evaluate a LoRA checkpoint
    python evaluate_llamafactory_models.py --model_path ./saves/phi3-variance-reg-redpajama
    
    # Evaluate with specific tasks
    python evaluate_llamafactory_models.py --model_path ./saves/phi3-variance-reg-redpajama --tasks "wikitext,arc_challenge,hellaswag"
    
    # Quick evaluation
    python evaluate_llamafactory_models.py --model_path ./saves/phi3-variance-reg-redpajama --preset quick_test
    
    # Full evaluation with all tasks
    python evaluate_llamafactory_models.py --model_path ./saves/phi3-variance-reg-redpajama --preset standard

Author: LLaMA-Factory Integration
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from datetime import datetime

# Add sparsity_subspace_fork to path for evaluation
sys.path.insert(0, str(Path(__file__).parent.parent / "sparsity_subspace_fork"))

try:
    from evaluation_suite import EvaluationSuite, EvaluationConfig
    from run_evaluation import run_evaluation_with_model
except ImportError as e:
    print(f"Error importing evaluation modules: {e}")
    print("Please ensure sparsity_subspace_fork is available in the parent directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLaMAFactoryModelLoader:
    """
    Loader for LLaMA-Factory fine-tuned models with support for various formats.
    """
    
    @staticmethod
    def detect_model_type(model_path: Path) -> str:
        """
        Detect the type of LLaMA-Factory model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Model type: 'lora', 'merged', 'full', or 'unknown'
        """
        if not model_path.exists():
            return 'unknown'
        
        # Check for LoRA adapter files
        if (model_path / "adapter_config.json").exists():
            return 'lora'
        
        # Check for merged model files
        if (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
            # Check if it's a full model or merged LoRA
            config_file = model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    # If it has _name_or_path, it's likely a merged model
                    if "_name_or_path" in config:
                        return 'merged'
                    else:
                        return 'full'
                except:
                    pass
        
        return 'unknown'
    
    @staticmethod
    def get_base_model_name(model_path: Path) -> str:
        """
        Get the base model name from LLaMA-Factory checkpoint.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Base model name for tokenizer loading
        """
        # Try adapter config first (for LoRA models)
        adapter_config_file = model_path / "adapter_config.json"
        if adapter_config_file.exists():
            try:
                with open(adapter_config_file, 'r') as f:
                    adapter_config = json.load(f)
                if "base_model_name_or_path" in adapter_config:
                    return adapter_config["base_model_name_or_path"]
            except Exception as e:
                logger.warning(f"Failed to read adapter config: {e}")
        
        # Try model config
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                if "_name_or_path" in config:
                    return config["_name_or_path"]
            except Exception as e:
                logger.warning(f"Failed to read model config: {e}")
        
        # Default fallback
        return "microsoft/Phi-3-mini-4k-instruct"
    
    @staticmethod
    def load_lora_model(model_path: Path, base_model_name: str) -> Any:
        """
        Load LoRA model using transformers and PEFT.
        
        Args:
            model_path: Path to LoRA adapter
            base_model_name: Base model name
            
        Returns:
            Loaded model with LoRA adapters
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Avoid DynamicCache issues
                use_cache=False
            )
            
            logger.info(f"Loading LoRA adapter from: {model_path}")
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(
                base_model,
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Merge adapters for evaluation (optional but recommended for speed)
            logger.info("Merging LoRA adapters for evaluation...")
            model = model.merge_and_unload()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
            raise
    
    @staticmethod
    def load_merged_model(model_path: Path) -> Any:
        """
        Load merged model directly.
        
        Args:
            model_path: Path to merged model
            
        Returns:
            Loaded model
        """
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info(f"Loading merged model from: {model_path}")
            
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",  # Avoid DynamicCache issues
                use_cache=False
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load merged model: {e}")
            raise
    
    @classmethod
    def load_model(cls, model_path: Path) -> tuple[Any, str]:
        """
        Load LLaMA-Factory model automatically detecting the format.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (loaded_model, base_model_name)
        """
        model_type = cls.detect_model_type(model_path)
        base_model_name = cls.get_base_model_name(model_path)
        
        logger.info(f"Detected model type: {model_type}")
        logger.info(f"Base model name: {base_model_name}")
        
        if model_type == 'lora':
            model = cls.load_lora_model(model_path, base_model_name)
        elif model_type in ['merged', 'full']:
            model = cls.load_merged_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model, base_model_name


class LLaMAFactoryEvaluator:
    """
    Main evaluator class for LLaMA-Factory models.
    """
    
    def __init__(self, model_path: str, output_dir: str = "./evaluation_results"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to LLaMA-Factory model
            output_dir: Directory to save evaluation results
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        logger.info(f"Initialized evaluator for: {self.model_path}")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def evaluate(self, 
                 tasks: Optional[List[str]] = None,
                 limit: Optional[int] = None,
                 preset: str = "standard") -> Dict[str, Any]:
        """
        Evaluate the model on specified tasks.
        
        Args:
            tasks: List of tasks to evaluate (None for preset default)
            limit: Limit number of samples per task
            preset: Evaluation preset to use
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting model evaluation...")
        
        # Load the model
        try:
            model, base_model_name = LLaMAFactoryModelLoader.load_model(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_output_dir = self.output_dir / f"evaluation_{timestamp}"
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation using the sparsity_subspace_fork evaluation system
        try:
            results = run_evaluation_with_model(
                model=model,
                model_name=base_model_name,
                output_dir=str(eval_output_dir),
                limit=limit or 1000
            )
            
            # Save additional metadata
            metadata = {
                "model_path": str(self.model_path),
                "base_model_name": base_model_name,
                "model_type": LLaMAFactoryModelLoader.detect_model_type(self.model_path),
                "evaluation_timestamp": timestamp,
                "preset": preset,
                "tasks": tasks,
                "limit": limit
            }
            
            metadata_file = eval_output_dir / "evaluation_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info("Evaluation completed successfully")
            return {
                "metadata": metadata,
                "results": results,
                "output_dir": str(eval_output_dir)
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def evaluate_multiple_checkpoints(self, 
                                    checkpoint_dirs: List[str],
                                    tasks: Optional[List[str]] = None,
                                    limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate multiple checkpoints and compare results.
        
        Args:
            checkpoint_dirs: List of checkpoint directories
            tasks: List of tasks to evaluate
            limit: Limit number of samples per task
            
        Returns:
            Combined evaluation results
        """
        all_results = {}
        
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_name = Path(checkpoint_dir).name
            logger.info(f"Evaluating checkpoint: {checkpoint_name}")
            
            try:
                # Create evaluator for this checkpoint
                evaluator = LLaMAFactoryEvaluator(
                    checkpoint_dir, 
                    str(self.output_dir / checkpoint_name)
                )
                
                # Run evaluation
                results = evaluator.evaluate(tasks=tasks, limit=limit)
                all_results[checkpoint_name] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate {checkpoint_name}: {e}")
                all_results[checkpoint_name] = {"error": str(e)}
        
        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_file = self.output_dir / f"combined_evaluation_{timestamp}.json"
        
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Combined results saved to: {combined_file}")
        return all_results


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluate LLaMA-Factory Fine-tuned Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        required=True,
        help="Path to LLaMA-Factory model directory"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to evaluate (e.g., 'wikitext,arc_challenge,hellaswag')"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick_test", "fast_test", "standard", "compression_analysis"],
        default="standard",
        help="Evaluation preset to use"
    )
    
    parser.add_argument(
        "--multiple_checkpoints",
        type=str,
        nargs="+",
        help="Evaluate multiple checkpoints and compare results"
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
    
    # Parse tasks if provided
    tasks = None
    if args.tasks:
        tasks = [task.strip() for task in args.tasks.split(',')]
        logger.info(f"Selected tasks: {tasks}")
    
    try:
        if args.multiple_checkpoints:
            # Evaluate multiple checkpoints
            logger.info(f"Evaluating {len(args.multiple_checkpoints)} checkpoints")
            
            evaluator = LLaMAFactoryEvaluator(
                args.multiple_checkpoints[0],  # Use first as base for output dir
                args.output_dir
            )
            
            results = evaluator.evaluate_multiple_checkpoints(
                args.multiple_checkpoints,
                tasks=tasks,
                limit=args.limit
            )
            
            print("\n" + "="*80)
            print("MULTIPLE CHECKPOINT EVALUATION COMPLETED")
            print("="*80)
            
            # Print summary
            successful_evals = sum(1 for r in results.values() if "error" not in r)
            total_evals = len(results)
            print(f"Successful evaluations: {successful_evals}/{total_evals}")
            
            for checkpoint_name, result in results.items():
                if "error" in result:
                    print(f"❌ {checkpoint_name}: FAILED - {result['error']}")
                else:
                    print(f"✅ {checkpoint_name}: SUCCESS")
                    if "results" in result and "perplexity" in result["results"]:
                        ppl = result["results"]["perplexity"]
                        print(f"   Perplexity: {ppl:.4f}")
        
        else:
            # Evaluate single checkpoint
            evaluator = LLaMAFactoryEvaluator(args.model_path, args.output_dir)
            
            results = evaluator.evaluate(
                tasks=tasks,
                limit=args.limit,
                preset=args.preset
            )
            
            print("\n" + "="*80)
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Model: {args.model_path}")
            print(f"Results saved to: {results['output_dir']}")
            
            if "results" in results and "perplexity" in results["results"]:
                ppl = results["results"]["perplexity"]
                print(f"Perplexity: {ppl:.4f}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()