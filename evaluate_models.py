#!/usr/bin/env python3

"""
LLaMA-Factory Model Evaluation CLI

This script provides a unified interface for evaluating LLaMA-Factory fine-tuned models
using LM evaluation harness and other comprehensive evaluation tools.

Usage:
    # Evaluate a single model with LM evaluation harness
    python evaluate_models.py --model_path ./saves/phi3-variance-reg-redpajama --tasks "hellaswag,arc_easy,winogrande"
    
    # Quick evaluation
    python evaluate_models.py --model_path ./saves/phi3-variance-reg-redpajama --quick
    
    # Compare multiple variance regularization models
    python evaluate_models.py --compare_variance_models ./saves/phi3-variance-reg-redpajama ./saves/phi3-variance-reg-identity
    
    # Evaluate with baseline comparison
    python evaluate_models.py --model_path ./saves/phi3-variance-reg-redpajama --baseline microsoft/Phi-3-mini-4k-instruct

Author: LLaMA-Factory Evaluation Integration
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))

try:
    from evaluate_llamafactory_models import LLaMAFactoryEvaluator
    from evaluate_variance_regularization import VarianceRegularizationEvaluator
except ImportError as e:
    print(f"Error importing evaluation modules: {e}")
    print("Please ensure the evaluation scripts are available in the evaluation/ directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_lm_eval_harness(model_path: str, 
                       tasks: list = None, 
                       output_dir: str = None,
                       limit: int = None,
                       quick: bool = False) -> dict:
    """Run LM evaluation harness on a model."""
    
    if quick:
        tasks = ["hellaswag", "arc_easy", "winogrande", "piqa"]
        limit = limit or 100
        print("🚀 Running quick evaluation...")
    elif not tasks:
        tasks = ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa", "boolq"]
        print("📊 Running standard evaluation...")
    
    print(f"Tasks: {', '.join(tasks)}")
    if limit:
        print(f"Sample limit: {limit}")
    
    # Use the existing LLaMA-Factory evaluator
    evaluator = LLaMAFactoryEvaluator(model_path, output_dir or "./evaluation_results")
    
    results = evaluator.evaluate(
        tasks=tasks,
        limit=limit,
        preset="quick_test" if quick else "standard"
    )
    
    return results


def run_variance_regularization_evaluation(model_paths: list,
                                         baseline: str = None,
                                         tasks: list = None,
                                         output_dir: str = None,
                                         limit: int = None) -> dict:
    """Run variance regularization specific evaluation."""
    
    print("🔧 Running variance regularization evaluation...")
    
    evaluator = VarianceRegularizationEvaluator(output_dir or "./variance_reg_evaluation")
    
    if len(model_paths) == 1:
        # Single model evaluation
        results = evaluator.evaluate_single_model(
            model_paths[0],
            tasks=tasks,
            limit=limit
        )
    else:
        # Multiple model comparison
        results = evaluator.compare_models(
            model_paths=model_paths,
            baseline_path=baseline,
            tasks=tasks,
            limit=limit
        )
    
    return results


def print_results_summary(results: dict, evaluation_type: str):
    """Print a summary of evaluation results."""
    
    print("\n" + "="*80)
    print(f"{evaluation_type.upper()} EVALUATION COMPLETED")
    print("="*80)
    
    if evaluation_type == "lm_harness":
        if "results" in results and "perplexity" in results["results"]:
            ppl = results["results"]["perplexity"]
            print(f"📈 Perplexity: {ppl:.4f}")
        
        if "results" in results and "accuracy" in results["results"]:
            acc = results["results"]["accuracy"]
            print(f"🎯 Accuracy: {acc:.4f}")
        
        print(f"📁 Results saved to: {results.get('output_dir', 'N/A')}")
    
    elif evaluation_type == "variance_regularization":
        if "comparison_summary" in results:
            summary = results["comparison_summary"]
            if "best_perplexity" in summary and summary["best_perplexity"]["model"]:
                best_model = summary["best_perplexity"]["model"]
                best_ppl = summary["best_perplexity"]["value"]
                print(f"🏆 Best Model: {best_model} (Perplexity: {best_ppl:.4f})")
        
        elif "evaluation_results" in results:
            eval_results = results["evaluation_results"]
            if "results" in eval_results and "perplexity" in eval_results["results"]:
                ppl = eval_results["results"]["perplexity"]
                print(f"📈 Perplexity: {ppl:.4f}")
            
            # Check for variance regularization detection
            if "variance_regularization_config" in results:
                config = results["variance_regularization_config"]
                if config.get("variance_regularization_detected", False):
                    print("✅ Variance regularization detected")
                    weight = config.get("variance_reg_weight", "N/A")
                    target = config.get("variance_reg_target", "N/A")
                    print(f"   Weight: {weight}, Target: {target}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="LLaMA-Factory Model Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model specification
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        help="Path to LLaMA-Factory model directory"
    )
    
    parser.add_argument(
        "--compare_variance_models",
        type=str,
        nargs="+",
        help="Compare multiple variance regularization models"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline model for comparison (e.g., microsoft/Phi-3-mini-4k-instruct)"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of tasks (e.g., 'hellaswag,arc_easy,winogrande')"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick evaluation with limited tasks and samples"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples per task"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


def main():
    """Main entry point."""
    
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
        if args.compare_variance_models:
            # Variance regularization comparison
            results = run_variance_regularization_evaluation(
                model_paths=args.compare_variance_models,
                baseline=args.baseline,
                tasks=tasks,
                output_dir=args.output_dir,
                limit=args.limit
            )
            print_results_summary(results, "variance_regularization")
        
        elif args.model_path:
            # Check if this is a variance regularization model
            model_path = Path(args.model_path)
            training_args_file = model_path / "training_args.json"
            
            is_variance_reg = False
            if training_args_file.exists():
                try:
                    import json
                    with open(training_args_file, 'r') as f:
                        training_args = json.load(f)
                    is_variance_reg = training_args.get("use_variance_regularization", False)
                except:
                    pass
            
            if is_variance_reg:
                # Use variance regularization evaluator
                results = run_variance_regularization_evaluation(
                    model_paths=[args.model_path],
                    baseline=args.baseline,
                    tasks=tasks,
                    output_dir=args.output_dir,
                    limit=args.limit
                )
                print_results_summary(results, "variance_regularization")
            else:
                # Use standard LM evaluation harness
                results = run_lm_eval_harness(
                    model_path=args.model_path,
                    tasks=tasks,
                    output_dir=args.output_dir,
                    limit=args.limit,
                    quick=args.quick
                )
                print_results_summary(results, "lm_harness")
        
        else:
            print("Error: Please specify either --model_path or --compare_variance_models")
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()