#!/usr/bin/env python3

"""
Specialized Evaluation Script for Variance Regularization Models

This script provides comprehensive evaluation specifically for models trained
with RMSNorm variance regularization in LLaMA-Factory.

Features:
- Automatic detection of variance regularization checkpoints
- Comparison with baseline models
- Analysis of regularization effects on model performance
- Comprehensive benchmarking with detailed metrics
- Support for multiple variance regularization configurations

Usage:
    # Evaluate variance regularization model
    python evaluate_variance_regularization.py --model_path ./saves/phi3-variance-reg-redpajama
    
    # Compare with baseline
    python evaluate_variance_regularization.py --model_path ./saves/phi3-variance-reg-redpajama --baseline microsoft/Phi-3-mini-4k-instruct
    
    # Evaluate multiple variance configurations
    python evaluate_variance_regularization.py --compare_models ./saves/phi3-variance-reg-redpajama ./saves/phi3-variance-reg-identity ./saves/phi3-variance-reg-ultrachat
    
    # Quick evaluation
    python evaluate_variance_regularization.py --model_path ./saves/phi3-variance-reg-redpajama --preset quick_test

Author: Variance Regularization Evaluation
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_llamafactory_models import LLaMAFactoryModelLoader, LLaMAFactoryEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VarianceRegularizationAnalyzer:
    """
    Analyzer for variance regularization effects on model performance.
    """
    
    def __init__(self, model_path: Path):
        """
        Initialize analyzer.
        
        Args:
            model_path: Path to variance regularization model
        """
        self.model_path = model_path
        self.model = None
        self.base_model_name = None
        
    def load_model(self) -> Tuple[Any, str]:
        """Load the variance regularization model."""
        if self.model is None:
            self.model, self.base_model_name = LLaMAFactoryModelLoader.load_model(self.model_path)
        return self.model, self.base_model_name
    
    def analyze_rmsnorm_weights(self) -> Dict[str, Any]:
        """
        Analyze RMSNorm weights to detect variance regularization effects.
        
        Returns:
            Dictionary containing RMSNorm weight analysis
        """
        model, _ = self.load_model()
        
        rmsnorm_analysis = {
            "layers_analyzed": [],
            "weight_statistics": {},
            "variance_patterns": {},
            "regularization_detected": False
        }
        
        try:
            # Analyze RMSNorm layers
            for name, module in model.named_modules():
                if "layernorm" in name.lower() or "rmsnorm" in name.lower():
                    if hasattr(module, 'weight') and module.weight is not None:
                        weight_data = module.weight.data.cpu().numpy()
                        
                        # Calculate statistics
                        stats = {
                            "mean": float(np.mean(weight_data)),
                            "std": float(np.std(weight_data)),
                            "min": float(np.min(weight_data)),
                            "max": float(np.max(weight_data)),
                            "variance": float(np.var(weight_data)),
                            "zero_count": int(np.sum(weight_data == 0.0)),
                            "near_zero_count": int(np.sum(np.abs(weight_data) < 1e-6)),
                            "shape": list(weight_data.shape)
                        }
                        
                        rmsnorm_analysis["layers_analyzed"].append(name)
                        rmsnorm_analysis["weight_statistics"][name] = stats
                        
                        # Check for regularization patterns
                        if stats["std"] < 0.1 or stats["zero_count"] > 0:
                            rmsnorm_analysis["regularization_detected"] = True
                        
                        logger.debug(f"Analyzed {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            
            logger.info(f"Analyzed {len(rmsnorm_analysis['layers_analyzed'])} RMSNorm layers")
            
        except Exception as e:
            logger.error(f"Failed to analyze RMSNorm weights: {e}")
            rmsnorm_analysis["error"] = str(e)
        
        return rmsnorm_analysis
    
    def detect_variance_regularization_config(self) -> Dict[str, Any]:
        """
        Detect variance regularization configuration from model artifacts.
        
        Returns:
            Dictionary containing detected configuration
        """
        config_info = {
            "variance_regularization_detected": False,
            "training_args": {},
            "adapter_config": {},
            "regularization_layers": [],
            "regularization_weight": None,
            "regularization_target": None
        }
        
        try:
            # Check training arguments
            training_args_file = self.model_path / "training_args.json"
            if training_args_file.exists():
                with open(training_args_file, 'r') as f:
                    training_args = json.load(f)
                    config_info["training_args"] = training_args
                    
                    # Look for variance regularization parameters
                    if "use_variance_regularization" in training_args:
                        config_info["variance_regularization_detected"] = training_args["use_variance_regularization"]
                    
                    for key in ["variance_reg_layers", "variance_reg_weight", "variance_reg_target", "variance_reg_norm_type"]:
                        if key in training_args:
                            config_info[key] = training_args[key]
            
            # Check adapter config
            adapter_config_file = self.model_path / "adapter_config.json"
            if adapter_config_file.exists():
                with open(adapter_config_file, 'r') as f:
                    adapter_config = json.load(f)
                    config_info["adapter_config"] = adapter_config
            
            logger.info(f"Variance regularization detected: {config_info['variance_regularization_detected']}")
            
        except Exception as e:
            logger.error(f"Failed to detect configuration: {e}")
            config_info["error"] = str(e)
        
        return config_info


class VarianceRegularizationEvaluator:
    """
    Comprehensive evaluator for variance regularization models.
    """
    
    def __init__(self, output_dir: str = "./variance_reg_evaluation"):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Variance regularization evaluator initialized")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def evaluate_single_model(self, 
                            model_path: str,
                            tasks: Optional[List[str]] = None,
                            limit: Optional[int] = None,
                            preset: str = "standard") -> Dict[str, Any]:
        """
        Evaluate a single variance regularization model.
        
        Args:
            model_path: Path to the model
            tasks: List of tasks to evaluate
            limit: Limit number of samples per task
            preset: Evaluation preset
            
        Returns:
            Comprehensive evaluation results
        """
        model_path = Path(model_path)
        model_name = model_path.name
        
        logger.info(f"Evaluating variance regularization model: {model_name}")
        
        # Create model-specific output directory
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "model_path": str(model_path),
            "model_name": model_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "preset": preset,
            "tasks": tasks,
            "limit": limit
        }
        
        try:
            # Analyze variance regularization configuration
            analyzer = VarianceRegularizationAnalyzer(model_path)
            config_analysis = analyzer.detect_variance_regularization_config()
            results["variance_regularization_config"] = config_analysis
            
            # Analyze RMSNorm weights
            rmsnorm_analysis = analyzer.analyze_rmsnorm_weights()
            results["rmsnorm_analysis"] = rmsnorm_analysis
            
            # Run standard evaluation
            evaluator = LLaMAFactoryEvaluator(str(model_path), str(model_output_dir))
            eval_results = evaluator.evaluate(tasks=tasks, limit=limit, preset=preset)
            results["evaluation_results"] = eval_results
            
            # Save results
            results_file = model_output_dir / "variance_reg_evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Evaluation completed for {model_name}")
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            results["error"] = str(e)
        
        return results
    
    def compare_models(self, 
                      model_paths: List[str],
                      baseline_path: Optional[str] = None,
                      tasks: Optional[List[str]] = None,
                      limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Compare multiple variance regularization models.
        
        Args:
            model_paths: List of model paths to compare
            baseline_path: Optional baseline model path
            tasks: List of tasks to evaluate
            limit: Limit number of samples per task
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(model_paths)} variance regularization models")
        
        comparison_results = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models_compared": len(model_paths),
            "baseline_model": baseline_path,
            "tasks": tasks,
            "limit": limit,
            "model_results": {},
            "comparison_summary": {}
        }
        
        # Evaluate each model
        for model_path in model_paths:
            model_name = Path(model_path).name
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                results = self.evaluate_single_model(
                    model_path, tasks=tasks, limit=limit, preset="standard"
                )
                comparison_results["model_results"][model_name] = results
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                comparison_results["model_results"][model_name] = {"error": str(e)}
        
        # Evaluate baseline if provided
        if baseline_path:
            logger.info(f"Evaluating baseline model: {baseline_path}")
            try:
                baseline_evaluator = LLaMAFactoryEvaluator(baseline_path, str(self.output_dir / "baseline"))
                baseline_results = baseline_evaluator.evaluate(tasks=tasks, limit=limit, preset="standard")
                comparison_results["baseline_results"] = baseline_results
            except Exception as e:
                logger.error(f"Failed to evaluate baseline: {e}")
                comparison_results["baseline_results"] = {"error": str(e)}
        
        # Generate comparison summary
        comparison_results["comparison_summary"] = self._generate_comparison_summary(comparison_results)
        
        # Save comparison results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = self.output_dir / f"variance_reg_comparison_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"Comparison results saved to: {comparison_file}")
        
        # Generate visualization
        self._create_comparison_plots(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of model comparison."""
        summary = {
            "best_perplexity": {"model": None, "value": float('inf')},
            "regularization_effects": {},
            "performance_ranking": []
        }
        
        try:
            perplexities = {}
            
            # Extract perplexities from model results
            for model_name, results in comparison_results["model_results"].items():
                if "error" not in results and "evaluation_results" in results:
                    eval_results = results["evaluation_results"]
                    if "results" in eval_results and "perplexity" in eval_results["results"]:
                        ppl = eval_results["results"]["perplexity"]
                        perplexities[model_name] = ppl
                        
                        if ppl < summary["best_perplexity"]["value"]:
                            summary["best_perplexity"]["model"] = model_name
                            summary["best_perplexity"]["value"] = ppl
            
            # Rank models by perplexity
            summary["performance_ranking"] = sorted(
                perplexities.items(), 
                key=lambda x: x[1]
            )
            
            # Analyze regularization effects
            for model_name, results in comparison_results["model_results"].items():
                if "variance_regularization_config" in results:
                    config = results["variance_regularization_config"]
                    if config.get("variance_regularization_detected", False):
                        summary["regularization_effects"][model_name] = {
                            "regularization_weight": config.get("variance_reg_weight"),
                            "regularization_target": config.get("variance_reg_target"),
                            "perplexity": perplexities.get(model_name, float('inf'))
                        }
            
        except Exception as e:
            logger.error(f"Failed to generate comparison summary: {e}")
            summary["error"] = str(e)
        
        return summary
    
    def _create_comparison_plots(self, comparison_results: Dict[str, Any]):
        """Create visualization plots for model comparison."""
        try:
            # Extract data for plotting
            model_names = []
            perplexities = []
            reg_weights = []
            
            for model_name, results in comparison_results["model_results"].items():
                if "error" not in results:
                    model_names.append(model_name)
                    
                    # Get perplexity
                    ppl = float('inf')
                    if "evaluation_results" in results and "results" in results["evaluation_results"]:
                        ppl = results["evaluation_results"]["results"].get("perplexity", float('inf'))
                    perplexities.append(ppl)
                    
                    # Get regularization weight
                    reg_weight = 0.0
                    if "variance_regularization_config" in results:
                        config = results["variance_regularization_config"]
                        reg_weight = config.get("variance_reg_weight", 0.0) or 0.0
                    reg_weights.append(reg_weight)
            
            if len(model_names) > 1:
                # Create plots directory
                plots_dir = self.output_dir / "comparison_plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Plot 1: Perplexity comparison
                plt.figure(figsize=(12, 6))
                bars = plt.bar(range(len(model_names)), perplexities, color='skyblue', alpha=0.7)
                plt.xlabel('Models')
                plt.ylabel('Perplexity')
                plt.title('Model Perplexity Comparison')
                plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, (bar, ppl) in enumerate(zip(bars, perplexities)):
                    if ppl != float('inf'):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{ppl:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(plots_dir / "perplexity_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot 2: Regularization weight vs Perplexity
                if any(w > 0 for w in reg_weights):
                    plt.figure(figsize=(10, 6))
                    scatter = plt.scatter(reg_weights, perplexities, c=range(len(model_names)), 
                                        cmap='viridis', s=100, alpha=0.7)
                    
                    # Add model name labels
                    for i, name in enumerate(model_names):
                        if perplexities[i] != float('inf'):
                            plt.annotate(name, (reg_weights[i], perplexities[i]), 
                                       xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    plt.xlabel('Variance Regularization Weight')
                    plt.ylabel('Perplexity')
                    plt.title('Regularization Weight vs Model Performance')
                    plt.colorbar(scatter, label='Model Index')
                    plt.tight_layout()
                    plt.savefig(plots_dir / "regularization_vs_performance.png", dpi=300, bbox_inches='tight')
                    plt.close()
                
                logger.info(f"Comparison plots saved to: {plots_dir}")
                
        except Exception as e:
            logger.error(f"Failed to create comparison plots: {e}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Evaluate Variance Regularization Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        help="Path to variance regularization model directory"
    )
    
    parser.add_argument(
        "--compare_models",
        type=str,
        nargs="+",
        help="Compare multiple variance regularization models"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline model for comparison (e.g., microsoft/Phi-3-mini-4k-instruct)"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./variance_reg_evaluation",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to evaluate"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick_test", "fast_test", "standard", "compression_analysis"],
        default="standard",
        help="Evaluation preset to use"
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
    
    # Initialize evaluator
    evaluator = VarianceRegularizationEvaluator(args.output_dir)
    
    try:
        if args.compare_models:
            # Compare multiple models
            logger.info(f"Comparing {len(args.compare_models)} variance regularization models")
            
            results = evaluator.compare_models(
                model_paths=args.compare_models,
                baseline_path=args.baseline,
                tasks=tasks,
                limit=args.limit
            )
            
            print("\n" + "="*80)
            print("VARIANCE REGULARIZATION MODEL COMPARISON COMPLETED")
            print("="*80)
            
            # Print summary
            summary = results.get("comparison_summary", {})
            if "best_perplexity" in summary and summary["best_perplexity"]["model"]:
                best_model = summary["best_perplexity"]["model"]
                best_ppl = summary["best_perplexity"]["value"]
                print(f"🏆 Best Model: {best_model} (Perplexity: {best_ppl:.4f})")
            
            if "performance_ranking" in summary:
                print("\n📊 Performance Ranking:")
                for i, (model, ppl) in enumerate(summary["performance_ranking"], 1):
                    print(f"  {i}. {model}: {ppl:.4f}")
            
            if "regularization_effects" in summary:
                print("\n🔧 Regularization Effects:")
                for model, effects in summary["regularization_effects"].items():
                    weight = effects.get("regularization_weight", "N/A")
                    target = effects.get("regularization_target", "N/A")
                    ppl = effects.get("perplexity", "N/A")
                    print(f"  {model}: weight={weight}, target={target}, ppl={ppl:.4f}")
        
        elif args.model_path:
            # Evaluate single model
            results = evaluator.evaluate_single_model(
                model_path=args.model_path,
                tasks=tasks,
                limit=args.limit,
                preset=args.preset
            )
            
            print("\n" + "="*80)
            print("VARIANCE REGULARIZATION MODEL EVALUATION COMPLETED")
            print("="*80)
            print(f"Model: {args.model_path}")
            
            # Print key results
            if "evaluation_results" in results and "results" in results["evaluation_results"]:
                eval_results = results["evaluation_results"]["results"]
                if "perplexity" in eval_results:
                    print(f"Perplexity: {eval_results['perplexity']:.4f}")
            
            # Print regularization info
            if "variance_regularization_config" in results:
                config = results["variance_regularization_config"]
                if config.get("variance_regularization_detected", False):
                    print("✅ Variance regularization detected")
                    weight = config.get("variance_reg_weight", "N/A")
                    target = config.get("variance_reg_target", "N/A")
                    print(f"   Weight: {weight}, Target: {target}")
                else:
                    print("❌ No variance regularization detected")
        
        else:
            print("Error: Please specify either --model_path or --compare_models")
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()