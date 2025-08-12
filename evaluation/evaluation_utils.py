#!/usr/bin/env python3

"""
Evaluation Utilities for LLaMA-Factory Models

This module provides utility functions and classes for evaluating
LLaMA-Factory fine-tuned models, including result analysis,
visualization, and comparison tools.

Author: LLaMA-Factory Evaluation Team
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import torch

# Setup logging
logger = logging.getLogger(__name__)


class ModelCheckpointManager:
    """
    Manager for LLaMA-Factory model checkpoints.
    """
    
    @staticmethod
    def find_checkpoints(base_dir: str, pattern: str = "*") -> List[Path]:
        """
        Find all model checkpoints in a directory.
        
        Args:
            base_dir: Base directory to search
            pattern: Pattern to match checkpoint directories
            
        Returns:
            List of checkpoint paths
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        checkpoints = []
        for path in base_path.glob(pattern):
            if path.is_dir() and ModelCheckpointManager.is_valid_checkpoint(path):
                checkpoints.append(path)
        
        return sorted(checkpoints)
    
    @staticmethod
    def is_valid_checkpoint(checkpoint_path: Path) -> bool:
        """
        Check if a directory is a valid LLaMA-Factory checkpoint.
        
        Args:
            checkpoint_path: Path to check
            
        Returns:
            True if valid checkpoint
        """
        # Check for essential files
        required_files = ["config.json"]
        optional_files = ["adapter_config.json", "pytorch_model.bin", "model.safetensors"]
        
        has_required = all((checkpoint_path / f).exists() for f in required_files)
        has_optional = any((checkpoint_path / f).exists() for f in optional_files)
        
        return has_required and has_optional
    
    @staticmethod
    def get_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
        """
        Extract information from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Dictionary containing checkpoint information
        """
        info = {
            "path": str(checkpoint_path),
            "name": checkpoint_path.name,
            "type": "unknown",
            "base_model": None,
            "training_args": {},
            "adapter_config": {},
            "created_time": None
        }
        
        try:
            # Get creation time
            if checkpoint_path.exists():
                info["created_time"] = datetime.fromtimestamp(
                    checkpoint_path.stat().st_mtime
                ).isoformat()
            
            # Check checkpoint type
            if (checkpoint_path / "adapter_config.json").exists():
                info["type"] = "lora"
                
                # Read adapter config
                with open(checkpoint_path / "adapter_config.json", 'r') as f:
                    adapter_config = json.load(f)
                    info["adapter_config"] = adapter_config
                    info["base_model"] = adapter_config.get("base_model_name_or_path")
            
            elif (checkpoint_path / "pytorch_model.bin").exists() or (checkpoint_path / "model.safetensors").exists():
                info["type"] = "merged"
            
            # Read training args if available
            training_args_file = checkpoint_path / "training_args.json"
            if training_args_file.exists():
                with open(training_args_file, 'r') as f:
                    info["training_args"] = json.load(f)
            
            # Read model config
            config_file = checkpoint_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if not info["base_model"] and "_name_or_path" in config:
                        info["base_model"] = config["_name_or_path"]
        
        except Exception as e:
            logger.error(f"Failed to read checkpoint info: {e}")
            info["error"] = str(e)
        
        return info


class EvaluationResultsAnalyzer:
    """
    Analyzer for evaluation results.
    """
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer.
        
        Args:
            results_file: Path to evaluation results JSON file
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load evaluation results from file."""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}
    
    def extract_metrics(self) -> pd.DataFrame:
        """
        Extract metrics into a pandas DataFrame.
        
        Returns:
            DataFrame with evaluation metrics
        """
        metrics_data = []
        
        try:
            if "results" in self.results:
                for group_name, group_results in self.results["results"].items():
                    if "results" in group_results:
                        for task_name, task_results in group_results["results"].items():
                            for metric_name, metric_value in task_results.items():
                                if isinstance(metric_value, (int, float)):
                                    metrics_data.append({
                                        "group": group_name,
                                        "task": task_name,
                                        "metric": metric_name,
                                        "value": metric_value
                                    })
            
            return pd.DataFrame(metrics_data)
        
        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
            return pd.DataFrame()
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get summary metrics from evaluation results.
        
        Returns:
            Dictionary of summary metrics
        """
        summary = {}
        
        try:
            df = self.extract_metrics()
            if not df.empty:
                # Get perplexity if available
                ppl_rows = df[df['metric'].str.contains('perplexity|ppl', case=False, na=False)]
                if not ppl_rows.empty:
                    summary['perplexity'] = ppl_rows['value'].iloc[0]
                
                # Get accuracy metrics
                acc_rows = df[df['metric'].str.contains('acc|accuracy', case=False, na=False)]
                if not acc_rows.empty:
                    summary['avg_accuracy'] = acc_rows['value'].mean()
                    summary['max_accuracy'] = acc_rows['value'].max()
                
                # Get task count
                summary['num_tasks'] = df['task'].nunique()
                summary['num_metrics'] = len(df)
        
        except Exception as e:
            logger.error(f"Failed to get summary metrics: {e}")
        
        return summary
    
    def compare_with_baseline(self, baseline_file: str) -> pd.DataFrame:
        """
        Compare results with a baseline.
        
        Args:
            baseline_file: Path to baseline results file
            
        Returns:
            DataFrame with comparison results
        """
        try:
            baseline_analyzer = EvaluationResultsAnalyzer(baseline_file)
            
            current_df = self.extract_metrics()
            baseline_df = baseline_analyzer.extract_metrics()
            
            # Merge on task and metric
            merged = pd.merge(
                current_df, baseline_df,
                on=['task', 'metric'],
                suffixes=('_current', '_baseline')
            )
            
            # Calculate differences
            merged['difference'] = merged['value_current'] - merged['value_baseline']
            merged['percent_change'] = (merged['difference'] / merged['value_baseline']) * 100
            
            return merged[['task', 'metric', 'value_current', 'value_baseline', 'difference', 'percent_change']]
        
        except Exception as e:
            logger.error(f"Failed to compare with baseline: {e}")
            return pd.DataFrame()


class EvaluationVisualizer:
    """
    Visualizer for evaluation results.
    """
    
    def __init__(self, output_dir: str = "./evaluation_plots"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_metrics_comparison(self, 
                              results_files: List[str],
                              model_names: Optional[List[str]] = None,
                              save_name: str = "metrics_comparison.png") -> None:
        """
        Plot comparison of metrics across multiple models.
        
        Args:
            results_files: List of result file paths
            model_names: Optional list of model names
            save_name: Name of saved plot file
        """
        try:
            all_metrics = []
            
            for i, results_file in enumerate(results_files):
                analyzer = EvaluationResultsAnalyzer(results_file)
                df = analyzer.extract_metrics()
                
                if not df.empty:
                    model_name = model_names[i] if model_names else f"Model_{i+1}"
                    df['model'] = model_name
                    all_metrics.append(df)
            
            if not all_metrics:
                logger.warning("No metrics data found for plotting")
                return
            
            combined_df = pd.concat(all_metrics, ignore_index=True)
            
            # Create subplots for different metric types
            metric_types = combined_df['metric'].unique()
            n_metrics = len(metric_types)
            
            if n_metrics == 0:
                return
            
            fig, axes = plt.subplots(
                nrows=(n_metrics + 2) // 3, 
                ncols=3, 
                figsize=(15, 5 * ((n_metrics + 2) // 3))
            )
            
            if n_metrics == 1:
                axes = [axes]
            elif n_metrics <= 3:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metric_types):
                if i >= len(axes):
                    break
                
                metric_data = combined_df[combined_df['metric'] == metric]
                
                if len(metric_data) > 1:
                    sns.barplot(
                        data=metric_data,
                        x='model',
                        y='value',
                        hue='task',
                        ax=axes[i]
                    )
                    axes[i].set_title(f'{metric.title()} Comparison')
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Metrics comparison plot saved to: {self.output_dir / save_name}")
        
        except Exception as e:
            logger.error(f"Failed to create metrics comparison plot: {e}")
    
    def plot_perplexity_trend(self,
                            results_files: List[str],
                            model_names: Optional[List[str]] = None,
                            save_name: str = "perplexity_trend.png") -> None:
        """
        Plot perplexity trend across models.
        
        Args:
            results_files: List of result file paths
            model_names: Optional list of model names
            save_name: Name of saved plot file
        """
        try:
            perplexities = []
            names = []
            
            for i, results_file in enumerate(results_files):
                analyzer = EvaluationResultsAnalyzer(results_file)
                summary = analyzer.get_summary_metrics()
                
                if 'perplexity' in summary:
                    perplexities.append(summary['perplexity'])
                    names.append(model_names[i] if model_names else f"Model_{i+1}")
            
            if not perplexities:
                logger.warning("No perplexity data found for plotting")
                return
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(names)), perplexities, color='skyblue', alpha=0.7)
            
            # Add value labels on bars
            for bar, ppl in zip(bars, perplexities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{ppl:.3f}', ha='center', va='bottom')
            
            plt.xlabel('Models')
            plt.ylabel('Perplexity')
            plt.title('Model Perplexity Comparison')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Perplexity trend plot saved to: {self.output_dir / save_name}")
        
        except Exception as e:
            logger.error(f"Failed to create perplexity trend plot: {e}")


class EvaluationReportGenerator:
    """
    Generator for comprehensive evaluation reports.
    """
    
    def __init__(self, output_dir: str = "./evaluation_reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_model_report(self,
                            results_file: str,
                            model_name: str,
                            checkpoint_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report for a single model.
        
        Args:
            results_file: Path to evaluation results
            model_name: Name of the model
            checkpoint_path: Optional path to model checkpoint
            
        Returns:
            Path to generated report file
        """
        try:
            analyzer = EvaluationResultsAnalyzer(results_file)
            summary = analyzer.get_summary_metrics()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.output_dir / f"{model_name}_report_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write(f"# Evaluation Report: {model_name}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if checkpoint_path:
                    f.write(f"**Model Path:** `{checkpoint_path}`\n\n")
                
                # Summary metrics
                f.write("## Summary Metrics\n\n")
                for metric, value in summary.items():
                    f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
                f.write("\n")
                
                # Detailed results
                f.write("## Detailed Results\n\n")
                df = analyzer.extract_metrics()
                
                if not df.empty:
                    # Group by task
                    for task in df['task'].unique():
                        task_data = df[df['task'] == task]
                        f.write(f"### {task.title()}\n\n")
                        
                        for _, row in task_data.iterrows():
                            f.write(f"- **{row['metric']}:** {row['value']:.4f}\n")
                        f.write("\n")
                
                # Model information
                if checkpoint_path:
                    checkpoint_info = ModelCheckpointManager.get_checkpoint_info(Path(checkpoint_path))
                    f.write("## Model Information\n\n")
                    f.write(f"- **Type:** {checkpoint_info.get('type', 'Unknown')}\n")
                    f.write(f"- **Base Model:** {checkpoint_info.get('base_model', 'Unknown')}\n")
                    
                    if checkpoint_info.get('training_args'):
                        f.write("- **Training Configuration:**\n")
                        training_args = checkpoint_info['training_args']
                        for key, value in training_args.items():
                            if key.startswith('variance_reg') or key in ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size']:
                                f.write(f"  - {key}: {value}\n")
                    f.write("\n")
            
            logger.info(f"Model report generated: {report_file}")
            return str(report_file)
        
        except Exception as e:
            logger.error(f"Failed to generate model report: {e}")
            return ""
    
    def generate_comparison_report(self,
                                 results_files: List[str],
                                 model_names: List[str],
                                 report_name: str = "model_comparison") -> str:
        """
        Generate a comparison report for multiple models.
        
        Args:
            results_files: List of result file paths
            model_names: List of model names
            report_name: Name of the report
            
        Returns:
            Path to generated report file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.output_dir / f"{report_name}_{timestamp}.md"
            
            with open(report_file, 'w') as f:
                f.write(f"# Model Comparison Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**Models Compared:** {len(model_names)}\n\n")
                
                # Summary table
                f.write("## Summary Comparison\n\n")
                f.write("| Model | Perplexity | Avg Accuracy | Tasks |\n")
                f.write("|-------|------------|--------------|-------|\n")
                
                summaries = []
                for results_file, model_name in zip(results_files, model_names):
                    analyzer = EvaluationResultsAnalyzer(results_file)
                    summary = analyzer.get_summary_metrics()
                    summaries.append((model_name, summary))
                    
                    ppl = summary.get('perplexity', 'N/A')
                    acc = summary.get('avg_accuracy', 'N/A')
                    tasks = summary.get('num_tasks', 'N/A')
                    
                    ppl_str = f"{ppl:.4f}" if isinstance(ppl, (int, float)) else str(ppl)
                    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
                    
                    f.write(f"| {model_name} | {ppl_str} | {acc_str} | {tasks} |\n")
                
                f.write("\n")
                
                # Best performing models
                f.write("## Best Performing Models\n\n")
                
                # Find best perplexity
                best_ppl = min(
                    (summary.get('perplexity', float('inf')), name) 
                    for name, summary in summaries 
                    if isinstance(summary.get('perplexity'), (int, float))
                )
                if best_ppl[0] != float('inf'):
                    f.write(f"- **Best Perplexity:** {best_ppl[1]} ({best_ppl[0]:.4f})\n")
                
                # Find best accuracy
                best_acc = max(
                    (summary.get('avg_accuracy', 0), name) 
                    for name, summary in summaries 
                    if isinstance(summary.get('avg_accuracy'), (int, float))
                )
                if best_acc[0] > 0:
                    f.write(f"- **Best Average Accuracy:** {best_acc[1]} ({best_acc[0]:.4f})\n")
                
                f.write("\n")
                
                # Individual model details
                f.write("## Individual Model Details\n\n")
                for model_name, summary in summaries:
                    f.write(f"### {model_name}\n\n")
                    for metric, value in summary.items():
                        if isinstance(value, (int, float)):
                            f.write(f"- **{metric.replace('_', ' ').title()}:** {value:.4f}\n")
                        else:
                            f.write(f"- **{metric.replace('_', ' ').title()}:** {value}\n")
                    f.write("\n")
            
            logger.info(f"Comparison report generated: {report_file}")
            return str(report_file)
        
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return ""


def setup_evaluation_environment():
    """Setup the evaluation environment with required dependencies."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        import seaborn as sns
        sns.set_style("whitegrid")
        
        logger.info("Evaluation environment setup completed")
        return True
    
    except ImportError as e:
        logger.error(f"Failed to setup evaluation environment: {e}")
        return False


def validate_evaluation_setup():
    """Validate that the evaluation setup is correct."""
    checks = {
        "sparsity_subspace_fork": False,
        "lm_evaluation_harness": False,
        "matplotlib": False,
        "seaborn": False,
        "pandas": False
    }
    
    try:
        # Check for sparsity_subspace_fork
        parent_dir = Path(__file__).parent.parent.parent
        if (parent_dir / "sparsity_subspace_fork").exists():
            checks["sparsity_subspace_fork"] = True
        
        # Check for lm-evaluation-harness
        if (parent_dir / "lm-evaluation-harness").exists():
            checks["lm_evaluation_harness"] = True
        
        # Check Python packages
        try:
            import matplotlib
            checks["matplotlib"] = True
        except ImportError:
            pass
        
        try:
            import seaborn
            checks["seaborn"] = True
        except ImportError:
            pass
        
        try:
            import pandas
            checks["pandas"] = True
        except ImportError:
            pass
    
    except Exception as e:
        logger.error(f"Error during validation: {e}")
    
    return checks