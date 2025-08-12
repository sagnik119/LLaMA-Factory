"""
LLaMA-Factory Model Evaluation Suite

This package provides comprehensive evaluation tools for models fine-tuned with LLaMA-Factory,
with special support for variance regularization and RMSNorm analysis.

Modules:
- evaluate_llamafactory_models: General evaluation for LLaMA-Factory models
- evaluate_variance_regularization: Specialized evaluation for variance regularization models
- evaluation_utils: Utility functions for evaluation tasks
- evaluation_pipeline: Automated evaluation pipeline
"""

__version__ = "1.0.0"
__author__ = "LLaMA-Factory Evaluation Team"

from .evaluate_llamafactory_models import LLaMAFactoryModelLoader, LLaMAFactoryEvaluator
from .evaluate_variance_regularization import VarianceRegularizationAnalyzer, VarianceRegularizationEvaluator

__all__ = [
    "LLaMAFactoryModelLoader",
    "LLaMAFactoryEvaluator", 
    "VarianceRegularizationAnalyzer",
    "VarianceRegularizationEvaluator"
]