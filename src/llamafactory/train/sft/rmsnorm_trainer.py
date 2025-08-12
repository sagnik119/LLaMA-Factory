# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from typing_extensions import override

from ...extras import logging
from .trainer import CustomSeq2SeqTrainer

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)


class RMSNormRegularizedTrainer(CustomSeq2SeqTrainer):
    r"""Custom trainer that adds RMSNorm regularization to the loss."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        **kwargs,
    ) -> None:
        super().__init__(finetuning_args=finetuning_args, **kwargs)
        self.rmsnorm_reg_layers = finetuning_args.rmsnorm_reg_layers
        self.rmsnorm_reg_weight = finetuning_args.rmsnorm_reg_weight
        self.rmsnorm_reg_target_norm = finetuning_args.rmsnorm_reg_target_norm
        self.use_rmsnorm_regularization = finetuning_args.use_rmsnorm_regularization
        
        # Variance regularization parameters
        self.use_variance_regularization = finetuning_args.use_variance_regularization
        self.variance_reg_layers = finetuning_args.variance_reg_layers
        self.variance_reg_weight = finetuning_args.variance_reg_weight
        self.variance_reg_target = finetuning_args.variance_reg_target
        self.variance_reg_norm_type = finetuning_args.variance_reg_norm_type
        
        # Store hooks for RMSNorm outputs
        self.rmsnorm_outputs = {}
        self.variance_outputs = {}  # Separate storage for variance regularization
        self.hooks = []
        self._hooks_registered = False
        
        # Don't register hooks during initialization to avoid distributed training issues
        # Hooks will be registered when training actually starts

    def _register_rmsnorm_hooks(self):
        """Register forward hooks to capture RMSNorm outputs from specified layers."""
        try:
            # Get the actual model (unwrap from DDP if needed)
            model = self.model
            if hasattr(model, 'module'):
                model = model.module
            
            # Initialize thread lock for distributed training
            if not hasattr(self, '_rmsnorm_outputs_lock'):
                import threading
                self._rmsnorm_outputs_lock = threading.Lock()
            
            # Register hooks for standard RMSNorm regularization
            if self.use_rmsnorm_regularization:
                self._register_norm_regularization_hooks(model, self.rmsnorm_reg_layers, "rmsnorm")
            
            # Register hooks for variance regularization
            if self.use_variance_regularization:
                self._register_norm_regularization_hooks(model, self.variance_reg_layers, "variance")
                    
        except Exception as e:
            logger.warning_rank0(f"Failed to register RMSNorm hooks: {e}")
            # Don't fail training if hook registration fails
            self.use_rmsnorm_regularization = False
            self.use_variance_regularization = False

    def _register_norm_regularization_hooks(self, model, layer_indices, hook_type):
        """Register hooks for norm regularization (either standard or variance)."""
        for layer_idx in layer_indices:
            try:
                # Determine which norm types to hook based on configuration
                norm_types = []
                if hook_type == "variance":
                    if self.variance_reg_norm_type == "post_attention_layernorm":
                        norm_types = ["post_attention_layernorm"]
                    elif self.variance_reg_norm_type == "input_layernorm":
                        norm_types = ["input_layernorm"]
                    elif self.variance_reg_norm_type == "both":
                        norm_types = ["post_attention_layernorm", "input_layernorm"]
                else:
                    norm_types = ["post_attention_layernorm"]  # Default for standard regularization
                
                for norm_type in norm_types:
                    # Try different possible paths for RMSNorm layers
                    possible_paths = [
                        f"model.layers.{layer_idx}.{norm_type}",
                        f"layers.{layer_idx}.{norm_type}",
                        f"transformer.h.{layer_idx}.{norm_type}",
                        f"model.layers.{layer_idx}.ln_2" if norm_type == "post_attention_layernorm" else f"model.layers.{layer_idx}.ln_1",
                        f"layers.{layer_idx}.ln_2" if norm_type == "post_attention_layernorm" else f"layers.{layer_idx}.ln_1"
                    ]
                    
                    rmsnorm_layer = None
                    for path in possible_paths:
                        try:
                            rmsnorm_layer = model
                            for attr in path.split('.'):
                                rmsnorm_layer = getattr(rmsnorm_layer, attr)
                            break
                        except AttributeError:
                            continue
                    
                    if rmsnorm_layer is not None:
                        def make_hook(layer_idx, norm_type, hook_type):
                            def hook(module, input, output):
                                # Store outputs in a thread-safe way for distributed training
                                with self._rmsnorm_outputs_lock:
                                    key = f"layer_{layer_idx}_{norm_type}"
                                    if hook_type == "variance":
                                        self.variance_outputs[key] = output.detach() if output is not None else None
                                    else:
                                        self.rmsnorm_outputs[key] = output.detach() if output is not None else None
                            return hook
                        
                        handle = rmsnorm_layer.register_forward_hook(make_hook(layer_idx, norm_type, hook_type))
                        self.hooks.append(handle)
                        logger.info_rank0(f"Registered {hook_type} hook for layer {layer_idx} {norm_type}")
                    else:
                        logger.warning_rank0(f"Could not find {norm_type} layer for layer {layer_idx}")
                        
            except Exception as e:
                logger.warning_rank0(f"Failed to register {hook_type} hook for layer {layer_idx}: {e}")

    def _compute_rmsnorm_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for RMSNorm outputs."""
        # Get device from model (handle DDP)
        device = next(self.model.parameters()).device
        
        if not self.rmsnorm_outputs:
            return torch.tensor(0.0, device=device)
        
        total_reg_loss = torch.tensor(0.0, device=device)
        
        # Thread-safe access to rmsnorm_outputs
        if hasattr(self, '_rmsnorm_outputs_lock'):
            with self._rmsnorm_outputs_lock:
                outputs_copy = dict(self.rmsnorm_outputs)
                self.rmsnorm_outputs.clear()
        else:
            outputs_copy = dict(self.rmsnorm_outputs)
            self.rmsnorm_outputs.clear()
        
        for layer_name, output in outputs_copy.items():
            if output is not None:
                # Compute row (token) norms - L2 norm across the feature dimension
                # output shape: [batch_size, seq_len, hidden_size]
                row_norms = torch.norm(output, p=2, dim=-1)  # [batch_size, seq_len]
                
                # Regularization: encourage row norms to be close to target norm
                target_norms = torch.full_like(row_norms, self.rmsnorm_reg_target_norm)
                reg_loss = F.mse_loss(row_norms, target_norms)
                
                total_reg_loss += reg_loss
        
        return total_reg_loss

    def _compute_variance_regularization_loss(self) -> torch.Tensor:
        """Compute variance regularization loss for RMSNorm outputs."""
        # Get device from model (handle DDP)
        device = next(self.model.parameters()).device
        
        if not self.variance_outputs:
            return torch.tensor(0.0, device=device)
        
        total_var_loss = torch.tensor(0.0, device=device)
        
        # Thread-safe access to variance_outputs
        if hasattr(self, '_rmsnorm_outputs_lock'):
            with self._rmsnorm_outputs_lock:
                outputs_copy = dict(self.variance_outputs)
                self.variance_outputs.clear()
        else:
            outputs_copy = dict(self.variance_outputs)
            self.variance_outputs.clear()
        
        for layer_name, output in outputs_copy.items():
            if output is not None:
                # Compute variance across the feature dimension for each token
                # output shape: [batch_size, seq_len, hidden_size]
                
                # Compute variance for each token (across hidden dimensions)
                token_variances = torch.var(output, dim=-1, unbiased=False)  # [batch_size, seq_len]
                
                # Target variance (regularize towards this value)
                target_variance = torch.full_like(token_variances, self.variance_reg_target)
                
                # Variance regularization loss: encourage variances to be close to target
                var_loss = F.mse_loss(token_variances, target_variance)
                
                total_var_loss += var_loss
        
        return total_var_loss

    @override
    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """Compute loss with RMSNorm regularization."""
        
        # Register hooks on first call (after distributed training is initialized)
        if (self.use_rmsnorm_regularization or self.use_variance_regularization) and not self._hooks_registered:
            self._register_rmsnorm_hooks()
            self._hooks_registered = True
        
        # Compute the standard loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
        
        total_loss = loss
        log_dict = {"train/base_loss": loss.item()}
        
        # Add RMSNorm regularization loss if enabled
        if self.use_rmsnorm_regularization and model.training:
            reg_loss = self._compute_rmsnorm_regularization_loss()
            total_loss = total_loss + self.rmsnorm_reg_weight * reg_loss
            log_dict["train/rmsnorm_reg_loss"] = reg_loss.item()
        
        # Add variance regularization loss if enabled
        if self.use_variance_regularization and model.training:
            var_loss = self._compute_variance_regularization_loss()
            total_loss = total_loss + self.variance_reg_weight * var_loss
            log_dict["train/variance_reg_loss"] = var_loss.item()
        
        # Update total loss in log
        log_dict["train/total_loss"] = total_loss.item()
        
        # Log all losses
        if hasattr(self, 'log') and model.training:
            self.log(log_dict)
        
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override save_model to handle Phi-3 model saving issues."""
        try:
            super().save_model(output_dir, _internal_call)
        except (FileNotFoundError, Exception) as e:
            # Handle all transformers library errors for Phi-3 model saving
            if isinstance(e, FileNotFoundError) or "custom_object_save" in str(e) or "phi3" in str(e).lower():
                logger.warning_rank0(f"Model saving failed due to transformers library issue: {e}")
                logger.warning_rank0("This is a known issue with Phi-3 models and doesn't affect training.")
                # Save only the state dict instead
                if output_dir is None:
                    output_dir = self.args.output_dir
                
                import torch
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                # Save only trainable parameters (RMSNorm weights)
                trainable_state_dict = {
                    name: param for name, param in self.model.named_parameters()
                    if param.requires_grad
                }
                torch.save(trainable_state_dict, os.path.join(output_dir, "rmsnorm_weights.pt"))
                logger.info_rank0(f"Saved RMSNorm weights to {os.path.join(output_dir, 'rmsnorm_weights.pt')}")
                
                # Also save training arguments and config
                import json
                config_dict = {
                    "rmsnorm_reg_layers": self.rmsnorm_reg_layers,
                    "rmsnorm_reg_weight": self.rmsnorm_reg_weight,
                    "rmsnorm_reg_target_norm": self.rmsnorm_reg_target_norm,
                    "use_rmsnorm_regularization": self.use_rmsnorm_regularization,
                    "use_variance_regularization": self.use_variance_regularization,
                    "variance_reg_layers": self.variance_reg_layers,
                    "variance_reg_weight": self.variance_reg_weight,
                    "variance_reg_target": self.variance_reg_target,
                    "variance_reg_norm_type": self.variance_reg_norm_type
                }
                with open(os.path.join(output_dir, "rmsnorm_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info_rank0(f"Saved RMSNorm config to {os.path.join(output_dir, 'rmsnorm_config.json')}")
            else:
                raise e

    def __del__(self):
        """Clean up hooks when trainer is destroyed."""
        for hook in self.hooks:
            hook.remove()