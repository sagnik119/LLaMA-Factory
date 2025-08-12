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
        
        # Store hooks for RMSNorm outputs
        self.rmsnorm_outputs = {}
        self.hooks = []
        
        if self.use_rmsnorm_regularization:
            self._register_rmsnorm_hooks()

    def _register_rmsnorm_hooks(self):
        """Register forward hooks to capture RMSNorm outputs from specified layers."""
        # Get the actual model (unwrap from DDP if needed)
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        
        # For Phi-3 models, the structure is typically:
        # model.layers[i].post_attention_layernorm
        for layer_idx in self.rmsnorm_reg_layers:
            try:
                # Try different possible paths for RMSNorm layers
                possible_paths = [
                    f"model.layers.{layer_idx}.post_attention_layernorm",
                    f"layers.{layer_idx}.post_attention_layernorm",
                    f"transformer.h.{layer_idx}.post_attention_layernorm",
                    f"model.layers.{layer_idx}.ln_2",
                    f"layers.{layer_idx}.ln_2"
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
                    def make_hook(layer_idx):
                        def hook(module, input, output):
                            # Store outputs in a thread-safe way for distributed training
                            if not hasattr(self, '_rmsnorm_outputs_lock'):
                                import threading
                                self._rmsnorm_outputs_lock = threading.Lock()
                            
                            with self._rmsnorm_outputs_lock:
                                self.rmsnorm_outputs[f"layer_{layer_idx}"] = output
                        return hook
                    
                    handle = rmsnorm_layer.register_forward_hook(make_hook(layer_idx))
                    self.hooks.append(handle)
                    logger.info_rank0(f"Registered RMSNorm hook for layer {layer_idx}")
                else:
                    logger.warning_rank0(f"Could not find RMSNorm layer for layer {layer_idx}")
                    
            except Exception as e:
                logger.warning_rank0(f"Failed to register hook for layer {layer_idx}: {e}")

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

    @override
    def compute_loss(
        self, 
        model: "PreTrainedModel", 
        inputs: Dict[str, Union[torch.Tensor, Any]], 
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """Compute loss with RMSNorm regularization."""
        
        # Compute the standard loss
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
        
        # Add RMSNorm regularization loss if enabled
        if self.use_rmsnorm_regularization and model.training:
            reg_loss = self._compute_rmsnorm_regularization_loss()
            total_loss = loss + self.rmsnorm_reg_weight * reg_loss
            
            # Log the regularization loss
            if hasattr(self, 'log'):
                self.log({
                    "train/rmsnorm_reg_loss": reg_loss.item(),
                    "train/base_loss": loss.item(),
                    "train/total_loss": total_loss.item()
                })
        else:
            total_loss = loss
        
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
                    "use_rmsnorm_regularization": self.use_rmsnorm_regularization
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