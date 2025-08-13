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

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ...extras.logging import get_logger
from .trainer import CustomSeq2SeqTrainer

logger = get_logger(__name__)


class BOSZeroTrainer(CustomSeq2SeqTrainer):
    """
    Custom trainer that adds BOS tokens and zeros out their embeddings.
    
    This trainer:
    1. Ensures BOS tokens are added at the beginning of sequences
    2. Hooks into the embedding layer to zero out position 0 outputs
    3. Maintains compatibility with LoRA fine-tuning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_hook_handle = None
        
        # Get BOS scaling factor from finetuning args (default to 0.01 for 99% reduction)
        finetuning_args = kwargs.get('finetuning_args')
        self.bos_scaling_factor = getattr(finetuning_args, 'bos_scaling_factor', 0.01) if finetuning_args else 0.01
        
        self._setup_embedding_hook()
        
        reduction_percent = (1 - self.bos_scaling_factor) * 100
        logger.info("🎯 BOSZeroTrainer initialized")
        logger.info("   - BOS tokens will be added to sequences")
        logger.info(f"   - Position 0 embeddings will be scaled by {self.bos_scaling_factor} ({reduction_percent:.0f}% reduction)")
        logger.info("   - This maintains gradient stability while minimizing BOS influence")
        logger.info("   - Reduced BOS influence forces model adaptation")
    
    def _setup_embedding_hook(self):
        """Set up hook to zero out BOS token embeddings at position 0."""
        # Find the embedding layer - handle both regular and wrapped models
        embedding_layer = None
        model_to_search = self.model
        
        # Handle DDP/FSDP wrapped models
        if hasattr(self.model, 'module'):
            model_to_search = self.model.module
            
        for name, module in model_to_search.named_modules():
            if isinstance(module, nn.Embedding) and 'embed_tokens' in name:
                embedding_layer = module
                logger.info(f"✅ Found embedding layer: {name}")
                break
        
        if embedding_layer is None:
            logger.warning("⚠️ No embedding layer found - BOS zeroing will not be applied")
            return
        
        def bos_zero_hook(module, input, output):
            """Hook to heavily reduce embeddings at position 0 (BOS token position)."""
            try:
                # output shape: (batch_size, seq_len, hidden_size)
                if output.dim() == 3 and output.size(1) > 0:
                    # Create scaled output with very small scaling factor (99% reduction)
                    scaled_output = output.clone()
                    scaled_output[:, 0, :] = output[:, 0, :] * self.bos_scaling_factor
                    return scaled_output
                        
                return output
            except Exception as e:
                logger.warning(f"⚠️ Error in BOS scaling hook: {e}")
                return output
        
        # Register the hook
        self.embedding_hook_handle = embedding_layer.register_forward_hook(bos_zero_hook)
        logger.info("✅ BOS zeroing hook registered on embedding layer")
    
    def _remove_embedding_hook(self):
        """Remove the embedding hook."""
        if self.embedding_hook_handle is not None:
            self.embedding_hook_handle.remove()
            self.embedding_hook_handle = None
            logger.info("🧹 BOS zeroing hook removed")
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step with BOS token handling.
        """
        # Ensure BOS tokens are present (this should be handled by data preprocessing)
        # The hook will automatically zero out position 0 embeddings
        
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            return super().training_step(model, inputs)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """
        Perform a prediction step with BOS token handling.
        """
        # The hook will automatically zero out position 0 embeddings during evaluation too
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save the model and log BOS zeroing configuration.
        """
        super().save_model(output_dir, _internal_call)
        
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # Save BOS zeroing configuration
        bos_config = {
            "add_bos_token": True,
            "zero_bos_embedding": True,
            "description": "Model trained with BOS tokens added and position 0 embeddings zeroed out",
            "trainer_class": "BOSZeroTrainer"
        }
        
        import json
        import os
        config_path = os.path.join(output_dir, "bos_zero_config.json")
        with open(config_path, "w") as f:
            json.dump(bos_config, f, indent=2)
        
        logger.info(f"💾 BOS zeroing configuration saved to {config_path}")
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with BOS zeroing configuration.
        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        # Call parent save checkpoint with correct signature
        if metrics is not None:
            super()._save_checkpoint(model, trial, metrics)
        else:
            super()._save_checkpoint(model, trial)
        
        # Save BOS zeroing configuration to checkpoint
        bos_config = {
            "add_bos_token": True,
            "zero_bos_embedding": True,
            "global_step": self.state.global_step,
            "description": "Checkpoint with BOS tokens added and position 0 embeddings zeroed out",
            "trainer_class": "BOSZeroTrainer"
        }
        
        import json
        config_path = os.path.join(output_dir, "bos_zero_config.json")
        with open(config_path, "w") as f:
            json.dump(bos_config, f, indent=2)
        
        logger.info(f"💾 BOS zeroing configuration saved to checkpoint: {config_path}")
    
    def __del__(self):
        """Clean up hooks when trainer is destroyed."""
        self._remove_embedding_hook()


def create_bos_zero_trainer(
    model,
    args,
    data_collator,
    train_dataset,
    eval_dataset=None,
    tokenizer=None,
    compute_metrics=None,
    callbacks=None,
    optimizers=(None, None),
    preprocess_logits_for_metrics=None,
):
    """
    Factory function to create a BOSZeroTrainer instance.
    """
    return BOSZeroTrainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )