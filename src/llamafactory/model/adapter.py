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

import re
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras import logging
from .model_utils.misc import find_all_linear_modules, find_expanded_modules
from .model_utils.quantization import QuantizationMethod
from .model_utils.unsloth import get_unsloth_peft_model, load_unsloth_peft_model
from .model_utils.visual import COMPOSITE_MODELS, get_forbidden_modules, patch_target_modules


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _setup_rmsnorm_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    """Setup RMSNorm-only fine-tuning by freezing all parameters except RMSNorm layers."""
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: RMSNorm-only")
    
    # Apply element-specific caps first (before freezing/unfreezing)
    _apply_rmsnorm_element_caps(model, finetuning_args)
    
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)
    
    # Then unfreeze only RMSNorm parameters
    rmsnorm_count = 0
    for name, param in model.named_parameters():
        # Look for RMSNorm layers (common patterns: "norm", "layer_norm", "post_attention_layernorm", "input_layernorm")
        if any(norm_name in name.lower() for norm_name in ["norm", "layer_norm", "post_attention_layernorm", "input_layernorm"]):
            param.requires_grad_(True)
            rmsnorm_count += 1
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
            logger.info_rank0(f"Unfrozen RMSNorm parameter: {name}")
    
    logger.info_rank0(f"Total RMSNorm parameters unfrozen: {rmsnorm_count}")

def _apply_rmsnorm_element_caps(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    """
    Apply element-specific caps to RMSNorm parameters and optionally freeze them.
    
    Args:
        model: The model to apply caps to
        finetuning_args: Fine-tuning arguments containing element cap specifications
    """
    if not hasattr(finetuning_args, 'rmsnorm_element_caps_dict') or not finetuning_args.rmsnorm_element_caps_dict:
        return
    
    logger.info_rank0("Applying RMSNorm element caps...")
    
    for param_spec, cap_value in finetuning_args.rmsnorm_element_caps_dict.items():
        # Parse parameter specification: layer.param_type.element_idx
        parts = param_spec.split('.')
        if len(parts) != 3:
            logger.warning_rank0(f"Invalid parameter specification: {param_spec}. Expected format: 'layer.param_type.element_idx'")
            continue
        
        layer_idx, param_type, element_idx = parts
        try:
            layer_idx = int(layer_idx)
            element_idx = int(element_idx)
        except ValueError:
            logger.warning_rank0(f"Invalid layer or element index in: {param_spec}")
            continue
        
        # Find the parameter in the model
        param_name = f"model.layers.{layer_idx}.{param_type}.weight"
        param = None
        
        for name, p in model.named_parameters():
            if name == param_name:
                param = p
                break
        
        if param is None:
            logger.warning_rank0(f"Parameter not found: {param_name}")
            continue
        
        # Check if element index is valid
        if element_idx >= param.size(0):
            logger.warning_rank0(f"Element index {element_idx} out of range for parameter {param_name} (size: {param.size(0)})")
            continue
        
        # Apply the cap
        with torch.no_grad():
            original_value = param.data[element_idx].item()
            param.data[element_idx] = cap_value
            logger.info_rank0(f"Capped {param_name}[{element_idx}] from {original_value:.6f} to {cap_value:.6f}")
        
        # Freeze the element if requested and parameter requires gradients
        if finetuning_args.freeze_capped_elements and param.requires_grad:
            # Create a custom hook to freeze this specific element
            def create_freeze_hook(param_tensor, idx):
                def freeze_element_hook(grad):
                    if grad is not None:
                        grad[idx] = 0.0
                    return grad
                return freeze_element_hook
            
            param.register_hook(create_freeze_hook(param, element_idx))
            logger.info_rank0(f"Frozen element {element_idx} of parameter {param_name}")
        elif finetuning_args.freeze_capped_elements and not param.requires_grad:
            logger.info_rank0(f"Skipping freeze hook for {param_name}[{element_idx}] - parameter doesn't require gradients")



def _setup_full_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Full")
    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    if not is_trainable:
        return

    logger.info_rank0("Fine-tuning method: Freeze")
    if hasattr(model.config, "text_config"):  # composite models
        config = getattr(model.config, "text_config")
    else:
        config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)

    model_type = getattr(model.config, "model_type", None)
    if not finetuning_args.freeze_multi_modal_projector and model_type in COMPOSITE_MODELS:
        trainable_layers.append(COMPOSITE_MODELS[model_type].projector_key)

    forbidden_modules = get_forbidden_modules(model.config, finetuning_args)
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) and not any(
            forbidden_module in name for forbidden_module in forbidden_modules
        ):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info_rank0("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    if is_trainable:
        logger.info_rank0("Fine-tuning method: {}".format("DoRA" if finetuning_args.use_dora else "LoRA"))

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False

        if model_args.use_unsloth:
            assert len(model_args.adapter_name_or_path) == 1, "Unsloth model only accepts a single adapter."
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        for adapter in adapter_to_merge:
            model: LoraModel = PeftModel.from_pretrained(model, adapter, **init_kwargs)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info_rank0(f"Merged {len(adapter_to_merge)} adapter(s).")

        if adapter_to_resume is not None:  # resume lora training
            if model_args.use_unsloth:
                model = load_unsloth_peft_model(config, model_args, finetuning_args, is_trainable=is_trainable)
            else:
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable, **init_kwargs)

        logger.info_rank0("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
        else:
            target_modules = finetuning_args.lora_target

        if finetuning_args.use_llama_pro:
            target_modules = find_expanded_modules(model, target_modules, finetuning_args.freeze_trainable_layers)

        target_modules = patch_target_modules(model, finetuning_args, target_modules)

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        if model_args.resize_vocab and finetuning_args.additional_target is None:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            module_names = set()
            for name, module in model.named_modules():
                if module in [input_embeddings, output_embeddings]:
                    module_names.add(name.split(".")[-1])

            finetuning_args.additional_target = module_names
            logger.warning_rank0("Vocab has been resized, add {} to trainable params.".format(",".join(module_names)))

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "use_rslora": finetuning_args.use_rslora,
            "use_dora": finetuning_args.use_dora,
            "modules_to_save": finetuning_args.additional_target,
        }

        if model_args.use_unsloth:
            model = get_unsloth_peft_model(model, model_args, peft_kwargs)
        else:
            if finetuning_args.pissa_init:
                if finetuning_args.pissa_iter == -1:
                    logger.info_rank0("Using PiSSA initialization.")
                    peft_kwargs["init_lora_weights"] = "pissa"
                else:
                    logger.info_rank0(f"Using PiSSA initialization with FSVD steps {finetuning_args.pissa_iter}.")
                    peft_kwargs["init_lora_weights"] = f"pissa_niter_{finetuning_args.pissa_iter}"

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **peft_kwargs,
            )
            model = get_peft_model(model, lora_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantized models can only be used for the LoRA tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if not is_trainable:
        pass
    elif finetuning_args.pure_bf16 or finetuning_args.use_badam:
        logger.info_rank0("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
    elif model_args.quantization_bit is None and is_deepspeed_zero3_enabled():
        logger.info_rank0("DeepSpeed ZeRO3 detected, remaining trainable params in float32.")
    else:
        logger.info_rank0("Upcasting trainable params to float32.")
        cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")
    
    # Apply RMSNorm-only training if specified
    if finetuning_args.rmsnorm_only_training and is_trainable:
        _setup_rmsnorm_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)

    return model
