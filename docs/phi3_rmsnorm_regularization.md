# Phi-3 Mini 4K Instruct RMSNorm Regularization

This document describes the new functionality for training Phi-3 Mini 4K Instruct models with RMSNorm regularization and RMSNorm-only parameter updates.

## Overview

The implementation adds two key features:

1. **RMSNorm-only Training**: Fine-tune only the RMSNorm parameters while freezing all other model parameters
2. **RMSNorm Output Regularization**: Apply regularization to post-attention RMSNorm outputs in specific layers to control row (token) norms

## New Configuration Parameters

### RMSNorm Regularization Arguments

- `use_rmsnorm_regularization` (bool, default: False): Enable RMSNorm output regularization
- `rmsnorm_reg_layers` (str, default: "2,4"): Comma-separated list of layer indices to apply regularization
- `rmsnorm_reg_weight` (float, default: 0.01): Weight for the regularization loss term
- `rmsnorm_reg_target_norm` (float, default: 1.0): Target norm value for regularization
- `rmsnorm_only_training` (bool, default: False): Train only RMSNorm parameters while freezing all others

## Usage Examples

### 1. Command Line Usage

```bash
llamafactory-cli train \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --stage sft \
    --do_train \
    --finetuning_type freeze \
    --freeze_trainable_modules all \
    --rmsnorm_only_training \
    --use_rmsnorm_regularization \
    --rmsnorm_reg_layers "2,4" \
    --rmsnorm_reg_weight 0.01 \
    --rmsnorm_reg_target_norm 0.0 \
    --dataset identity \
    --template phi \
    --cutoff_len 1024 \
    --output_dir ./saves/phi3-rmsnorm-reg \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16
```

### 2. YAML Configuration

Use the provided configuration file:

```bash
llamafactory-cli train examples/train_lora/phi3_rmsnorm_regularization.yaml
```

### 3. Python Script

```bash
python examples/train_lora/train_phi3_rmsnorm.py
```

### 4. Multi-GPU Training

For distributed training across multiple GPUs, use the Gloo backend for optimal stability:

```bash
llamafactory-cli train \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --stage sft \
    --do_train \
    --finetuning_type freeze \
    --rmsnorm_only_training \
    --use_rmsnorm_regularization \
    --rmsnorm_reg_layers "2,4" \
    --rmsnorm_reg_weight 0.01 \
    --rmsnorm_reg_target_norm 0.0 \
    --dataset identity \
    --template phi \
    --output_dir ./saves/phi3-rmsnorm-reg-multi \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-5 \
    --max_steps 100 \
    --bf16 \
    --ddp_backend gloo
```

**Important Multi-GPU Notes:**
- **Use Gloo Backend**: Add `--ddp_backend gloo` for stable distributed training
- **NCCL Issues**: The default NCCL backend may cause hanging issues with Phi-3 models
- **GPU Selection**: Use `CUDA_VISIBLE_DEVICES=0,1,2,3` to specify which GPUs to use
- **Batch Size Scaling**: Total effective batch size = `per_device_train_batch_size × num_gpus`
- **Synchronization**: RMSNorm hooks are automatically synchronized across all GPUs

## Technical Details

### RMSNorm-only Training

When `rmsnorm_only_training` is enabled, the system:

1. Freezes all model parameters by setting `requires_grad=False`
2. Identifies RMSNorm layers by searching for parameters with names containing:
   - "norm"
   - "layer_norm" 
   - "post_attention_layernorm"
   - "input_layernorm"
3. Unfreezes only these RMSNorm parameters by setting `requires_grad=True`

### RMSNorm Output Regularization

The regularization mechanism:

1. **Hook Registration**: Forward hooks are registered on post-attention RMSNorm layers in the specified layers (2 and 4 by default)
2. **Output Capture**: During forward pass, RMSNorm outputs are captured and stored
3. **Norm Computation**: Row (token) norms are computed using L2 norm across the feature dimension
4. **Regularization Loss**: MSE loss is computed between actual norms and target norms
5. **Loss Integration**: Regularization loss is added to the main training loss with the specified weight

### Model Architecture Support

The implementation supports various model architectures by trying multiple possible paths for RMSNorm layers:
- `model.layers.{layer_idx}.post_attention_layernorm` (Phi-3, LLaMA-style)
- `layers.{layer_idx}.post_attention_layernorm`
- `transformer.h.{layer_idx}.post_attention_layernorm` (GPT-style)
- `model.layers.{layer_idx}.ln_2`
- `layers.{layer_idx}.ln_2`

## Implementation Files

### Core Implementation
- `src/llamafactory/hparams/finetuning_args.py`: New configuration parameters
- `src/llamafactory/model/adapter.py`: RMSNorm-only training setup
- `src/llamafactory/train/sft/rmsnorm_trainer.py`: Custom trainer with regularization
- `src/llamafactory/train/sft/workflow.py`: Integration with SFT workflow

### Examples and Documentation
- `examples/train_lora/phi3_rmsnorm_regularization.yaml`: YAML configuration
- `examples/train_lora/train_phi3_rmsnorm.py`: Python training script
- `docs/phi3_rmsnorm_regularization.md`: This documentation

## Expected Behavior

### Training Logs
When using RMSNorm regularization, you should see additional metrics in the training logs:
- `train/rmsnorm_reg_loss`: The regularization loss component
- `train/base_loss`: The original training loss
- `train/total_loss`: Combined loss (base + regularization)

### Parameter Updates
With `rmsnorm_only_training=True`, only RMSNorm parameters will be updated during training. You can verify this by checking the parameter gradients or by monitoring which parameters change between epochs.

### Regularization Effect
The regularization should encourage the row norms of post-attention RMSNorm outputs in layers 2 and 4 to approach the target norm value (0.0 in the example configuration).
## GUI-Based Training

The RMSNorm regularization functionality is fully integrated into LLaMA-Factory's web-based GUI interface.

### Launching the Web UI

```bash
llamafactory-cli webui
```

### Using RMSNorm Regularization in the GUI

1. **Navigate to the Train tab** in the web interface
2. **Expand the "RMSNorm Regularization" accordion** (located after the SwanLab section)
3. **Configure the following parameters**:
   - **RMSNorm Only Training**: Enable to train only RMSNorm parameters
   - **Use RMSNorm Regularization**: Enable regularization of RMSNorm outputs
   - **RMSNorm Reg Layers**: Specify layers to regularize (e.g., "2,4" or "1,3,5")
   - **RMSNorm Reg Weight**: Set regularization weight (slider: 0.0 to 1.0, default: 0.01)
   - **RMSNorm Reg Target Norm**: Set target norm value (slider: 0.0 to 10.0, default: 0.0)

### GUI Configuration Example

For Phi-3 Mini 4K Instruct with RMSNorm regularization:

1. **Model Settings**:
   - Model: `microsoft/Phi-3-mini-4k-instruct`
   - Finetuning Type: `freeze`
   - Template: `phi`

2. **Training Settings**:
   - Learning Rate: `5e-5`
   - Epochs: `3.0`
   - Batch Size: `1`
   - Gradient Accumulation: `8`

3. **Freeze Settings**:
   - Freeze Trainable Modules: `all`

4. **RMSNorm Regularization Settings**:
   - ✅ RMSNorm Only Training
   - ✅ Use RMSNorm Regularization
   - RMSNorm Reg Layers: `2,4`
   - RMSNorm Reg Weight: `0.01`
   - RMSNorm Reg Target Norm: `0.0`

5. **Distributed Training** (for multi-GPU):
   - Add `"ddp_backend": "gloo"` to Extra Args field

### GUI Benefits

- **Visual Parameter Tuning**: Easy adjustment of regularization parameters with sliders
- **Real-time Validation**: Immediate feedback on configuration errors
- **Integrated Monitoring**: Built-in loss visualization and training progress tracking
- **Configuration Saving**: Save and load training configurations for reproducibility


## Troubleshooting

### Common Issues

1. **Hook Registration Failures**: If hooks fail to register, check the model architecture and adjust the layer paths in the trainer
2. **Memory Issues**: RMSNorm regularization adds computational overhead; reduce batch size if needed
3. **Convergence Issues**: Adjust the regularization weight if the regularization term dominates the loss

### Debugging Tips

1. Enable verbose logging to see which RMSNorm parameters are unfrozen
2. Monitor the regularization loss to ensure it's being computed correctly
3. Check that only RMSNorm parameters have `requires_grad=True`

## Performance Considerations

- RMSNorm-only training significantly reduces the number of trainable parameters
- Regularization adds minimal computational overhead during forward pass
- Memory usage is slightly increased due to storing intermediate RMSNorm outputs
- Training speed should be comparable to standard fine-tuning approaches