# RedPajama v2 Dataset with Variance Regularization

This document provides complete instructions for training Phi-3 Mini 4K Instruct with variance regularization using the massive RedPajama v2 dataset (1T+ tokens, 100K+ samples).

## Dataset Information

- **Dataset**: `togethercomputer/RedPajama-Data-V2`
- **Size**: Over 1 trillion tokens from diverse web sources
- **Format**: Raw text content (pre-training format)
- **Quality**: High-quality web data, deduplicated and filtered
- **Samples**: Virtually unlimited (use `max_samples` to control)

## Configuration Files

### 1. YAML Configuration
- **File**: `examples/train_lora/phi3_variance_reg_redpajama.yaml`
- **Usage**: Standard YAML-based training configuration

### 2. Python Script
- **File**: `examples/train_lora/train_phi3_variance_reg_redpajama.py`
- **Usage**: Direct Python execution with embedded configuration

## Training Commands

### Single GPU Training

```bash
# Using YAML configuration
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml

# Using Python script
python examples/train_lora/train_phi3_variance_reg_redpajama.py

# With specific GPU
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml
```

### Multi-GPU Training (Highly Recommended for RedPajama)

```bash
# 2 GPUs with automatic distributed training
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml

# 4 GPUs for faster processing
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml

# 8 GPUs for maximum throughput
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml

# Force torchrun for explicit distributed training
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml
```

### Advanced Multi-GPU with Custom Settings

```bash
# Large-scale training with 500K samples
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    max_samples=500000 \
    per_device_train_batch_size=4 \
    gradient_accumulation_steps=2

# High-throughput training with optimized batch size
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    per_device_train_batch_size=8 \
    gradient_accumulation_steps=1 \
    learning_rate=3e-5

# Custom variance regularization settings
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    variance_reg_weight=0.005 \
    variance_reg_target=1.2 \
    variance_reg_layers="model.layers.0.post_attention_layernorm,model.layers.1.post_attention_layernorm,model.layers.2.post_attention_layernorm,model.layers.3.post_attention_layernorm"
```

## Key Parameters

### Dataset Parameters
- `dataset: redpajama_v2` - Uses the RedPajama v2 dataset
- `max_samples: 100000` - Limits to 100K samples (can increase significantly)
- `cutoff_len: 1024` - Maximum sequence length
- `template: phi` - Uses Phi-3 chat template
- `preprocessing_num_workers: 16` - Parallel data processing

### Variance Regularization Parameters
- `use_variance_regularization: true` - Enables variance regularization
- `variance_reg_layers: "model.layers.0.post_attention_layernorm,model.layers.1.post_attention_layernorm,model.layers.2.post_attention_layernorm"` - Target layers
- `variance_reg_weight: 0.01` - Regularization strength
- `variance_reg_target: 1.0` - Target variance value
- `variance_reg_norm_type: "l2"` - Norm type for variance computation

### Training Parameters
- `per_device_train_batch_size: 2` - Batch size per GPU
- `gradient_accumulation_steps: 4` - Effective batch size = 2 × 4 × num_gpus
- `learning_rate: 5.0e-05` - Learning rate
- `num_train_epochs: 3.0` - Number of epochs
- `bf16: true` - Mixed precision training
- `ddp_timeout: 180000000` - Extended timeout for large dataset loading

## Expected Training Time

With RedPajama v2 dataset (100K samples):

### Single GPU (RTX 4090/A100)
- **Time**: ~14-18 hours
- **Memory**: ~22-24GB VRAM
- **Effective batch size**: 8 (2 × 4)
- **Throughput**: ~5-6K samples/hour

### Dual GPU (2x RTX 4090/A100)
- **Time**: ~7-9 hours
- **Memory**: ~22-24GB VRAM per GPU
- **Effective batch size**: 16 (2 × 4 × 2)
- **Throughput**: ~11-12K samples/hour

### Quad GPU (4x RTX 4090/A100)
- **Time**: ~3.5-4.5 hours
- **Memory**: ~22-24GB VRAM per GPU
- **Effective batch size**: 32 (2 × 4 × 4)
- **Throughput**: ~22-25K samples/hour

### Octa GPU (8x RTX 4090/A100)
- **Time**: ~1.8-2.3 hours
- **Memory**: ~22-24GB VRAM per GPU
- **Effective batch size**: 64 (2 × 4 × 8)
- **Throughput**: ~44-50K samples/hour

## Monitoring Training

### Loss Components
The training will show three loss components:
1. **Base Loss**: Standard language modeling loss
2. **Variance Regularization Loss**: Penalty for variance deviation
3. **Total Loss**: Combined loss (base + variance regularization)

### Example Log Output
```
Step 10: loss=3.124, base_loss=3.112, variance_reg_loss=0.012, total_loss=3.124
Step 20: loss=2.987, base_loss=2.976, variance_reg_loss=0.011, total_loss=2.987
Step 50: loss=2.654, base_loss=2.645, variance_reg_loss=0.009, total_loss=2.654
```

### Performance Metrics
- **Tokens/second**: Monitor throughput
- **GPU utilization**: Should be >90% during training
- **Memory usage**: Monitor for OOM issues
- **Loss convergence**: Base loss should decrease steadily

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=8

# Enable gradient checkpointing
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    gradient_checkpointing=true

# Reduce sequence length
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    cutoff_len=512
```

### Slow Dataset Loading
```bash
# Increase preprocessing workers
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    preprocessing_num_workers=32

# Use cached dataset after first run
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    overwrite_cache=false

# Reduce max_samples for testing
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    max_samples=10000
```

### Multi-GPU Issues
```bash
# Use Gloo backend for better compatibility
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    ddp_backend=gloo

# Increase timeout for large datasets
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    ddp_timeout=300000000

# Force synchronization
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    dataloader_pin_memory=false
```

### Network/Download Issues
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/large/storage
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml

# Use offline mode if dataset is cached
export HF_DATASETS_OFFLINE=1
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml
```

## Output Structure

```
saves/phi3-variance-reg-redpajama/
├── adapter_config.json          # LoRA adapter configuration
├── adapter_model.safetensors    # LoRA weights
├── trainer_state.json           # Training state
├── training_args.json           # Training arguments
├── training_loss.png            # Loss curves
└── runs/                        # TensorBoard logs
    └── events.out.tfevents.*    # TensorBoard event files
```

## Scaling Recommendations

### For 100K Samples
- **Minimum**: 2 GPUs (RTX 4090/A100)
- **Recommended**: 4 GPUs
- **Optimal**: 8 GPUs for fastest training

### For 500K Samples
- **Minimum**: 4 GPUs
- **Recommended**: 8 GPUs
- **Optimal**: 16+ GPUs with multi-node setup

### For 1M+ Samples
- **Multi-node setup required**
- **16+ GPUs recommended**
- **Consider using DeepSpeed ZeRO-3**

## Advanced Configuration

### DeepSpeed Integration
```bash
# Use DeepSpeed ZeRO-3 for very large scale
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    deepspeed=examples/deepspeed/ds_z3_config.json
```

### Custom Variance Regularization
```bash
# Target more layers for stronger regularization
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    variance_reg_layers="model.layers.0.post_attention_layernorm,model.layers.1.post_attention_layernorm,model.layers.2.post_attention_layernorm,model.layers.3.post_attention_layernorm,model.layers.4.post_attention_layernorm"

# Adjust regularization strength
llamafactory-cli train examples/train_lora/phi3_variance_reg_redpajama.yaml \
    variance_reg_weight=0.02 \
    variance_reg_target=0.8
```

## Next Steps

After training completes:

1. **Merge LoRA weights**:
   ```bash
   llamafactory-cli export examples/merge_lora/phi3_lora_sft.yaml \
       adapter_name_or_path=./saves/phi3-variance-reg-redpajama
   ```

2. **Test inference**:
   ```bash
   llamafactory-cli chat examples/inference/phi3_lora_sft.yaml \
       adapter_name_or_path=./saves/phi3-variance-reg-redpajama
   ```

3. **Evaluate model**:
   ```bash
   llamafactory-cli eval examples/train_lora/phi3_lora_eval.yaml \
       adapter_name_or_path=./saves/phi3-variance-reg-redpajama
   ```

## Performance Tips

1. **Use multiple GPUs** - RedPajama is massive, multi-GPU is essential
2. **Enable mixed precision** (`bf16: true`) to maximize throughput
3. **Optimize batch size** based on GPU memory and throughput
4. **Monitor GPU utilization** - should be >90% during training
5. **Use fast storage** - NVMe SSD recommended for dataset caching
6. **Increase preprocessing workers** for faster data loading
7. **Consider gradient checkpointing** if memory is limited

The RedPajama v2 dataset provides diverse, high-quality web content that will significantly improve your model's general knowledge and language understanding capabilities.