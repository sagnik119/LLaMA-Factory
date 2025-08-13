# BOS Token Zeroing Training Commands

Quick reference for running the BOS token zeroing experiment with LoRA fine-tuning.

## ⚠️ Recent Fixes Applied

✅ **Fixed training_step signature** - Resolved `TypeError: BOSZeroTrainer.training_step() takes 3 positional arguments but 4 were given`

✅ **Improved BOS token processing** - Enhanced data preprocessing to properly add BOS tokens

✅ **Enhanced multi-GPU support** - Improved hook handling for DDP/FSDP wrapped models

✅ **Fixed NaN gradient issue** - Changed from complete zeroing to scaling (0.1 factor) to maintain gradient flow

✅ **Added configurable scaling** - New `bos_scaling_factor` parameter for fine-tuning the BOS reduction

## Training Commands

### Method 1: Using the Training Script
```bash
python examples/train_lora/train_phi3_bos_zero.py
```

### Method 2: Using LLaMA-Factory CLI
```bash
llamafactory-cli train examples/train_lora/phi3_bos_zero_lora.yaml
```

### Method 3: Direct Command Line
```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --dataset alpaca_en \
    --template phi \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --use_bos_zero_training \
    --output_dir saves/phi3-bos-zero-lora-alpaca \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16
```

## Evaluation Commands

### Evaluate Trained Model
```bash
python evaluate_rmsnorm_llamafactory.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --adapter_path saves/phi3-bos-zero-lora-alpaca \
    --output_dir ./bos_zero_eval_results \
    --generate_samples
```

### Evaluate Baseline (No BOS Zeroing)
```bash
python evaluate_rmsnorm_llamafactory.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --output_dir ./baseline_eval_results \
    --generate_samples
```

## Matrix Capture Commands

### Capture Matrices from BOS-Zero Model
```bash
# First, merge the LoRA adapter (if needed)
llamafactory-cli export \
    --model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --adapter_name_or_path saves/phi3-bos-zero-lora-alpaca \
    --template phi \
    --finetuning_type lora \
    --export_dir saves/phi3-bos-zero-merged \
    --export_size 2 \
    --export_device cpu

# Then capture matrices
python capture_phi3_with_rmsnorm_weights.py \
    --model-id saves/phi3-bos-zero-merged \
    --output-dir ./phi3_bos_zero_matrices \
    --seq-len 1024
```

### Capture Matrices from Baseline Model
```bash
python capture_phi3_with_rmsnorm_weights.py \
    --model-id microsoft/Phi-3-mini-4k-instruct \
    --output-dir ./phi3_baseline_matrices \
    --seq-len 1024
```

## Monitoring Commands

### Check Training Progress
```bash
# View training logs
tail -f saves/phi3-bos-zero-lora-alpaca/trainer_log.jsonl

# Plot training loss
python -c "
from llamafactory.extras.ploting import plot_loss
plot_loss('saves/phi3-bos-zero-lora-alpaca', keys=['loss', 'eval_loss'])
"
```

### Check Model Size
```bash
# Check LoRA adapter size
du -sh saves/phi3-bos-zero-lora-alpaca/

# List checkpoint files
ls -la saves/phi3-bos-zero-lora-alpaca/
```

## Debugging Commands

### Test BOS Token Processing
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
text = 'Hello world'
tokens = tokenizer.encode(text)
print(f'Original tokens: {tokens}')
print(f'BOS token ID: {tokenizer.bos_token_id}')
print(f'Decoded: {tokenizer.decode(tokens)}')
"
```

### Verify Training Configuration
```bash
# Check if BOS zero training is enabled
python -c "
import yaml
with open('examples/train_lora/phi3_bos_zero_lora.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'BOS zero training: {config.get(\"use_bos_zero_training\", False)}')
print(f'LoRA target: {config.get(\"lora_target\", \"not set\")}')
print(f'LoRA rank: {config.get(\"lora_rank\", \"not set\")}')
"
```

## Expected Output Patterns

### Training Start
```
🚀 Starting Phi-3 BOS Zero LoRA Training
📋 Config: examples/train_lora/phi3_bos_zero_lora.yaml
🎯 Features:
   - BOS tokens added to all sequences
   - Position 0 embeddings zeroed out
   - LoRA fine-tuning on all linear layers
```

### Trainer Initialization
```
🎯 BOSZeroTrainer initialized
   - BOS tokens will be added to sequences
   - Position 0 embeddings will be zeroed out
✅ Found embedding layer: model.embed_tokens
✅ BOS zeroing hook registered on embedding layer
```

### Training Progress
```
{'loss': 1.234, 'learning_rate': 0.0005, 'epoch': 1.0}
{'eval_loss': 1.123, 'eval_accuracy': 0.85, 'epoch': 1.0}
```

### Training Completion
```
💾 BOS zeroing configuration saved to saves/phi3-bos-zero-lora-alpaca/bos_zero_config.json
Training completed successfully!
```

## File Structure After Training

```
saves/phi3-bos-zero-lora-alpaca/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # LoRA weights
├── bos_zero_config.json         # BOS zeroing configuration
├── trainer_log.jsonl            # Training logs
├── training_args.bin            # Training arguments
├── training_loss.png            # Loss plot
└── checkpoint-*/                # Training checkpoints
    ├── adapter_model.safetensors
    ├── bos_zero_config.json
    └── trainer_state.json
```

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --per_device_train_batch_size 1
   --gradient_accumulation_steps 8
   ```

2. **BOS Token Not Found**
   ```bash
   # Check tokenizer
   python -c "
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
   print(f'BOS token: {tokenizer.bos_token}')
   print(f'BOS token ID: {tokenizer.bos_token_id}')
   "
   ```

3. **Hook Not Registered**
   ```bash
   # Check model architecture
   python -c "
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
   for name, module in model.named_modules():
       if 'embed' in name:
           print(f'Found: {name} - {type(module)}')
   "
   ```

## Performance Expectations

- **Training Time**: ~2-4 hours on single GPU (depending on dataset size)
- **Memory Usage**: ~8-12GB GPU memory with batch_size=2
- **Model Size**: LoRA adapter ~50-100MB (much smaller than full model)
- **Convergence**: Should see loss decrease within first few hundred steps

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. TypeError: training_step() takes 3 positional arguments but 4 were given
**Status**: ✅ **FIXED**
- **Cause**: Incorrect method signature in BOSZeroTrainer
- **Solution**: Updated training_step method to handle both old and new transformers versions

#### 2. BOS tokens not being added to sequences
**Symptoms**: Training example shows sequences not starting with BOS token (ID: 1)
**Status**: ✅ **FIXED**
- **Cause**: BOS processing not integrated into data pipeline
- **Solution**: Enhanced SupervisedDatasetProcessor with BOS token handling

#### 3. Multi-GPU training failures
**Status**: ✅ **FIXED**
- **Cause**: Hook registration issues with DDP/FSDP wrapped models
- **Solution**: Improved embedding layer detection for wrapped models

### Monitoring Success

Look for these log messages to confirm proper operation:

```
[INFO] ✅ Found embedding layer: base_model.model.model.embed_tokens
[INFO] ✅ BOS zeroing hook registered on embedding layer
[INFO] 🎯 BOSZeroTrainer initialized
[INFO] ✅ Added BOS token (1) to sequence of length X -> Y
```

### Testing the Implementation

Run the debug script to test with minimal resources:
```bash
python debug_bos_zero.py
```

### Performance Tips

1. **Start with single GPU** for initial testing:
   ```bash
   CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3_bos_zero_lora.yaml
   ```

2. **Use smaller datasets** for debugging:
   ```yaml
   max_samples: 100  # Add to YAML config
   ```

3. **Monitor GPU memory** usage during training

### Expected Training Behavior

- **BOS Token Addition**: Sequences will be prepended with BOS token (ID: 1)
- **Position 0 Zeroing**: Embeddings at position 0 will be zeroed during forward pass
- **Label Masking**: BOS token positions will have labels set to -100 (ignored in loss)
- **LoRA Training**: Only LoRA parameters will be updated, not the full model


#### 4. NaN gradients causing training failure
**Status**: ✅ **FIXED**
- **Cause**: Complete zeroing of embeddings broke gradient flow
- **Solution**: Changed to scaling approach (default 0.1 factor) instead of complete zeroing
- **Configuration**: Adjustable via `bos_scaling_factor` parameter (0.0 = complete zeroing, 1.0 = no scaling)

### Configuration Options

#### BOS Scaling Factor
```yaml
bos_scaling_factor: 0.1  # Default: 90% reduction while maintaining gradients
```

**Recommended values:**
- `0.1` (default): 90% reduction, stable gradients
- `0.05`: 95% reduction, more aggressive
- `0.2`: 80% reduction, more conservative
- `0.0`: Complete zeroing (may cause NaN gradients)
