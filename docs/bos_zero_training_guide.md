# BOS Zero Training Guide

## Overview

BOS Zero Training is a novel fine-tuning approach that adds BOS (Beginning of Sequence) tokens to input sequences and then reduces their embedding influence during training. This forces the model to adapt to minimal first-token information, potentially improving robustness and attention distribution patterns.

## Key Features

- **Configurable BOS Scaling**: Adjustable scaling factor for BOS token influence
- **Gradient Stability**: Maintains stable gradients through scaling rather than complete zeroing
- **LoRA Integration**: Works seamlessly with Low-Rank Adaptation fine-tuning
- **Evaluation Consistency**: Evaluation applies the same BOS processing as training

## Configuration Parameters

### `use_bos_zero_training`
- **Type**: `bool`
- **Default**: `false`
- **Description**: Enables BOS zero training mode

### `bos_scaling_factor`
- **Type**: `float`
- **Default**: `0.01`
- **Range**: `0.0` to `1.0`
- **Description**: Scaling factor applied to position 0 embeddings
- **Examples**:
  - `0.01` = 99% reduction (very aggressive)
  - `0.05` = 95% reduction (aggressive)
  - `0.1` = 90% reduction (moderate)
  - `0.5` = 50% reduction (mild)

## Training Configuration

### Basic Configuration
```yaml
# Enable BOS zero training
use_bos_zero_training: true
bos_scaling_factor: 0.05  # 95% reduction

# LoRA settings (recommended)
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 32
```

### Advanced Configuration
```yaml
# More aggressive BOS reduction
use_bos_zero_training: true
bos_scaling_factor: 0.01  # 99% reduction

# Higher rank LoRA for complex adaptations
lora_rank: 64
lora_alpha: 32
lora_dropout: 0.1

# Training settings
learning_rate: 5.0e-4
num_train_epochs: 3.0
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
```

## Usage Examples

### 1. Basic Training
```bash
# Train with default 99% BOS reduction
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3_bos_zero_lora.yaml
```

### 2. Custom Scaling Factor
Create a custom configuration file:
```yaml
# custom_bos_config.yaml
model_name_or_path: microsoft/Phi-3-mini-4k-instruct
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

# Custom BOS scaling
use_bos_zero_training: true
bos_scaling_factor: 0.1  # 90% reduction

# Other settings...
dataset: alpaca_en
template: phi
output_dir: saves/phi3-bos-custom
```

### 3. Evaluation
```bash
# Evaluate with same scaling factor as training
python evaluate_bos_zero_lora.py \
    --model_path microsoft/Phi-3-mini-4k-instruct \
    --adapter_path saves/phi3-bos-zero-lora-alpaca \
    --scaling_factor 0.05 \
    --generate_samples

# Baseline comparison (no BOS scaling)
python evaluate_bos_zero_lora.py \
    --model_path microsoft/Phi-3-mini-4k-instruct \
    --adapter_path saves/phi3-bos-zero-lora-alpaca \
    --no_bos_scaling
```

## Technical Implementation

### Training Process
1. **Data Preprocessing**: BOS tokens are added to all input sequences
2. **Embedding Hook**: Forward hook intercepts embedding layer output
3. **BOS Scaling**: Position 0 embeddings are scaled by the configured factor
4. **Gradient Flow**: Scaling maintains gradient stability unlike complete zeroing

### Hook Implementation
```python
def bos_scaling_hook(module, input, output):
    if output.dim() == 3 and output.size(1) > 0:
        scaled_output = output.clone()
        scaled_output[:, 0, :] = output[:, 0, :] * self.bos_scaling_factor
        return scaled_output
    return output
```

### Multi-GPU Compatibility
The implementation handles DDP/FSDP wrapped models:
```python
# Handle wrapped models
if hasattr(model, 'module'):
    actual_model = model.module
else:
    actual_model = model
```

## Scaling Factor Guidelines

### Choosing the Right Factor

| Factor | Reduction | Use Case | Gradient Stability |
|--------|-----------|----------|-------------------|
| 0.01   | 99%       | Research, extreme adaptation | Stable |
| 0.05   | 95%       | Aggressive training | Stable |
| 0.1    | 90%       | Moderate adaptation | Very Stable |
| 0.2    | 80%       | Conservative approach | Very Stable |
| 0.5    | 50%       | Mild BOS reduction | Very Stable |

### Recommendations
- **Start with 0.05** for most experiments
- **Use 0.01** for maximum BOS influence reduction
- **Use 0.1+** if training instability occurs
- **Never use 0.0** (complete zeroing causes NaN gradients)

## Troubleshooting

### Common Issues

#### NaN Gradients
- **Cause**: Scaling factor too low (approaching 0.0)
- **Solution**: Increase scaling factor to 0.01 or higher

#### Poor Convergence
- **Cause**: Too aggressive BOS reduction for the model/dataset
- **Solution**: Increase scaling factor to 0.05 or 0.1

#### Memory Issues
- **Cause**: Hook overhead with large models
- **Solution**: Use gradient checkpointing or reduce batch size

### Debugging Commands
```bash
# Check training logs for hook registration
grep "BOS.*hook" logs/training.log

# Monitor gradient norms
python -c "
import torch
# Add gradient monitoring to training script
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: {param.grad.norm().item():.6f}')
"
```

## Research Applications

### Attention Analysis
BOS zero training enables studying:
- Attention redistribution patterns
- Model robustness to first-token information
- Transformer adaptation mechanisms

### Comparative Studies
- BOS zero vs. standard fine-tuning
- Different scaling factors effects
- Task-specific BOS importance

### Evaluation Metrics
- Perplexity comparison
- Generation quality assessment
- Attention visualization
- Downstream task performance

## File Structure

```
├── src/llamafactory/
│   ├── hparams/finetuning_args.py     # Configuration parameters
│   ├── train/sft/bos_zero_trainer.py  # Custom trainer implementation
│   ├── train/sft/workflow.py          # Training pipeline integration
│   └── data/processor/supervised.py   # Data preprocessing
├── examples/train_lora/
│   └── phi3_bos_zero_lora.yaml        # Training configuration
├── evaluate_bos_zero_lora.py          # Evaluation script
└── docs/bos_zero_training_guide.md    # This guide
```

## Advanced Usage

### Custom Hook Implementation
For research purposes, you can modify the hook behavior:
```python
class CustomBOSHook:
    def __init__(self, scaling_factor, apply_noise=False):
        self.scaling_factor = scaling_factor
        self.apply_noise = apply_noise
    
    def custom_hook(self, module, input, output):
        scaled_output = output.clone()
        scaled_output[:, 0, :] = output[:, 0, :] * self.scaling_factor
        
        if self.apply_noise:
            noise = torch.randn_like(scaled_output[:, 0, :]) * 0.01
            scaled_output[:, 0, :] += noise
            
        return scaled_output
```

### Batch Processing
For large-scale evaluation:
```python
# Batch evaluation script
for scaling_factor in [0.01, 0.05, 0.1, 0.2]:
    results = evaluate_model(
        model_path="microsoft/Phi-3-mini-4k-instruct",
        adapter_path="saves/phi3-bos-zero-lora",
        scaling_factor=scaling_factor
    )
    save_results(f"results_scale_{scaling_factor}.json", results)
```

## Contributing

To extend BOS zero training:
1. Modify [`bos_zero_trainer.py`](../src/llamafactory/train/sft/bos_zero_trainer.py) for training changes
2. Update [`finetuning_args.py`](../src/llamafactory/hparams/finetuning_args.py) for new parameters
3. Extend [`evaluate_bos_zero_lora.py`](../evaluate_bos_zero_lora.py) for evaluation features
4. Add tests and documentation

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)