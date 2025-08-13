# Phi-3 BOS Token Zeroing with LoRA Fine-tuning

This document describes the BOS token zeroing experiment for Phi-3 Mini 4K Instruct model using LoRA fine-tuning.

## Overview

This experiment implements a novel training approach that:

1. **Adds BOS tokens** at the beginning of every training sequence
2. **Zeros out the embedding output** for position 0 (the BOS token position)
3. **Fine-tunes using LoRA** on all linear layers of the model

The goal is to study how the model adapts when the first token's information is systematically removed during training.

## Experimental Design

### Key Components

1. **BOS Token Addition**: Every training sample gets a BOS (Beginning of Sequence) token prepended
2. **Embedding Zeroing**: A forward hook zeros out the embedding layer output at position 0
3. **LoRA Fine-tuning**: Low-Rank Adaptation is applied to all linear layers
4. **Supervised Fine-tuning**: Standard instruction-following training on Alpaca dataset

### Technical Implementation

#### Custom Trainer: `BOSZeroTrainer`
- Inherits from `CustomSeq2SeqTrainer`
- Registers a forward hook on the embedding layer
- Automatically zeros out position 0 embeddings during forward passes
- Maintains compatibility with LoRA and other fine-tuning methods

#### Data Processing: `preprocess_bos_supervised_dataset`
- Ensures BOS tokens are added to all sequences
- Sets labels to -100 (ignore) for BOS token positions
- Handles tokenization and sequence length management

#### Configuration Integration
- New parameter: `use_bos_zero_training: true`
- Seamlessly integrates with existing LLaMA-Factory workflow
- Compatible with all LoRA configurations

## Usage

### Training Command

```bash
# Using the training script
python examples/train_lora/train_phi3_bos_zero.py

# Or using LLaMA-Factory CLI directly
llamafactory-cli train examples/train_lora/phi3_bos_zero_lora.yaml
```

### Configuration File

```yaml
# examples/train_lora/phi3_bos_zero_lora.yaml
model_name_or_path: microsoft/Phi-3-mini-4k-instruct
stage: sft
finetuning_type: lora
lora_target: all
use_bos_zero_training: true  # Enable BOS zeroing
```

## Expected Behavior

### During Training
1. **Data Preprocessing**: BOS tokens added to all sequences
2. **Forward Pass**: Position 0 embeddings automatically zeroed
3. **Loss Calculation**: BOS token position ignored (label = -100)
4. **LoRA Updates**: Only LoRA parameters updated, base model frozen

### Model Adaptation
The model should learn to:
- Rely less on the first token for context
- Develop alternative attention patterns
- Potentially improve robustness to sequence variations

## Files Created

### Core Implementation
- `src/llamafactory/train/sft/bos_zero_trainer.py` - Custom trainer
- `src/llamafactory/data/processor/bos_processor.py` - Data preprocessing
- `src/llamafactory/hparams/finetuning_args.py` - Configuration parameter

### Training Configuration
- `examples/train_lora/phi3_bos_zero_lora.yaml` - Training config
- `examples/train_lora/train_phi3_bos_zero.py` - Training script

### Integration
- `src/llamafactory/train/sft/workflow.py` - Workflow integration

## Monitoring and Debugging

### Training Logs
The trainer provides detailed logging:
```
🎯 BOSZeroTrainer initialized
   - BOS tokens will be added to sequences
   - Position 0 embeddings will be zeroed out
✅ Found embedding layer: model.embed_tokens
✅ BOS zeroing hook registered on embedding layer
```

### Saved Artifacts
- **Model checkpoints**: Standard LoRA adapter weights
- **Configuration**: `bos_zero_config.json` with experiment details
- **Training logs**: Standard LLaMA-Factory logging

### Verification
Check that BOS tokens are properly handled:
```python
# Sample tokens should start with BOS
📝 Sample tokens (first 10): ['<|endoftext|>', 'The', 'quick', 'brown', ...]
✅ BOS token (32000) found at position 0
✅ Position 0 label is -100 (ignored for BOS token)
```

## Research Questions

This experiment can help answer:

1. **Adaptation Mechanisms**: How does the model compensate for missing first-token information?
2. **Attention Patterns**: Do attention weights redistribute when position 0 is zeroed?
3. **Performance Impact**: How does BOS zeroing affect downstream task performance?
4. **Robustness**: Does the model become more robust to input variations?

## Evaluation

### Baseline Comparison
Compare against:
- Standard LoRA fine-tuning (without BOS zeroing)
- Full fine-tuning approaches
- Original Phi-3 model performance

### Metrics
- **Perplexity**: Language modeling capability
- **Task Performance**: Instruction-following quality
- **Generation Quality**: Text coherence and relevance
- **Attention Analysis**: Visualization of attention patterns

### Evaluation Commands
```bash
# Evaluate the fine-tuned model
python evaluate_rmsnorm_llamafactory.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --adapter_path saves/phi3-bos-zero-lora-alpaca \
    --output_dir ./bos_zero_eval_results

# Compare with baseline
python evaluate_rmsnorm_llamafactory.py \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --output_dir ./baseline_eval_results
```

## Matrix Capture

To analyze internal representations:

```bash
# Capture matrices from BOS-zero trained model
python capture_phi3_with_rmsnorm_weights.py \
    --model_path saves/phi3-bos-zero-lora-alpaca \
    --output_dir ./phi3_bos_zero_matrices
```

## Troubleshooting

### Common Issues

1. **BOS Token Not Found**
   - Check tokenizer configuration
   - Verify BOS token ID in logs

2. **Hook Not Registered**
   - Ensure embedding layer is found
   - Check model architecture compatibility

3. **Memory Issues**
   - Reduce batch size
   - Use gradient checkpointing

### Debug Mode
Enable detailed logging:
```yaml
logging_steps: 1
save_steps: 100
eval_steps: 50
```

## Future Extensions

### Possible Variations
1. **Different Zero Positions**: Zero other positions (1, 2, etc.)
2. **Partial Zeroing**: Zero only specific embedding dimensions
3. **Dynamic Zeroing**: Randomly zero different positions
4. **Multi-token Zeroing**: Zero multiple consecutive positions

### Advanced Analysis
1. **Attention Visualization**: Analyze attention pattern changes
2. **Representation Analysis**: Study embedding space modifications
3. **Gradient Analysis**: Examine gradient flow patterns
4. **Ablation Studies**: Test different zeroing strategies

## Conclusion

This BOS token zeroing experiment provides a novel way to study model adaptation and robustness. By systematically removing information from the first token position, we can gain insights into how transformer models process sequential information and adapt to constraints.

The implementation is fully integrated with LLaMA-Factory's training pipeline, making it easy to reproduce and extend for further research.