#!/usr/bin/env python3
"""
Comprehensive matrix capture script for Phi-3 Mini 4k Instruct model with fine-tuned RMSNorm weights.
This script loads the base Phi-3 model, applies fine-tuned RMSNorm weights, and then captures matrices 
at input and output of every stage of every layer including:
- Input RMSNorm
- Q, K, V projections (unfused from qkv_proj)
- QK.T (attention scores)
- Causal masking
- Softmax layer
- Attention operation
- Post attention RMSNorm
- Gate projection, Up projection (unfused from gate_up_proj)
- SiLU activation
- SwiGLU elementwise multiply
- Down projection

Usage:
    python capture_phi3_with_rmsnorm_weights.py --rmsnorm-weights-path saves/phi3-variance-reg-alpaca-layers-2-4/checkpoint-500/rmsnorm_weights.pt --output-dir ./phi3_rmsnorm_matrices
    python capture_phi3_with_rmsnorm_weights.py --rmsnorm-weights-path saves/phi3-variance-reg-alpaca-layers-2-4/checkpoint-500/rmsnorm_weights.pt --output-dir ./phi3_rmsnorm_matrices --seq-len 512
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

class Phi3MatrixCaptureWithRMSNorm:
    def __init__(self, model, tokenizer, rmsnorm_weights_path: str, output_dir: str, seq_len: int = 1024, data_seed: int = 42):
        self.model = model
        self.tokenizer = tokenizer
        self.rmsnorm_weights_path = rmsnorm_weights_path
        self.output_dir = Path(output_dir)
        self.seq_len = seq_len
        self.data_seed = data_seed
        self.captured_matrices = {}
        self.hooks = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model configuration
        self.config = model.config
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, 'num_key_value_heads', self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = self.config.intermediate_size
        self.num_layers = self.config.num_hidden_layers
        
        print(f"🔧 Phi-3 Model Configuration:")
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Attention heads: {self.num_attention_heads}")
        print(f"   Key-value heads: {self.num_key_value_heads}")
        print(f"   Head dimension: {self.head_dim}")
        print(f"   Intermediate size: {self.intermediate_size}")
        print(f"   Number of layers: {self.num_layers}")
        print(f"   Sequence length: {self.seq_len}")
        
        # Apply RMSNorm weights first
        self._apply_rmsnorm_weights()
        
    def _apply_rmsnorm_weights(self):
        """Apply fine-tuned RMSNorm weights to the model."""
        if not os.path.exists(self.rmsnorm_weights_path):
            raise FileNotFoundError(f"RMSNorm weights file not found: {self.rmsnorm_weights_path}")
        
        print(f"🔄 Loading and applying RMSNorm weights from: {self.rmsnorm_weights_path}")
        
        try:
            # Load the saved RMSNorm weights
            device = next(self.model.parameters()).device
            rmsnorm_weights = torch.load(self.rmsnorm_weights_path, map_location=device)
            print(f"   Loaded {len(rmsnorm_weights)} RMSNorm parameters")
            
            # Apply weights to model
            applied_count = 0
            for name, param in self.model.named_parameters():
                if name in rmsnorm_weights:
                    print(f"   ✅ Applying RMSNorm weight: {name}")
                    param.data = rmsnorm_weights[name].to(param.device, param.dtype)
                    applied_count += 1
            
            print(f"✅ Successfully applied {applied_count} RMSNorm weights to the model")
            
            if applied_count == 0:
                print("⚠️ No RMSNorm weights were applied. Check parameter name matching.")
                self._debug_parameter_names(rmsnorm_weights)
                
        except Exception as e:
            print(f"❌ Error loading RMSNorm weights: {e}")
            raise
    
    def _debug_parameter_names(self, rmsnorm_weights: Dict[str, torch.Tensor]):
        """Debug parameter name matching."""
        print("=== Parameter Name Debugging ===")
        print("RMSNorm weights keys:")
        for key in sorted(rmsnorm_weights.keys()):
            print(f"  {key}")
        
        print("\nModel RMSNorm parameters:")
        for name, param in self.model.named_parameters():
            if "norm" in name.lower():
                print(f"  {name} - shape: {param.shape}")
        
    def save_matrix(self, matrix: torch.Tensor, name: str, layer_idx: Optional[int] = None):
        """Save a matrix to disk with proper naming convention."""
        if layer_idx is not None:
            filename = f"layer_{layer_idx:02d}_{name}.pt"
        else:
            filename = f"{name}.pt"
        
        filepath = self.output_dir / filename
        torch.save(matrix.detach().cpu(), filepath)
        
        # Also save shape and statistics
        stats = {
            'shape': list(matrix.shape),
            'dtype': str(matrix.dtype),
        }
        
        # Only compute statistics for floating-point tensors
        temp1 = 0.0  # top 20
        temp2 = 0.0  # top 100
        temp3 = 0.0  # top 2
        temp4 = 0.0  # top 5
        temp5 = 0.0  # top 500
        temp6 = 0.0  # top 1000
        
        if matrix.dtype.is_floating_point and matrix.numel() > 0:
            try:
                # Flatten matrix and get absolute values
                flat_abs = matrix.abs().reshape(-1)
                
                # Get top 2 values if matrix has enough elements
                if flat_abs.numel() >= 2:
                    top_2_indices = torch.argsort(flat_abs)[-2:]
                    top_2_values = flat_abs[top_2_indices]
                    temp3 = float(top_2_values.max().item() / top_2_values.mean().item())
                
                # Get top 5 values if matrix has enough elements
                if flat_abs.numel() >= 5:
                    top_5_indices = torch.argsort(flat_abs)[-5:]
                    top_5_values = flat_abs[top_5_indices]
                    temp4 = float(top_5_values.max().item() / top_5_values.mean().item())
                
                # Get top 20 values if matrix has enough elements
                if flat_abs.numel() >= 20:
                    top_20_indices = torch.argsort(flat_abs)[-20:]
                    top_20_values = flat_abs[top_20_indices]
                    temp1 = float(top_20_values.max().item() / top_20_values.mean().item())
                
                # Get top 100 values if matrix has enough elements
                if flat_abs.numel() >= 100:
                    top_100_indices = torch.argsort(flat_abs)[-100:]
                    top_100_values = flat_abs[top_100_indices]
                    temp2 = float(top_100_values.max().item() / top_100_values.mean().item())
                
                # Get top 500 values if matrix has enough elements
                if flat_abs.numel() >= 500:
                    top_500_indices = torch.argsort(flat_abs)[-500:]
                    top_500_values = flat_abs[top_500_indices]
                    temp5 = float(top_500_values.max().item() / top_500_values.mean().item())
                    
                # Get top 1000 values if matrix has enough elements
                if flat_abs.numel() >= 1000:
                    top_1000_indices = torch.argsort(flat_abs)[-1000:]
                    top_1000_values = flat_abs[top_1000_indices]
                    temp6 = float(top_1000_values.max().item() / top_1000_values.mean().item())
                
            except Exception as e:
                print(f"Warning: Could not compute top-k statistics for {name}: {e}")
                temp1 = temp2 = temp3 = temp4 = temp5 = temp6 = 0.0

        if matrix.dtype.is_floating_point:
            # Calculate absolute median
            abs_matrix = matrix.abs()
            absolute_median = float(torch.median(abs_matrix).item())
            absolute_max = float(abs_matrix.max().item())
            
            # Calculate absolute max/median ratio (avoid division by zero)
            absolute_max_median_ratio = float(absolute_max / absolute_median) if absolute_median > 1e-10 else float('inf')
            
            stats.update({
                'mean': float(matrix.mean().item()),
                'std': float(matrix.std().item()),
                'min': float(matrix.min().item()),
                'max': float(matrix.max().item()),
                'norm': float(matrix.norm().item()),
                'absolute_mean': float(matrix.abs().mean().item()),
                'absolute_max': float(matrix.abs().max().item()),
                'absolute_median': absolute_median,
                'absolute_norm': float(matrix.abs().norm().item()),
                'absolute max/mean ratio': float(matrix.abs().max().item() / matrix.abs().mean().item()),
                'absolute max/median ratio': absolute_max_median_ratio,
                'absolute_norm/mean ratio': float(matrix.abs().norm().item() / matrix.abs().mean().item()),
                'absolute max/mean of top 2 magnitude entries': temp3,
                'absolute max/mean of top 5 magnitude entries': temp4,
                'absolute max/mean of top 20 magnitude entries': temp1,
                'absolute max/mean of top 100 magnitude entries': temp2,
                'absolute max/mean of top 500 magnitude entries': temp5,
                'absolute max/mean of top 1000 magnitude entries': temp6,
            })
        else:
            # For integer tensors (like token IDs), compute basic statistics
            stats.update({
                'min': int(matrix.min().item()),
                'max': int(matrix.max().item()),
                'unique_values': int(torch.unique(matrix).numel()) if matrix.numel() < 10000 else "too_many_to_count"
            })
        
        stats_filename = filename.replace('.pt', '_stats.json')
        with open(self.output_dir / stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 Saved {name} with shape {matrix.shape} to {filename}")
        
    def unfuse_qkv_weights(self, qkv_weight: torch.Tensor) -> tuple:
        """Unfuse QKV projection weights into separate Q, K, V components."""
        # Calculate dimensions
        q_size = self.hidden_size
        kv_size = self.num_key_value_heads * self.head_dim
        
        # Verify the split is correct
        expected_total = q_size + kv_size + kv_size
        actual_total = qkv_weight.shape[0]
        
        if expected_total != actual_total:
            print(f"⚠️ QKV split mismatch: expected {expected_total}, got {actual_total}")
            # Fall back to naive split
            qkv_size = qkv_weight.shape[0] // 3
            q_size = kv_size = qkv_size
            print(f"   Using naive split: {qkv_size} each")
        
        # Extract components
        q_weight = qkv_weight[:q_size, :]
        k_weight = qkv_weight[q_size:q_size+kv_size, :]
        v_weight = qkv_weight[q_size+kv_size:, :]
        
        return q_weight, k_weight, v_weight
    
    def unfuse_gate_up_weights(self, gate_up_weight: torch.Tensor) -> tuple:
        """Unfuse gate_up projection weights into separate gate and up components."""
        intermediate_size = gate_up_weight.shape[0] // 2
        gate_weight = gate_up_weight[:intermediate_size, :]
        up_weight = gate_up_weight[intermediate_size:, :]
        return gate_weight, up_weight
    
    def create_attention_hooks(self, layer_idx: int):
        """Create hooks for attention mechanism components."""
        layer = self.model.model.layers[layer_idx]
        
        # Hook for input RMSNorm
        def input_rmsnorm_hook(module, input, output):
            self.save_matrix(input[0], f"input_rmsnorm_input", layer_idx)
            self.save_matrix(output, f"input_rmsnorm_output", layer_idx)
            # Save the weight matrix
            self.save_matrix(module.weight.data, f"input_rmsnorm_weight", layer_idx)
        
        hook_handle = layer.input_layernorm.register_forward_hook(input_rmsnorm_hook)
        self.hooks.append(hook_handle)
        
        # Hook for QKV projection (fused)
        def qkv_proj_hook(module, input, output):
            # Save fused input and output
            self.save_matrix(input[0], f"qkv_proj_input", layer_idx)
            self.save_matrix(output, f"qkv_proj_output_fused", layer_idx)
            
            # Unfuse the weights and compute individual projections
            q_weight, k_weight, v_weight = self.unfuse_qkv_weights(module.weight.data)
            
            # Compute individual projections
            input_tensor = input[0]
            q_output = F.linear(input_tensor, q_weight, None)
            k_output = F.linear(input_tensor, k_weight, None)
            v_output = F.linear(input_tensor, v_weight, None)
            
            # Save unfused components
            self.save_matrix(q_weight, f"q_proj_weight", layer_idx)
            self.save_matrix(k_weight, f"k_proj_weight", layer_idx)
            self.save_matrix(v_weight, f"v_proj_weight", layer_idx)
            self.save_matrix(q_output, f"q_proj_output", layer_idx)
            self.save_matrix(k_output, f"k_proj_output", layer_idx)
            self.save_matrix(v_output, f"v_proj_output", layer_idx)
            
            # Reshape for attention computation
            batch_size, seq_len, _ = input_tensor.shape
            q = q_output.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
            k = k_output.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = v_output.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            # Save reshaped Q, K, V
            self.save_matrix(q, f"q_reshaped", layer_idx)
            self.save_matrix(k, f"k_reshaped", layer_idx)
            self.save_matrix(v, f"v_reshaped", layer_idx)
            
            # Handle grouped query attention by repeating K and V if needed
            if self.num_key_value_heads != self.num_attention_heads:
                # Repeat K and V to match Q's head count
                k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
                v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1)
                self.save_matrix(k, f"k_repeated_for_gqa", layer_idx)
                self.save_matrix(v, f"v_repeated_for_gqa", layer_idx)
            
            # Compute attention scores (QK.T)
            scale = 1.0 / (self.head_dim ** 0.5)
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            self.save_matrix(attention_scores, f"attention_scores_qk_transpose_pre_mask", layer_idx)
            
            # Apply causal mask
            seq_len_attn = attention_scores.size(-1)
            causal_mask = torch.tril(torch.ones(seq_len_attn, seq_len_attn, device=attention_scores.device, dtype=attention_scores.dtype))
            # Convert mask to additive form (0 for allowed, -inf for masked)
            mask_value = torch.finfo(attention_scores.dtype).min
            causal_mask = causal_mask.masked_fill(causal_mask == 0, mask_value)
            causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
            self.save_matrix(causal_mask, f"causal_mask", layer_idx)
            
            # Apply causal mask to attention scores
            masked_attention_scores = attention_scores + causal_mask
            self.save_matrix(masked_attention_scores, f"attention_scores_qk_transpose_masked", layer_idx)
            
            # Apply softmax to masked scores
            attention_probs = F.softmax(masked_attention_scores, dim=-1)
            self.save_matrix(attention_probs, f"attention_softmax_output", layer_idx)
            
            # Apply attention to values
            attention_output = torch.matmul(attention_probs, v)
            self.save_matrix(attention_output, f"attention_operation_output", layer_idx)
            
            # Reshape back
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            self.save_matrix(attention_output, f"attention_reshaped_output", layer_idx)
        
        hook_handle = layer.self_attn.qkv_proj.register_forward_hook(qkv_proj_hook)
        self.hooks.append(hook_handle)
        
        # Hook for output projection
        def o_proj_hook(module, input, output):
            self.save_matrix(input[0], f"o_proj_input", layer_idx)
            self.save_matrix(output, f"o_proj_output", layer_idx)
            # Save the weight matrix
            self.save_matrix(module.weight.data, f"o_proj_weight", layer_idx)
        
        hook_handle = layer.self_attn.o_proj.register_forward_hook(o_proj_hook)
        self.hooks.append(hook_handle)
        
        # Hook for post attention RMSNorm
        def post_attn_rmsnorm_hook(module, input, output):
            self.save_matrix(input[0], f"post_attn_rmsnorm_input", layer_idx)
            self.save_matrix(output, f"post_attn_rmsnorm_output", layer_idx)
            # Save the weight matrix
            self.save_matrix(module.weight.data, f"post_attn_rmsnorm_weight", layer_idx)
        
        hook_handle = layer.post_attention_layernorm.register_forward_hook(post_attn_rmsnorm_hook)
        self.hooks.append(hook_handle)
    
    def create_mlp_hooks(self, layer_idx: int):
        """Create hooks for MLP components."""
        layer = self.model.model.layers[layer_idx]
        
        # Hook for gate_up projection (fused)
        def gate_up_proj_hook(module, input, output):
            # Save fused input and output
            self.save_matrix(input[0], f"gate_up_proj_input", layer_idx)
            self.save_matrix(output, f"gate_up_proj_output_fused", layer_idx)
            
            # Unfuse the weights and compute individual projections
            gate_weight, up_weight = self.unfuse_gate_up_weights(module.weight.data)
            
            # Compute individual projections
            input_tensor = input[0]
            gate_output = F.linear(input_tensor, gate_weight, None)
            up_output = F.linear(input_tensor, up_weight, None)
            
            # Save unfused components
            self.save_matrix(gate_weight, f"gate_proj_weight", layer_idx)
            self.save_matrix(up_weight, f"up_proj_weight", layer_idx)
            self.save_matrix(gate_output, f"gate_proj_output_pre_activation", layer_idx)
            self.save_matrix(up_output, f"up_proj_output", layer_idx)
            
            # Compute and save elementwise multiplication of gate and up weights
            gate_up_elementwise = gate_weight * up_weight
            self.save_matrix(gate_up_elementwise, f"gate_proj_up_proj_elementwise_multiplied", layer_idx)
            
            # Apply SiLU activation to gate projection
            gate_activation_output = F.silu(gate_output)
            self.save_matrix(gate_activation_output, f"gate_proj_silu_output", layer_idx)
            
            # SwiGLU: elementwise multiplication
            swiglu_output = gate_activation_output * up_output
            self.save_matrix(swiglu_output, f"swiglu_elementwise_multiply_output_silu", layer_idx)
        
        hook_handle = layer.mlp.gate_up_proj.register_forward_hook(gate_up_proj_hook)
        self.hooks.append(hook_handle)
        
        # Hook for down projection
        def down_proj_hook(module, input, output):
            self.save_matrix(input[0], f"down_proj_input", layer_idx)
            self.save_matrix(output, f"down_proj_output", layer_idx)
            # Save the weight matrix
            self.save_matrix(module.weight.data, f"down_proj_weight", layer_idx)
        
        hook_handle = layer.mlp.down_proj.register_forward_hook(down_proj_hook)
        self.hooks.append(hook_handle)
    
    def register_all_hooks(self):
        """Register hooks for all layers."""
        print(f"🔗 Registering hooks for {self.num_layers} layers...")
        
        for layer_idx in range(self.num_layers):
            print(f"   Registering hooks for layer {layer_idx}")
            self.create_attention_hooks(layer_idx)
            self.create_mlp_hooks(layer_idx)
        
        print(f"✅ All hooks registered successfully")
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("🧹 All hooks removed")
    
    def generate_sample_input(self) -> torch.Tensor:
        """Generate a diverse sample input of the specified sequence length with no repeated sentences."""
        # Set random seed for reproducible sentence selection
        import random
        random.seed(self.data_seed)
        
        # Create a diverse set of sentences covering different topics and structures
        diverse_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the way we work and live.",
            "Climate change poses significant challenges for future generations.",
            "The ancient pyramids of Egypt continue to fascinate archaeologists.",
            "Modern technology has revolutionized communication across the globe.",
            "Ocean currents play a crucial role in regulating Earth's climate.",
            "Space exploration has led to numerous scientific breakthroughs.",
            "Renewable energy sources are becoming increasingly cost-effective.",
            "The human brain contains approximately 86 billion neurons.",
            "Quantum computing promises to solve complex problems exponentially faster.",
            "Biodiversity loss threatens ecosystems worldwide.",
            "Machine learning algorithms can identify patterns in vast datasets.",
            "The invention of the printing press democratized access to knowledge.",
            "Genetic engineering offers potential solutions to inherited diseases.",
            "Urban planning must balance growth with environmental sustainability.",
            "The internet has created unprecedented opportunities for global collaboration.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Cryptocurrency represents a new paradigm in digital finance.",
            "The discovery of antibiotics revolutionized modern medicine.",
            "Virtual reality technology is creating immersive educational experiences.",
            "Coral reefs support an extraordinary diversity of marine life.",
            "Autonomous vehicles may reshape transportation infrastructure.",
            "The scientific method provides a systematic approach to understanding nature.",
            "Social media platforms have transformed how people share information.",
            "Nanotechnology enables manipulation of matter at the atomic scale.",
            "The water cycle is essential for maintaining life on Earth.",
            "Blockchain technology ensures secure and transparent transactions.",
            "Archaeological evidence reveals insights into ancient civilizations.",
            "Renewable energy storage solutions are critical for grid stability.",
            "The immune system protects the body from harmful pathogens.",
            "Artificial neural networks mimic the structure of biological brains.",
            "Deforestation contributes to habitat loss and climate change.",
            "Precision medicine tailors treatments to individual genetic profiles.",
            "The theory of evolution explains the diversity of life on Earth.",
            "Robotics automation is transforming manufacturing processes.",
            "Sustainable agriculture practices help preserve soil health.",
            "The periodic table organizes elements by their atomic properties.",
            "Cloud computing provides scalable and flexible IT infrastructure.",
            "Gravitational waves offer new insights into cosmic phenomena.",
            "Biotechnology applications range from medicine to environmental cleanup.",
            "The carbon cycle regulates atmospheric carbon dioxide levels.",
            "Cybersecurity measures protect digital systems from malicious attacks.",
            "Stem cell research holds promise for regenerative medicine.",
            "The electromagnetic spectrum encompasses radio waves to gamma rays.",
            "Data visualization helps communicate complex information effectively.",
            "Plate tectonics explains the movement of Earth's crustal plates.",
            "Artificial photosynthesis could provide clean energy solutions.",
            "The nervous system coordinates responses to environmental stimuli.",
            "Distributed computing harnesses the power of multiple processors.",
            "Conservation efforts aim to protect endangered species from extinction."
        ]
        
        # Build text by cycling through sentences without immediate repeats
        sample_text = ""
        sentence_index = 0
        
        # Keep adding sentences until we have enough text for tokenization
        while len(sample_text.split()) < self.seq_len * 2:  # Rough estimate for tokenization
            sample_text += diverse_sentences[sentence_index] + " "
            sentence_index = (sentence_index + 1) % len(diverse_sentences)
        
        # Tokenize
        inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=self.seq_len,
            truncation=True,
            padding="max_length"
        )
        
        return inputs.input_ids
    
    def capture_matrices(self):
        """Main function to capture all matrices."""
        print(f"🚀 Starting matrix capture for Phi-3 Mini 4k Instruct with RMSNorm weights")
        print(f"   RMSNorm weights: {self.rmsnorm_weights_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Sequence length: {self.seq_len}")
        
        # Register hooks
        self.register_all_hooks()
        
        try:
            # Generate sample input
            print(f"📝 Generating sample input...")
            input_ids = self.generate_sample_input()
            print(f"   Input shape: {input_ids.shape}")
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Save input
            self.save_matrix(input_ids, "model_input")
            
            # Run forward pass with compatibility settings
            print(f"🔄 Running forward pass...")
            with torch.no_grad():
                # Use past_key_values=None to avoid cache compatibility issues
                outputs = self.model(input_ids, past_key_values=None, use_cache=False)
            
            # Save final output
            self.save_matrix(outputs.logits, "model_output_logits")
            
            print(f"✅ Matrix capture completed successfully!")
            print(f"📁 All matrices saved to: {self.output_dir}")
            
            # Create summary
            self.create_summary()
            
        except Exception as e:
            print(f"❌ Error during matrix capture: {e}")
            raise
        finally:
            # Clean up hooks
            self.remove_all_hooks()
    
    def create_summary(self):
        """Create a summary of all captured matrices."""
        summary = {
            "model_config": {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "intermediate_size": self.intermediate_size,
                "num_layers": self.num_layers,
                "sequence_length": self.seq_len
            },
            "rmsnorm_weights_path": str(self.rmsnorm_weights_path),
            "captured_matrices": {},
            "matrix_flow": {
                "attention_flow": [
                    "input_rmsnorm_input", "input_rmsnorm_output",
                    "qkv_proj_input", "qkv_proj_output_fused",
                    "q_proj_output", "k_proj_output", "v_proj_output",
                    "q_reshaped", "k_reshaped", "v_reshaped",
                    "attention_scores_qk_transpose_pre_mask",
                    "causal_mask", "attention_scores_qk_transpose_masked",
                    "attention_softmax_output", "attention_operation_output",
                    "attention_reshaped_output", "o_proj_input", "o_proj_output",
                    "post_attn_rmsnorm_input", "post_attn_rmsnorm_output"
                ],
                "mlp_flow": [
                    "gate_up_proj_input", "gate_up_proj_output_fused",
                    "gate_proj_output_pre_activation", "up_proj_output",
                    "gate_proj_silu_output", "swiglu_elementwise_multiply_output_silu",
                    "down_proj_input", "down_proj_output"
                ]
            }
        }
        
        # List all saved matrices
        for file_path in self.output_dir.glob("*.pt"):
            matrix_name = file_path.stem
            stats_file = self.output_dir / f"{matrix_name}_stats.json"
            
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                summary["captured_matrices"][matrix_name] = stats
        
        # Save summary
        summary_path = self.output_dir / "capture_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary saved to: {summary_path}")
        print(f"📈 Total matrices captured: {len(summary['captured_matrices'])}")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture matrices from Phi-3 Mini 4k Instruct model with fine-tuned RMSNorm weights")
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Model ID to load"
    )
    parser.add_argument(
        "--rmsnorm-weights-path",
        type=str,
        required=True,
        help="Path to the saved RMSNorm weights (.pt file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./phi3_rmsnorm_matrices",
        help="Output directory for saved matrices"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for input sample"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model"
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Random seed for reproducible data generation (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"🔧 Loading Phi-3 Mini 4k Instruct model with RMSNorm weights...")
    print(f"   Model ID: {args.model_id}")
    print(f"   RMSNorm weights: {args.rmsnorm_weights_path}")
    print(f"   Device: {args.device}")
    print(f"   Data type: {args.dtype}")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with compatibility settings
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=True,
        attn_implementation="eager"  # Use eager attention to avoid flash-attention compatibility issues
    )
    model.eval()
    
    print(f"✅ Model loaded successfully")
    
    # Create matrix capture instance
    capture = Phi3MatrixCaptureWithRMSNorm(
        model=model,
        tokenizer=tokenizer,
        rmsnorm_weights_path=args.rmsnorm_weights_path,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        data_seed=args.data_seed
    )
    
    # Capture matrices
    capture.capture_matrices()
    
    print(f"🎉 Matrix capture completed!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()