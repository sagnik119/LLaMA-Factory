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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from ...extras.logging import get_logger
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values
from .supervised import preprocess_supervised_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ...hparams import DataArguments


logger = get_logger(__name__)


def preprocess_bos_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    """
    Preprocesses supervised dataset with BOS token addition.
    
    This function ensures that:
    1. BOS tokens are added at the beginning of every sequence
    2. The data is properly formatted for the BOS zeroing trainer
    """
    logger.info("🎯 Preprocessing dataset with BOS token addition")
    
    # First, preprocess using the standard supervised preprocessing
    processed_examples = preprocess_supervised_dataset(
        examples=examples,
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args
    )
    
    # Ensure BOS tokens are added to input_ids
    if "input_ids" in processed_examples:
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            # If no BOS token is defined, use the EOS token or pad token as fallback
            bos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
            if bos_token_id is None:
                logger.warning("⚠️ No BOS, EOS, or PAD token found. Using token ID 1 as BOS.")
                bos_token_id = 1
        
        logger.info(f"✅ Using BOS token ID: {bos_token_id}")
        
        modified_input_ids = []
        modified_attention_mask = []
        modified_labels = []
        
        for i, input_ids in enumerate(processed_examples["input_ids"]):
            # Check if BOS token is already at the beginning
            if len(input_ids) > 0 and input_ids[0] != bos_token_id:
                # Add BOS token at the beginning
                new_input_ids = [bos_token_id] + input_ids
                
                # Adjust attention mask
                if "attention_mask" in processed_examples:
                    attention_mask = processed_examples["attention_mask"][i]
                    new_attention_mask = [1] + attention_mask
                else:
                    new_attention_mask = [1] * len(new_input_ids)
                
                # Adjust labels
                if "labels" in processed_examples:
                    labels = processed_examples["labels"][i]
                    # Add -100 (ignore index) for the BOS token position
                    new_labels = [-100] + labels
                else:
                    new_labels = [-100] + input_ids
                
                # Truncate if necessary to maintain max length
                max_length = data_args.cutoff_len
                if len(new_input_ids) > max_length:
                    new_input_ids = new_input_ids[:max_length]
                    new_attention_mask = new_attention_mask[:max_length]
                    new_labels = new_labels[:max_length]
                
                modified_input_ids.append(new_input_ids)
                modified_attention_mask.append(new_attention_mask)
                modified_labels.append(new_labels)
                
            else:
                # BOS token already present or empty sequence
                modified_input_ids.append(input_ids)
                if "attention_mask" in processed_examples:
                    modified_attention_mask.append(processed_examples["attention_mask"][i])
                else:
                    modified_attention_mask.append([1] * len(input_ids))
                
                if "labels" in processed_examples:
                    modified_labels.append(processed_examples["labels"][i])
                else:
                    modified_labels.append(input_ids)
        
        # Update the processed examples
        processed_examples["input_ids"] = modified_input_ids
        processed_examples["attention_mask"] = modified_attention_mask
        if "labels" in processed_examples:
            processed_examples["labels"] = modified_labels
        
        logger.info(f"✅ Added BOS tokens to {len(modified_input_ids)} examples")
        
        # Log sample for verification
        if len(modified_input_ids) > 0:
            sample_tokens = tokenizer.convert_ids_to_tokens(modified_input_ids[0][:10])
            logger.info(f"📝 Sample tokens (first 10): {sample_tokens}")
    
    return processed_examples


def print_bos_supervised_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    """
    Prints a formatted example from the BOS supervised dataset.
    """
    print("=" * 50)
    print("BOS SUPERVISED DATASET EXAMPLE")
    print("=" * 50)
    
    if "input_ids" in example:
        input_ids = example["input_ids"]
        print(f"Input IDs length: {len(input_ids)}")
        print(f"First 20 tokens: {input_ids[:20]}")
        
        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[:20])
        print(f"First 20 decoded: {tokens}")
        
        # Check if BOS token is at position 0
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None and len(input_ids) > 0:
            if input_ids[0] == bos_token_id:
                print(f"✅ BOS token ({bos_token_id}) found at position 0")
            else:
                print(f"⚠️ BOS token ({bos_token_id}) NOT at position 0. Found: {input_ids[0]}")
        
        # Full text preview
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Full text preview: {full_text[:200]}...")
    
    if "labels" in example:
        labels = example["labels"]
        print(f"Labels length: {len(labels)}")
        print(f"First 20 labels: {labels[:20]}")
        
        # Check if position 0 is ignored (-100)
        if len(labels) > 0:
            if labels[0] == -100:
                print("✅ Position 0 label is -100 (ignored for BOS token)")
            else:
                print(f"⚠️ Position 0 label is {labels[0]} (should be -100 for BOS token)")
    
    print("=" * 50)