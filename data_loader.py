import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any

class ActivationsDataLoader:
    def __init__(self, cfg):
        """
        Class to load text dataset and language model, and generate activation batches.
        
        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.device = cfg["device"]
        self.layer_idx = cfg.get("layer_idx", -1)
        self.llm_batch_size = cfg.get("llm_batch_size", 1)
        self.batch_size = cfg.get("batch_size", 32)
        
        # Load dataset
        self.dataset = load_dataset("open-thoughts/OpenThoughts-114k")["train"]
        
        # Load model and tokenizer
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Define dtype mapping
        dtypes = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtypes[cfg.get("model_dtype", "float16")],
            device_map=self.device
        )
        
        # Take a subset of examples
        num_examples = min(cfg.get("max_examples", 1000), len(self.dataset))
        self.indices = list(range(num_examples))
        
        print(f"Dataset initialized with {len(self.indices)} examples")
        
        # Initialize iteration counter and shuffled indices
        self._shuffle_indices()
        self.iter_position = 0
    
    def _shuffle_indices(self):
        """Shuffle the indices to randomize batch generation"""
        self.shuffled_indices = self.indices.copy()
        np.random.shuffle(self.shuffled_indices)
    
    def get_activation_batch(self, indices):
        """
        Get a batch of activations for the given indices
        
        Args:
            indices: List of indices from the dataset
            
        Returns:
            Tensor of shape [batch_size, seq_len, d_in]
        """
        # Get texts for the given indices
        texts = [self.dataset[idx]["text"] for idx in indices]
        
        # Process texts in batches based on llm_batch_size
        all_activations = []
        
        for i in range(0, len(texts), self.llm_batch_size):
            batch_texts = texts[i:i + self.llm_batch_size]
            batch_activations = self._process_text_batch(batch_texts)
            all_activations.extend(batch_activations)
            
        # Stack all activations into a single tensor
        # Each activation has shape [1, seq_len, d_in]
        return torch.cat(all_activations, dim=0)
    
    def _process_text_batch(self, texts):
        """
        Process a batch of texts to extract activations
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tensors, each with shape [1, seq_len, d_in]
        """
        # Tokenize all texts in the batch
        batch_tokens = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        
        # Calculate the actual layer index
        num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding layer
        actual_layer_idx = self.layer_idx if self.layer_idx >= 0 else num_layers + self.layer_idx
        
        # Process the batch
        with torch.no_grad():
            outputs = self.model(
                **batch_tokens,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True
            )
        
        # Extract activations
        hidden_states_batch = outputs.hidden_states[actual_layer_idx]
        
        # Split the batch back into individual examples
        return [hidden_states_batch[j:j+1] for j in range(hidden_states_batch.size(0))]
    
    def get_next_batch(self):
        """
        Get the next batch of activations
        
        Returns:
            Tensor of shape [batch_size, seq_len, d_in]
        """
        # Get the next batch of indices
        start_pos = self.iter_position
        end_pos = min(start_pos + self.batch_size, len(self.shuffled_indices))
        batch_indices = self.shuffled_indices[start_pos:end_pos]
        
        # Update position for next call
        self.iter_position = end_pos
        
        # Reshuffle and reset position if we've gone through all indices
        if self.iter_position >= len(self.shuffled_indices):
            self._shuffle_indices()
            self.iter_position = 0
        
        return self.get_activation_batch(batch_indices)
    
    def get_data_size(self):
        """Get the number of examples"""
        return len(self.indices)
    
    def get_num_batches(self):
        """Get the number of batches in one epoch"""
        return (len(self.indices) + self.batch_size - 1) // self.batch_size