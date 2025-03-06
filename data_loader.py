import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np

class OnDeviceActivationsDataset(Dataset):
    def __init__(self, dataset, indices, model, tokenizer, cfg):
        """
        Dataset class that computes activations on-demand and keeps them on device
        
        Args:
            dataset: HuggingFace dataset
            indices: List of indices to use from the dataset
            model: Language model to extract activations from
            tokenizer: Tokenizer for the language model
            cfg: Configuration dictionary
        """
        self.dataset = dataset
        self.indices = indices
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = cfg["device"]
        self.layer_idx = cfg.get("layer_idx", -1)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the example from the dataset
        example_idx = self.indices[idx]
        example = self.dataset[example_idx]
        text = example["text"]
        
        # Extract activations from the model (kept on device)
        return self.extract_activations(text)
    
    def extract_activations(self, text):
        """
        Extract activations from a specific layer in the model and keep them on device
        
        Returns:
            Tensor of shape [1, seq_len, d_in] - representing a batch size of 1
        """
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Calculate the actual layer index, converting negative indexing to positive
        num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding layer
        actual_layer_idx = self.layer_idx if self.layer_idx >= 0 else num_layers + self.layer_idx
        
        # Only compute hidden states up to the required layer to save memory
        with torch.no_grad():
            outputs = self.model(
                **tokens, 
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True
            )
        
        # Extract activations from specified layer and keep on device
        # Shape: [1, seq_len, d_in]
        hidden_states = outputs.hidden_states[actual_layer_idx]
        
        # Ensure consistent shape [batch_size=1, seq_len, d_in]
        if hidden_states.dim() == 2:
            # If it's [seq_len, d_in], add batch dimension
            hidden_states = hidden_states.unsqueeze(0)
            
        return hidden_states

def load_data_and_representations(cfg):
    """
    Load the OpenThoughts-114k dataset and DeepSeek-R1-Distill-Llama-8B model,
    and prepare on-device datasets for training.
    
    Args:
        cfg: Configuration dictionary containing parameters
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Load OpenThoughts-114k dataset
    dataset = load_dataset("open-thoughts/OpenThoughts-114k")["train"]
    
    # Load DeepSeek-R1-Distill-Llama-8B model and tokenizer
    device = cfg["device"]
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Define dtype mapping
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtypes[cfg.get("model_dtype", "float16")],
        device_map=device
    )
    
    # Take a subset of examples (adjust based on resources)
    num_examples = min(cfg.get("max_examples", 1000), len(dataset))
    all_indices = list(range(num_examples))
    
    # Split into training and validation sets
    split_idx = int(num_examples * cfg.get("train_ratio", 0.8))
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"Creating dataset with {len(train_indices)} training examples and {len(val_indices)} validation examples")
    
    # Create datasets that compute activations on-demand
    train_dataset = OnDeviceActivationsDataset(dataset, train_indices, model, tokenizer, cfg)
    val_dataset = OnDeviceActivationsDataset(dataset, val_indices, model, tokenizer, cfg)
    
    # Define collate function to properly batch tensors
    def collate_activations(batch):
        """
        Custom collate function to batch together tensors while ensuring
        consistent shape [batch_size, seq_len, d_in]
        """
        # Each item in batch is already [1, seq_len, d_in]
        # We need to concatenate along batch dimension
        return torch.cat(batch, dim=0)
    
    # Create dataloaders with fewer workers for GPU dataset and custom collate function
    # When keeping data on device, we need to use fewer workers to avoid CUDA issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        # For on-device computation, using multiple workers can cause CUDA errors
        # so we use fewer workers or none
        num_workers=0,
        collate_fn=collate_activations
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_activations
    )
    
    return train_loader, val_loader