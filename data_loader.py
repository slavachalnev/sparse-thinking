import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np

class ReasoningStepDataset(Dataset):
    def __init__(self, representations, cfg):
        """
        Dataset class for reasoning step representations
        
        Args:
            representations: Tensor of shape [num_examples, num_steps, d_in]
            cfg: Configuration dictionary
        """
        self.representations = representations
        self.cfg = cfg
    
    def __len__(self):
        return len(self.representations)
    
    def __getitem__(self, idx):
        return self.representations[idx]

def extract_activations(model, tokenizer, text, layer_idx=-1, device="cuda"):
    """
    Extract activations from a specific layer in the model
    
    Args:
        model: DeepSeek model
        tokenizer: DeepSeek tokenizer
        text: Input text
        layer_idx: Index of the layer to extract activations from (-1 for last layer)
        device: Device to run the model on
        
    Returns:
        Tensor containing activations for each reasoning step
    """
    tokens = tokenizer(text, return_tensors="pt").to(device)
    
    # Get all hidden states
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    # Extract activations from specified layer
    hidden_states = outputs.hidden_states[layer_idx]
    
    return hidden_states.cpu()

def load_data_and_model(cfg):
    """
    Load the OpenThoughts-114k dataset and DeepSeek-R1-Distill-Llama-8B model,
    and prepare the data for training.
    
    Args:
        cfg: Configuration dictionary containing parameters
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        model_sae: Initialized SplitSAE model
    """
    # Load OpenThoughts-114k dataset
    dataset = load_dataset("open-thoughts/OpenThoughts-114k")
    
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
    
    # Process examples through the model
    representations = []
    
    # Check if cached representations exist
    cache_path = cfg.get("cache_path", "representations_cache.pt")
    if os.path.exists(cache_path) and cfg.get("use_cache", True):
        print(f"Loading cached representations from {cache_path}")
        representations = torch.load(cache_path)
    else:
        print("Processing examples and extracting representations...")
        # Take a subset for processing (adjust based on resources)
        num_examples = min(cfg.get("max_examples", 1000), len(dataset["train"]))
        
        for i in range(num_examples):
            example = dataset["train"][i]
            # Extract reasoning problem text
            text = example["text"]
            
            # Extract activations from the model
            activations = extract_activations(
                model, 
                tokenizer, 
                text, 
                layer_idx=cfg.get("layer_idx", -1),
                device=device
            )
            
            # Store as representation
            representations.append(activations)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_examples} examples")
        
        # Stack representations
        representations = torch.stack(representations)
        
        # Cache representations
        if cfg.get("cache_representations", True):
            print(f"Caching representations to {cache_path}")
            torch.save(representations, cache_path)
    
    # Split into training and validation sets
    split_idx = int(len(representations) * cfg.get("train_ratio", 0.8))
    train_representations = representations[:split_idx]
    val_representations = representations[split_idx:]
    
    # Create datasets
    train_dataset = ReasoningStepDataset(train_representations, cfg)
    val_dataset = ReasoningStepDataset(val_representations, cfg)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4)
    )
    
    # Import the SplitSAE model here to avoid circular imports
    from model import SplitSAE
    
    # Initialize the SplitSAE model
    model_sae = SplitSAE(cfg)
    
    return train_loader, val_loader, model_sae