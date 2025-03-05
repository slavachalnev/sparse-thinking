# Plan Document for Implementing SAE-like Architecture for Reasoning Step Representations

## Overview

We want to implement a sparse autoencoder (SAE)-like architecture to analyze representations from the DeepSeek-R1-Distill-Llama-8B reasoning model. Each reasoning task consists of multiple reasoning steps (tokens), each represented as a vector (e.g., hidden-layer activations or residual states). 

We'll use the OpenThoughts-114k dataset, which contains 114k high-quality examples covering math, science, code, and puzzles, as our source of reasoning problems.

We define two distinct latent codes:

- **Shared latent code**: captures stable, repeated features across all reasoning steps. Minimal or no sparsity penalty (just to improve reconstruction).
- **Unique latent code**: captures features specific to each reasoning step. Strong L1 sparsity penalty to encourage monosemantic features.

The main goal is to reconstruct each step's input representation as accurately as possible, while explicitly separating shared and step-specific information.

### Key simplification:

We'll use **simple mean pooling** across reasoning steps to generate the input for the shared component, since our goal for shared latent is just to reduce reconstruction error rather than interpretability or temporal complexity.

## Dataset and Model Selection

### Dataset: OpenThoughts-114k
- `open-thoughts/OpenThoughts-114k`
- 114k high-quality synthetic reasoning examples
- Covers diverse domains: math, science, code, and puzzles
- Verified for correctness through rigorous data generation pipeline

### Model: DeepSeek-R1-Distill-Llama-8B
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- Distilled from DeepSeek-R1, a model trained via large-scale reinforcement learning for reasoning

## Data Processing Pipeline

1. **Extract reasoning examples from OpenThoughts-114k**
   - Load examples from the dataset
   - Filter for diverse problem types to ensure comprehensive analysis

2. **Process through DeepSeek-R1-Distill-Llama-8B**
   - For each reasoning example, run inference through the model
   - Extract hidden-layer activations or residual states at each reasoning step
   - Store these vector representations for training

3. **Prepare training data**
   - Format the extracted representations as tensors for input to our SAE architecture
   - Split into training and validation sets

## Inputs and Outputs:

### Inputs:
- Processed dataset containing multiple reasoning problems (examples) from OpenThoughts-114k.
- Each example is a sequence of \( T \) reasoning step vectors extracted from DeepSeek-R1-Distill-Llama-8B:
  ```
  reasoning_steps = tensor of shape [batch_size, num_steps, input_dim]
  ```

### Outputs:
- Reconstructed reasoning step representations:
  ```
  reconstructed_steps = tensor of shape [batch_size, num_steps, input_dim]
  ```
- Sparse latent codes for unique features, and dense (or less sparse) latent codes for shared features.

## Model Components:

The model consists of three neural network components:

### 1. Shared Encoder (`EncoderShared`)
- Input: mean-pooled reasoning steps vector `[batch_size, input_dim]`
- Output: shared latent representation `[batch_size, shared_latent_dim]`
- Low sparsity regularization (or none).

### 2. Unique (Step-specific) Encoder (`EncoderUnique`)
- Input: individual reasoning steps `[batch_size, num_steps, input_dim]`
- Output: unique latent representations `[batch_size, num_steps, unique_latent_dim]`
- High sparsity regularization (L1).

### 3. Decoder (`Decoder`)
- Input: concatenated shared and unique latent features (`shared + unique`)
  `[batch_size, num_steps, shared_latent_dim + unique_latent_dim]`
- Output: reconstructed reasoning steps `[batch_size, num_steps, input_dim]`

## Evaluation and Analysis:

- Evaluate reconstruction error to ensure the model correctly reconstructs reasoning step representations.
- Inspect sparsity of unique latent codes (`z_unique`). Confirm they're sparse and potentially interpretable.
- Compare reasoning patterns across different domains (math vs. code vs. science) to identify domain-specific reasoning features.
- Analyze the shared latent space to identify common reasoning strategies used across problems.
- Optionally use linear probes or visualization techniques to analyze latent features.
