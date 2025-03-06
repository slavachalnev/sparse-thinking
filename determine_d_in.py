import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_dimensions():
    # Load DeepSeek-R1-Distill-Llama-8B model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model in eval mode with float16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # Let the model decide where to put things
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Test with a small reasoning example
    test_text = "Let's solve the following problem step-by-step: What is 25 + 30?"
    print(f"\nTest input: {test_text}")
    
    # Tokenize input
    tokens = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    # Get model information
    print("\nModel architecture:")
    # Check model type
    model_type = type(model).__name__
    print(f"Model type: {model_type}")
    
    # Extract and print hidden dimension sizes
    config = model.config
    print(f"Hidden size: {config.hidden_size}")
    
    # Generate with output_hidden_states=True to get the hidden states
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    # For each layer, print the shape of hidden states
    print("\nHidden states shapes for each layer:")
    for i, hidden_state in enumerate(outputs.hidden_states):
        print(f"Layer {i}: {hidden_state.shape}")
    
    # The last hidden state (usually what we'd use for the SAE)
    last_hidden = outputs.hidden_states[-1]
    print(f"\nLast hidden state shape: {last_hidden.shape}")
    
    # This shape gives us the dimension we need for d_in
    d_in = last_hidden.shape[-1]
    print(f"\nRecommended d_in value: {d_in}")
    
    return d_in

if __name__ == "__main__":
    d_in = get_model_dimensions()
    print("\n" + "="*50)
    print(f"To configure your SAE model, use d_in = {d_in}")
    print("="*50)