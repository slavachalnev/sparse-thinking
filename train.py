import torch
import torch.optim as optim
import argparse
import os
import json
from tqdm import tqdm
import wandb
import numpy as np
from data_loader import load_data_and_model

def train(cfg):
    # Initialize wandb
    if cfg.get("use_wandb", True):
        wandb.init(
            project=cfg.get("wandb_project", "sparse-thinking"),
            name=cfg.get("wandb_run_name", f"run_{cfg['seed']}"),
            config=cfg
        )
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg["seed"])
    
    # Load data and model
    train_loader, val_loader, model = load_data_and_model(cfg)
    
    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0)
    )
    
    # Training loop
    device = cfg["device"]
    num_epochs = cfg["num_epochs"]
    
    # Create directory for saving checkpoints
    os.makedirs(cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_unique_l1 = 0
        epoch_unique_l0 = 0
        epoch_recon_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            batch = batch.to(device)
            
            # Forward pass
            losses = model.get_losses(batch)
            
            # Compute total loss with L1 regularization
            total_loss = losses["recon_loss"] + cfg["l1_coef"] * losses["unique_l1"]
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_recon_loss += losses["recon_loss"].item()
            epoch_unique_l1 += losses["unique_l1"].item()
            epoch_unique_l0 += losses["unique_l0"].item()
        
        # Average metrics
        epoch_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_unique_l1 /= len(train_loader)
        epoch_unique_l0 /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_unique_l1 = 0
        val_unique_l0 = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                batch = batch.to(device)
                
                # Forward pass
                losses = model.get_losses(batch)
                
                # Compute total loss
                total_loss = losses["recon_loss"] + cfg["l1_coef"] * losses["unique_l1"]
                
                # Track metrics
                val_loss += total_loss.item()
                val_recon_loss += losses["recon_loss"].item()
                val_unique_l1 += losses["unique_l1"].item()
                val_unique_l0 += losses["unique_l0"].item()
            
            # Average metrics
            val_loss /= len(val_loader)
            val_recon_loss /= len(val_loader)
            val_unique_l1 /= len(val_loader)
            val_unique_l0 /= len(val_loader)
        
        # Log metrics to wandb
        if cfg.get("use_wandb", True):
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": epoch_loss,
                "train/recon_loss": epoch_recon_loss,
                "train/unique_l1": epoch_unique_l1,
                "train/unique_l0": epoch_unique_l0,
                "val/loss": val_loss,
                "val/recon_loss": val_recon_loss,
                "val/unique_l1": val_unique_l1,
                "val/unique_l0": val_unique_l0,
            })
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Unique L1: {epoch_unique_l1:.6f}, "
              f"Unique L0: {epoch_unique_l0:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.get("checkpoint_interval", 5) == 0:
            checkpoint_path = os.path.join(
                cfg.get("checkpoint_dir", "checkpoints"), 
                f"model_epoch_{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Log model checkpoint to wandb
            if cfg.get("use_wandb", True) and cfg.get("log_model", True):
                wandb.save(checkpoint_path)
    
    # Close wandb
    if cfg.get("use_wandb", True):
        wandb.finish()
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Sparse Thinking SAE model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    # Train the model
    model = train(cfg)
    
    # Save final model
    final_model_path = os.path.join(
        cfg.get("checkpoint_dir", "checkpoints"), 
        "model_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()