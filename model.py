import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

class SplitSAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_in = self.cfg["d_in"]  # input dimension from DeepSeek
        d_shared = self.cfg["d_shared"]  # shared latent dimension
        d_unique = self.cfg["d_unique"]  # unique latent dimension
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        
        # Shared encoder weights
        self.W_shared_enc = nn.Parameter(
            torch.empty(d_in, d_shared, dtype=self.dtype)
        )
        self.b_shared_enc = nn.Parameter(torch.zeros(d_shared, dtype=self.dtype))
        
        # Unique encoder weights
        self.W_unique_enc = nn.Parameter(
            torch.empty(d_in, d_unique, dtype=self.dtype)
        )
        self.b_unique_enc = nn.Parameter(torch.zeros(d_unique, dtype=self.dtype))
        
        # Decoder weights
        self.W_dec = nn.Parameter(
            torch.empty(d_shared + d_unique, d_in, dtype=self.dtype)
        )
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=self.dtype))
        
        # Initialize decoder norm
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * self.cfg["dec_init_norm"]
        )
        
        # Initialize encoders using decoder's transpose
        self.W_shared_enc.data = self.W_dec.data[:d_shared].T
        self.W_unique_enc.data = self.W_dec.data[d_shared:].T
        
        self.to(self.cfg["device"])

    def encode_shared(self, x):
        # x: [batch, d_in]
        x_enc = einops.einsum(
            x,
            self.W_shared_enc,
            "batch d_in, d_in d_shared -> batch d_shared"
        )
        return F.relu(x_enc + self.b_shared_enc)

    def encode_unique(self, x):
        # x: [batch, num_steps, d_in]
        x_enc = einops.einsum(
            x,
            self.W_unique_enc,
            "batch steps d_in, d_in d_unique -> batch steps d_unique"
        )
        return F.relu(x_enc + self.b_unique_enc)

    def decode(self, z_shared, z_unique):
        # z_shared: [batch, d_shared]
        # z_unique: [batch, num_steps, d_unique]
        z_combined = torch.cat([z_shared.unsqueeze(1).expand(-1, z_unique.size(1), -1), z_unique], dim=-1)
        # z_combined: [batch, num_steps, d_shared + d_unique]
        
        x_dec = einops.einsum(
            z_combined,
            self.W_dec,
            "batch steps d_combined, d_combined d_in -> batch steps d_in"
        )
        return x_dec + self.b_dec

    def forward(self, x):
        # x: [batch, num_steps, d_in]
        # Mean pool for shared features
        x_mean = x.mean(dim=1)  # [batch, d_in]
        
        # Encode
        z_shared = self.encode_shared(x_mean)
        z_unique = self.encode_unique(x)
        
        # Decode
        x_recon = self.decode(z_shared, z_unique)
        
        return x_recon, z_shared, z_unique

    def get_losses(self, x):
        x = x.to(self.dtype)
        x_recon, z_shared, z_unique = self.forward(x)
        
        # Reconstruction loss
        recon_loss = (x_recon - x).pow(2).mean()
        
        # L1 sparsity on unique features
        unique_l1 = z_unique.abs().sum(-1).mean()
        
        # L0 sparsity on unique features
        unique_l0 = (z_unique > 0).float().sum(-1).mean()
        
        return {
            "recon_loss": recon_loss,
            "unique_l1": unique_l1,
            "unique_l0": unique_l0
        } 