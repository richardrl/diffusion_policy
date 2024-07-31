import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # x: (B), timesteps, 1
        # emb: 1, 1, embedding_dim/2
        # broadcast -> (B), timesteps, embedding_dim/2

        unsqueezed_emb = emb

        for _ in range(len(x.shape)):
            unsqueezed_emb = unsqueezed_emb.unsqueeze(0)
        emb = x[..., None] * unsqueezed_emb

        # -> num_timesteps, embedding_dim
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
