import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from sub_models.attention_blocks import PositionalEncoding1D, AttentionBlock, AttentionBlockKVCache


class StochasticTransformer(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout):
        super().__init__()
        self.action_dim = action_dim

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6)  # TODO: check if this is necessary

        self.head = nn.Linear(feat_dim, stoch_dim)

    def forward(self, samples, action, mask):
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for enc_layer in self.layer_stack:
            feats, attn = enc_layer(feats, mask)

        feat = self.head(feats)
        return feat


class StochasticTransformerKVCache(nn.Module):
    def __init__(self, stoch_dim, action_dim, feat_dim, num_layers, num_heads, max_length, dropout, conf):
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.max_cache_length = None

        # mix image_embedding and action
        self.stem = nn.Sequential(
            nn.Linear(stoch_dim+action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim)
        )
        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=feat_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlockKVCache(feat_dim=feat_dim, hidden_dim=feat_dim*2, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(feat_dim, eps=1e-6) 
        # self.root_hyper_sphere_r = conf.Models.WorldModel.HyperSphereR ** 0.5

    def forward(self, samples, action, mask):
        '''
        Normal forward pass
        '''
        action = F.one_hot(action.long(), self.action_dim).float() # (B, L, action_dim)
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats) # (B, L, feat_dim)

        for layer in self.layer_stack:
            feats, attn = layer(feats, feats, feats, mask)
        
        return feats

    def reset_kv_cache_list(self, batch_size, dtype, max_cache_length=None):
        '''
        Reset self.kv_cache_list
        Each element is a tuple of (k_cache, v_cache) for each layer
        k_cache, v_cache: (B, n_head, L_cache, d_k/d_v)
        '''
        self.max_cache_length = max_cache_length
            
        self.kv_cache_list = []
        num_heads = self.layer_stack[0].slf_attn.n_head
        d_k = self.layer_stack[0].slf_attn.d_k
        d_v = self.layer_stack[0].slf_attn.d_v
        
        for layer in self.layer_stack:
            # Initialize empty cache (B, n_head, 0, d_k/d_v)
            k_cache = torch.zeros(size=(batch_size, num_heads, 0, d_k), dtype=dtype, device="cuda")
            v_cache = torch.zeros(size=(batch_size, num_heads, 0, d_v), dtype=dtype, device="cuda")
            self.kv_cache_list.append((k_cache, v_cache))

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        Each cache element is a tuple of (k_cache, v_cache) containing projected K, V
        '''
        assert samples.shape[1] == 1
        
        # Get current cache length
        current_cache_len = self.kv_cache_list[0][0].shape[2]  # k_cache.shape = (B, n_head, L, d_k)
        # Use None as mask - attention to all previous tokens (including current) is allowed
        # The causal constraint is already enforced by only having access to past KV cache
        mask = None
        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=current_cache_len)
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            k_cache, v_cache = self.kv_cache_list[idx]
            
            # Use efficient KV cache with projected K, V
            feats, attn, new_k, new_v = layer.forward_with_kv_cache(feats, k_cache, v_cache, mask)
            
            # Append new projected K, V to cache
            updated_k_cache = torch.cat([k_cache, new_k], dim=2)  # (B, n_head, L+1, d_k)
            updated_v_cache = torch.cat([v_cache, new_v], dim=2)  # (B, n_head, L+1, d_v)
            
            # Remove old entries if cache exceeds max_cache_length (FIFO)
            if self.max_cache_length is not None and updated_k_cache.shape[2] > self.max_cache_length:
                updated_k_cache = updated_k_cache[:, :, -self.max_cache_length:, :]
                updated_v_cache = updated_v_cache[:, :, -self.max_cache_length:, :]
            
            self.kv_cache_list[idx] = (updated_k_cache, updated_v_cache)

        return feats
