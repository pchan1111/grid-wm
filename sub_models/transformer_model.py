import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from sub_models.attention_blocks import get_vector_mask
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
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads

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

    def reset_kv_cache_list(self, batch_size, dtype, max_length=None):
        '''
        Reset self.kv_cache_list
        '''
        self.max_cache_length = max_length
        self.key_cache_list = []
        self.value_cache_list = []
        for layer in self.layer_stack:
            self.key_cache_list.append(torch.zeros(size=(batch_size, self.num_heads, 0, self.head_dim), dtype=dtype, device="cuda"))
            self.value_cache_list.append(torch.zeros(size=(batch_size, self.num_heads, 0, self.head_dim), dtype=dtype, device="cuda"))

    def forward_with_kv_cache(self, samples, action):
        '''
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        '''
        assert samples.shape[1] == 1

        action = F.one_hot(action.long(), self.action_dim).float()
        feats = self.stem(torch.cat([samples, action], dim=-1))
        feats = self.position_encoding.forward_with_position(feats, position=self.key_cache_list[0].shape[2])
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            feats, attn = layer.forward_with_kv_cache(feats, self.key_cache_list, self.value_cache_list, idx)

            if self.max_cache_length is not None and self.key_cache_list[idx].shape[1] > self.max_cache_length:
                self.key_cache_list[idx] = self.key_cache_list[idx][:, -self.max_cache_length:, :]
                self.value_cache_list[idx] = self.value_cache_list[idx][:, -self.max_cache_length:, :]

        return feats