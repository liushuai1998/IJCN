# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        
        # Using Attn Mask to distinguish different subwindows
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]



class ChannelMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(ChannelMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim

        self.channel_pos = nn.Parameter(torch.zeros(1, 1, 1, input_dim))
        trunc_normal_(self.channel_pos, std=.02)
        
        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x + self.channel_pos
        
        qkv = self.qkv(x).reshape(B, H*W, 3, self.n_heads, C // self.n_heads)
        qkv = rearrange(qkv, 'b hw three n d -> three b n d hw')
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v)
        x = rearrange(x, 'b n d hw -> b hw (n d)')
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        return self.proj(x)


class ParallelBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super(ParallelBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        
        self.ln_channel = nn.LayerNorm(input_dim)
        self.channel_msa = ChannelMSA(input_dim, input_dim, head_dim, window_size, self.type)
        
        self.fusion = nn.Sequential(nn.Linear(input_dim * 2, input_dim), nn.GELU())
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 2 * input_dim), nn.GELU(), nn.Linear(2 * input_dim, output_dim))

    def forward(self, x):
        identity = x
        x1 = self.ln1(x)
        x1 = self.msa(x1)
        x1 = x + self.drop_path(x1)
        
        x2 = self.ln_channel(x)
        x2 = self.channel_msa(x2)
        x2 = x + self.drop_path(x2)
        concat_features = torch.cat([x1, x2], dim=-1)
        fused = self.fusion(concat_features)
        x = identity + fused

        x = x + self.drop_path(self.mlp(self.ln2(x)))
        
        return x


class DJViT(nn.Module):
    def __init__(self, dct_size=16, channel=320, patch_size=1, window_size=4, head_dim=32, depth=4, drop_path_rate=0.1):
        super(DJViT, self).__init__()
        self.dct_size = dct_size
        self.channel = channel
        self.patch_size = patch_size 
        self.head_dim = head_dim
        self.window_size = window_size
        self.depth = depth
        
        self.input_resolution = dct_size // patch_size
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        
        self.blocks = nn.ModuleList([
            ParallelBlock(
                input_dim=channel,
                output_dim=channel,
                head_dim=self.head_dim,
                window_size=self.window_size,
                drop_path=dpr[i],
                type='W' if i % 2 == 0 else 'SW',
                input_resolution=self.input_resolution
            ) for i in range(self.depth)
        ])
        
        self.norm = nn.LayerNorm(channel)
        
        self.conv_out = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input: [bs, 320, H/16, W/16]
        x = rearrange(x, 'b c h w -> b h w c')
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv_out(x)

        return x