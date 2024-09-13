import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def create_pad_mask(tensor, padding_value=0):
    """
    Create a padding mask tensor for a given tensor.

    Args:
    - tensor (torch.Tensor): The input tensor with shape [batch, sequence_length, token_embedding_size].
    - padding_value (int): The padding value used in the tensor. Default is 0.

    Returns:
    - torch.Tensor: A padding mask tensor with shape [batch, sequence_length, 1],
      where pad positions are marked with False and other positions are marked with True.
    """
    # Create a tensor that is True where the tensor is not equal to the padding value
    not_padding = tensor != padding_value

    # Check if at least one element in each sequence is not equal to the padding value
    # If at least one element is not padding, the sequence is marked with True
    pad_mask = not_padding.any(dim=-1, keepdim=True)
    
    # Convert the mask to a boolean tensor
    pad_mask = pad_mask.bool()
    
    # Expand the mask to match the tensor shape by adding an extra dimension at the end
    pad_mask = pad_mask.all(dim=-1).unsqueeze(-1)
    # pad_mask = pad_mask.unsqueeze(-1)
    
    return pad_mask

def attention(query:Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    sqrt_dim_head = query.shape[-1]**0.5

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head
    # Shape of scores [batch_size, num_heads, sequence_length, sequence_length]

    if mask is not None:
        scores = scores.masked_fill(mask==0, -5e4)

    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_embed: int, max_len: int=1024, drop_prob: float =0.1) -> None:
        super(PositionalEncoding, self).__init__()

        assert dim_embed % 2 == 0

        self.dim_embed = dim_embed
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        dim_pair = torch.arange(0, dim_embed, 2)
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / dim_embed))

        pe = torch.zeros(max_len, dim_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_positions, dim_embed)
        pe = pe.unsqueeze(0)

        # Register as non-learnable parameters
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(drop_prob)


    def forward(self, x: Tensor):
        x = x + self.pe[:, :self.max_len]/10
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_embed: int, drop_prob: float) -> None:
        super().__init__()
        assert dim_embed % num_heads == 0

        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        self.query  = nn.Linear(dim_embed, dim_embed)
        self.key    = nn.Linear(dim_embed, dim_embed)
        self.value  = nn.Linear(dim_embed, dim_embed)
        self.output = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor=None) -> Tensor:
        query   = self.query(x)
        key     = self.key(y)
        value   = self.value(y)

        batch_size = x.size(0)
        query   = query .view(batch_size, -1, self.num_heads, self.dim_head)
        key     = key   .view(batch_size, -1, self.num_heads, self.dim_head)
        value   = value .view(batch_size, -1, self.num_heads, self.dim_head)

        # Reshape into the number of heads (batch_size, num_heads, -1, dim_head)
        query   = query.transpose(1,2)
        key     = key.transpose(1,2)
        value   = value.transpose(1,2)

        if mask is not None:
            # Give the mask one extra dimension to be broadcastable across multiple heads
            mask = mask.unsqueeze(1)

        attn = attention(query, key, value, mask) # [batch_size, num_heads, sequence_length_q, sequence_length_k]
        attn = attn.transpose(1, 2) # [batch_size, sequence_length_q, num_heads, sequence_length_k]
        attn = attn.contiguous().view(batch_size, -1, self.dim_embed) # [batch_size, sequence_length_q, token_embedding_size]

        out = self.dropout(self.output(attn))

        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float) -> None:
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn, bias=False),
            nn.SiLU(),
            # nn.Dropout(drop_prob),
            nn.Linear(dim_pffn, dim_embed),
            # nn.Dropout(drop_prob),
        )

    def forward(self, x:Tensor) -> Tensor:
        return self.pffn(x)
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scaler timesteps into vector representations:
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # self.mlp.to('cuda')
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=1000):
        """
        Create sinusodial timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # print(t_freq.device)
        t_emb = self.mlp(t_freq)
        return t_emb
    


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    
class MusTDiffBlock(nn.Module):
    '''
    A MustDiff Block with adaptive layer norm zero (adaLN-Zero) conditioning.
    '''
    def __init__(self,num_heads, dim_embed, mlp_ratio=4.0, drop_prob=0.1)-> None:
        super().__init__()
        # self.norm1 = nn.LayerNorm(dim_embed, elementwise_affine=False, eps=1e-4)
        self.norm1 = RMSNorm(dim_embed, eps=1e-5)
        self.self_atten = MultiHeadAttention(num_heads=num_heads, dim_embed=dim_embed, drop_prob = drop_prob)
        # self.norm2 = nn.LayerNorm(dim_embed, elementwise_affine=False, eps=1e-4)
        self.norm2 = RMSNorm(dim_embed, eps=1e-5)
        dim_pwff = int(dim_embed * mlp_ratio)
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 6 * dim_embed, bias=True)
        )

    def forward(self, x, c, mask=None):
        shift_mha, scale_mha, gate_mha, shift_ffd, scale_ffd, gate_ffd = self.adaLN_modulation(c).chunk(6, dim=1)
        normalized_x_1 = modulate(self.norm1(x), shift_mha, scale_mha)
        x = x + gate_mha.unsqueeze(1) * self.self_atten(normalized_x_1, normalized_x_1, mask)
        normalized_x_2 = modulate(self.norm2(x), shift_ffd, scale_ffd)
        x = x + gate_ffd.unsqueeze(1) * self.feed_forward(normalized_x_2)

        return x

class FinalLayer(nn.Module):
    """
    The final layer of MusTDiff
    """
    def __init__(self, dim_embed):
        super().__init__()
        # self.norm_final = nn.LayerNorm(dim_embed, elementwise_affine=False, eps=1e-4)
        self.norm_final = RMSNorm(dim_embed, eps=1e-5)
        self.linear = nn.Linear(dim_embed, dim_embed, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_embed, 2 * dim_embed, bias=False)
        )
        # self.scale = MinMaxScalingLayer()
    def forward(self,x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # x = self.scale(x)
        # x = torch.sigmoid(x)
        return x
    
class MusTDiff(nn.Module):
    """
    Transformer-based diffusion
    """
    def __init__(
            self,
            dim_embed = 1024, 
            num_heads = 16,
            depth = 28, # N: The number of repetitions of the MusTDiff module
            mlp_ratio = 4.0,
            ):
        super().__init__()
        self.num_heads = num_heads
        # self.layernorm = nn.LayerNorm(dim_embed, elementwise_affine=False, eps=1e-6)
        self.x_embedder = PositionalEncoding(dim_embed)
        self.t_embedder = TimestepEmbedder(dim_embed)
        self.blocks = nn.ModuleList([
            MusTDiffBlock(num_heads=self.num_heads, dim_embed=dim_embed, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(dim_embed)
        self.initialize_weights()


    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.002) # mean = 0 default
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.002)

        # Zero-out adaLN modulation layers in MusTDiff blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Forward pass of MusTDiff.
        x: (batch_size, equence_length, token_embedding_dim) tensor of (noised) latent music representation
        t: (batch_size, ) tensor of diffusion timesteps
        For short
        x: (N, T, D)
        t: (N,)
        """
        # print(f"before embed: {x.shape}")
        mask = create_pad_mask(x)       # (N, T, D)
        # x = self.layernorm(x)
        x = self.x_embedder(x)    # (N, T, D)
        t = self.t_embedder(t)          # (N, D)
        c = t                           # (N, D)
        # print(f"x = self.x_embedder(x), shape: {x.shape}")
        for block in self.blocks:
            x = block(x, c, mask)             # (N, T, D)

        x = self.final_layer(x, c)      # (N, T, D)

        return x
    