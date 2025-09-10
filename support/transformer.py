import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce, repeat


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        ## Construct a linear transformation module. This function should accept the following parameters:
        self.in_features = in_features ## final dimension of the input
        self.out_features = out_features ## final dimension of the output
        self.device = device ## Device to store the parameters on
        self.dtype = dtype ## Data type of the parameters

        w = torch.empty(out_features, in_features, device=self.device, dtype=self.dtype)
        std = torch.sqrt(torch.tensor(2.0/(in_features+out_features)))
        self.weights = nn.Parameter(nn.init.trunc_normal_(w, mean=0.0, std=std.item(),a=-3*std.item(),b=3*std.item()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Apply the linear transformation to the input
        output = einsum(
            x, self.weights,
            "... in_dim, out_dim in_dim -> ... out_dim"
        )
        return output
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        ## Construct an embedding module
        super().__init__()
        self.num_embeddings = num_embeddings ## Size of the vocabulary
        self.embedding_dim = embedding_dim ## Dimension of the embedding vectors
        self.device = device ## Device to store the parameters on
        self.dtype = dtype ## Data type of the parameters
        
        w = torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype)
        std = 1.0
        self.weights = nn.Parameter(nn.init.trunc_normal_(w, mean=0.0, std=std,a=-3,b=3))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ## Lookup the embedding vectors for the given token IDs.
        return self.weights[token_ids]
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        ## Construct the RMSNorm module.
        super().__init__()
        self.d_model = d_model ## Hidden dimension of the model
        self.eps = eps ## Epsilon value for numerical stability
        self.device = device ## Device to store the parameters on
        self.dtype = dtype ## Data type of the parameters

        self.weights = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Process an input tensor of shape
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squaremean = reduce(
            x**2, "... d_model -> ... 1", 'mean'
        )
        x_RMS = (x_squaremean+self.eps).sqrt()
        result = x / x_RMS * self.weights
        return result.to(in_dtype)
    
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        self.d_model = d_model ## Hidden dimension of the model
        if d_ff is None:
            q = round(d_model*8/3/64)
            self.d_ff = q*64
        else:
            self.d_ff = d_ff
        
        self.w1_weight = nn.Parameter(torch.randn(self.d_ff, self.d_model))
        self.w2_weight = nn.Parameter(torch.randn(self.d_model, self.d_ff))
        self.w3_weight = nn.Parameter(torch.randn(self.d_ff, self.d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(
            self.w1_weight, x,
            "d_ff d_model, ... d_model -> ... d_ff"
        )
        w3x = einsum(
            self.w3_weight, x,
            "d_ff d_model, ... d_model -> ... d_ff"
        )
        SiLUw1x = w1x*torch.sigmoid(w1x)
        part2 = SiLUw1x * w3x
        result = einsum(
            self.w2_weight, part2,
            "d_model d_ff, ... d_ff -> ... d_model"
        )
        return result
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        ## Construct the RoPE module and create buffers if needed.
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even head dimension (pairs of features)"
        self.theta = theta ## $\\Theta$ value for the RoPE
        self.d_k = d_k ## dimension of query and key vectors
        self.max_seq_len = max_seq_len ## Maximum sequence length that will be inputted
        self.device = device ## Device to store the buffer on

        dim_index = torch.arange(self.d_k // 2, dtype=torch.float32)
        position_index = torch.arange(self.max_seq_len, dtype=torch.float32)
        theta_inv_index = self.theta**(-2*dim_index/d_k)
        theta_ik = einsum(
            position_index, theta_inv_index,
            "s, d -> s d"
        )
        sin = torch.sin(theta_ik)
        cos = torch.cos(theta_ik)
        
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        assert x.shape[-1] == self.d_k, "The last dim of input should be equal to dim of embedding."
        assert x.shape[-2] == token_positions.shape[-1], "token_positions length must match sequence length"
        sin_expend = self.sin[token_positions]
        cos_expend = self.cos[token_positions]

        x_even = x[...,::2]
        x_odd = x[...,1::2]

        y_even = x_even*cos_expend-x_odd*sin_expend
        y_odd = x_even*sin_expend+x_odd*cos_expend
        y = rearrange(torch.stack([y_even, y_odd], dim=-1), '... s d two -> ... s (d two)')
        return y