import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        ## Construct a linear transformation module. This function should accept the following parameters:
        self.in_features = in_features ## final dimension of the input
        self.out_features = out_features ## final dimension of the output
        self.device = device ## Device to store the parameters on
        self.dtype = dtype ## Data type of the parameters

        w = torch.empty(out_features, in_features)
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
        
        w = torch.empty(num_embeddings, embedding_dim)
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

        self.weights = nn.Parameter(torch.ones(d_model))
        
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
        