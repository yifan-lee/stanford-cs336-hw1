import torch
import torch.nn as nn
from einops import rearrange, einsum


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
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0.0, std=std.item(),a=-3*std.item(),b=3*std.item()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Apply the linear transformation to the input
        output = einsum(
            x, self.weight,
            "... in_dim, out_dim in_dim -> ... out_dim"
        )
        return output