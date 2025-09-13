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

        self.weights = nn.Parameter(
            torch.empty(out_features, in_features, device=self.device, dtype=self.dtype)
        )
        std = torch.sqrt(torch.tensor(2.0/(in_features+out_features)))
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std.item(), a=-3*std.item(), b=3*std.item())
    
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
        
        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype)
        )
        std = 1.0
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std,a=-3,b=3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ## Lookup the embedding vectors for the given token IDs.
        return self.weights[token_ids]
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
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
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model ## Hidden dimension of the model
        self.device = device
        self.dtype = dtype
        if d_ff is None:
            q = round(d_model*8/3/64)
            self.d_ff = q*64
        else:
            self.d_ff = d_ff
        
        self.w1_weight = Linear(self.d_model,self.d_ff, device=self.device, dtype=self.dtype)
        self.w2_weight = Linear(self.d_ff, self.d_model, device=self.device, dtype=self.dtype)
        self.w3_weight = Linear(self.d_model,self.d_ff, device=self.device, dtype=self.dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1_weight(x)
        w3x = self.w3_weight(x)
        SiLUw1x = w1x*torch.sigmoid(w1x)
        part2 = SiLUw1x * w3x
        result = self.w2_weight(part2)
        return result
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, rope_theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        ## Construct the RoPE module and create buffers if needed.
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even head dimension (pairs of features)"
        self.rope_theta = rope_theta ## $\\Theta$ value for the RoPE
        self.d_k = d_k ## dimension of query and key vectors
        self.max_seq_len = max_seq_len ## Maximum sequence length that will be inputted
        self.device = device ## Device to store the buffer on

        dim_index = torch.arange(self.d_k // 2, device=self.device, dtype=torch.float32)
        position_index = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        theta_inv_index = self.rope_theta**(-2*dim_index/d_k)
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
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_subtract_max = x-x_max
    x_subtract_max_exp = torch.exp(x_subtract_max)
    x_subtract_max_exp_sum = torch.sum(x_subtract_max_exp, dim=dim, keepdim=True)
    y = x_subtract_max_exp/x_subtract_max_exp_sum
    return y

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    QK = einsum(
        Q, K, "... n d_k, ... m d_k -> ... n m"
    )
    QK_scaled = QK/torch.tensor(d_k).sqrt()
    if mask is not None:
        M = torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))
        QK_scaled += M
    QK_scaled_softmax = softmax(QK_scaled, Q.dim()-1)
    y = einsum(
        QK_scaled_softmax, V, "... n m, ... m d_v -> ... n d_v"
    )
    return y



class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                rope_theta: float|None=None, max_seq_len: int|None=None,
                device: torch.device | None = None,dtype: torch.dtype | None = None,):
        super().__init__()
        ## mplement causal multi-head self-attention
        self.d_model = d_model ## final dimension of the input
        self.num_heads = num_heads ## number of heads
        self.device = device ## Device to store the parameters on
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        
        assert d_model%num_heads == 0, "d_model/num_heads need to be int"
        self.d_k = d_model//num_heads
        self.d_v = d_model//num_heads
        
        self.W_q = Linear(d_model,self.d_k * num_heads,device,dtype)
        self.W_k = Linear(d_model,self.d_k * num_heads,device,dtype)
        self.W_v = Linear(d_model,self.d_v * num_heads,device,dtype)
        self.W_o = Linear(self.d_k * num_heads,d_model,device,dtype)
        
        
        self.rope = None
        if (rope_theta is not None) and (max_seq_len is not None):
            self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len,device,dtype)
    
    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor|None=None) -> torch.Tensor:
        seq_len = in_features.shape[-2]
        mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool))
        Q = self.W_q(in_features)
        Q_head = rearrange(
            Q, "... seq_len (n d_k) -> ... n seq_len d_k", n = self.num_heads
        )
        K = self.W_k(in_features)
        K_head = rearrange(
            K, "... seq_len (n d_k) -> ... n seq_len d_k", n = self.num_heads
        )
        if self.rope is not None:
            if token_positions is not None:
                position = repeat(
                    token_positions, " ... seq_len -> ... n seq_len", n = self.num_heads
                )
            else:
                position = torch.arange(seq_len)
                position_expend_shape = (Q_head.shape[:-1])
                position = position.expand(position_expend_shape)
            Q_head = self.rope(Q_head, position)
            K_head = self.rope(K_head, position)
        V = self.W_v(in_features)
        V_head = rearrange(
            V, "... seq_len (n d_v) -> ... n seq_len d_v", n = self.num_heads
        )
        expend_shape = (*Q_head.shape[:-1], seq_len)
        mask_boardcasted = mask.expand(expend_shape)
        head = scaled_dot_product_attention(Q_head,K_head,V_head,mask_boardcasted)
        head = rearrange(
            head, "... n seq_len d_v -> ... seq_len (n d_v)"
        )
        attention = self.W_o(head)
        return attention
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                rope_theta: float|None=None, max_seq_len: int|None=None,
                device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        
        self.rms_norm1 = RMSNorm(d_model, device = device, dtype = dtype)
        self.rms_norm2 = RMSNorm(d_model, device = device, dtype = dtype)
        self.mha = MultiheadSelfAttention(
            d_model, 
            num_heads, 
            rope_theta=rope_theta,
            max_seq_len=max_seq_len, 
            device = device, 
            dtype = dtype
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff ,device = device, dtype = dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.mha(self.rms_norm1(x))
        x += self.ffn(self.rms_norm2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
        d_model: int, num_heads: int, rope_theta: float, d_ff:int,
        device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.device = device
        self.dtype = dtype
        
        
        self.embedding = Embedding(
            num_embeddings = vocab_size,
            embedding_dim = d_model,
            device=device, dtype=dtype
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device, dtype)
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model,
            device=device, dtype=dtype)
        self.linear = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device, dtype=dtype
        )
        
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(in_indices)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        x = self.linear(x)
        return x
        
        
def cross_entropy(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    inputs_subtract_max = inputs - inputs_max
    inputs_exp = inputs_subtract_max.exp()
    inputs_exp_sum = torch.sum(inputs_exp, dim=-1, keepdim=True)

    target_expanded = rearrange(target, 'n -> n 1')
    target_logits = torch.gather(inputs_subtract_max, 1, target_expanded)
    
    loss = - (target_logits - torch.log(inputs_exp_sum))
    return loss.mean()