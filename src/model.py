"""
GPT Model Implementation
Based on Nanochat architecture with modern features
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 5000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    sequence_len: int = 128
    dropout: float = 0.1
    bias: bool = False  # No bias in linear layers (Nanochat style)


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm without learnable parameters (Nanochat style).
    Compatible with all PyTorch versions.
    
    Args:
        x: Input tensor
        
    Returns:
        Normalized tensor
    """
    # RMSNorm: x / sqrt(mean(x^2) + eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Provides relative positional information without learned parameters.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor [B, T, H, D]
        cos: Cosine embeddings [1, T, 1, D]
        sin: Sine embeddings [1, T, 1, D]
        
    Returns:
        Tensor with rotary embeddings applied
    """
    assert x.ndim == 4  # [B, T, H, D]
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos[..., :d] - x2 * sin[..., :d]
    y2 = x1 * sin[..., :d] + x2 * cos[..., :d]
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with:
    - Rotary position embeddings
    - QK normalization
    - Optional Multi-Query Attention (MQA)
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Query, Key, Value projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C]
            cos_sin: Tuple of (cos, sin) rotary embeddings
            
        Returns:
            Output tensor [B, T, C]
        """
        B, T, C = x.size()
        
        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)  # [B, T, H, D]
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)  # [B, T, H, D]
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)  # [B, T, H, D]
        
        # Apply rotary embeddings
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # QK normalization (Nanochat style)
        q = norm(q)
        k = norm(k)
        
        # Transpose for attention: [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Causal self-attention using Flash Attention when available
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with ReLU² activation (Nanochat style).
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            Output tensor [B, T, C]
        """
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU² activation (Nanochat style)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with pre-norm architecture.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, C]
            cos_sin: Rotary embeddings
            
        Returns:
            Output tensor [B, T, C]
        """
        # Pre-norm architecture
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model with Nanochat architecture features:
    - Rotary position embeddings (no learned positional embeddings)
    - RMSNorm without learnable parameters
    - QK normalization
    - ReLU² activation
    - Untied token embedding and LM head weights
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Rotary position embeddings
        self.rope = RotaryEmbedding(
            dim=config.n_embd // config.n_head,
            max_seq_len=config.sequence_len
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Language model head (untied from token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"✓ GPT model initialized: {n_params:,} parameters")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional, for training)
            
        Returns:
            Tuple of (logits, loss)
            - logits: [B, T, vocab_size]
            - loss: scalar (if targets provided)
        """
        B, T = idx.size()
        assert T <= self.config.sequence_len, f"Sequence length {T} exceeds maximum {self.config.sequence_len}"
        
        # Token embeddings
        x = self.wte(idx)  # [B, T, C]
        
        # Apply initial norm (Nanochat style)
        x = norm(x)
        
        # Get rotary embeddings for current sequence length
        cos_sin = self.rope(T)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        
        # Final norm
        x = norm(x)
        
        # Language model head
        logits = self.lm_head(x)  # [B, T, vocab_size]
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = no filtering)
            
        Returns:
            Generated token indices [B, T + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.sequence_len else idx[:, -self.config.sequence_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())

