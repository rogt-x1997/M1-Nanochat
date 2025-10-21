"""
Google Colab Handler
Generates executable Colab notebooks with complete training pipeline
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ColabJobConfig:
    """Configuration for a Colab training job."""
    job_id: str
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    sequence_len: int
    learning_rate: float
    batch_size: int
    max_iters: int
    eval_interval: int
    warmup_steps: int
    checkpoint_dir: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ColabHandler:
    """
    Handler for Google Colab training jobs.
    Generates complete, executable Colab notebooks.
    """
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.drive_mount_path = "/content/drive/MyDrive/nanochat"
    
    def generate_notebook(self, config: ColabJobConfig) -> Dict[str, Any]:
        """
        Generate a complete Colab notebook in JSON format.
        
        Args:
            config: Job configuration
            
        Returns:
            Notebook dictionary (can be saved as .ipynb)
        """
        cells = []
        
        # Cell 1: Setup and dependencies
        cells.append(self._create_code_cell(f"""# Nanochat Training - Job {config.job_id}
# Generated: {config.timestamp}

# Install dependencies
!pip install -q torch numpy regex requests tqdm

print("✓ Dependencies installed")
"""))
        
        # Cell 2: Mount Google Drive
        cells.append(self._create_code_cell("""# Mount Google Drive for checkpoint storage
from google.colab import drive
drive.mount('/content/drive')

import os
project_dir = "/content/drive/MyDrive/nanochat"
os.makedirs(project_dir, exist_ok=True)
os.makedirs(f"{project_dir}/checkpoints", exist_ok=True)
os.makedirs(f"{project_dir}/models", exist_ok=True)
os.makedirs(f"{project_dir}/data", exist_ok=True)
os.makedirs(f"{project_dir}/logs", exist_ok=True)

print(f"✓ Google Drive mounted at {project_dir}")
"""))
        
        # Cell 3: GPU check and imports
        cells.append(self._create_code_cell("""# Check GPU and import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"PyTorch version: {torch.__version__}")
"""))
        
        # Cell 4: Tokenizer implementation
        cells.append(self._create_code_cell(self._get_tokenizer_code()))
        
        # Cell 5: Model implementation
        cells.append(self._create_code_cell(self._get_model_code(config)))
        
        # Cell 6: Training functions
        cells.append(self._create_code_cell(self._get_training_code(config)))
        
        # Cell 7: Data preparation
        cells.append(self._create_code_cell("""# Prepare training data
print("Preparing training data...")

# Create sample dataset
sample_text = \"\"\"
Once upon a time, there was a little girl named Lucy. Lucy loved to play in the park.
One day, Lucy found a magic stone. The stone was shiny and beautiful.
Lucy took the stone home and showed it to her mom. Her mom smiled and said it was special.
That night, Lucy dreamed of magical adventures. She flew through the sky and met friendly dragons.
When she woke up, Lucy felt happy. She knew that every day could be an adventure.

There was a boy named Tom. Tom liked to build things with blocks.
He built tall towers and long bridges. His favorite color was blue.
One sunny day, Tom went to the beach. He built a big sandcastle.
The waves came and washed it away, but Tom just laughed and built another one.
Tom learned that it's fun to create new things every day.

A cat named Whiskers lived in a cozy house. Whiskers loved to chase butterflies.
In the garden, there were many colorful flowers. Whiskers would play among them.
Sometimes Whiskers would take naps in the warm sunshine.
Life was good for Whiskers the cat.
\"\"\" * 200  # Repeat for more data

# Train tokenizer
print("Training tokenizer...")
tokenizer = SimpleTokenizer()
tokenizer.train(sample_text)
print(f"Vocabulary size: {len(tokenizer)}")

# Save tokenizer
tokenizer_path = f"{project_dir}/models/tokenizer.pkl"
tokenizer.save(tokenizer_path)
print(f"✓ Tokenizer saved to {tokenizer_path}")

# Tokenize data
tokens = tokenizer.encode(sample_text)
data = torch.tensor(tokens, dtype=torch.long)
print(f"Total tokens: {len(data):,}")

# Split train/val
split_idx = int(len(data) * 0.95)
train_data = data[:split_idx]
val_data = data[split_idx:]
print(f"Train tokens: {len(train_data):,}")
print(f"Val tokens: {len(val_data):,}")
"""))
        
        # Cell 8: Initialize and train model
        cells.append(self._create_code_cell(f"""# Initialize model and train
print("Initializing model...")
config = GPTConfig(
    vocab_size=len(tokenizer),
    n_layer={config.n_layer},
    n_head={config.n_head},
    n_embd={config.n_embd},
    sequence_len={config.sequence_len}
)

model = GPT(config).to(device)
print(f"Model parameters: {{model.get_num_params():,}}")

# Training configuration
training_config = {{
    'learning_rate': {config.learning_rate},
    'batch_size': {config.batch_size},
    'seq_length': {config.sequence_len},
    'max_iters': {config.max_iters},
    'eval_interval': {config.eval_interval},
    'eval_iters': 10,
    'gradient_accumulation_steps': 1,
    'warmup_steps': {config.warmup_steps},
    'checkpoint_dir': f'{{project_dir}}/checkpoints',
    'log_dir': f'{{project_dir}}/logs'
}}

# Train model
print("\\nStarting training...")
engine = TrainingEngine(model, train_data, val_data, training_config, device)
stats = engine.train()

print("\\n" + "="*60)
print("Training Complete!")
print(f"Best validation loss: {{stats['best_val_loss']:.4f}}")
print(f"Total time: {{stats['total_time']:.1f}}s")
print("="*60)
"""))
        
        # Cell 9: Inference and text generation
        cells.append(self._create_code_cell("""# Test text generation
print("\\nTesting text generation...")

model.eval()

def generate_text(prompt, max_tokens=100, temperature=0.8):
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    
    generated = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=20
    )
    
    return tokenizer.decode(generated[0].tolist())

# Generate samples
test_prompts = ["Once upon a time", "There was a", "The cat"]

print("\\nGenerated samples:")
print("="*60)
for prompt in test_prompts:
    generated = generate_text(prompt, max_tokens=50, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print("-"*60)

print("\\n✓ All done! Model and checkpoints saved to Google Drive")
print(f"Checkpoint location: {project_dir}/checkpoints/best_model.pt")
"""))
        
        # Create notebook structure
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "name": f"nanochat_training_{config.job_id}.ipynb",
                    "provenance": [],
                    "gpuType": "T4",
                    "accelerator": "GPU"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "cells": cells
        }
        
        return notebook
    
    def _create_code_cell(self, code: str) -> Dict[str, Any]:
        """Create a code cell for the notebook."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.split('\n')
        }
    
    def _get_tokenizer_code(self) -> str:
        """Get tokenizer implementation code."""
        return """# Simple Character-Level Tokenizer
import pickle

class SimpleTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def train(self, text):
        unique_chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(unique_chars)
    
    def encode(self, text):
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens):
        return ''.join(self.idx_to_char.get(idx, '') for idx in tokens)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.vocab_size = data['vocab_size']
    
    def __len__(self):
        return self.vocab_size

print("✓ Tokenizer class defined")
"""
    
    def _get_model_code(self, config: ColabJobConfig) -> str:
        """Get model implementation code."""
        return """# GPT Model Implementation
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 5000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    sequence_len: int = 128
    dropout: float = 0.1
    bias: bool = False

def norm(x):
    # RMSNorm: x / sqrt(mean(x^2) + eps) - Compatible with all PyTorch versions
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, None, :])
        self.register_buffer('sin_cached', emb.sin()[None, :, None, :])
    
    def forward(self, seq_len):
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos[..., :d] - x2 * sin[..., :d]
    y2 = x1 * sin[..., :d] + x2 * cos[..., :d]
    return torch.cat([y1, y2], dim=-1).to(x.dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
    
    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                          dropout_p=self.dropout if self.training else 0.0, 
                                          is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.rope = RotaryEmbedding(config.n_embd // config.n_head, config.sequence_len)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.wte(idx)
        x = norm(x)
        cos_sin = self.rope(T)
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.sequence_len else idx[:, -self.config.sequence_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

print("✓ Model classes defined")
"""
    
    def _get_training_code(self, config: ColabJobConfig) -> str:
        """Get training engine code."""
        return """# Training Engine
class TrainingEngine:
    def __init__(self, model, train_data, val_data, config, device):
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = device
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.seq_length = config['seq_length']
        self.max_iters = config['max_iters']
        self.eval_interval = config['eval_interval']
        self.eval_iters = config['eval_iters']
        self.warmup_steps = config['warmup_steps']
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.iteration = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.max_iters - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + np.cos(progress * np.pi)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.seq_length - 1, (self.batch_size,))
        x = torch.stack([data[i:i + self.seq_length] for i in ix])
        y = torch.stack([data[i + 1:i + self.seq_length + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
    
    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = {}
        for split in ['train', 'val']:
            batch_losses = []
            for _ in range(self.eval_iters):
                x, y = self.get_batch(split)
                _, loss = self.model(x, y)
                batch_losses.append(loss.item())
            losses[split] = sum(batch_losses) / len(batch_losses)
        self.model.train()
        return losses
    
    def train(self):
        self.model.train()
        start_time = time.time()
        for iter_num in range(self.max_iters):
            self.iteration = iter_num
            x, y = self.get_batch('train')
            _, loss = self.model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            if iter_num % self.eval_interval == 0 or iter_num == self.max_iters - 1:
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                print(f"Iter {iter_num:6d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s")
                
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    checkpoint_path = self.checkpoint_dir / 'best_model.pt'
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'iteration': iter_num,
                        'best_val_loss': self.best_val_loss,
                        'config': self.config
                    }, checkpoint_path)
                    print(f"  ✓ Best model saved")
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_time': time.time() - start_time,
            'iterations': self.max_iters
        }

print("✓ Training engine defined")
"""
    
    def submit_job(self, config: ColabJobConfig) -> str:
        """
        Create a Colab training job.
        
        Args:
            config: Job configuration
            
        Returns:
            Job ID
        """
        job_id = config.job_id
        
        # Generate notebook
        notebook = self.generate_notebook(config)
        
        # Store job info
        self.jobs[job_id] = {
            'config': asdict(config),
            'status': 'ready_to_run',
            'created_at': datetime.now().isoformat(),
            'backend': 'colab',
            'checkpoint_path': f"{self.drive_mount_path}/checkpoints/best_model.pt",
            'notebook': notebook
        }
        
        print(f"✓ Colab job created: {job_id}")
        return job_id
    
    def get_notebook(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get notebook for a job."""
        job = self.jobs.get(job_id)
        if job:
            return job['notebook']
        return None
    
    def save_notebook(self, job_id: str, output_path: str) -> bool:
        """
        Save notebook to file.
        
        Args:
            job_id: Job ID
            output_path: Path to save notebook
            
        Returns:
            True if successful
        """
        notebook = self.get_notebook(job_id)
        if notebook:
            with open(output_path, 'w') as f:
                json.dump(notebook, f, indent=2)
            print(f"✓ Notebook saved to {output_path}")
            return True
        return False
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        return self.jobs.get(job_id, {'status': 'not_found'})
    
    def get_all_jobs(self) -> list:
        """Get all jobs."""
        return list(self.jobs.values())


def create_colab_handler():
    """Factory function to create Colab handler."""
    return ColabHandler()

