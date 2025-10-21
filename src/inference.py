"""
Inference Engine for Text Generation
Handles loading models and generating text with various sampling strategies
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
from pathlib import Path


class InferenceEngine:
    """
    Inference engine for text generation from trained GPT models.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda'
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Trained GPT model
            tokenizer: Tokenizer instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.tokenizer = tokenizer
        self.device = device
        
        print(f"✓ Inference engine initialized")
        print(f"  Device: {device}")
        print(f"  Model parameters: {model.get_num_params():,}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (None = disabled)
            top_p: Nucleus sampling threshold (None = disabled)
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            stop_tokens: List of tokens that stop generation
            
        Returns:
            Generated text
        """
        # Encode prompt
        if prompt:
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = [0]  # Start with a default token if no prompt
        
        # Convert to tensor
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = []
        
        # Generate tokens
        for _ in range(max_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.model.config.sequence_len else idx[:, -self.model.config.sequence_len:]
            
            # Forward pass
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]  # Get logits for last position
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and generated_tokens:
                for token in set(generated_tokens):
                    logits[0, token] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            generated_tokens.append(idx_next.item())
            
            # Check for stop tokens
            if stop_tokens:
                current_text = self.tokenizer.decode(idx[0].tolist())
                if any(stop in current_text for stop in stop_tokens):
                    break
        
        # Decode to text
        generated_text = self.tokenizer.decode(idx[0].tolist())
        
        return generated_text
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            generated = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            results.append(generated)
        
        return results
    
    @torch.no_grad()
    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of the model on given text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score
        """
        # Encode text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return float('inf')
        
        # Convert to tensor
        data = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        # Compute loss in chunks
        seq_len = self.model.config.sequence_len
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(0, len(data) - 1, seq_len):
            # Get chunk
            chunk_len = min(seq_len, len(data) - i - 1)
            x = data[i:i + chunk_len].unsqueeze(0)
            y = data[i + 1:i + chunk_len + 1].unsqueeze(0)
            
            # Forward pass
            _, loss = self.model(x, y)
            
            total_loss += loss.item() * chunk_len
            total_tokens += chunk_len
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded from {checkpoint_path}")
        
        if 'iteration' in checkpoint:
            print(f"  Trained for {checkpoint['iteration']} iterations")
        if 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")


def create_inference_engine(model, tokenizer, device='cuda'):
    """
    Factory function to create an inference engine.
    
    Args:
        model: GPT model
        tokenizer: Tokenizer instance
        device: Device to run on
        
    Returns:
        InferenceEngine instance
    """
    return InferenceEngine(model, tokenizer, device)


def load_model_for_inference(checkpoint_path: str, model, tokenizer, device='cuda'):
    """
    Load a model from checkpoint and create inference engine.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: GPT model (architecture must match)
        tokenizer: Tokenizer instance
        device: Device to run on
        
    Returns:
        InferenceEngine instance with loaded model
    """
    engine = create_inference_engine(model, tokenizer, device)
    engine.load_checkpoint(checkpoint_path)
    return engine

