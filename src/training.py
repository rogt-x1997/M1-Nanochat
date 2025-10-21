"""
Training Engine for GPT Model
Handles the complete training loop with logging, checkpointing, and evaluation
"""

import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime


class TrainingEngine:
    """
    Training engine for GPT models.
    Handles training loop, optimization, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize training engine.
        
        Args:
            model: GPT model to train
            train_data: Training data tensor
            val_data: Validation data tensor
            config: Training configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.device = device
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 32)
        self.seq_length = config.get('seq_length', 128)
        self.max_iters = config.get('max_iters', 10000)
        self.eval_interval = config.get('eval_interval', 500)
        self.eval_iters = config.get('eval_iters', 10)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.warmup_steps = config.get('warmup_steps', 100)
        
        # Directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.log_dir = Path(config.get('log_dir', './logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.iteration = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'iteration': []
        }
        
        print(f"✓ Training engine initialized")
        print(f"  Device: {device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max iterations: {self.max_iters}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.01},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        
        return optimizer
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create cosine annealing learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            # Warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_iters - self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def get_batch(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            Tuple of (input_ids, target_ids)
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Random starting positions
        ix = torch.randint(len(data) - self.seq_length - 1, (self.batch_size,))
        
        # Create input and target sequences
        x = torch.stack([data[i:i + self.seq_length] for i in ix])
        y = torch.stack([data[i + 1:i + self.seq_length + 1] for i in ix])
        
        return x.to(self.device), y.to(self.device)
    
    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Estimate loss on train and val sets.
        
        Returns:
            Dictionary with 'train' and 'val' losses
        """
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
    
    def train_step(self) -> float:
        """
        Perform a single training step.
        
        Returns:
            Training loss for this step
        """
        # Get batch
        x, y = self.get_batch('train')
        
        # Forward pass
        _, loss = self.model(x, y)
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Optimizer step (only every gradient_accumulation_steps)
        if (self.iteration + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Scheduler step
            self.scheduler.step()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def save_checkpoint(self, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_iter_{self.iteration}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved: {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from iteration {self.iteration}")
    
    def save_training_log(self) -> None:
        """Save training history to JSON file."""
        log_path = self.log_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        # Convert any tensors to Python floats
        history_serializable = {
            k: [float(v) if torch.is_tensor(v) else v for v in vals]
            for k, vals in self.training_history.items()
        }
        with open(log_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
    
    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.
        
        Returns:
            Training statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {self.max_iters} iterations")
        print(f"{'='*60}\n")
        
        self.model.train()
        start_time = time.time()
        
        for iter_num in range(self.iteration, self.max_iters):
            self.iteration = iter_num
            
            # Training step
            train_loss = self.train_step()
            
            # Evaluation and logging
            if iter_num % self.eval_interval == 0 or iter_num == self.max_iters - 1:
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                
                # Update history
                self.training_history['iteration'].append(iter_num)
                self.training_history['train_loss'].append(losses['train'])
                self.training_history['val_loss'].append(losses['val'])
                self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print progress
                print(f"Iter {iter_num:6d} | "
                      f"Train Loss: {losses['train']:.4f} | "
                      f"Val Loss: {losses['val']:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {elapsed:.1f}s")
                
                # Save checkpoint if best
                is_best = losses['val'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint(is_best=True)
                
                # Regular checkpoint
                if iter_num % (self.eval_interval * 5) == 0:
                    self.save_checkpoint(is_best=False)
        
        # Final checkpoint and log
        self.save_checkpoint(is_best=False)
        self.save_training_log()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        print(f"  Final learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'iterations': self.max_iters,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1]
        }


def create_training_engine(model, train_data, val_data, config, device='cuda'):
    """
    Factory function to create a training engine.
    
    Args:
        model: GPT model
        train_data: Training data tensor
        val_data: Validation data tensor
        config: Training configuration
        device: Device to train on
        
    Returns:
        TrainingEngine instance
    """
    return TrainingEngine(model, train_data, val_data, config, device)

