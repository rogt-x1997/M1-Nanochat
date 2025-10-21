"""
Data Loader for LLM Training
Handles dataset downloading, tokenization, and efficient batching
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import requests
from tqdm import tqdm


class DataLoader:
    """
    Data loader for LLM training.
    Handles dataset preparation, tokenization, and batching.
    """
    
    def __init__(
        self,
        tokenizer,
        data_dir: str = './data',
        shard_size_mb: int = 500,
        train_split: float = 0.95
    ):
        """
        Initialize data loader.
        
        Args:
            tokenizer: Tokenizer instance
            data_dir: Directory for data storage
            shard_size_mb: Size of each data shard in MB
            train_split: Fraction of data for training (rest for validation)
        """
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_mb = shard_size_mb
        self.train_split = train_split
        
        self.train_data = None
        self.val_data = None
        
        print(f"✓ Data loader initialized")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Train/val split: {train_split:.2%}/{1-train_split:.2%}")
    
    def download_sample_dataset(self, dataset_name: str = 'tinystories') -> str:
        """
        Download a sample dataset for training.
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            Path to downloaded dataset
        """
        print(f"Downloading {dataset_name} dataset...")
        
        if dataset_name == 'tinystories':
            # TinyStories dataset (small, good for testing)
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
            output_path = self.data_dir / 'tinystories.txt'
            
            if output_path.exists():
                print(f"  Dataset already exists: {output_path}")
                return str(output_path)
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                print(f"  ✓ Dataset downloaded: {output_path}")
                return str(output_path)
            
            except Exception as e:
                print(f"  ✗ Download failed: {e}")
                # Create a small sample dataset for testing
                return self._create_sample_dataset()
        
        else:
            print(f"  Unknown dataset: {dataset_name}")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> str:
        """
        Create a small sample dataset for testing.
        
        Returns:
            Path to sample dataset
        """
        print("  Creating sample dataset for testing...")
        
        sample_text = """
        Once upon a time, there was a little girl named Lucy. Lucy loved to play in the park.
        One day, Lucy found a magic stone. The stone was shiny and beautiful.
        Lucy took the stone home and showed it to her mom. Her mom smiled and said it was special.
        That night, Lucy dreamed of magical adventures. She flew through the sky and met friendly dragons.
        When she woke up, Lucy felt happy. She knew that every day could be an adventure.
        The end.
        
        There was a boy named Tom. Tom liked to build things with blocks.
        He built tall towers and long bridges. His favorite color was blue.
        One sunny day, Tom went to the beach. He built a big sandcastle.
        The waves came and washed it away, but Tom just laughed and built another one.
        Tom learned that it's fun to create new things every day.
        
        A cat named Whiskers lived in a cozy house. Whiskers loved to chase butterflies.
        In the garden, there were many colorful flowers. Whiskers would play among them.
        Sometimes Whiskers would take naps in the warm sunshine.
        Life was good for Whiskers the cat.
        """ * 100  # Repeat to make it larger
        
        output_path = self.data_dir / 'sample_dataset.txt'
        with open(output_path, 'w') as f:
            f.write(sample_text)
        
        print(f"  ✓ Sample dataset created: {output_path}")
        return str(output_path)
    
    def prepare_data(self, text_file: str, force_retokenize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training and validation data from text file.
        
        Args:
            text_file: Path to text file
            force_retokenize: Whether to force re-tokenization
            
        Returns:
            Tuple of (train_data, val_data) tensors
        """
        print(f"Preparing data from {text_file}...")
        
        # Check for cached tokenized data
        cache_path = self.data_dir / 'tokenized_data.pt'
        
        if cache_path.exists() and not force_retokenize:
            print("  Loading cached tokenized data...")
            cached = torch.load(cache_path)
            self.train_data = cached['train']
            self.val_data = cached['val']
            print(f"  ✓ Loaded from cache")
            print(f"    Train tokens: {len(self.train_data):,}")
            print(f"    Val tokens: {len(self.val_data):,}")
            return self.train_data, self.val_data
        
        # Read text file
        print("  Reading text file...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"  Text length: {len(text):,} characters")
        
        # Tokenize
        print("  Tokenizing...")
        tokens = self.tokenizer.encode(text)
        print(f"  ✓ Tokenized: {len(tokens):,} tokens")
        
        # Convert to tensor
        data = torch.tensor(tokens, dtype=torch.long)
        
        # Split into train and validation
        split_idx = int(len(data) * self.train_split)
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]
        
        # Cache the tokenized data
        print("  Caching tokenized data...")
        torch.save({
            'train': self.train_data,
            'val': self.val_data
        }, cache_path)
        
        print(f"  ✓ Data prepared")
        print(f"    Train tokens: {len(self.train_data):,}")
        print(f"    Val tokens: {len(self.val_data):,}")
        
        return self.train_data, self.val_data
    
    def get_batch(self, split: str, batch_size: int, seq_length: int, device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data.
        
        Args:
            split: 'train' or 'val'
            batch_size: Batch size
            seq_length: Sequence length
            device: Device to put tensors on
            
        Returns:
            Tuple of (input_ids, target_ids)
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Random starting positions
        ix = torch.randint(len(data) - seq_length - 1, (batch_size,))
        
        # Create input and target sequences
        x = torch.stack([data[i:i + seq_length] for i in ix])
        y = torch.stack([data[i + 1:i + seq_length + 1] for i in ix])
        
        return x.to(device), y.to(device)
    
    def get_stats(self) -> dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        if self.train_data is not None:
            stats['train_tokens'] = len(self.train_data)
            stats['train_unique_tokens'] = len(torch.unique(self.train_data))
        
        if self.val_data is not None:
            stats['val_tokens'] = len(self.val_data)
            stats['val_unique_tokens'] = len(torch.unique(self.val_data))
        
        if self.train_data is not None and self.val_data is not None:
            stats['total_tokens'] = len(self.train_data) + len(self.val_data)
            stats['vocab_coverage'] = len(torch.unique(torch.cat([self.train_data, self.val_data])))
        
        return stats


def create_data_loader(tokenizer, data_dir='./data', train_split=0.95):
    """
    Factory function to create a data loader.
    
    Args:
        tokenizer: Tokenizer instance
        data_dir: Data directory
        train_split: Train/val split ratio
        
    Returns:
        DataLoader instance
    """
    return DataLoader(tokenizer, data_dir, train_split=train_split)


def download_and_prepare_data(tokenizer, dataset_name='tinystories', data_dir='./data'):
    """
    Convenience function to download and prepare data in one step.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_name: Name of dataset
        data_dir: Data directory
        
    Returns:
        Tuple of (data_loader, train_data, val_data)
    """
    loader = create_data_loader(tokenizer, data_dir)
    text_file = loader.download_sample_dataset(dataset_name)
    train_data, val_data = loader.prepare_data(text_file)
    return loader, train_data, val_data

