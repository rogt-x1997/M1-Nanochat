"""
BPE Tokenizer Implementation
Based on the Byte Pair Encoding algorithm
"""

import pickle
import regex as re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer for text processing.
    
    This implements the BPE algorithm to learn a vocabulary from a text corpus
    and provides encoding/decoding functionality.
    """
    
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (including base 256 bytes)
        """
        self.vocab_size = vocab_size
        # Start with base 256 byte tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []  # List of (pair, new_token_id) tuples
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def train(self, text: str, verbose: bool = True) -> None:
        """
        Train the BPE tokenizer on a text corpus.
        
        Args:
            text: Training text corpus
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Training BPE tokenizer with vocab_size={self.vocab_size}")
        
        # Number of merges to perform
        num_merges = self.vocab_size - 256
        
        # Split text into words using the pattern
        words = re.findall(self.pattern, text)
        
        # Convert words to byte sequences
        word_bytes = [list(word.encode('utf-8')) for word in words]
        
        # Perform BPE merges
        for i in range(num_merges):
            # Count pair frequencies across all words
            pair_freqs = defaultdict(int)
            for word in word_bytes:
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pair_freqs[pair] += 1
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Create new token ID
            new_token_id = 256 + i
            
            # Record the merge
            self.merges.append((best_pair, new_token_id))
            
            # Update vocabulary
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply merge to all words
            new_word_bytes = []
            for word in word_bytes:
                new_word = self._merge_pair(word, best_pair, new_token_id)
                new_word_bytes.append(new_word)
            word_bytes = new_word_bytes
            
            if verbose and (i + 1) % 500 == 0:
                print(f"  Completed {i + 1}/{num_merges} merges")
        
        if verbose:
            print(f"✓ Training complete: {len(self.vocab)} tokens in vocabulary")
    
    def _merge_pair(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Merge a specific pair in a sequence of token IDs.
        
        Args:
            ids: List of token IDs
            pair: Pair to merge (a, b)
            new_id: New token ID to replace the pair
            
        Returns:
            List of token IDs with pair merged
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to a list of token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Split text into words
        words = re.findall(self.pattern, text)
        
        # Convert to token IDs
        tokens = []
        for word in words:
            # Start with byte-level encoding
            word_tokens = list(word.encode('utf-8'))
            
            # Apply learned merges
            for (pair, new_id) in self.merges:
                word_tokens = self._merge_pair(word_tokens, pair, new_id)
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text string
        """
        # Convert token IDs to bytes
        byte_array = b''.join(self.vocab[token] for token in tokens if token in self.vocab)
        
        # Decode bytes to string, replacing invalid UTF-8 sequences
        text = byte_array.decode('utf-8', errors='replace')
        return text
    
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file.
        
        Args:
            path: File path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'vocab': self.vocab,
                'merges': self.merges
            }, f)
        print(f"✓ Tokenizer saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the tokenizer from a file.
        
        Args:
            path: File path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = data['merges']
        print(f"✓ Tokenizer loaded from {path}")
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)


# Simple tokenizer for quick testing (character-level)
class SimpleTokenizer:
    """
    Simple character-level tokenizer for testing purposes.
    """
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def train(self, text: str) -> None:
        """Build vocabulary from unique characters."""
        unique_chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(unique_chars)
        print(f"✓ Simple tokenizer trained: {self.vocab_size} unique characters")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.idx_to_char.get(idx, '') for idx in tokens)
    
    def save(self, path: str) -> None:
        """Save tokenizer."""
        with open(path, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path: str) -> None:
        """Load tokenizer."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.vocab_size = data['vocab_size']
    
    def __len__(self) -> int:
        return self.vocab_size

