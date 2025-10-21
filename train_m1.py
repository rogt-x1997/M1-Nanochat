"""
M1 MacBook Optimized Training Script
Trains a GPT model on MacBook Air M1 in under 1 hour
"""

import torch
import json
import time
from pathlib import Path
from src import (
    SimpleTokenizer,
    GPT,
    GPTConfig,
    create_training_engine,
    create_data_loader,
    create_inference_engine
)


def get_device():
    """Get the best available device for M1 Mac."""
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal Performance Shaders) available - using GPU acceleration")
        return torch.device("mps")
    else:
        print("⚠ MPS not available - falling back to CPU")
        return torch.device("cpu")


def load_config(config_path='configs/m1_macbook_config.json'):
    """Load M1-optimized configuration."""
    # Try to load from file, otherwise use defaults
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Return default configuration
        return {
            "model_name": "nanochat-m1-optimized",
            "description": "Optimized for MacBook Air M1 - trains in ~30-45 minutes",
            "model": {
                "vocab_size": 5000,
                "n_layer": 6,
                "n_head": 6,
                "n_embd": 384,
                "sequence_len": 256,
                "dropout": 0.1
            },
            "training": {
                "learning_rate": 3e-4,
                "batch_size": 16,
                "seq_length": 256,
                "max_iters": 5000,
                "eval_interval": 250,
                "eval_iters": 20,
                "gradient_accumulation_steps": 2,
                "warmup_steps": 500,
                "weight_decay": 0.1,
                "grad_clip": 1.0
            },
            "device": {
                "backend": "mps",
                "fallback_to_cpu": True,
                "compile_model": False
            },
            "data": {
                "dataset": "tinystories",
                "train_split": 0.95,
                "max_tokens": 1000000,
                "cache_tokenized": True
            },
            "checkpointing": {
                "save_best_only": True,
                "checkpoint_dir": "./checkpoints/m1_optimized",
                "log_dir": "./logs/m1_optimized"
            },
            "inference": {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "estimated_time": "30-45 minutes",
            "estimated_memory": "4-6 GB",
            "expected_loss": "< 1.5"
        }


def prepare_training_data(tokenizer, max_tokens=1000000):
    """Prepare training data optimized for M1."""
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    
    # Use TinyStories-style data (good for quick training)
    sample_text = """
Once upon a time, there was a little girl named Lucy. She loved to explore the forest near her home.
One sunny day, Lucy found a magical stone that glowed with a soft blue light. She picked it up carefully.
The stone felt warm in her hands. Suddenly, she heard a gentle voice saying, "Thank you for finding me."
Lucy was surprised but not scared. The voice belonged to a tiny fairy who lived inside the stone.
The fairy told Lucy that she had been waiting for someone kind to free her from the stone.
Lucy and the fairy became best friends. They had many adventures together in the magical forest.
They helped lost animals find their way home and planted beautiful flowers everywhere they went.
The forest became more beautiful each day because of their kindness and friendship.

There was a boy named Tom who loved to build things. He would spend hours creating with blocks and wood.
One day, Tom decided to build a treehouse in the big oak tree in his backyard.
He worked hard every day, measuring, cutting, and hammering. His father helped him with the difficult parts.
After two weeks, the treehouse was finally complete. It had windows, a door, and even a small ladder.
Tom invited all his friends to see his creation. They were amazed by how wonderful it looked.
They spent the whole afternoon playing in the treehouse, telling stories and eating snacks.
Tom felt proud of what he had accomplished through hard work and determination.

A curious cat named Whiskers lived in a cozy house with a loving family.
Whiskers loved to watch birds from the window and chase butterflies in the garden.
One morning, Whiskers discovered a small mouse hiding behind the bookshelf.
Instead of chasing it, Whiskers decided to become friends with the little mouse.
The mouse was nervous at first, but Whiskers was gentle and kind.
They played together every day, sharing food and keeping each other company.
The family was surprised to see a cat and mouse being such good friends.
It taught everyone that friendship can happen in the most unexpected ways.

In a small village, there lived a wise old woman who knew many things about nature.
Children would come to her to learn about plants, animals, and the changing seasons.
She taught them how to grow vegetables, identify different birds, and respect all living things.
One spring, she helped the children plant a community garden in the village square.
Everyone worked together, digging, planting, and watering the seeds they had sown.
As summer came, the garden bloomed with colorful flowers and fresh vegetables.
The village became a happier place because people learned to work together and share.
The wise woman smiled, knowing that she had taught them the most important lesson of all.

A young rabbit named Rosie loved to hop through the meadow every morning.
She would greet all the other animals and share the fresh clover she found.
One day, Rosie noticed that an old turtle was struggling to climb a small hill.
Without hesitation, Rosie helped push the turtle up the hill, even though it took a long time.
The turtle was very grateful and thanked Rosie for her kindness and patience.
From that day on, Rosie and the turtle became close friends.
They would meet every day to share stories and enjoy the beautiful meadow together.
Rosie learned that helping others brings the greatest joy of all.
""" * 200  # Repeat to get ~1M tokens
    
    print(f"Text length: {len(sample_text):,} characters")
    
    # Train tokenizer
    print("\nTraining tokenizer...")
    tokenizer.train(sample_text)
    print(f"✓ Vocabulary size: {len(tokenizer)}")
    
    # Tokenize
    print("\nTokenizing text...")
    tokens = tokenizer.encode(sample_text)
    
    # Limit to max_tokens for faster training
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"✓ Limited to {max_tokens:,} tokens for optimal training time")
    
    data = torch.tensor(tokens, dtype=torch.long)
    
    # Split
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"\n✓ Data prepared:")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")
    
    return train_data, val_data, tokenizer


def create_model(config, vocab_size):
    """Create M1-optimized GPT model."""
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    
    model_config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        sequence_len=config['model']['sequence_len'],
        dropout=config['model']['dropout']
    )
    
    model = GPT(model_config)
    
    print(f"✓ Model created:")
    print(f"  Layers: {model_config.n_layer}")
    print(f"  Heads: {model_config.n_head}")
    print(f"  Hidden dim: {model_config.n_embd}")
    print(f"  Sequence length: {model_config.sequence_len}")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Estimated size: {model.get_num_params() * 4 / 1e6:.1f} MB")
    
    return model


def train_model(model, train_data, val_data, config, device):
    """Train the model with M1 optimizations."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    training_config = {
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'seq_length': config['training']['seq_length'],
        'max_iters': config['training']['max_iters'],
        'eval_interval': config['training']['eval_interval'],
        'eval_iters': config['training']['eval_iters'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'warmup_steps': config['training']['warmup_steps'],
        'checkpoint_dir': config['checkpointing']['checkpoint_dir'],
        'log_dir': config['checkpointing']['log_dir']
    }
    
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Sequence length: {training_config['seq_length']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Max iterations: {training_config['max_iters']}")
    print(f"  Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    
    # Estimate time
    estimated_time = config.get('estimated_time', '30-45 minutes')
    print(f"\n⏱  Estimated training time: {estimated_time}")
    print(f"💾 Estimated memory usage: {config.get('estimated_memory', '4-6 GB')}")
    
    input("\nPress Enter to start training...")
    
    # Create training engine
    engine = create_training_engine(model, train_data, val_data, training_config, str(device))
    
    # Train
    start_time = time.time()
    stats = engine.train()
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Best validation loss: {stats['best_val_loss']:.4f}")
    print(f"✓ Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"✓ Iterations: {stats['iterations']}")
    print(f"✓ Tokens/second: {len(train_data) * stats['iterations'] / total_time:.0f}")
    
    return engine, stats


def test_inference(model, tokenizer, config, device):
    """Test text generation."""
    print("\n" + "="*70)
    print("TESTING TEXT GENERATION")
    print("="*70)
    
    inf_engine = create_inference_engine(model, tokenizer, str(device))
    
    test_prompts = [
        "Once upon a time",
        "There was a little",
        "In a magical forest",
        "A brave knight",
        "The wise old"
    ]
    
    print("\nGenerated samples:\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] Prompt: '{prompt}'")
        
        generated = inf_engine.generate(
            prompt=prompt,
            max_tokens=80,
            temperature=config['inference']['temperature'],
            top_k=config['inference']['top_k'],
            top_p=config['inference']['top_p']
        )
        
        print(f"Generated:\n{generated}\n")
        print("-" * 70)
    
    return inf_engine


def main():
    """Main training workflow for M1 MacBook."""
    print("\n" + "="*70)
    print("🚀 NANOCHAT - M1 MACBOOK OPTIMIZED TRAINING")
    print("="*70)
    print("\nThis script will:")
    print("  1. Prepare training data (~1M tokens)")
    print("  2. Create a 6-layer GPT model (~3M parameters)")
    print("  3. Train for 5000 iterations (~30-45 minutes)")
    print("  4. Generate sample text")
    print("\nOptimized for: MacBook Air M1 with MPS acceleration")
    
    # Get device
    device = get_device()
    
    # Load config
    config = load_config()
    print(f"\n✓ Loaded configuration: {config['model_name']}")
    
    # Prepare data
    tokenizer = SimpleTokenizer()
    train_data, val_data, tokenizer = prepare_training_data(
        tokenizer, 
        max_tokens=config['data']['max_tokens']
    )
    
    # Save tokenizer
    Path(config['checkpointing']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(config['checkpointing']['checkpoint_dir']) / 'tokenizer.pkl'
    tokenizer.save(str(tokenizer_path))
    print(f"\n✓ Tokenizer saved to {tokenizer_path}")
    
    # Create model
    model = create_model(config, len(tokenizer))
    
    # Train
    engine, stats = train_model(model, train_data, val_data, config, device)
    
    # Test inference
    inf_engine = test_inference(engine.model, tokenizer, config, device)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print(f"\nModel saved to: {config['checkpointing']['checkpoint_dir']}/best_model.pt")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"\nTo use this model later:")
    print(f"  1. Load tokenizer: tokenizer.load('{tokenizer_path}')")
    print(f"  2. Load checkpoint: torch.load('{config['checkpointing']['checkpoint_dir']}/best_model.pt')")
    print(f"  3. Generate text with the inference engine")
    print(f"\nOr run: streamlit run app.py")
    print(f"  Then use 'Local Training' tab to load and test the model")
    print("\n🎉 Happy generating! 🎉\n")


if __name__ == "__main__":
    main()

