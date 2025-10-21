"""
M1 MacBook Optimized Streamlit UI
Simplified interface for local training on MacBook Air M1
"""

import streamlit as st
import os
import json
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src import (
    SimpleTokenizer,
    GPT,
    GPTConfig,
    create_training_engine,
    create_inference_engine
)

# Page config
st.set_page_config(
    page_title="Nanochat M1 - Local Training",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'device' not in st.session_state:
    if torch.backends.mps.is_available():
        st.session_state.device = 'mps'
    else:
        st.session_state.device = 'cpu'

# Header
st.markdown("# 🚀 Nanochat - M1 MacBook Edition")
st.markdown("*Optimized for MacBook Air M1 - Train in under 1 hour*")

# Device info
col1, col2, col3 = st.columns(3)
with col1:
    device_emoji = "⚡" if st.session_state.device == 'mps' else "🐌"
    st.metric("Device", f"{device_emoji} {st.session_state.device.upper()}")
with col2:
    if st.session_state.model:
        st.metric("Model Status", "✅ Ready")
    else:
        st.metric("Model Status", "⏳ Not Trained")
with col3:
    if st.session_state.training_complete:
        st.metric("Training", "✅ Complete")
    else:
        st.metric("Training", "⏸️ Not Started")

st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")
    
    # Preset configurations
    preset = st.selectbox(
        "Configuration Preset",
        ["Fast (~15 min)", "Balanced (~30 min)", "Quality (~45 min)", "Custom"],
        help="Choose a preset optimized for training time vs quality"
    )
    
    if preset == "Fast (~15 min)":
        default_layers = 4
        default_heads = 4
        default_dim = 256
        default_seq = 128
        default_iters = 2000
        default_batch = 12
    elif preset == "Balanced (~30 min)":
        default_layers = 6
        default_heads = 6
        default_dim = 384
        default_seq = 256
        default_iters = 5000
        default_batch = 16
    elif preset == "Quality (~45 min)":
        default_layers = 8
        default_heads = 8
        default_dim = 512
        default_seq = 256
        default_iters = 8000
        default_batch = 12
    else:  # Custom
        default_layers = 6
        default_heads = 6
        default_dim = 384
        default_seq = 256
        default_iters = 5000
        default_batch = 16
    
    st.markdown("---")
    st.markdown("### Architecture")
    
    n_layer = st.slider("Layers", 2, 12, default_layers)
    n_head = st.slider("Attention Heads", 2, 12, default_heads)
    n_embd = st.selectbox("Hidden Dimension", [128, 256, 384, 512, 768], 
                          index=[128, 256, 384, 512, 768].index(default_dim))
    seq_length = st.selectbox("Sequence Length", [64, 128, 256, 512],
                              index=[64, 128, 256, 512].index(default_seq))
    
    st.markdown("---")
    st.markdown("### Training")
    
    max_iters = st.number_input("Training Iterations", 1000, 20000, default_iters, step=1000)
    batch_size = st.slider("Batch Size", 4, 32, default_batch)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-4, 3e-4, 5e-4, 1e-3],
        value=3e-4,
        format_func=lambda x: f"{x:.0e}"
    )
    
    # Estimate parameters and time
    est_params = n_layer * n_embd * n_embd * 12
    est_time_min = max_iters / 200  # Rough estimate
    
    st.markdown("---")
    st.markdown("### Estimates")
    st.info(f"""
    **Parameters:** ~{est_params/1e6:.1f}M  
    **Memory:** ~{est_params*4/1e9:.1f} GB  
    **Time:** ~{est_time_min:.0f} minutes
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📚 Training", "🎯 Inference", "📊 Model Info"])

# Tab 1: Training
with tab1:
    st.subheader("Train Your Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Training Data")
        
        data_source = st.radio(
            "Data Source",
            ["Sample Stories (Built-in)", "Custom Text"],
            help="Choose your training data source"
        )
        
        if data_source == "Custom Text":
            custom_text = st.text_area(
                "Enter your training text:",
                height=200,
                placeholder="Paste your text here... (minimum 1000 characters recommended)"
            )
        else:
            st.info("Using built-in sample stories dataset (~200K tokens)")
    
    with col2:
        st.markdown("### Quick Actions")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if data_source == "Custom Text" and len(custom_text) < 100:
                st.error("Please provide more training text (at least 100 characters)")
            else:
                # Prepare data
                with st.spinner("Preparing training data..."):
                    if data_source == "Custom Text":
                        text = custom_text * 10  # Repeat for more data
                    else:
                        # Use sample data
                        text = """
Once upon a time, there was a little girl named Lucy. She loved to explore the forest near her home.
One sunny day, Lucy found a magical stone that glowed with a soft blue light.
The fairy told Lucy that she had been waiting for someone kind to free her from the stone.
Lucy and the fairy became best friends. They had many adventures together in the magical forest.

There was a boy named Tom who loved to build things. He would spend hours creating with blocks.
Tom decided to build a treehouse in the big oak tree in his backyard.
After two weeks, the treehouse was finally complete. It had windows, a door, and even a ladder.
Tom felt proud of what he had accomplished through hard work and determination.

A curious cat named Whiskers lived in a cozy house with a loving family.
Whiskers loved to watch birds from the window and chase butterflies in the garden.
Instead of chasing the mouse, Whiskers decided to become friends with it.
It taught everyone that friendship can happen in the most unexpected ways.
""" * 300
                    
                    # Train tokenizer
                    tokenizer = SimpleTokenizer()
                    tokenizer.train(text)
                    
                    # Tokenize
                    tokens = tokenizer.encode(text)
                    data = torch.tensor(tokens, dtype=torch.long)
                    
                    # Limit tokens for reasonable training time
                    max_tokens = 500000
                    if len(data) > max_tokens:
                        data = data[:max_tokens]
                    
                    split_idx = int(len(data) * 0.95)
                    train_data = data[:split_idx]
                    val_data = data[split_idx:]
                    
                    st.session_state.tokenizer = tokenizer
                    
                    st.success(f"✓ Data prepared: {len(train_data):,} training tokens")
                
                # Create model
                with st.spinner("Creating model..."):
                    config = GPTConfig(
                        vocab_size=len(tokenizer),
                        n_layer=n_layer,
                        n_head=n_head,
                        n_embd=n_embd,
                        sequence_len=seq_length,
                        dropout=0.1
                    )
                    model = GPT(config)
                    st.success(f"✓ Model created: {model.get_num_params():,} parameters")
                
                # Train
                st.markdown("### Training Progress")
                
                training_config = {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'max_iters': max_iters,
                    'eval_interval': max(max_iters // 10, 100),
                    'eval_iters': 10,
                    'gradient_accumulation_steps': 2,
                    'warmup_steps': max_iters // 10,
                    'checkpoint_dir': './checkpoints/m1_streamlit',
                    'log_dir': './logs/m1_streamlit'
                }
                
                device = st.session_state.device
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                
                engine = create_training_engine(model, train_data, val_data, training_config, device)
                
                start_time = time.time()
                stats = engine.train()
                elapsed = time.time() - start_time
                
                progress_bar.progress(100)
                
                st.session_state.model = model
                st.session_state.training_complete = True
                
                st.success(f"""
                ✅ **Training Complete!**
                
                - Best validation loss: {stats['best_val_loss']:.4f}
                - Training time: {elapsed/60:.1f} minutes
                - Tokens/second: {len(train_data) * max_iters / elapsed:.0f}
                """)
                
                st.balloons()
        
        if st.button("💾 Save Model", use_container_width=True):
            if st.session_state.model is None:
                st.warning("No model to save. Train a model first!")
            else:
                save_path = "./saved_models/my_model.pt"
                Path("./saved_models").mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': st.session_state.model.state_dict(),
                    'config': {
                        'vocab_size': len(st.session_state.tokenizer),
                        'n_layer': n_layer,
                        'n_head': n_head,
                        'n_embd': n_embd,
                        'sequence_len': seq_length
                    }
                }, save_path)
                st.success(f"✓ Model saved to {save_path}")

# Tab 2: Inference
with tab2:
    st.subheader("Generate Text")
    
    if st.session_state.model is None:
        st.info("👈 Train a model first in the Training tab!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "Enter your prompt:",
                value="Once upon a time",
                height=100
            )
            
            max_tokens = st.slider("Max tokens to generate", 20, 200, 80)
        
        with col2:
            st.markdown("### Generation Settings")
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
            top_k = st.slider("Top-K", 1, 100, 40)
            top_p = st.slider("Top-P", 0.1, 1.0, 0.9, 0.05)
        
        if st.button("✨ Generate", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                inf_engine = create_inference_engine(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    st.session_state.device
                )
                
                generated = inf_engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                st.markdown("### Generated Text")
                st.markdown(f"```\n{generated}\n```")
        
        # Batch generation
        st.markdown("---")
        st.markdown("### Batch Generation")
        
        if st.button("🎲 Generate 5 Samples", use_container_width=True):
            with st.spinner("Generating samples..."):
                inf_engine = create_inference_engine(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    st.session_state.device
                )
                
                sample_prompts = [
                    "Once upon a time",
                    "In a magical forest",
                    "There was a brave",
                    "The wise old",
                    "A curious little"
                ]
                
                for i, p in enumerate(sample_prompts, 1):
                    gen = inf_engine.generate(p, max_tokens=60, temperature=0.8, top_k=40)
                    st.markdown(f"**Sample {i}:** {gen}")
                    st.markdown("---")

# Tab 3: Model Info
with tab3:
    st.subheader("Model Information")
    
    if st.session_state.model is None:
        st.info("No model loaded yet.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Architecture")
            st.json({
                "Layers": n_layer,
                "Attention Heads": n_head,
                "Hidden Dimension": n_embd,
                "Sequence Length": seq_length,
                "Total Parameters": f"{st.session_state.model.get_num_params():,}",
                "Model Size (MB)": f"{st.session_state.model.get_num_params() * 4 / 1e6:.1f}"
            })
        
        with col2:
            st.markdown("### Tokenizer")
            if st.session_state.tokenizer:
                st.json({
                    "Vocabulary Size": len(st.session_state.tokenizer),
                    "Type": "Character-level",
                    "Encoding": "Simple mapping"
                })
        
        st.markdown("### Device Information")
        st.json({
            "Device": st.session_state.device.upper(),
            "MPS Available": torch.backends.mps.is_available(),
            "PyTorch Version": torch.__version__
        })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Nanochat M1 Edition | Optimized for MacBook Air M1 | 🚀</p>
</div>
""", unsafe_allow_html=True)

