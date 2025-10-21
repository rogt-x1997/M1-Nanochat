"""
Nanochat Pipeline - Streamlit UI
Complete interface for LLM training pipeline
"""

import streamlit as st
import os
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    create_pipeline,
    SimpleTokenizer,
    GPT,
    GPTConfig,
    create_training_engine,
    create_data_loader,
    create_inference_engine
)

# Page config
st.set_page_config(
    page_title="Nanochat LLM Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = create_pipeline(backend='colab')

if 'active_jobs' not in st.session_state:
    st.session_state.active_jobs = {}

if 'local_model' not in st.session_state:
    st.session_state.local_model = None

if 'local_tokenizer' not in st.session_state:
    st.session_state.local_tokenizer = None

# Sidebar
with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    
    # Backend selection
    backend = st.radio(
        "Training Backend:",
        ["Google Colab (FREE)", "RunPod GPU (PAID)"],
        help="Choose where to run training"
    )
    
    selected_backend = 'colab' if 'Colab' in backend else 'runpod'
    
    if selected_backend != st.session_state.pipeline.backend:
        if st.session_state.pipeline.switch_backend(selected_backend):
            st.success(f"✓ Switched to {selected_backend}")
    
    st.markdown("---")
    
    # Model configuration
    st.markdown("## 🎛️ Model Settings")
    
    vocab_size = st.selectbox(
        "Vocabulary Size",
        [5000, 10000, 25000, 50000],
        help="Number of tokens in vocabulary"
    )
    
    num_layers = st.slider("Layers", 2, 12, 4, help="Number of transformer layers")
    num_heads = st.slider("Attention Heads", 2, 8, 4, help="Number of attention heads")
    hidden_dim = st.selectbox("Hidden Dimension", [128, 256, 512, 768], index=1)
    seq_length = st.selectbox("Sequence Length", [64, 128, 256, 512], index=1)
    
    # Update config
    st.session_state.pipeline.update_config({
        'vocab_size': vocab_size,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'hidden_dim': hidden_dim,
        'seq_length': seq_length
    })
    
    st.markdown("---")
    
    # Training settings
    st.markdown("## 🏋️ Training Settings")
    
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}"
    )
    
    batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)
    max_iters = st.number_input("Max Iterations", 100, 100000, 10000, step=100)
    
    st.session_state.pipeline.update_config({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_iters': max_iters
    })

# Main content
st.markdown("# 🤖 Nanochat LLM Pipeline")
st.markdown("*End-to-End Training: Tokenizer → Pretraining → Inference*")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    backend_display = "Colab 🆓" if st.session_state.pipeline.backend == 'colab' else "RunPod 💰"
    st.metric("Backend", backend_display)
with col2:
    st.metric("Model Size", f"{num_layers}L-{hidden_dim}H")
with col3:
    st.metric("Parameters", f"~{(num_layers * hidden_dim * hidden_dim * 12) // 1000}K")
with col4:
    st.metric("Vocab", f"{vocab_size:,}")

st.markdown("---")

# Tabs for different stages
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Setup & Training",
    "💻 Local Training",
    "🎯 Inference",
    "📊 Jobs & Monitoring"
])

# Tab 1: Setup & Training
with tab1:
    st.subheader("Cloud Training Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Backend Info")
        if st.session_state.pipeline.backend == 'colab':
            st.info("""
            **Google Colab (FREE)**
            - GPU: T4/A100 (variable)
            - Memory: 12-40GB VRAM
            - Time: 12 hrs/session
            - Cost: Free
            
            Perfect for learning and small models!
            """)
        else:
            st.info("""
            **RunPod GPU (PAID)**
            - GPU: Configurable (A4000, A100, H100)
            - Memory: Up to 80GB per GPU
            - Time: Unlimited
            - Cost: ~$0.50-$3/hr
            
            For serious training and large models.
            """)
    
    with col2:
        st.markdown("### Current Configuration")
        config_display = {
            'Layers': num_layers,
            'Heads': num_heads,
            'Hidden Dim': hidden_dim,
            'Sequence Length': seq_length,
            'Vocab Size': vocab_size,
            'Learning Rate': f"{learning_rate:.0e}",
            'Batch Size': batch_size,
            'Max Iterations': max_iters
        }
        st.json(config_display)
    
    st.markdown("---")
    
    # Submit training job
    st.markdown("### Submit Training Job")
    
    stage_name = st.text_input("Job Name", "pretraining-run1")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Submit Training Job", type="primary", use_container_width=True):
            with st.spinner("Submitting job..."):
                job_id = st.session_state.pipeline.submit_training_job(stage=stage_name)
                
                if job_id:
                    st.session_state.active_jobs[job_id] = {
                        'stage': stage_name,
                        'submitted_at': datetime.now().isoformat(),
                        'backend': st.session_state.pipeline.backend
                    }
                    
                    # Save notebook if Colab
                    if st.session_state.pipeline.backend == 'colab':
                        notebook_path = f"notebooks/{job_id}.ipynb"
                        os.makedirs("notebooks", exist_ok=True)
                        st.session_state.pipeline.save_notebook(job_id, notebook_path)
                        
                        st.success(f"✓ Job submitted: {job_id}")
                        st.info(f"📓 Notebook saved to: {notebook_path}")
                        st.markdown(f"**Next steps:**\n1. Open the notebook in Google Colab\n2. Run all cells\n3. Monitor training progress")
                    else:
                        st.success(f"✓ RunPod job submitted: {job_id}")
                else:
                    st.error("Failed to submit job")
    
    with col2:
        if st.button("📥 Download Notebook", use_container_width=True):
            if st.session_state.pipeline.backend == 'colab':
                jobs = st.session_state.pipeline.get_all_jobs()
                if jobs:
                    latest_job = jobs[-1]
                    job_id = latest_job['config']['job_id']
                    notebook_path = f"notebooks/{job_id}.ipynb"
                    os.makedirs("notebooks", exist_ok=True)
                    if st.session_state.pipeline.save_notebook(job_id, notebook_path):
                        st.success(f"✓ Notebook saved to {notebook_path}")
                else:
                    st.warning("No jobs available")
            else:
                st.warning("Notebooks only available for Colab backend")
    
    with col3:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

# Tab 2: Local Training
with tab2:
    st.subheader("Local Training (CPU/GPU)")
    st.markdown("Train a small model locally for testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Preparation")
        
        use_sample = st.checkbox("Use sample dataset", value=True)
        
        if st.button("📥 Prepare Data", use_container_width=True):
            with st.spinner("Preparing data..."):
                # Create tokenizer
                tokenizer = SimpleTokenizer()
                
                # Sample text
                sample_text = """
                Once upon a time, there was a little girl named Lucy. Lucy loved to play in the park.
                One day, Lucy found a magic stone. The stone was shiny and beautiful.
                Lucy took the stone home and showed it to her mom. Her mom smiled and said it was special.
                That night, Lucy dreamed of magical adventures. She flew through the sky and met friendly dragons.
                When she woke up, Lucy felt happy. She knew that every day could be an adventure.
                """ * 50
                
                tokenizer.train(sample_text)
                st.session_state.local_tokenizer = tokenizer
                
                # Prepare data
                import torch
                tokens = tokenizer.encode(sample_text)
                data = torch.tensor(tokens, dtype=torch.long)
                
                split_idx = int(len(data) * 0.95)
                st.session_state.train_data = data[:split_idx]
                st.session_state.val_data = data[split_idx:]
                
                st.success(f"✓ Data prepared")
                st.info(f"Vocab size: {len(tokenizer)}\nTrain tokens: {len(st.session_state.train_data):,}\nVal tokens: {len(st.session_state.val_data):,}")
    
    with col2:
        st.markdown("### Model Training")
        
        local_iters = st.number_input("Training Iterations", 100, 5000, 500, step=100)
        
        if st.button("🏋️ Train Model", type="primary", use_container_width=True):
            if st.session_state.local_tokenizer is None:
                st.error("Please prepare data first")
            else:
                with st.spinner("Training model..."):
                    import torch
                    
                    # Create model
                    config = GPTConfig(
                        vocab_size=len(st.session_state.local_tokenizer),
                        n_layer=2,  # Small for local training
                        n_head=2,
                        n_embd=128,
                        sequence_len=64
                    )
                    model = GPT(config)
                    
                    # Training config
                    training_config = {
                        'learning_rate': 1e-3,
                        'batch_size': 4,
                        'seq_length': 64,
                        'max_iters': local_iters,
                        'eval_interval': local_iters // 5,
                        'eval_iters': 5,
                        'gradient_accumulation_steps': 1,
                        'warmup_steps': local_iters // 10,
                        'checkpoint_dir': './local_checkpoints',
                        'log_dir': './local_logs'
                    }
                    
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    
                    # Train
                    engine = create_training_engine(
                        model,
                        st.session_state.train_data,
                        st.session_state.val_data,
                        training_config,
                        device
                    )
                    
                    # Progress placeholder
                    progress_text = st.empty()
                    
                    stats = engine.train()
                    
                    st.session_state.local_model = model
                    
                    st.success(f"✓ Training complete!")
                    st.metric("Best Val Loss", f"{stats['best_val_loss']:.4f}")
                    st.metric("Training Time", f"{stats['total_time']:.1f}s")

# Tab 3: Inference
with tab3:
    st.subheader("Text Generation")
    
    if st.session_state.local_model is None:
        st.info("Train a local model first to test inference, or load a checkpoint")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area("Enter prompt:", "Once upon a time", height=100)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                max_tokens = st.slider("Max Tokens", 10, 200, 50)
            with col_b:
                temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
            with col_c:
                top_k = st.slider("Top-K", 1, 100, 20)
        
        with col2:
            st.markdown("### Settings")
            st.caption(f"Max tokens: {max_tokens}")
            st.caption(f"Temperature: {temperature}")
            st.caption(f"Top-K: {top_k}")
        
        if st.button("🎯 Generate Text", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                import torch
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                engine = create_inference_engine(
                    st.session_state.local_model,
                    st.session_state.local_tokenizer,
                    device
                )
                
                generated = engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )
                
                st.markdown("### Generated Text")
                st.markdown(f"```\n{generated}\n```")

# Tab 4: Jobs & Monitoring
with tab4:
    st.subheader("Job Monitoring")
    
    # Get all jobs
    all_jobs = st.session_state.pipeline.get_all_jobs()
    
    if all_jobs:
        st.markdown(f"**Total Jobs: {len(all_jobs)}**")
        
        for job in all_jobs:
            with st.expander(f"Job: {job['config']['job_id']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Configuration:**")
                    st.json({
                        'Job ID': job['config']['job_id'],
                        'Backend': job.get('backend', 'unknown'),
                        'Status': job.get('status', 'unknown'),
                        'Created': job.get('created_at', 'N/A')
                    })
                
                with col2:
                    st.markdown("**Model Settings:**")
                    st.json({
                        'Layers': job['config']['n_layer'],
                        'Heads': job['config']['n_head'],
                        'Hidden Dim': job['config']['n_embd'],
                        'Vocab Size': job['config']['vocab_size']
                    })
                
                # Actions
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"📥 Download Notebook", key=f"dl_{job['config']['job_id']}"):
                        notebook_path = f"notebooks/{job['config']['job_id']}.ipynb"
                        os.makedirs("notebooks", exist_ok=True)
                        if st.session_state.pipeline.save_notebook(job['config']['job_id'], notebook_path):
                            st.success(f"✓ Saved to {notebook_path}")
                
                with col_b:
                    if st.button(f"🗑️ Remove", key=f"rm_{job['config']['job_id']}"):
                        st.warning("Job removal not implemented yet")
    else:
        st.info("No jobs submitted yet. Submit a training job to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Nanochat Pipeline v1.0 | Built with Streamlit | Based on Nanochat by Andrej Karpathy</p>
</div>
""", unsafe_allow_html=True)

