Nanochat M1 - Train GPT Models Locally on MacBook Air

Train a real GPT-style language model from scratch on your MacBook Air M1 in under 30 minutes. This implementation combines Andrej Karpathy's nanochat architecture with Apple Silicon optimizations, enabling local LLM training without cloud GPUs. Features include automatic MPS acceleration, three preset configurations (Fast/Balanced/Quality), and a complete Streamlit UI for interactive training and text generation.
Quick Start: pip install torch streamlit && streamlit run app_m1.py
Key Features: 3M parameter transformer • Character-level tokenization • Real gradient descent training • Autoregressive text generation • No API keys or cloud costs required
Hardware: Optimized for MacBook Air M1 (8GB RAM) • Uses Metal Performance Shaders for GPU acceleration • Comparable speed to Google Colab T4 • Unlimited training time

Learn More: Check M1_QUICKSTART.md for detailed setup instructions.
