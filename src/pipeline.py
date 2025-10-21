"""
Pipeline Orchestrator
Main controller for the end-to-end LLM training pipeline
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from .colab_handler import ColabHandler, ColabJobConfig
from .runpod_handler import RunPodHandler, RunPodJobConfig


class PipelineOrchestrator:
    """
    Main pipeline orchestrator for LLM training.
    Manages backend selection, job submission, and monitoring.
    """
    
    def __init__(self, backend: str = 'colab', config_path: str = 'pipeline_config.json'):
        """
        Initialize pipeline orchestrator.
        
        Args:
            backend: 'colab' or 'runpod'
            config_path: Path to configuration file
        """
        self.backend = backend.lower()
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize backend handler
        if self.backend == 'colab':
            self.handler = ColabHandler()
        elif self.backend == 'runpod':
            api_key = os.getenv('RUNPOD_API_KEY')
            if not api_key:
                raise ValueError("RUNPOD_API_KEY not set in environment")
            self.handler = RunPodHandler(api_key)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        print(f"✓ Pipeline initialized with {self.backend} backend")
    
    def load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration."""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"✓ Config loaded from {self.config_path}")
            return config
        else:
            # Default configuration
            config = {
                'model_name': 'nanochat',
                'model_size': 'd4',
                'vocab_size': 5000,
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 4,
                'seq_length': 128,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'max_iters': 10000,
                'eval_interval': 500,
                'warmup_steps': 100,
                'output_dir': './outputs',
                'checkpoints_dir': './checkpoints',
                'logs_dir': './logs',
                'created_at': datetime.now().isoformat()
            }
            self.save_config(config)
            print(f"✓ Default config created")
            return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Config saved to {self.config_path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(updates)
        self.save_config()
    
    def submit_training_job(self, stage: str = 'pretraining') -> Optional[str]:
        """
        Submit a training job to the backend.
        
        Args:
            stage: Training stage name
            
        Returns:
            Job ID if successful
        """
        job_id = f"{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if self.backend == 'colab':
            config = ColabJobConfig(
                job_id=job_id,
                vocab_size=self.config.get('vocab_size', 5000),
                n_layer=self.config.get('num_layers', 4),
                n_head=self.config.get('num_heads', 4),
                n_embd=self.config.get('hidden_dim', 256),
                sequence_len=self.config.get('seq_length', 128),
                learning_rate=self.config.get('learning_rate', 1e-4),
                batch_size=self.config.get('batch_size', 32),
                max_iters=self.config.get('max_iters', 10000),
                eval_interval=self.config.get('eval_interval', 500),
                warmup_steps=self.config.get('warmup_steps', 100),
                checkpoint_dir='/content/drive/MyDrive/nanochat/checkpoints'
            )
            
            job_id = self.handler.submit_job(config)
            print(f"✓ Colab job submitted: {job_id}")
            return job_id
        
        elif self.backend == 'runpod':
            config = RunPodJobConfig(
                job_id=job_id,
                gpu_type=self.config.get('gpu_type', 'NVIDIA RTX A4000'),
                gpu_count=self.config.get('gpu_count', 1),
                vocab_size=self.config.get('vocab_size', 5000),
                n_layer=self.config.get('num_layers', 4),
                n_head=self.config.get('num_heads', 4),
                n_embd=self.config.get('hidden_dim', 256),
                sequence_len=self.config.get('seq_length', 128),
                learning_rate=self.config.get('learning_rate', 1e-4),
                batch_size=self.config.get('batch_size', 32),
                max_iters=self.config.get('max_iters', 10000),
                budget_per_hr=self.config.get('budget_per_hr', 1.0)
            )
            
            job_id = self.handler.submit_job(config)
            if job_id:
                print(f"✓ RunPod job submitted: {job_id}")
            return job_id
        
        return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job."""
        return self.handler.get_job_status(job_id)
    
    def get_all_jobs(self) -> list:
        """Get all jobs."""
        return self.handler.get_all_jobs()
    
    def terminate_job(self, job_id: str) -> bool:
        """Terminate a job."""
        if self.backend == 'runpod':
            return self.handler.terminate_pod(job_id)
        else:
            print("Colab jobs must be terminated manually in the notebook")
            return True
    
    def switch_backend(self, new_backend: str) -> bool:
        """
        Switch to a different backend.
        
        Args:
            new_backend: 'colab' or 'runpod'
            
        Returns:
            True if successful
        """
        new_backend = new_backend.lower()
        
        if new_backend == self.backend:
            print(f"Already using {new_backend} backend")
            return True
        
        try:
            if new_backend == 'colab':
                self.handler = ColabHandler()
            elif new_backend == 'runpod':
                api_key = os.getenv('RUNPOD_API_KEY')
                if not api_key:
                    print("✗ RUNPOD_API_KEY not set in environment")
                    return False
                self.handler = RunPodHandler(api_key)
            else:
                print(f"✗ Unknown backend: {new_backend}")
                return False
            
            self.backend = new_backend
            print(f"✓ Switched to {new_backend} backend")
            return True
        
        except Exception as e:
            print(f"✗ Failed to switch backend: {e}")
            return False
    
    def get_notebook(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get Colab notebook for a job."""
        if self.backend == 'colab':
            return self.handler.get_notebook(job_id)
        return None
    
    def save_notebook(self, job_id: str, output_path: str) -> bool:
        """Save Colab notebook to file."""
        if self.backend == 'colab':
            return self.handler.save_notebook(job_id, output_path)
        return False


def create_pipeline(backend: str = 'colab', config_path: str = 'pipeline_config.json'):
    """
    Factory function to create pipeline orchestrator.
    
    Args:
        backend: 'colab' or 'runpod'
        config_path: Path to configuration file
        
    Returns:
        PipelineOrchestrator instance
    """
    return PipelineOrchestrator(backend, config_path)


def load_env_config(env_path: str = '.env') -> Dict[str, Any]:
    """
    Load environment configuration from .env file.
    
    Args:
        env_path: Path to .env file
        
    Returns:
        Configuration dictionary
    """
    load_dotenv(env_path)
    
    return {
        'backend': os.getenv('TRAINING_BACKEND', 'colab'),
        'runpod_api_key': os.getenv('RUNPOD_API_KEY'),
        'gpu_type': os.getenv('RUNPOD_GPU_TYPE', 'NVIDIA RTX A4000'),
        'gpu_count': int(os.getenv('RUNPOD_GPU_COUNT', '1')),
        'model_name': os.getenv('MODEL_NAME', 'nanochat'),
        'vocab_size': int(os.getenv('VOCAB_SIZE', '5000')),
        'hidden_dim': int(os.getenv('HIDDEN_DIM', '256')),
        'num_layers': int(os.getenv('NUM_LAYERS', '4')),
        'num_heads': int(os.getenv('NUM_HEADS', '4')),
        'seq_length': int(os.getenv('SEQ_LENGTH', '128')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '0.0001')),
        'batch_size': int(os.getenv('BATCH_SIZE', '32')),
        'max_iters': int(os.getenv('MAX_ITERS', '10000'))
    }

