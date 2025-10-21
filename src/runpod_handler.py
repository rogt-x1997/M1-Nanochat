"""
RunPod Handler
Manages GPU pods and training jobs via RunPod API
"""

import os
import requests
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class RunPodJobConfig:
    """Configuration for a RunPod training job."""
    job_id: str
    gpu_type: str
    gpu_count: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    sequence_len: int
    learning_rate: float
    batch_size: int
    max_iters: int
    budget_per_hr: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class RunPodHandler:
    """
    Handler for RunPod GPU training jobs.
    Manages pod creation, monitoring, and termination via RunPod API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize RunPod handler.
        
        Args:
            api_key: RunPod API key
        """
        self.api_key = api_key
        self.base_url = "https://api.runpod.io/graphql"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        print(f"✓ RunPod handler initialized")
    
    def _make_request(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make GraphQL request to RunPod API.
        
        Args:
            query: GraphQL query string
            variables: Query variables
            
        Returns:
            Response dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key  # RunPod uses 'api-key' header
        }
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "errors" in result:
                print(f"RunPod API error: {result['errors']}")
                return {"errors": result['errors']}
            
            return result
        
        except requests.exceptions.RequestException as e:
            print(f"RunPod request failed: {str(e)}")
            return {"errors": [{"message": str(e)}]}
    
    def create_pod(self, config: RunPodJobConfig) -> Optional[str]:
        """
        Create a GPU pod for training.
        
        Args:
            config: Job configuration
            
        Returns:
            Pod ID if successful, None otherwise
        """
        print(f"Creating RunPod pod: {config.gpu_count}x{config.gpu_type}")
        
        # GraphQL mutation to create pod
        query = """
        mutation {
          podFindAndDeployOnDemand(input: {
            cloudType: SECURE
            gpuCount: %d
            gpuTypeId: "%s"
            volumeInGb: 50
            containerDiskInGb: 20
            minVcpuCount: 8
            minMemoryInGb: 32
            dockerArgs: ""
            ports: "8888/http,22/tcp"
            volumeMountPath: "/workspace"
            env: [
              {key: "JUPYTER_ENABLE_LAB", value: "yes"}
            ]
            templateId: "runpod-torch"
          }) {
            id
            imageName
            gpuCount
            costPerHr
            machineId
            machine {
              gpuDisplayName
            }
          }
        }
        """ % (config.gpu_count, config.gpu_type)
        
        result = self._make_request(query)
        
        if "data" in result and result["data"].get("podFindAndDeployOnDemand"):
            pod = result["data"]["podFindAndDeployOnDemand"]
            pod_id = pod["id"]
            
            # Store job info
            self.jobs[pod_id] = {
                'config': asdict(config),
                'pod_id': pod_id,
                'status': 'initializing',
                'created_at': datetime.now().isoformat(),
                'last_update': datetime.now().isoformat(),
                'gpu_type': config.gpu_type,
                'gpu_count': config.gpu_count,
                'cost_per_hr': pod.get('costPerHr', config.budget_per_hr),
                'metrics': {}
            }
            
            print(f"✓ Pod created: {pod_id}")
            print(f"  GPU: {config.gpu_count}x{config.gpu_type}")
            print(f"  Cost: ${pod.get('costPerHr', config.budget_per_hr)}/hr")
            
            return pod_id
        else:
            print(f"✗ Failed to create pod: {result}")
            return None
    
    def get_pod_status(self, pod_id: str) -> Dict[str, Any]:
        """
        Get status of a pod.
        
        Args:
            pod_id: Pod ID
            
        Returns:
            Pod status dictionary
        """
        query = """
        query {
          pod(input: {podId: "%s"}) {
            id
            name
            runtime {
              uptimeInSeconds
              ports {
                ip
                isIpPublic
                privatePort
                publicPort
                type
              }
              gpus {
                id
                gpuUtilPercent
                memoryUtilPercent
              }
            }
            desiredStatus
            costPerHr
            gpuCount
          }
        }
        """ % pod_id
        
        result = self._make_request(query)
        
        if "data" in result and result["data"].get("pod"):
            pod = result["data"]["pod"]
            
            # Update job info
            if pod_id in self.jobs:
                self.jobs[pod_id].update({
                    'status': pod.get('desiredStatus', 'unknown'),
                    'last_update': datetime.now().isoformat(),
                    'uptime_seconds': pod.get('runtime', {}).get('uptimeInSeconds', 0),
                    'ports': pod.get('runtime', {}).get('ports', []),
                    'gpu_metrics': pod.get('runtime', {}).get('gpus', [])
                })
            
            return self.jobs.get(pod_id, {})
        else:
            return {'status': 'error', 'pod_id': pod_id, 'error': result.get('errors')}
    
    def terminate_pod(self, pod_id: str) -> bool:
        """
        Terminate a pod.
        
        Args:
            pod_id: Pod ID
            
        Returns:
            True if successful
        """
        print(f"Terminating pod: {pod_id}")
        
        query = """
        mutation {
          podTerminate(input: {podId: "%s"})
        }
        """ % pod_id
        
        result = self._make_request(query)
        
        if "data" in result:
            if pod_id in self.jobs:
                self.jobs[pod_id]['status'] = 'terminated'
                self.jobs[pod_id]['terminated_at'] = datetime.now().isoformat()
            
            print(f"✓ Pod terminated: {pod_id}")
            return True
        else:
            print(f"✗ Failed to terminate pod: {result}")
            return False
    
    def get_all_pods(self) -> List[Dict[str, Any]]:
        """
        Get all pods for the account.
        
        Returns:
            List of pod dictionaries
        """
        query = """
        query {
          myself {
            pods {
              id
              name
              runtime {
                uptimeInSeconds
              }
              desiredStatus
              costPerHr
              gpuCount
            }
          }
        }
        """
        
        result = self._make_request(query)
        
        if "data" in result and result["data"].get("myself"):
            pods = result["data"]["myself"].get("pods", [])
            return pods
        else:
            return []
    
    def submit_job(self, config: RunPodJobConfig) -> Optional[str]:
        """
        Submit a training job (creates pod).
        
        Args:
            config: Job configuration
            
        Returns:
            Job ID (pod ID) if successful
        """
        return self.create_pod(config)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status (alias for get_pod_status)."""
        return self.get_pod_status(job_id)
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all tracked jobs."""
        return list(self.jobs.values())
    
    def wait_for_pod_ready(self, pod_id: str, timeout: int = 300) -> bool:
        """
        Wait for pod to be ready.
        
        Args:
            pod_id: Pod ID
            timeout: Maximum wait time in seconds
            
        Returns:
            True if pod is ready, False if timeout
        """
        print(f"Waiting for pod {pod_id} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_pod_status(pod_id)
            
            if status.get('status') == 'RUNNING':
                print(f"✓ Pod is ready!")
                return True
            
            time.sleep(10)
        
        print(f"✗ Timeout waiting for pod to be ready")
        return False


def create_runpod_handler(api_key: Optional[str] = None):
    """
    Factory function to create RunPod handler.
    
    Args:
        api_key: RunPod API key (if None, reads from environment)
        
    Returns:
        RunPodHandler instance
    """
    if api_key is None:
        api_key = os.getenv('RUNPOD_API_KEY')
    
    if not api_key:
        raise ValueError("RunPod API key not provided and RUNPOD_API_KEY not set in environment")
    
    return RunPodHandler(api_key)

