"""
Ollama Progress Monitoring Utility
Real-time model download progress tracking for enhanced user experience
"""

import subprocess
import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class ModelDownloadProgress:
    """Represents the download progress of a single model"""
    model_name: str
    status: str  # "downloading", "pulling", "verifying", "success", "error"
    progress_percent: Optional[float] = None
    downloaded_mb: Optional[float] = None
    total_mb: Optional[float] = None
    speed_mbps: Optional[float] = None
    eta_seconds: Optional[int] = None
    error_message: Optional[str] = None

class OllamaProgressMonitor:
    """Monitor Ollama model download progress in real-time"""
    
    def __init__(self):
        self.active_downloads = {}
        self.completed_models = set()
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model is already installed"""
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                # Check if model name appears in the list
                return model_name.lower() in result.stdout.lower()
        except Exception:
            pass
        return False
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        models = []
        try:
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        # Extract model name (first column)
                        model_name = line.split()[0]
                        if model_name and not model_name.startswith("NAME"):
                            models.append(model_name)
        except Exception:
            pass
        return models
    
    def start_model_download(self, model_name: str) -> None:
        """Start downloading a model (non-blocking)"""
        if self.check_model_exists(model_name):
            return  # Model already exists
        
        try:
            # Start the download process in the background
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.active_downloads[model_name] = {
                "process": process,
                "start_time": time.time(),
                "last_update": time.time()
            }
            
        except Exception as e:
            # Store error information
            self.active_downloads[model_name] = {
                "error": str(e),
                "start_time": time.time()
            }
    
    def get_download_progress(self, model_name: str) -> Optional[ModelDownloadProgress]:
        """Get current download progress for a specific model"""
        
        # Check if model is already completed
        if self.check_model_exists(model_name):
            if model_name not in self.completed_models:
                self.completed_models.add(model_name)
            return ModelDownloadProgress(
                model_name=model_name,
                status="success",
                progress_percent=100.0
            )
        
        # Check if we have an active download
        if model_name not in self.active_downloads:
            return ModelDownloadProgress(
                model_name=model_name,
                status="not_started"
            )
        
        download_info = self.active_downloads[model_name]
        
        # Check for errors
        if "error" in download_info:
            return ModelDownloadProgress(
                model_name=model_name,
                status="error",
                error_message=download_info["error"]
            )
        
        # Check process status
        process = download_info.get("process")
        if not process:
            return ModelDownloadProgress(
                model_name=model_name,
                status="error",
                error_message="Process not found"
            )
        
        # Try to read progress from process output
        try:
            # Check if process is still running
            if process.poll() is None:
                # Process is still running - try to read output
                return ModelDownloadProgress(
                    model_name=model_name,
                    status="downloading",
                    progress_percent=None  # We can't get detailed progress without parsing ollama output
                )
            else:
                # Process has finished
                if process.returncode == 0:
                    # Success
                    self.completed_models.add(model_name)
                    return ModelDownloadProgress(
                        model_name=model_name,
                        status="success",
                        progress_percent=100.0
                    )
                else:
                    # Error
                    return ModelDownloadProgress(
                        model_name=model_name,
                        status="error",
                        error_message=f"Download failed with code {process.returncode}"
                    )
        
        except Exception as e:
            return ModelDownloadProgress(
                model_name=model_name,
                status="error",
                error_message=str(e)
            )
    
    def get_all_progress(self) -> Dict[str, ModelDownloadProgress]:
        """Get progress for all tracked models"""
        progress = {}
        
        # Check active downloads
        for model_name in self.active_downloads.keys():
            progress[model_name] = self.get_download_progress(model_name)
        
        # Check completed models
        for model_name in self.completed_models:
            if model_name not in progress:
                progress[model_name] = ModelDownloadProgress(
                    model_name=model_name,
                    status="success",
                    progress_percent=100.0
                )
        
        return progress
    
    def cleanup_completed(self):
        """Clean up completed download processes"""
        to_remove = []
        for model_name, download_info in self.active_downloads.items():
            if "process" in download_info:
                process = download_info["process"]
                if process.poll() is not None:  # Process has finished
                    to_remove.append(model_name)
        
        for model_name in to_remove:
            del self.active_downloads[model_name]
    
    def estimate_download_time(self, model_size_gb: float, speed_mbps: float = 50) -> str:
        """Estimate download time based on model size and connection speed"""
        if speed_mbps <= 0:
            return "Unknown"
        
        # Convert GB to Mb (Gigabytes to Megabits)
        model_size_mb = model_size_gb * 8000  # 1 GB = 8000 Mb
        
        # Calculate time in seconds
        time_seconds = model_size_mb / speed_mbps
        
        # Format as human-readable time
        if time_seconds < 60:
            return f"~{int(time_seconds)} seconds"
        elif time_seconds < 3600:
            minutes = int(time_seconds / 60)
            return f"~{minutes} minutes"
        else:
            hours = int(time_seconds / 3600)
            minutes = int((time_seconds % 3600) / 60)
            return f"~{hours}h {minutes}m"

# Global instance for easy access
ollama_progress_monitor = OllamaProgressMonitor()