"""
Workflow Orchestrator
Provides simplified, reusable workflow management for complex multi-step processes.

Version: 1.0.0
Date: 2025-08-13
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass 
class WorkflowStep:
    """Defines a single step in a workflow."""
    id: str
    title: str
    description: str
    execute_function: Optional[Callable] = None
    requirements: List[str] = None  # List of required session state keys
    auto_advance: bool = True
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []

class WorkflowState(Enum):
    """Standard workflow states."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    ERROR = "error"

class WorkflowOrchestrator:
    """
    Simplified workflow management for complex multi-step processes.
    
    This orchestrator separates business logic from UI presentation,
    making workflows easier to understand, test, and maintain.
    """
    
    def __init__(self, workflow_name: str, steps: List[WorkflowStep]):
        """
        Initialize workflow orchestrator.
        
        Args:
            workflow_name: Name of the workflow for logging/session state
            steps: List of workflow steps in order
        """
        self.workflow_name = workflow_name
        self.steps = {step.id: step for step in steps}
        self.step_order = [step.id for step in steps]
        self.current_step_id = None
        self.workflow_data = {}
        self.status_callback = None
        
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback function for status updates."""
        self.status_callback = callback
        
    def _log_status(self, message: str):
        """Log status message using callback or logger."""
        if self.status_callback:
            self.status_callback(message)
        else:
            logger.info(message)
    
    def initialize_workflow(self, session_data: Dict[str, Any]) -> str:
        """
        Initialize or resume workflow from session data.
        
        Args:
            session_data: Current session state data
            
        Returns:
            ID of current step
        """
        # Determine current step from session data
        step_key = f"{self.workflow_name}_current_step"
        
        if step_key in session_data and session_data[step_key] in self.steps:
            self.current_step_id = session_data[step_key]
        else:
            # Start from first step
            self.current_step_id = self.step_order[0]
            session_data[step_key] = self.current_step_id
            
        # Load workflow data from session
        data_key = f"{self.workflow_name}_data"
        if data_key in session_data:
            self.workflow_data = session_data[data_key]
        else:
            self.workflow_data = {}
            session_data[data_key] = self.workflow_data
            
        self._log_status(f"Initialized {self.workflow_name}: Step {self.current_step_id}")
        return self.current_step_id
    
    def get_current_step(self) -> WorkflowStep:
        """Get current workflow step."""
        return self.steps[self.current_step_id]
    
    def get_step_progress(self) -> Dict[str, Any]:
        """Get workflow progress information."""
        current_index = self.step_order.index(self.current_step_id)
        
        return {
            "current_step": self.current_step_id,
            "current_step_title": self.steps[self.current_step_id].title,
            "step_number": current_index + 1,
            "total_steps": len(self.steps),
            "progress": (current_index + 1) / len(self.steps),
            "completed_steps": self.step_order[:current_index],
            "remaining_steps": self.step_order[current_index + 1:]
        }
    
    def can_execute_step(self, step_id: str, session_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if step can be executed based on requirements.
        
        Args:
            step_id: Step to check
            session_data: Current session state
            
        Returns:
            Tuple of (can_execute, missing_requirements)
        """
        step = self.steps[step_id]
        missing_requirements = []
        
        for requirement in step.requirements:
            if requirement not in session_data or not session_data[requirement]:
                missing_requirements.append(requirement)
                
        return len(missing_requirements) == 0, missing_requirements
    
    def execute_step(self, step_id: str = None, session_data: Dict[str, Any] = None, 
                    **kwargs) -> Dict[str, Any]:
        """
        Execute a workflow step.
        
        Args:
            step_id: Step to execute (defaults to current)
            session_data: Session state data
            **kwargs: Additional arguments to pass to step function
            
        Returns:
            Dict containing execution results
        """
        if step_id is None:
            step_id = self.current_step_id
            
        step = self.steps[step_id]
        
        if not step.execute_function:
            return {"status": "error", "message": f"No execute function defined for step {step_id}"}
        
        # Check requirements
        if session_data:
            can_execute, missing = self.can_execute_step(step_id, session_data)
            if not can_execute:
                return {
                    "status": "error",
                    "message": f"Missing requirements: {', '.join(missing)}",
                    "missing_requirements": missing
                }
        
        try:
            self._log_status(f"Executing step: {step.title}")
            
            # Execute step function
            result = step.execute_function(**kwargs)
            
            if isinstance(result, dict) and result.get("status") == "success":
                # Store results in workflow data
                self.workflow_data[f"{step_id}_result"] = result
                
                # Auto-advance if configured
                if step.auto_advance:
                    self.advance_to_next_step(session_data)
                
                self._log_status(f"Completed step: {step.title}")
                
            return result
            
        except Exception as e:
            error_msg = f"Step execution failed: {str(e)}"
            self._log_status(f"❌ {error_msg}")
            logger.error(f"Step {step_id} execution error: {e}")
            
            return {
                "status": "error",
                "message": error_msg,
                "step_id": step_id
            }
    
    def advance_to_next_step(self, session_data: Dict[str, Any]) -> Optional[str]:
        """
        Advance to next step in workflow.
        
        Args:
            session_data: Session state to update
            
        Returns:
            Next step ID or None if at end
        """
        current_index = self.step_order.index(self.current_step_id)
        
        if current_index < len(self.step_order) - 1:
            next_step_id = self.step_order[current_index + 1]
            self.current_step_id = next_step_id
            
            # Update session state
            step_key = f"{self.workflow_name}_current_step"
            session_data[step_key] = next_step_id
            
            self._log_status(f"Advanced to step: {self.steps[next_step_id].title}")
            return next_step_id
        else:
            self._log_status("Workflow completed!")
            return None
    
    def reset_workflow(self, session_data: Dict[str, Any]):
        """Reset workflow to beginning."""
        self.current_step_id = self.step_order[0]
        self.workflow_data = {}
        
        # Clear session data
        step_key = f"{self.workflow_name}_current_step"
        data_key = f"{self.workflow_name}_data"
        
        session_data[step_key] = self.current_step_id
        session_data[data_key] = self.workflow_data
        
        self._log_status(f"Reset {self.workflow_name} to beginning")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get complete workflow summary for display/export."""
        return {
            "workflow_name": self.workflow_name,
            "current_step": self.current_step_id,
            "progress": self.get_step_progress(),
            "data": self.workflow_data,
            "steps": [
                {
                    "id": step.id,
                    "title": step.title,
                    "description": step.description,
                    "completed": step.id in [s for s in self.step_order 
                                           if self.step_order.index(s) < self.step_order.index(self.current_step_id)]
                }
                for step in self.steps.values()
            ]
        }