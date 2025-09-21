# Resource Monitor - System Resource Monitoring and Alerting
# Version: v1.0.0
# Date: 2025-08-30
# Purpose: Monitor system resources and provide user-friendly alerts about model performance

import streamlit as st
import psutil
import logging
from typing import Dict, Tuple
from cortex_engine.utils.smart_model_selector import smart_selector, MODEL_MEMORY_REQUIREMENTS

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources and provide user alerts"""
    
    def __init__(self):
        self.selector = smart_selector

    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory usage status"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent
        }

    def check_system_health(self) -> Dict[str, any]:
        """Check overall system health for AI workloads"""
        memory_status = self.get_memory_status()
        recommendations = self.selector.get_recommended_models()
        
        # Determine health status
        health_level = "good"
        warnings = []
        
        if memory_status["percent_used"] > 85:
            health_level = "critical"
            warnings.append("⚠️ System memory usage is critically high (>85%)")
            
        elif memory_status["percent_used"] > 70:
            health_level = "warning" 
            warnings.append("⚠️ System memory usage is high (>70%)")
            
        if memory_status["available_gb"] < 4.0:
            health_level = "critical"
            warnings.append("⚠️ Less than 4GB memory available - AI operations may fail")
            
        return {
            "health_level": health_level,
            "memory_status": memory_status,
            "warnings": warnings,
            "recommended_tier": recommendations["tier"],
            "recommended_model": recommendations["text_model"]
        }

    def display_resource_warning_if_needed(self):
        """Display Streamlit warning if system resources are constrained"""
        health = self.check_system_health()
        
        if health["health_level"] == "critical":
            st.error(f"""
            🚨 **Critical Resource Warning**
            
            {' '.join(health['warnings'])}
            
            **Current Status:**
            - Memory Usage: {health['memory_status']['percent_used']:.1f}%
            - Available: {health['memory_status']['available_gb']:.1f}GB
            
            **Recommendations:**
            - Close unnecessary applications to free memory
            - Consider restarting the application
            - Using efficient model: `{health['recommended_model']}`
            """)
            
        elif health["health_level"] == "warning":
            st.warning(f"""
            ⚠️ **Resource Usage High**
            
            {' '.join(health['warnings'])}
            
            System is automatically using efficient models to prevent crashes.
            Currently selected: `{health['recommended_model']}`
            """)

    def display_model_selection_info(self):
        """Display information about automatic model selection"""
        recommendations = self.selector.get_recommended_models()
        system_summary = self.selector.get_system_summary()
        
        with st.expander("🧠 Model Selection Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**System Resources:**")
                st.write(f"• Total Memory: {system_summary['system_memory_gb']:.1f}GB")
                if system_summary['is_docker']:
                    docker_limit = system_summary.get('docker_memory_limit_gb')
                    if docker_limit:
                        st.write(f"• Docker Limit: {docker_limit:.1f}GB")
                    else:
                        st.write("• Docker: No memory limit")
                st.write(f"• Available: {system_summary['available_memory_gb']:.1f}GB")
                
            with col2:
                st.markdown("**Selected Models:**")
                st.write(f"• Text: `{recommendations['text_model']}`")
                st.write(f"• Vision: `{recommendations['vision_model']}`")
                st.write(f"• Tier: **{recommendations['tier'].title()}**")
                
            st.info(f"💡 {recommendations['description']}")
            
            if recommendations['tier'] == 'efficient':
                st.markdown("""
                **Why efficient models?** Your system has been optimized for:
                - ✅ Stable performance without memory crashes
                - ✅ Fast inference times
                - ✅ Excellent quality for most tasks
                
                To use powerful models (mistral-small3.2), ensure 32GB+ RAM is available.
                """)

    def check_model_compatibility(self, model_name: str) -> Tuple[bool, str]:
        """Check if a specific model is compatible with current system"""
        return self.selector.can_run_model(model_name)

    def display_model_compatibility_warning(self, model_name: str):
        """Display warning if attempting to use incompatible model"""
        can_run, message = self.check_model_compatibility(model_name)
        
        if not can_run:
            st.error(f"""
            🚫 **Model Compatibility Issue**
            
            {message}
            
            **Alternative:** The system will automatically use `{self.selector.get_recommended_models()['text_model']}` instead.
            """)
            return False
        return True

# Global instance for easy importing
resource_monitor = ResourceMonitor()

def display_resource_status():
    """Convenience function to display resource status in Streamlit"""
    resource_monitor.display_resource_warning_if_needed()
    resource_monitor.display_model_selection_info()

def check_model_before_use(model_name: str) -> bool:
    """Convenience function to check model compatibility before use"""
    return resource_monitor.display_model_compatibility_warning(model_name)