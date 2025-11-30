# ## File: cortex_engine/idea_generator.py
# Version: 2.0.0 (Refactored Modular Architecture)
# Date: 2025-08-08
# Purpose: Compatibility wrapper for refactored Idea Generator module
#          Maintains backward compatibility while using new modular structure

"""
Idea Generator - Refactored Modular Architecture

This module has been refactored from a monolithic 2,429-line file into
a clean modular structure for better maintainability:

- idea_generator/core.py: Main IdeaGenerator class
- idea_generator/double_diamond.py: Double Diamond methodology  
- idea_generator/agents.py: Multi-agent ideation system
- idea_generator/export.py: Results export functionality

This file serves as a compatibility wrapper to maintain existing imports.
"""

# Import the refactored classes
from .idea_generator.core import IdeaGenerator
from .idea_generator.double_diamond import DoubleDiamondProcessor
from .idea_generator.agents import IdeationAgents
from .idea_generator.export import IdeaExporter

# Maintain backward compatibility
__all__ = [
    'IdeaGenerator',
    'DoubleDiamondProcessor', 
    'IdeationAgents',
    'IdeaExporter'
]

# Legacy alias for backward compatibility
IdeaGeneratorEngine = IdeaGenerator