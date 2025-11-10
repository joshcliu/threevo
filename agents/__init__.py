"""
ThreEvo Agents Module

This module contains the three core agents and orchestration logic:
- BaseAgent: Abstract base class for all agents
- CoderAgent: Generates code solutions
- TesterAgent: Generates test suites
- ReasoningAgent: Independently validates through reasoning
- ThreEvoOrchestration: Three-way validation and feedback logic
"""

from .base_agent import BaseAgent
from .coder_agent import CoderAgent
from .tester_agent import TesterAgent
from .reasoning_agent import ReasoningAgent
from .orchestration import ThreEvoOrchestration

__all__ = [
    'BaseAgent',
    'CoderAgent',
    'TesterAgent',
    'ReasoningAgent',
    'ThreEvoOrchestration',
]
