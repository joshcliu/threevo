"""
Base Agent Class

Abstract base class that defines the interface for all agents in ThreEvo.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


class BaseAgent(ABC):
    """Abstract base class for all agents in ThreEvo"""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        initial_prompt: str = "",
        api_key: Optional[str] = None
    ):
        """
        Initialize the base agent.

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for generation
            initial_prompt: Initial system prompt for the agent
            api_key: API key for the LLM service
        """
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = initial_prompt
        self.history: List[Dict[str, Any]] = []
        self.feedback_cache: List[str] = []

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    @abstractmethod
    def generate(self, problem: str) -> Any:
        """
        Generate output based on current prompt.

        Args:
            problem: Problem specification string

        Returns:
            Generated output (type varies by agent)
        """
        pass

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: User prompt to send to LLM
            system_prompt: Optional system prompt (uses self.prompt if not provided)

        Returns:
            LLM response as string
        """
        messages = []

        if system_prompt or self.prompt:
            messages.append(SystemMessage(content=system_prompt or self.prompt))

        messages.append(HumanMessage(content=prompt))

        response = self.llm.invoke(messages)
        return response.content

    def save_state(self) -> Dict[str, Any]:
        """
        Save agent state for checkpointing.

        Returns:
            Dictionary containing agent state
        """
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'prompt': self.prompt,
            'history': self.history,
            'feedback_cache': self.feedback_cache
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore agent from checkpoint.

        Args:
            state: Dictionary containing saved agent state
        """
        self.model_name = state.get('model_name', self.model_name)
        self.temperature = state.get('temperature', self.temperature)
        self.prompt = state.get('prompt', self.prompt)
        self.history = state.get('history', [])
        self.feedback_cache = state.get('feedback_cache', [])

    def add_feedback(self, feedback: str) -> None:
        """
        Add feedback to the cache.

        Args:
            feedback: Feedback string to cache
        """
        self.feedback_cache.append(feedback)

    def clear_feedback(self) -> None:
        """Clear the feedback cache."""
        self.feedback_cache = []
