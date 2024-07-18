from abc import ABC, abstractmethod
from typing import Any

from langchain.agents.agent import AgentExecutor
from langchain_core.prompts import BasePromptTemplate


class BaseAgent(ABC):

    @abstractmethod
    def create_agent(self, **kwargs: Any) -> AgentExecutor:
        pass

    @abstractmethod
    def get_prompt(self, **kwargs: Any) -> BasePromptTemplate:
        pass
