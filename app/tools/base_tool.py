from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Abstract base class for every tool.
    """

    name = ""
    description = ""

    @abstractmethod
    def run(self, **kwargs):
        """
        Execute the tool.
        """
        pass