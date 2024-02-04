from abc import ABC, abstractmethod

class SnakeController(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, data) -> dict:
        pass