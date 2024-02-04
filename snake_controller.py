from abc import ABC, abstractmethod

class SnakeController(ABC):
    def __init__(self) -> None:
        self.moves_made = {}

    @abstractmethod
    def act(self, data) -> dict:
        pass
    
    def get_last_move_made(self, game) -> int:
        return self.moves_made[game.id][-1]