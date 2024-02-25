import random
import constants
from snake_controller import SnakeController

class LoopController (SnakeController):
    def __init__(self, seed=1):
        super().__init__("loop")

        self.random = random.Random(seed)        

    def name(self) -> str:
        return "LoopController"

    def act(self, data) -> dict:
        # random move for first turn
        if (data['turn'] == 0):
            move = self.random.choice(['up', 'down', 'left', 'right'])
        else:
            you = data['you']

            snakeHead = you['body'][0]
            snakeHead = (snakeHead['x'], snakeHead['y'])
            
            snakeNeck = you['body'][1]
            snakeNeck = (snakeNeck['x'], snakeNeck['y'])

            local_dir = constants.LocalDirection.RIGHT

            move = self.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck)

        return {
            'move': move
        }
