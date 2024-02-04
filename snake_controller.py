from abc import ABC, abstractmethod

import constants

class SnakeController(ABC):
    def __init__(self) -> None:
        self.moves_made = {}

    @abstractmethod
    def act(self, data) -> dict:
        pass
    
    def get_last_move_made(self, game) -> int:
        return self.moves_made[game.id][-1]
    
    # Take head and neck coordinates and convert a local direction to 
    # actual BattleSnake move
    def getLocalDirectionAsMove(self, dir, snakeHead, snakeNeck):
        head_x, head_y = snakeHead[0], snakeHead[1]
        neck_x, neck_y = snakeNeck[0], snakeNeck[1]

        if (dir == constants.LocalDirection.STRAIGHT):
            # Go straight         
            if (neck_y == (head_y + 1)): # snake is facing UP
                return 'up'
            elif (neck_y == (head_y - 1)): # snake is facing DOWN
                return 'down'
            elif (neck_x == (head_x - 1)): # snake is facing RIGHT
                return 'right'
            else: # snake is facing LEFT
                return 'left'
        elif (dir == constants.LocalDirection.LEFT):
            # Turn Left
            if (neck_y == (head_y + 1)): # snake is facing UP
                return 'left'
            elif (neck_y == (head_y - 1)): # snake is facing DOWN
                return 'right'
            elif (neck_x == (head_x - 1)): # snake is facing RIGHT
                return 'up'
            else: # snake is facing LEFT
                return 'down'
        elif (dir == constants.LocalDirection.RIGHT):
            # Turn Left
            if (neck_y == (head_y + 1)): # snake is facing UP
                return 'right'
            elif (neck_y == (head_y - 1)): # snake is facing DOWN
                return 'left'
            elif (neck_x == (head_x - 1)): # snake is facing RIGHT
                return 'down'
            else: # snake is facing LEFT
                return 'up'
        
        return 'up'