from abc import ABC, abstractmethod

import constants

class SnakeController(ABC):
    def __init__(self, nickname) -> None:
        self.moves_made:dict = {}
        self.nickname = nickname

    @abstractmethod
    def act(self, data) -> dict:
        pass
    
    def get_last_move(self, game) -> dict:
        return self.moves_made[game.id][-1]
    
    def get_last_local_direction(self, game) -> int:
        return self.moves_made[game.id][-1]['local_direction']
    
    @abstractmethod
    def name(self) -> str:
        pass        
    
    def store_move(self, game_id, move, local_direction, q_values=None) -> dict:
        move_obj = {
            'move': move,
            'local_direction': local_direction,
            'q_values': q_values
        }
        
        if (game_id not in self.moves_made):
            self.moves_made[game_id] = []

        self.moves_made[game_id].append(move_obj)

        return move_obj
    
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
    
    def get_snakes_together(self, snakes):
        snakesTogether = []

        for s in snakes:
             for b in s["body"]:
                snakesTogether.append(( b['x'], b['y'] ))

        return snakesTogether
    
    def dont_hit_wall(self, moves, head, walls):
        if head[0] == walls[0]-1 and 'right' in moves:
            moves.remove('right')

        elif head[0] == 0 and 'left' in moves:
            moves.remove('left')

        if head[1] == 0 and 'up' in moves:
            moves.remove('up')

        elif head[1] == walls[1]-1 and 'down' in moves:
            moves.remove('down')

        return moves