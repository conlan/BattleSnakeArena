import random
from snake_controller import SnakeController

import uuid

FOOD_THRESHOLD = 30
FOOD_SEARCH_DIST = 10

class SimpleController (SnakeController):
    def __init__(self, seed=1):
        self.random = random.Random()
        self.random.seed(seed)

    def act(self, data) -> dict:
        # Get all the data
        you = data['you']
        you['body'] = [ (b['x'], b['y']) for b in you['body'] ]
        you['head'] = you['body'][0]
        you['size'] = len(you['body'])
        # health = you["health"]
        walls = (data['board']['width'], data['board']['height'])
        snakesTogether = []
        [ [ snakesTogether.append(( b['x'], b['y'] )) for b in s["body"] ] for s in data['board']['snakes'] ]

        snakes = data["board"]['snakes']
        for s in snakes:
            s['size'] = len(s['body'])
            s['body'] = [ (b['x'], b['y']) for b in s['body'] ]
            s['head'] = s['body'][0]
            s['tail'] = s['body'][-1]

        food = [(f['x'], f['y']) for f in data['board']['food']]

        moves = ['left', 'right', 'up', 'down']

        # Don't hit the walls
        movesTmp = self.dont_hit_wall(moves, you['head'], walls)
        if movesTmp != []:
            moves = movesTmp

        # Don't hit other snakes
        movesTmp = self.dont_hit_snakes(moves, you['head'], snakesTogether, [])
        if movesTmp != []:
            moves = movesTmp

        # Don't get eaten
        movesTmp = self.dont_get_eaten(moves, you, snakes)
        if movesTmp != []:
            moves = movesTmp

        # Search for food if your health is low
        if you['size'] < FOOD_THRESHOLD:
            for i in range(1, FOOD_SEARCH_DIST):
                movesTmp = self.get_food(moves, you['head'], food, i)
                if movesTmp != []:
                    moves = movesTmp
                    break

        # Choose the previous move or a random one
        if you['size'] != 1:
            previous_move = self.get_previous_move(you['head'], you['body'][1])
        else:
            previous_move = 'noMoveYall'
        if previous_move in moves:
            move = previous_move
        elif moves != []:
            move = self.random.choice(moves)
        else:
            # if we've eliminated all possible moves then
            # default to a move where you have at least a chance at survival
            # e.g. don't hit a snake
            chance_of_survival_moves = self.dont_hit_snakes(['left', 'right', 'up', 'down'], you['head'], snakesTogether, [])
            # and don't hit a wall
            chance_of_survival_moves = self.dont_hit_wall(chance_of_survival_moves, you['head'], walls)
            
            if (chance_of_survival_moves != []):
                move = self.random.choice(chance_of_survival_moves)
            else:
                # if no possible moves then at this point just move up
                move = 'up'

        return {
            'move': move
        }
    
    def get_food(self, moves, head, food, dist):
        validMoves = []
        for f in food:
            xdist = f[0]-head[0]
            ydist = f[1]-head[1]

            if (abs(xdist) + abs(ydist)) <= dist:

                if xdist > 0 and 'right' in moves:
                    validMoves.append('right')

                elif xdist < 0 and 'left' in moves:
                    validMoves.append('left')

                elif ydist > 0 and 'down' in moves:
                    validMoves.append('down')

                elif ydist < 0 and 'up' in moves:
                    validMoves.append('up')

        return validMoves
    
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


    def dont_hit_snakes(self, moves, head, snakesTogether, ignore):
        if self.get_space(head, 'left') in snakesTogether and 'left' in moves:
            moves.remove('left')

        if self.get_space(head, 'right') in snakesTogether and 'right' in moves:
            moves.remove('right')

        if self.get_space(head, 'up') in snakesTogether and 'up' in moves:
            moves.remove('up')

        if self.get_space(head, 'down') in snakesTogether and 'down' in moves:
            moves.remove('down')

        return moves
    
    def get_space(self, space, move):
        if move == 'left':
            return (space[0] - 1, space[1])

        elif move == 'right':
            return (space[0] + 1, space[1])

        elif move == 'up':
            return (space[0], space[1] - 1)

        else:
            return (space[0], space[1] + 1)


    def dont_get_eaten(self, moves, you, snakes, sameSize=True):
        for s in snakes:
            if (s['size'] >= you['size']) and sameSize or \
            (s['size'] > you['size']) and not sameSize:
                xdist = s['head'][0]-you['head'][0]
                ydist = s['head'][1]-you['head'][1]

                if abs(xdist) == 1 and abs(ydist) == 1:
                    if xdist > 0 and 'right' in moves:
                        moves.remove('right')

                    elif xdist < 0 and 'left' in moves:
                        moves.remove('left')

                    if ydist > 0 and 'down' in moves:
                        moves.remove('down')

                    elif ydist < 0 and 'up' in moves:
                        moves.remove('up')

                elif (abs(xdist) == 2 and ydist == 0) or (abs(ydist) == 2 and xdist == 0):
                    if xdist == 2 and 'right' in moves:
                        moves.remove('right')

                    elif xdist == -2 and 'left' in moves:
                        moves.remove('left')

                    elif ydist == 2 and 'down' in moves:
                        moves.remove('down')

                    elif ydist == -2 and 'up' in moves:
                        moves.remove('up')

        return moves
    
    def get_previous_move(self, head, second):
        if head[0] == second[0]:
            if head[1] > second[1]:
                return 'down'

            else:
                return 'up'
        else:
            if head[0] > second[0]:
                return 'right'

            else:
                return 'left'