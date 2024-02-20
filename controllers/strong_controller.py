"""
Logic taken from Josh Hartmann's battleJake2019 2019 entry.
"""

import random
from snake_controller import SnakeController

# The higher the quicker we start searching for food
HUNGRY = 60
# Starting search radius
FOOD_MIN = 1
# When we start searching our max radius
STARVING = 40
# Max searching radius
FOOD_MAX = 40

SIZE_THRESHOLD = 0

# TODO factor reused opponent controller logic into a base controller class

class StrongController (SnakeController):
    def __init__(self, nickname, seed=1):
        super().__init__(nickname)

        self.random = random.Random()
        self.random.seed(seed)

        self.ate_food_last_turn = False
        
    def name(self) -> str:
        return "StrongController"

    def act(self, data) -> dict:
        # Get all the data
        you = data['you']
        you['body'] = [ (b['x'], b['y']) for b in you['body'] ]
        you['head'] = you['body'][0]
        you['size'] = len(you['body'])
        # health = you["health"]
        walls = (data['board']['width'], data['board']['height'])
        
        snakesTogether = []
        
        for s in data['board']['snakes']:
             for b in s["body"]:
                snakesTogether.append(( b['x'], b['y'] ))

        snakes = data["board"]['snakes']
        for s in snakes:
            s['size'] = len(s['body'])
            s['body'] = [ (b['x'], b['y']) for b in s['body'] ]
            s['head'] = s['body'][0]
            s['tail'] = s['body'][-1]

        food = [(f['x'], f['y']) for f in data['board']['food']]

        move = None
        moves = ['left', 'right', 'up', 'down']

        # Moving restrictions
        if self.ate_food_last_turn:
            moves = self.dont_hit_wall(moves, you['head'], walls)
            
            moves = self.dont_hit_snakes(moves, you['head'], snakesTogether, [])
            
            moves = self.dont_get_eaten(moves, you, snakes)            
        else:
            moves = self.dont_hit_wall(moves, you['head'], walls)
            moves = self.dont_hit_snakes(moves, you['head'], snakesTogether, [you['body'][-1]])
            moves = self.dont_get_eaten(moves, you, snakes)

        # Don't choose nothing that'll kill you next time
        if len(moves) > 1:
            tmpMoves = list(moves)
            for m in moves:
                nextHead = self.get_space(you['head'], m)
                nextMoves = ['left', 'right', 'up', 'down']
                nextMoves = self.dont_hit_wall(nextMoves, nextHead, walls)
                nextMoves = self.dont_hit_snakes(nextMoves, nextHead, snakesTogether + [you['head']], [])
                if nextMoves == []:
                    tmpMoves.remove(m)
            if tmpMoves != []:
                moves = tmpMoves

        # Take food as first preference if I'm smol
        if you['size'] < 6:
            you["health"] = you["health"]/2

        # Take food as preference as I get more hungry
        if self.have_choice(move, moves) and (you["health"] < HUNGRY):
            maxFood = round( (1 - ((you["health"]-STARVING) / (HUNGRY-STARVING))) * (FOOD_MAX-FOOD_MIN) )

            for i in reversed(range(1, maxFood)):
                if self.have_choice(move, moves):
                    moves = self.get_food(moves, you['head'], food, i)

        if self.have_choice(moves, moves):
            move = self.strangle_others(moves, you['head'], you['size'], you['body'], snakes, walls)

        # Flee from a wall as preference
        if self.have_choice(move, moves):
            moves = self.flee_wall(moves, walls, you['head'])

        # Flee others (including yourself) as preference
        if self.have_choice(move, moves):
            moves = self.flee_others(moves, [you['body'][0], you['body'][-1]], snakesTogether,you['head'], 1)

        # Take killing others as preference
        if self.have_choice(move, moves):
            moves = self.eat_others(moves, you['head'], you['size'], snakes)

        # Move away from the heads of others
        if self.have_choice(move, moves):
            moves = self.flee_heads(moves, snakes, you['head'], dist=3)

        # Go straight as preference
        if self.have_choice(move, moves):
            move = self.go_straight(moves, you['head'], you['body'])

        # Flee heads
        if self.have_choice(move, moves):
            moves = self.flee_heads(moves, snakes, you['head'])

        if self.have_choice(move, moves):
            move = self.random.choice(moves)

        # No suggested moves
        if move == None:
            # There is only one choice
            if len(moves) == 1:
                move = moves[0]

            # There is no choice
            else:
                moves = self.eat_tail(you['head'], snakes)
                
                moves = self.dont_get_eaten(moves, you, snakes, sameSize=False)
                if moves != []:
                    move = moves[0]

                if move == None:
                    moves = ['left', 'right', 'up', 'down']
                    moves = self.dont_hit_wall(moves, you['head'], walls)

                    moves = self.dont_hit_snakes(moves, you['head'], snakesTogether, [])
                    moves = self.dont_get_eaten(moves, you, snakes, sameSize=False)

                    if moves == []:
                        move = 'up'
                    else:
                        move = self.random.choice(moves)
        return {
            'move': move
        }
    
    def go_straight(self, moves, head, body):
        if len(body) > 1:
            pm = self.get_previous_move(head, body[1])
            if pm in moves:
                return pm
    
    def flee_heads(self, moves, snakes, head, dist=999):
        headManhattan = [ abs(s['head'][0]-head[0]) + abs(s['head'][1]-head[1]) for s in snakes]
        closestSnakes = sorted( [(x,i) for (i,x) in enumerate(headManhattan)] )

        tmpMoves = list(moves)

        for s in closestSnakes:
            snake = snakes[s[1]]
            xdist = head[0] - snake['body'][0][0]
            ydist = head[1] - snake['body'][0][1]

            if len(moves) == 1:
                return moves

            if abs(xdist) < abs(ydist) and xdist < dist:
                if ('left' in moves) and (xdist > 0):
                    moves.remove('left')

                if ('right' in moves) and (xdist < 0):
                    moves.remove('right')
            elif ydist < dist:
                if ('down' in moves) and (ydist < 0):
                    moves.remove('down')

                if ('up' in moves) and (ydist > 0):
                    moves.remove('up')
            else:
                return moves

        if moves == []:
            moves = tmpMoves
        return moves
    
    def eat_others(self, moves, head, mySize, snakes):
        validMoves = []
        for s in snakes:

            if s['size'] < mySize-1:
                xdist = s['head'][0]-head[0]
                ydist = s['head'][1]-head[1]

                if (abs(xdist) == 1) and (abs(ydist) == 1):
                    if xdist > 0 and 'right' in moves:
                        validMoves.append('right')

                    elif xdist < 0 and 'left' in moves:
                        validMoves.append('left')

                    if ydist > 0 and 'down' in moves:
                        validMoves.append('down')

                    elif ydist < 0 and 'up' in moves:
                        validMoves.append('up')

                elif (abs(xdist) == 2 and ydist == 0) or (abs(ydist) == 2 and xdist == 0):
                    if xdist == 2 and 'right' in moves:
                        validMoves.append('right')

                    elif xdist == -2 and 'left' in moves:
                        validMoves.append('left')

                    elif ydist == 2 and 'down' in moves:
                        validMoves.append('down')

                    elif 'up' in moves:
                        validMoves.append('up')

        if validMoves == []:
            return moves
        return list(set(validMoves))
    
    def flee_others(self, moves, delMoves, snakesTogether, head, dist):
        prevMoves = list(moves)
        validMoves = list(moves)
        for s in snakesTogether:
            if s not in delMoves:
                for m in moves:
                    fh = self.get_space(head, m)
                    xdist = s[0]-fh[0]
                    ydist = s[1]-fh[1]

                    # If the future head is beside a snake
                    if (abs(xdist) == dist and ydist == 0) or (abs(ydist) == dist and xdist == 0):
                        validMoves.remove(m)
                moves = validMoves

        if moves == []:
            return prevMoves
        return moves
    
    def eat_tail(self, head, snakes):
        moves = []
        for s in snakes:
            xdist = head[0] - s['tail'][0]
            ydist = head[1] - s['tail'][1]
            if abs(xdist) == 1 and ydist == 0:
                if xdist > 0:
                    moves.append('left')
                else:
                    moves.append('right')

            if abs(ydist) == 1 and xdist == 0:
                if ydist > 0:
                    moves.append('up')
                else:
                    moves.append('down')
        return moves
    
    def flee_wall(self, moves, walls, head):
        # Flee the wall if I'm against it
        if head[0] >= walls[0]-1:
            if 'left' in moves:
                return ['left']

        elif head[0] <= 0:
            if 'right' in moves:
                return ['right']

        if head[1] <= 0:
            if 'down' in moves:
                return ['down']

        elif head[1] >= walls[1]-1:
            if 'up' in moves:
                return ['up']

        validMoves = list(moves)

        # Keep 1 space buffer between you and the wall
        if head[0] >= walls[0]-2:
            if 'right' in moves:
                validMoves.remove('right')

        elif head[0] <= 1:
            if 'left' in moves:
                validMoves.remove('left')

        if len(moves) > 1:

            if head[1] <= 1:
                if 'up' in moves:
                    validMoves.remove('up')

            elif head[1] >= walls[1]-2:
                if 'down' in moves:
                    validMoves.remove('down')


        if validMoves == []:
            return moves
        return validMoves

    def have_choice(self, move, moves):
        if move != None:
            return False
        if len(moves) <= 1:
            return False
        return True
    
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