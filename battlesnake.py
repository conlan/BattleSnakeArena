import random
import time
import traceback
import uuid
import argparse
import tqdm
import requests
import traceback
from threading import Thread
from multiprocessing import Pool

import snakes

FOOD_COLOR = snakes.COLORS["green"]
BORDER_COLOR = snakes.COLORS["grey"]
DEFAULT_COLOR = snakes.COLORS["default"]

# TOOD make constants
SNAKE_START_SIZE = 3

BOARD_SIZE_SMALL = 7
BOARD_SIZE_MEDIUM = 11

MAX_SNAKE_HEALTH = 100

GAME_RESULT_DRAW = "DRAW"

VERBOSE = False

class BattleSnake():

    def __init__(self, dims=(BOARD_SIZE_MEDIUM,BOARD_SIZE_MEDIUM), food_rate=0.02, seed=None):
        self.seed = seed if seed else int(time.time()*10000)
        random.seed(self.seed)
        self.width = dims[0]
        self.height = dims[1]
        self.snakes = []
        self.turn = 0
        self.food = []
        self.food_prob = 0
        self.food_rate = food_rate

    def get_unoccupied_points(self, includePossibleMoves):
        unoccupiedPoints = []

        pointIsOccupied = {}

        for food in self.food:
            pointIsOccupied[food] = True

        for snake in self.snakes:
            # if we're including possible moves then look at where this snake can go
            if (includePossibleMoves):
                head_point = snake.body[0]

                nextMovePoints = [
                    (head_point[0] - 1, head_point[1]),
                    (head_point[0] + 1, head_point[1]),
                    (head_point[0], head_point[1] + 1),
                    (head_point[0], head_point[1] - 1)
                ]

                for possible_next_move in nextMovePoints:
                    pointIsOccupied[possible_next_move] = True
            
            for b in snake.body:
                pointIsOccupied[b] = True

        for y in range(self.height):
            for x in range(self.width):
                point = (x, y)

                if (point not in pointIsOccupied):
                    unoccupiedPoints.append(point)
                

        return unoccupiedPoints

    def place_food(self):
        # Follow standard placement rules at
        # https://github.com/BattlesnakeOfficial/rules/blob/main/board.go#L387
        centerCoord = ((self.width - 1) // 2, (self.height - 1) // 2)

        isSmallBoard = (self.width * self.height) < (BOARD_SIZE_MEDIUM * BOARD_SIZE_MEDIUM)

        if (len(self.snakes) <= 4) or not isSmallBoard:
            # place 1 food within exactly 2 moves of each snake, but never towards the center or in a corner
            for snake in self.snakes:
                snakeHead = snake.body[0]

                possibleFoodLocations = [
                    (snakeHead[0] - 1, snakeHead[1] - 1),
                    (snakeHead[0] - 1, snakeHead[1] + 1),
                    (snakeHead[0] + 1, snakeHead[1] - 1),
                    (snakeHead[0] + 1, snakeHead[1] + 1)
                ]

                availableFoodLocations = []

                for possibleFoodLocation in possibleFoodLocations:
                    # don't place food on center
                    if (possibleFoodLocation == centerCoord):
                        continue
                    # or on top of existing food
                    if (possibleFoodLocation in self.food):
                        continue

                    isAwayFromCenter = False

                    if (possibleFoodLocation[0] < snakeHead[0] and snakeHead[0] < centerCoord[0]):
                        isAwayFromCenter = True
                    elif (centerCoord[0] < snakeHead[0] and snakeHead[0] < possibleFoodLocation[0]):
                        isAwayFromCenter = True
                    elif (possibleFoodLocation[1] < snakeHead[1] and snakeHead[1] < centerCoord[1]):
                        isAwayFromCenter = True
                    elif (centerCoord[1] < snakeHead[1] and snakeHead[1] < possibleFoodLocation[1]):
                        isAwayFromCenter = True
                    
                    if not isAwayFromCenter:
                        continue
                        
                    # Don't spawn food in corners
                    if (possibleFoodLocation[0] == 0 or possibleFoodLocation[0] == (self.width-1)) and (possibleFoodLocation[1] == 0 or possibleFoodLocation[1] == (self.height-1)):
                        continue
                    
                    availableFoodLocations.append(possibleFoodLocation)

                # put 1 food on one of the available locations
                self.food.append(random.choice(availableFoodLocations))                                        
    
        # Place 1 food in center for dramatic purposes        
        if (centerCoord in self.get_unoccupied_points(True)):
            self.food.append(centerCoord)
            

    def place_snakes(self, snakes):
        # Place According to Standard Rules
        # https://github.com/BattlesnakeOfficial/rules/blob/main/board.go#L130
        min, mid, max = 1, (self.width-1) // 2, self.width - 2

        cornerPoints = [
		    [min, min],
		    [min, max],
		    [max, min],
		    [max, max]
        ]

        cardinalPoints = [
		    [min, mid],
		    [mid, min],
		    [mid, max],
		    [max, mid]
        ]

        if (len(snakes) > (len(cornerPoints) + len(cardinalPoints))):
            raise ValueError("Too many snakes for the board size")                

        random.shuffle(cornerPoints)
        random.shuffle(cardinalPoints)        

        startPoints = []

        if (random.random() < 0.5):
            startPoints = cornerPoints + cardinalPoints
        else:
            startPoints = cardinalPoints + cornerPoints

        for i in range(len(snakes)):
            snake = snakes[i]

            spot = tuple(startPoints[i])

            snake.body.append(spot)

            self.snakes.append(snake)

    def start_game(self, speed=50, output_board=True, debug=False):
        is_solo = (len(self.snakes) == 1)

        json = self._get_board_json()
        for s in self.snakes: s.start(json)

        while(True):
            t1 = time.time()
            self._move_snakes(debug=debug)
            self._detect_death()
            self._check_food()
            self._add_food()

            self.turn += 1

            if output_board: self.print_board()

            if self._check_winner(is_solo):
                break

            while(time.time()-t1 <= float(100-speed)/float(100)): pass
        
        if (len(self.snakes) == 0):
            return GAME_RESULT_DRAW
        
        return self.snakes[0].name if not is_solo else None


    def print_board(self):
        snakes = []
        for s in self.snakes:
            snakes.append(s.body)

        ywall = " " * 2 * self.width + "  "
        print(f"{BORDER_COLOR}{ywall}{DEFAULT_COLOR}") # Y Border
        for j in range(self.height):
            print(f"{BORDER_COLOR} {DEFAULT_COLOR}", end="") # X Border
            for i in range(self.width):
                if (i, j) in self.food:
                    print(f"{FOOD_COLOR}  {DEFAULT_COLOR}", end="") # Food
                else:
                    no_snake = True
                    for ind, s in enumerate(snakes):
                        if (i, j) in s:
                            if s[0] == (i, j):
                                print(f"{self.snakes[ind].color}OO{DEFAULT_COLOR}", end="") # Head
                            else:
                                print(f"{self.snakes[ind].color}  {DEFAULT_COLOR}", end="") # Body
                            no_snake = False
                    if no_snake:
                        print(f"{DEFAULT_COLOR}  {DEFAULT_COLOR}", end="") # Empty
            print(f"{BORDER_COLOR} {DEFAULT_COLOR}") # X Border
        print(f"{BORDER_COLOR}{ywall}{DEFAULT_COLOR}") # Y Border


    def _add_food(self):
        if random.random() < self.food_prob:
            self.food.append(self._empty_spot())
            self.food_prob = self.food_rate
        else:
            self.food_prob += self.food_rate


    def _empty_spot(self):
        taken = list(self.food)
        for s in self.snakes:
            taken.extend(s.body)

        spot = (random.choice(range(self.width)),
                random.choice(range(self.height)))
        while spot in taken:
            spot = (random.choice(range(self.width)),
                    random.choice(range(self.height)))
        return spot


    def _move_snakes(self, debug=False):        
        threads = []
        for snake in self.snakes:
            json = self._get_board_json()
            process = Thread(target=snake.move, args=(json,))
            process.start()
            threads.append(process)

        for process in threads:
            process.join()


    def _delete_snakes(self, snakes, reason=None):
        if not snakes == []:
            for s in snakes:
                if s in self.snakes:
                    s.end(self._get_board_json())
                    self.snakes.remove(s)


    def _resolve_head_collisions(self):
        del_snakes = []
        for s1 in self.snakes:
            for s2 in self.snakes:
                if s1 != s2:
                    if s2.body[0] == s1.body[0]:
                        if len(s1.body) > len(s2.body):
                            del_snakes.append(s2)

                        elif len(s1.body) < len(s2.body):
                            del_snakes.append(s1)
                        else:
                            del_snakes.append(s1)
                            del_snakes.append(s2)

        self._delete_snakes(del_snakes, reason="HEAD-ON-HEAD")


    def _detect_snake_collision(self):
        all_snakes = []
        for s in self.snakes:
            all_snakes.extend(s.body[1:])

        del_snakes = []
        for s in self.snakes:
            head = s.body[0]
            if head in all_snakes:
                del_snakes.append(s)

        self._delete_snakes(del_snakes, reason="SNAKE COLLISION")


    def _detect_wall_collision(self):
        del_snakes = []
        for s in self.snakes:
            head = s.body[0]
            if( head[0] < 0 or head[1] < 0 or
                head[0] >= self.width or
                head[1] >= self.height):
                del_snakes.append(s)

        self._delete_snakes(del_snakes, reason="WALL COLLISION")


    def _detect_starvation(self):
        del_snakes = []
        
        for s in self.snakes:
            if(s.health <= 0):
                del_snakes.append(s)

        self._delete_snakes(del_snakes, reason="STARVATION")


    def _check_food(self):
        removed_food = []

        for f in self.food:
            for s in self.snakes:
                if f in s.body:
                    s.health = MAX_SNAKE_HEALTH
                    s.ate_food = True

                    removed_food.append(f)
                    break

        self.food = [f for f in self.food if f not in removed_food]


    def _get_board_json(self):
        jsonobj = {}
        jsonobj["turn"] = self.turn
        jsonobj["board"] = {}
        jsonobj["board"]["height"] = self.height
        jsonobj["board"]["width"] = self.width
        jsonobj["board"]["snakes"] = [s.jsonize() for s in self.snakes]
        jsonobj["board"]["food"] = [{"x":f[0], "y":f[1]} for f in self.food]
        return jsonobj


    def _detect_death(self):
        self._detect_starvation()
        self._detect_wall_collision()
        self._detect_snake_collision()
        self._resolve_head_collisions()


    def _check_winner(self, is_solo):
        return (len(self.snakes) == 1 and not is_solo) or (len(self.snakes) == 0)







class Snake():

    def __init__(self, name=None, id=None, color=None, move=None, end=None, start=None, server=None, **kwargs):
        self.body = []
        self.health = MAX_SNAKE_HEALTH
        self.ate_food = False
        self.color = color if color else snakes.COLORS["red"]
        self.id = id if id else str(uuid.uuid4())
        self.name = name if name else self.id
        self._move = move
        self._start = start
        self._end = end
        self.server = server
        self.kwargs = kwargs


    def jsonize(self):
        jsonobj = {}
        jsonobj["health"] = self.health
        jsonobj["body"] = [{"x": b[0], "y": b[1]} for b in self.body]
        jsonobj["id"] = self.id
        jsonobj["name"] = self.name
        return jsonobj


    def move(self, data):
        data["you"] = self.jsonize()
        try:
            if self._move:
                r = self._move(data)
            elif self.server:
                url = self.server + "/move"
                r = requests.post(url, json=data).json()
        except Exception as e:
            traceback.print_exc()
            r = {"move": "up"}
        self._move_snake(r["move"])


    def start(self, data):
        data["you"] = self.jsonize()
        try:
            if self._start:
                self._start(data)
            elif self.server:
                url = self.server + "/start"
                requests.post(url, json=data)
        except Exception as e:
            traceback.print_exc()


    def end(self, data):
        data["you"] = self.jsonize()
        try:
            if self._end:
                self._end(data)
            elif self.server:
                url = self.server + "/end"
                requests.post(url, json=data)
        except Exception as e:
            traceback.print_exc()


    def _move_snake(self, mv):
        head = self.body[0]

        if(mv == "left"):
            self.body = [(head[0]-1, head[1])] + self.body
        elif(mv == "right"):
            self.body = [(head[0]+1, head[1])] + self.body
        elif(mv == "down"):
            self.body =[(head[0], head[1]+1)] + self.body
        else:
            self.body = [(head[0], head[1]-1)] + self.body


        if len(self.body) > SNAKE_START_SIZE and not self.ate_food:
            self.body = self.body[:-1]
        self.ate_food = False
        self.health = self.health -1


    def reset(self):
        self.body = []
        self.health = MAX_SNAKE_HEALTH
        self.ate_food = False








def verbose_print(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def run_game(snakes, food_rate=0.005, dims=(BOARD_SIZE_MEDIUM,BOARD_SIZE_MEDIUM), suppress_board=False, speed=80, quiet=False, seed=None):
    game = BattleSnake(food_rate=food_rate, dims=dims, seed=seed)    
    game.place_snakes(snakes)    
    game.place_food()

    game_results = {}
    game_results["winner"] = game.start_game(speed=speed, output_board=suppress_board, debug=True)
    game_results["turns"] = game.turn
    game_results["seed"] = game.seed

    if not quiet:
        print("Winner: {}, Turns: {}, Seed: {}".format(game_results["winner"], game_results["turns"], game_results["seed"] ))

    return game_results


def _run_game_from_args(args):
    return run_game(
        snakes=args.snakes,
        food_rate=args.food_rate,
        dims=args.dims,
        suppress_board=args.suppress_board,
        speed=args.speed,
        quiet=args.silent,
        seed=args.seed)


def parse_args(sysargs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--food_rate", help="Rate of food spawning", type=float, default=0.02)
    parser.add_argument("-s", "--snakes", nargs='+', help="Snakes to battle", type=str, default=["simpleJake", "battleJake2019"])
    parser.add_argument("-d", "--dims", nargs='+', help="Dimensions of the board in x,y", type=int, default=[BOARD_SIZE_MEDIUM,BOARD_SIZE_MEDIUM])
    parser.add_argument("-p", "--silent", help="Print information about the game", action="store_true", default=False)
    parser.add_argument("-g", "--games", help="Number of games to play", type=int, default=1)
    parser.add_argument("-b", "--suppress-board", help="Don't print the board", action="store_false", default=True)
    parser.add_argument("-t", "--threads", help="Number of threads to run multiple games on", type=int, default=4)
    parser.add_argument("-i", "--seed", help="Game seed", type=int, default=None)
    parser.add_argument("-sp", "--speed", help="Speed of the game", type=int, default=90)
    if sysargs:
        args = parser.parse_args(sysargs)
    else:
        args = parser.parse_args()

    if len(args.dims) == 1:
        args.dims = (args.dims[0], args.dims[0])
    elif len(args.dims) == 2:
        args.dims = tuple(args.dims)

    battle_snakes = []
    for input_snake in args.snakes:
        snek = [k for k in snakes.SNAKES if input_snake == k['name']]
        if len(snek) == 1:
            s = snek[0]
            battle_snakes.append(Snake(**s))
        else:
            verbose_print("Malformed snakes.py file or snakes input argument.")
    args.snakes = battle_snakes

    return args


def main():
    args = parse_args()

    if args.games == 1 or args.suppress_board:
        for i in range(args.games):
            _run_game_from_args(args)
    else:
        args.silent = True
        with Pool(args.threads) as p:
            outputs = list(tqdm.tqdm(p.imap_unordered(_run_game_from_args, [args for i in range(args.games)]), total=args.games))

        winners = [d["winner"] for d in outputs]

        for winner in set(winners):
            if (winner == GAME_RESULT_DRAW):
                print("Games Tied: {}".format(sum([1 for s in winners if s == winner])))
            else:
                print("{}, Games Won: {}".format(winner, sum([1 for s in winners if s == winner])))


if __name__ == "__main__":
    main()
