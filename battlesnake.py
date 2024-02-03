import random
import time
import traceback
import uuid
import argparse
import requests
import traceback
from threading import Thread
from multiprocessing import Pool

import snakes
import rl_utils

FOOD_COLOR = snakes.COLORS["green"]
BORDER_COLOR = snakes.COLORS["grey"]
DEFAULT_COLOR = snakes.COLORS["default"]
DEFAULT_SPEED = 90

# TOOD make constants
SNAKE_START_SIZE = 3

BOARD_SIZE_SMALL = 7
BOARD_SIZE_MEDIUM = 11

MAX_SNAKE_HEALTH = 100

GAME_RESULT_DRAW = "DRAW"

VERBOSE = False

class BattleSnake():
    def __init__(self, dims=(BOARD_SIZE_MEDIUM,BOARD_SIZE_MEDIUM), food_spawn_chance=0, min_food=1, seed=None):
        self.seed = seed if seed else int(time.time()*10000)
        self.random = random.Random(self.seed)
        
        self.width = dims[0]
        self.height = dims[1]
        self.snakes = []
        self.turn = 0
        self.food = []
        self.min_food = min_food
        self.food_spawn_chance = food_spawn_chance

    def get_unoccupied_points(self, includePossibleMoves):
        occupied_points = list(self.food)

        for snake in self.snakes:
            occupied_points.extend(snake.body)

            # if we're including possible moves then look at where this snake can go
            if (includePossibleMoves):
                head_point = snake.body[0]

                nextMovePoints = [
                    (head_point[0] - 1, head_point[1]),
                    (head_point[0] + 1, head_point[1]),
                    (head_point[0], head_point[1] + 1),
                    (head_point[0], head_point[1] - 1)
                ]

                occupied_points.extend(nextMovePoints)

        unoccupiedPoints = []

        for y in range(self.height):
            for x in range(self.width):
                point = (x, y)

                if (point not in occupied_points):
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
                self.food.append(self.random.choice(availableFoodLocations))                                        
    
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

        self.random.shuffle(cornerPoints)
        self.random.shuffle(cardinalPoints)        

        startPoints = []

        if (self.random.random() < 0.5):
            startPoints = cornerPoints + cardinalPoints
        else:
            startPoints = cardinalPoints + cornerPoints

        for i in range(len(snakes)):
            snake = snakes[i]

            spot = tuple(startPoints[i])

            snake.body.append(spot)

            self.snakes.append(snake)

    def start_game(self, speed=DEFAULT_SPEED, output_board=True, train_reinforcement=False, record_train_reinforcement_video=False):
        is_solo_game = (len(self.snakes) == 1)
        
        for s in self.snakes:
            json = self._get_board_json()
            json["model_save_path"] = s.model_save_path
            s.start(json)

        # if we want to record a video of the training snake
        board_image_frames_for_recording = []

        # keep a reference to the training snake it case it gets killed
        snake_in_training = self.snakes[0]

        while (True):                       
            # if we're training RL then convert the current board to an image
            rl_state_image, snakes_health = None, None    
            if (train_reinforcement) or (record_train_reinforcement_video):
                rl_state_image, snakes_health = rl_utils.convertBoardToState(self._get_board_json(), snake_in_training.id)                
                
                if record_train_reinforcement_video:
                    board_image_frames_for_recording.append(rl_state_image)

            t1 = time.time()
            self._move_snakes()            
            self._detect_death()
            self._check_for_eaten_food()
            self._spawn_food()

            is_game_over = self._check_winner(is_solo_game)

            # if we're training RL then grab an updated board image as the next state
            # OR if we're recording RL video and this is the final turn after game over
            if (train_reinforcement) or (record_train_reinforcement_video):
                # if the training snake was killed
                training_snake_was_killed = (snake_in_training not in self.snakes)
                # generate a new state after all the snakes have moved
                next_rl_state_image, next_snake_healths = rl_utils.convertBoardToState(self._get_board_json(), snake_in_training.id)                

                if (train_reinforcement):
                    # if the training snake was just killed its health will not be in the health array so insert it here
                    if (training_snake_was_killed):
                        next_snake_healths.insert(0, 0.0)                        
                            
                    # determine reward for snake
                    training_reward = 0
                    
                    if (training_snake_was_killed):
                        training_reward = rl_utils.get_training_reward( \
                            snake_in_training.training_reward_index, "REWARD_FOR_DEATH")
                    else:
                        training_reward = rl_utils.get_training_reward( \
                            snake_in_training.training_reward_index, "REWARD_FOR_SURVIVAL")

                        # if food eaten
                        if (snake_in_training.ate_food):
                            training_reward += rl_utils.get_training_reward( \
                                snake_in_training.training_reward_index, "REWARD_FOR_FOOD")

                        # add training reward if snake wins (not applicable in solo games)                    
                        if (is_game_over and not training_snake_was_killed):
                            training_reward += rl_utils.get_training_reward( \
                                snake_in_training.training_reward_index, "REWARD_FOR_VICTORY")
            
                    # if game is over OR if the training snake was killed
                    training_is_done = is_game_over or training_snake_was_killed

                    snake_in_training.total_accumulated_reward += training_reward

                    # send to snake in training for cache
                    rl_state_obj = {
                        "image" : rl_state_image,
                        "health" : snakes_health
                    }
                    next_rl_state_obj = {
                        "image" : next_rl_state_image,
                        "health" : next_snake_healths
                    }
                    snake_in_training.cache(rl_state_obj, next_rl_state_obj, training_reward, training_is_done)

                # when snakes > 2, end game if training_snake_was_killed
                if not is_solo_game and training_snake_was_killed:
                    is_game_over = True

                # append final frame if we're recording video
                if (record_train_reinforcement_video and is_game_over):
                    board_image_frames_for_recording.append(next_rl_state_image)  

            self.turn += 1

            if output_board: self.print_board()

            if (is_game_over):
                break

            if (speed < 100):
                while(time.time()-t1 <= float(100-speed)/float(100)): pass

        if (record_train_reinforcement_video):
            rl_utils.output_to_video(board_image_frames_for_recording)
        
        # if there's no snakes left
        if (len(self.snakes) == 0):
            # if this was a solo game then the first snake won
            if (is_solo_game):
                return snake_in_training.name
            else:
                return GAME_RESULT_DRAW
            
        # TODO append remaining snake names together here
        
        return self.snakes[0].name if not is_solo_game else None


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


    def _spawn_food(self):
        # Following Standard rules at
        # https://github.com/BattlesnakeOfficial/rules/blob/main/standard.go#L368
        numCurrFood = len(self.food)

        if (numCurrFood < self.min_food):
            self._place_food_randomly(self.min_food - numCurrFood)
        elif (self.food_spawn_chance > 0):            
            if (self.random.random() < self.food_spawn_chance):
                self._place_food_randomly(1)

    def _place_food_randomly(self, num_food):
        for _ in range(num_food):
            unoccupiedPoints = self.get_unoccupied_points(False)

            if (len(unoccupiedPoints) > 0):
                spot = self.random.choice(unoccupiedPoints)
                self.food.append(spot)

    def _empty_spot(self):
        unoccupied_points = self.get_unoccupied_points(False)        

        spot = (self.random.choice(range(self.width)),
                self.random.choice(range(self.height)))
        while spot not in unoccupied_points:
            spot = (self.random.choice(range(self.width)),
                    self.random.choice(range(self.height)))
        return spot

    def _move_snakes(self):
        threads = []
        
        for snake in self.snakes:
            json = self._get_board_json()
            process = Thread(target=snake.move, args=(json,))
            threads.append(process)

        # Start these threads outside of the setup loop, otherwise
        # they may get called early and effect the board json for
        # subsequent snakes which throws off their move calcaluations
        for process in threads:
            process.start()
            
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


    def _check_for_eaten_food(self):
        removed_food = []

        for f in self.food:
            for s in self.snakes:
                if f in s.body:
                    s.health = MAX_SNAKE_HEALTH
                    s.ate_food = True
                    s.num_food_consumed += 1

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

    def _check_winner(self, is_solo_game):
        return (len(self.snakes) == 1 and not is_solo_game) or (len(self.snakes) == 0)

class Snake():
    def __init__(self, name=None, id=None, color=None, move=None, end=None, start=None, cache=None, server=None, training_reward_index=None, **kwargs):
        self.body = []
        self.health = MAX_SNAKE_HEALTH
        self.ate_food = False
        self.num_food_consumed = 0
        self.last_move_local_direction = 0
        self.color = color if color else snakes.COLORS["red"]
        self.id = id if id else str(uuid.uuid4())
        self.name = name if name else self.id
        self._move = move
        self._cache = cache
        
        self.training_losses = []
        self.training_epsilon = 0
        self.training_curr_step = 0
        self.training_reward_index = training_reward_index if training_reward_index else 0
        self.total_accumulated_reward = 0
        self.model_save_path = kwargs.get("model_save_path", None)

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

        snake_move = r["move"]
        snake_move_local_direction = r["local_direction"] if "local_direction" in r else None
        
        self._move_snake(snake_move, snake_move_local_direction)

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

    def cache(self, rl_state_obj, next_rl_state_obj, reward, done):
        if (self.last_move_local_direction is None):
            return
        
        action = [0, 0, 0]
        action[self.last_move_local_direction] = 1
                
        try:
            if (self._cache):
                results = self._cache(rl_state_obj, next_rl_state_obj, reward, action, done)
                
                if ('loss' in results):
                    loss = results['loss']
                    
                    self.training_losses.append(float("{:.4f}".format(loss)))

                if ('epsilon' in results):
                    self.training_epsilon = results['epsilon']

                if ('curr_step' in results):
                    self.training_curr_step = results['curr_step']
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

    def _move_snake(self, mv, local_dir):
        self.last_move_local_direction = local_dir

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

def run_game(snake_types, food_spawn_chance, min_food, dims=(BOARD_SIZE_MEDIUM,BOARD_SIZE_MEDIUM), suppress_board=False, train_reinforcement=False, record_train_reinforcement_video=False, randomize_opponent_count=False, speed=DEFAULT_SPEED, silent=False, seed=None):
    snakes = [Snake(**snake_type) for snake_type in snake_types]

    if (randomize_opponent_count):
        # determine a random number of opponents for each game
        num_opponents = random.randint(1, 3)
        # pop snakes until we have the right number of opponents
        while (len(snakes) > (num_opponents + 1)):
            snakes.pop(random.randint(1, len(snakes) - 1))

    game = BattleSnake(food_spawn_chance=food_spawn_chance, min_food=min_food, dims=dims, seed=seed)
    game.place_snakes(snakes)    
    game.place_food()

    game_results = {}
    game_results["winner"] = game.start_game(speed=speed, output_board=(not suppress_board), train_reinforcement=train_reinforcement, record_train_reinforcement_video=record_train_reinforcement_video)
    game_results["turns"] = game.turn
    game_results["seed"] = game.seed
    game_results["num_snakes"] = len(snakes)

    # Stats for training reporting
    snake_in_training = snakes[0]

    game_results["training_losses"] = snake_in_training.training_losses
    game_results["training_epsilon"] = snake_in_training.training_epsilon
    game_results["training_curr_step"] = snake_in_training.training_curr_step
    game_results["training_food_consumed"] = snake_in_training.num_food_consumed
    game_results["total_accumulated_reward"] = snake_in_training.total_accumulated_reward

    if not silent:
        print("Winner: {}, Turns: {}, Reward: {}, Seed: {}".format(game_results["winner"], \
            game_results["turns"], game_results["total_accumulated_reward"], game_results["seed"] ))

    return game_results

def _run_game_from_args(args):
    return run_game(
        snake_types=args.snake_types,
        food_spawn_chance=args.food_spawn_chance,
        min_food=args.min_food,
        dims=args.dims,
        suppress_board=args.suppress_board,
        train_reinforcement=args.train_reinforcement,
        record_train_reinforcement_video=args.record_train_reinforcement_video,
        randomize_opponent_count=args.randomize_opponent_count,
        speed=args.speed,
        silent=args.silent,
        seed=args.seed)

def parse_args(sysargs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--food_spawn_chance", help="Chance of food spawning", type=float, default=0.15)
    parser.add_argument("-mf", "--min_food", help="Minimum number of food", type=float, default=1)
    parser.add_argument("-s", "--snakes", nargs='+', help="Snakes to battle", type=str, default=["simpleJake", "battleJake2019", "battleJake2019", "battleJake2019", "DQNConlan2024"])
    parser.add_argument("-d", "--dims", nargs='+', help="Dimensions of the board in x,y", type=int, default=[BOARD_SIZE_MEDIUM,BOARD_SIZE_MEDIUM])
    parser.add_argument("-p", "--silent", help="Print information about the game", action="store_true", default=False)
    parser.add_argument("-g", "--games", help="Number of games to play", type=int, default=1)
    parser.add_argument("-dis", "--discord_webhook_url", nargs='+', help="Discord webhook for reporting", type=str, default=None)
    parser.add_argument("-tb", "--tensor_board_dir", nargs='+', help="Output data to tensorboard dir", type=str, default=None)
    parser.add_argument("-b", "--suppress_board", help="Don't print the board", action="store_false", default=True)
    parser.add_argument("-rl", "--train_reinforcement", help="Whether we should run in RL mode", action="store_true", default=False)
    parser.add_argument("-roc", "--randomize_opponent_count", help="Whether we should randomize opponent counts during RL training", action="store_true", default=False)
    parser.add_argument("-rec", "--record_train_reinforcement_video", help="Whether we should record a video of the reinforcement train", action="store_true", default=False)
    parser.add_argument("-i", "--seed", help="Game seed", type=int, default=None)
    parser.add_argument("-sp", "--speed", help="Speed of the game", type=int, default=DEFAULT_SPEED)
    if sysargs:
        args = parser.parse_args(sysargs)
    else:
        args = parser.parse_args()

    if len(args.dims) == 1:
        args.dims = (args.dims[0], args.dims[0])
    elif len(args.dims) == 2:
        args.dims = tuple(args.dims)

    if (args.discord_webhook_url):
        args.discord_webhook_url = args.discord_webhook_url[0]

    if (args.tensor_board_dir):
        args.tensor_board_dir = args.tensor_board_dir[0]
    
    snake_types = []
    for input_snake in args.snakes:
        snek = [k for k in snakes.SNAKES if input_snake == k['name']]
        if len(snek) == 1:
            s = snek[0]
            snake_types.append(s)            
        else:
            verbose_print("Malformed snakes.py file or snakes input argument.")
    args.snake_types = snake_types

    return args

def main():
    args = parse_args()

    running_winners = {}
    
    for i in range(args.games):
        game_results = _run_game_from_args(args)

        num_snakes = game_results["num_snakes"]

        # winner tracking
        winner = game_results["winner"]
        if (num_snakes not in running_winners):
            running_winners[num_snakes] = []
        running_winners[num_snakes].append(winner)    

    for snake_count in running_winners:
        winners = running_winners[snake_count]

        for winner in set(winners):
            if (winner == GAME_RESULT_DRAW):
                print(f'{snake_count}-player, Games Tied: {sum([1 for s in winners if s == winner])}')
            else:
                print(f'{snake_count}-player, {winner} Won: {sum([1 for s in winners if s == winner])}')

if __name__ == "__main__":
    main()