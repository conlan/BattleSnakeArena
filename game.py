import time
import random
# import constants
from threading import Thread

import uuid

# from snake import Snake

class GameParameters():
    def __init__(self, config, seed=None) -> None:
        self.seed = seed if seed else int(time.time()*10000)
        self.random = random.Random(self.seed)

        dims = config["board_size"]
        self.width = dims[0]
        self.height = dims[1]

        self.food_spawn_chance = config["food_spawn_chance"]
        self.min_food = config["min_food"]

    def is_small_board(self) -> bool:
        return (self.width * self.height) < (11 * 11) # TODO pull out magic number

class Game():
    def __init__(self, parameters, snakes) -> None:
        self.parameters = parameters
        self.id = "game-" + str(uuid.uuid4())
        self.all_snakes = snakes
        
        self.reset()

    def reset(self) -> bool:
        self.turn = 0
        self.food : list[tuple] = [] 
        self.live_snakes = []
        self.is_solo_game = len(self.all_snakes) == 1

        for snake in self.all_snakes:
            snake.reset()
            self.live_snakes.append(snake)

        self._place_snakes()
        self._place_food()

        return False
    
    def step(self) -> bool:
        self._move_snakes()            
        self._detect_death()
        self._check_for_eaten_food()
        self._spawn_food()

        self.turn += 1
        
        is_game_over = self._check_winner()
        
        return is_game_over
    
    def get_game_results(self) -> dict:
        is_game_over = self._check_winner()

        if (not is_game_over):
            return {}
        
        winning_snake = None

        if self.is_solo_game:
            winning_snake = self.all_snakes[0]
        else:
            if len(self.live_snakes) == 1:
                winning_snake = self.live_snakes[0]
            
        return {
            "id" : self.id,
            "turns" : self.turn,
            "winner" : winning_snake,
            "snakes" : self.all_snakes
        }
    
    def _check_winner(self) -> bool:
        return (len(self.live_snakes) == 1 and not self.is_solo_game) or (len(self.live_snakes) == 0)
    
    def _spawn_food(self) -> None:
        # Following Standard rules at
        # https://github.com/BattlesnakeOfficial/rules/blob/main/standard.go#L368
        numCurrFood = len(self.food)

        if (numCurrFood < self.parameters.min_food):
            self._place_food_randomly(self.parameters.min_food - numCurrFood)
        elif (self.parameters.food_spawn_chance > 0):            
            if (self.parameters.random.random() < self.parameters.food_spawn_chance):
                self._place_food_randomly(1)

    def _place_food_randomly(self, num_food):
        for _ in range(num_food):
            unoccupiedPoints = self._get_unoccupied_points(False)

            if (len(unoccupiedPoints) > 0):
                spot = self.parameters.random.choice(unoccupiedPoints)
                self.food.append(spot)
    
    def _move_snakes(self) -> None:
        # prepare moves for snakes
        for snake in self.live_snakes:
            snake_pov_json = self.get_board_json(snake)

            snake.prepare_move(snake_pov_json)          

        # execute moves for snakes
        for snake in self.live_snakes:
            snake.execute_move()        

    def _check_for_eaten_food(self):
        removed_food = []

        for f in self.food:
            for s in self.live_snakes:
                if f in s.body:
                    s.eat()                    

                    removed_food.append(f)
                    break

        self.food = [f for f in self.food if f not in removed_food]

    def _detect_death(self):
        self._detect_starvation()
        self._detect_wall_collision()
        self._detect_snake_collision()
        self._resolve_head_collisions()

    def _resolve_head_collisions(self):
        del_snakes = []

        for s1 in self.live_snakes:
            if (s1 in del_snakes):
                continue
            
            for s2 in self.live_snakes:
                if (s2 in del_snakes):
                    continue

                if s1 != s2:
                    if s2.head() == s1.head():
                        if (s1.length() > s2.length()):
                            del_snakes.append(s2)

                        elif (s1.length()) < (s2.length()):
                            del_snakes.append(s1)
                        else:
                            del_snakes.append(s1)
                            del_snakes.append(s2)

        self._delete_snakes(del_snakes, reason="HEAD")

    def _detect_snake_collision(self):
        snake_bodies = []

        for s in self.live_snakes:
            snake_bodies.extend(s.body[1:])

        del_snakes = []
        for s in self.live_snakes:
            head = s.head()
            if head in snake_bodies:
                del_snakes.append(s)

        self._delete_snakes(del_snakes, reason="COLLIDE")
    
    def _detect_starvation(self):
        del_snakes = []

        for s in self.live_snakes:
            if(s.health <= 0):
                del_snakes.append(s)
        
        self._delete_snakes(del_snakes, reason="STARVE")

    def _detect_wall_collision(self):
        del_snakes = []

        for s in self.live_snakes:
            head = s.head()
            if( head[0] < 0 or head[1] < 0 or
                head[0] >= self.width() or
                head[1] >= self.height()):
                del_snakes.append(s)          

        self._delete_snakes(del_snakes, reason="WALL")

    def _delete_snakes(self, snakes, reason):
        for s in snakes:
            s.kill(reason)
            self.live_snakes.remove(s)    

    def get_board_json(self, pov_snake):
        jsonobj = {}
        jsonobj["game"] = {
            "id" : self.id
        }
        jsonobj["turn"] = self.turn
        jsonobj["board"] = {}
        jsonobj["board"]["height"] = self.height()
        jsonobj["board"]["width"] = self.width()
        jsonobj["board"]["snakes"] = [s.jsonize() for s in self.live_snakes]
        jsonobj["board"]["food"] = [{"x":f[0], "y":f[1]} for f in self.food]
        
        jsonobj["you"] = pov_snake.jsonize()
            
        return jsonobj
    
    def width(self) -> int:
        return self.parameters.width
    
    def height(self) -> int:
        return self.parameters.height
    
    def _get_unoccupied_points(self, includePossibleMoves):
        occupied_points = list(self.food)

        for snake in self.live_snakes:
            occupied_points.extend(snake.body)

            # if we're including possible moves then look at where this snake can go
            if (includePossibleMoves):
                head_point = snake.head()

                nextMovePoints = [
                    (head_point[0] - 1, head_point[1]),
                    (head_point[0] + 1, head_point[1]),
                    (head_point[0], head_point[1] + 1),
                    (head_point[0], head_point[1] - 1)
                ]

                occupied_points.extend(nextMovePoints)

        unoccupiedPoints = []

        for y in range(self.height()):
            for x in range(self.width()):
                point = (x, y)

                if (point not in occupied_points):
                    unoccupiedPoints.append(point)
                
        return unoccupiedPoints

    # TODO move into rules module
    def _place_food(self) -> None:
        # Follow standard placement rules at
        # https://github.com/BattlesnakeOfficial/rules/blob/main/board.go#L387
        centerCoord = ((self.parameters.width - 1) // 2, (self.parameters.height - 1) // 2)

        if (len(self.live_snakes) <= 4) or not self.parameters.is_small_board():
            # place 1 food within exactly 2 moves of each snake, but never towards the center or in a corner
            for snake in self.live_snakes:
                snakeHead = snake.head()

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
                    if (possibleFoodLocation[0] == 0 or possibleFoodLocation[0] == (self.parameters.width-1)) \
                            and (possibleFoodLocation[1] == 0 or possibleFoodLocation[1] == (self.parameters.height-1)):
                        continue
                    
                    availableFoodLocations.append(possibleFoodLocation)

                # put 1 food on one of the available locations
                self.food.append(self.parameters.random.choice(availableFoodLocations))                                        
    
        # Place 1 food in center for dramatic purposes        
        if (centerCoord in self._get_unoccupied_points(True)):
            self.food.append(centerCoord)

    def _place_snakes(self) -> None:
        # Place According to Standard Rules
        # https://github.com/BattlesnakeOfficial/rules/blob/main/board.go#L130
        min, mid, max = 1, (self.parameters.width-1) // 2, self.parameters.width - 2

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

        if (len(self.live_snakes) > (len(cornerPoints) + len(cardinalPoints))):
            raise ValueError("Too many snakes for the board size")                

        self.parameters.random.shuffle(cornerPoints)
        self.parameters.random.shuffle(cardinalPoints)        

        startPoints = []

        if (self.parameters.random.random() < 0.5):
            startPoints = cornerPoints + cardinalPoints
        else:
            startPoints = cardinalPoints + cornerPoints

        for i in range(len(self.live_snakes)):
            snake = self.live_snakes[i]

            spot = tuple(startPoints[i])

            snake.place_at_spot(spot)