import time
import random

class ArenaParameters():
    def __init__(self, dims, food_spawn_chance, seed=None) -> None:
        self.seed = seed if seed else int(time.time()*10000)
        self.random = random.Random(self.seed)

        self.width = dims[0]
        self.height = dims[1]

        self.food_spawn_chance = food_spawn_chance

    def is_small_board(self) -> bool:
        return (self.width * self.height) < (11 * 11) # TODO pull out magic number

class Arena():
    def __init__(self, parameters, snakes) -> None:
        self.parameters = parameters
        self.snakes = snakes
        self.food = []

    def reset(self) -> bool:
        for snake in self.snakes:
            snake.reset()

        self._place_snakes()
        self._place_food()

        return False
    
    def step(self) -> bool:
        print("Step")
        return True
    
    def width(self) -> int:
        return self.parameters.width
    
    def height(self) -> int:
        return self.parameters.height
    
    def _get_unoccupied_points(self, includePossibleMoves):
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

        for y in range(self.parameters.height):
            for x in range(self.parameters.width):
                point = (x, y)

                if (point not in occupied_points):
                    unoccupiedPoints.append(point)
                
        return unoccupiedPoints

    def _place_food(self) -> None:
        # Follow standard placement rules at
        # https://github.com/BattlesnakeOfficial/rules/blob/main/board.go#L387
        centerCoord = ((self.parameters.width - 1) // 2, (self.parameters.height - 1) // 2)

        if (len(self.snakes) <= 4) or not self.parameters.is_small_board():
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

        if (len(self.snakes) > (len(cornerPoints) + len(cardinalPoints))):
            raise ValueError("Too many snakes for the board size")                

        self.parameters.random.shuffle(cornerPoints)
        self.parameters.random.shuffle(cardinalPoints)        

        startPoints = []

        if (self.parameters.random.random() < 0.5):
            startPoints = cornerPoints + cardinalPoints
        else:
            startPoints = cardinalPoints + cornerPoints

        for i in range(len(self.snakes)):
            snake = self.snakes[i]

            spot = tuple(startPoints[i])

            snake.place_at_spot(spot)