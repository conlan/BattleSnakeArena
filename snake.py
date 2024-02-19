import constants
import uuid

from typing import List, TypedDict, Any 

JsonizedSnakeObj = TypedDict('JsonizedSnakeObj', {
    'health': int, 
    'body': List[dict[str, int]], 
    'id': str, 
    'name': str})

class Snake():
    def __init__(self, name, color, controller) -> None:
        self.name = name        
        self.color = color
        self.controller = controller
        self.id = "snake" + "-" + str(uuid.uuid4())        
        
        self.reset()        

    def reset(self) -> None:
        self.body:list[tuple] = []
        
        # this gets set and then every snake acts simultaneously
        self.prepared_move = None
        
        self.health = constants.MAX_SNAKE_HEALTH
        self.ate_food = False
        self.num_food_consumed = 0

        self.is_dead = False
        self.death_reason = None

        self.collected_reward = 0

    def is_alive(self) -> bool:
        return not self.is_dead
    
    def collect_reward(self, reward):
        self.collected_reward += reward
    
    def eat(self) -> None:
        self.health = constants.MAX_SNAKE_HEALTH
        self.ate_food = True
        self.num_food_consumed += 1

        # add another body segment at same position of the tail
        self.body.append(self.body[-1])
    
    def head(self) -> tuple:
        return self.body[0]
    
    def length(self) -> int:
        return len(self.body)

    def kill(self, reason) -> None:
        self.is_dead = True
        self.death_reason = reason

    # this gets called before move() so that all snakes can prepare their moves without affecting the game state
    def prepare_move(self, data) -> dict:
        # query controller for move
        self.prepared_move = self.controller.act(data)

    def execute_move(self) -> None:
        if (self.prepared_move == None):
            raise ValueError("Snake move not prepared")
        
        move_obj = self.prepared_move
        
        mv = move_obj["move"]
        
        head = self.body[0]

        if(mv == "left"):
            self.body = [(head[0]-1, head[1])] + self.body
        elif(mv == "right"):
            self.body = [(head[0]+1, head[1])] + self.body
        elif(mv == "down"):
            self.body =[(head[0], head[1]+1)] + self.body
        else:
            self.body = [(head[0], head[1]-1)] + self.body

        # remove the tail
        self.body = self.body[:-1]

        self.ate_food = False
        self.health = self.health -1

        # reset this back to None
        self.prepared_move = None
    
    def jsonize(self) -> JsonizedSnakeObj:
        jsonobj:JsonizedSnakeObj = {
            "health" : self.health,
            "body" : [{"x": b[0], "y": b[1]} for b in self.body],
            "id" : self.id,
            "name" : self.name
        }
        return jsonobj

    def place_at_spot(self, spot) -> None:
        if (len(self.body) > 0):
            raise ValueError("Snake already placed")
        
        for i in range (constants.SNAKE_START_SIZE):
            self.body.append(spot)