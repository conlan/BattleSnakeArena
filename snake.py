import constants
import uuid

class Snake():
    def __init__(self, name, color, controller) -> None:
        self.name = name
        self.id = self.name + "-" + str(uuid.uuid4())
        self.color = color
        self.controller = controller
        
        self.reset()        

    def reset(self) -> None:
        self.body = []
        
        self.health = constants.MAX_SNAKE_HEALTH
        self.ate_food = False
        self.num_food_consumed = 0

        self.is_dead = False
        self.death_reason = None

    def is_alive(self) -> bool:
        return not self.is_dead
    
    def eat(self) -> None:
        self.health = constants.MAX_SNAKE_HEALTH
        self.ate_food = True
        self.num_food_consumed += 1
    
    def head(self) -> tuple:
        return self.body[0]

    def kill(self, reason) -> None:
        self.is_dead = True
        self.death_reason = reason

    def move(self, data) -> None:
        move_obj = self.controller.move(data)
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

        if len(self.body) > constants.SNAKE_START_SIZE and not self.ate_food:
            self.body = self.body[:-1]

        self.ate_food = False
        self.health = self.health -1
    
    def jsonize(self) -> dict:
        jsonobj = {}
        jsonobj["health"] = self.health
        jsonobj["body"] = [{"x": b[0], "y": b[1]} for b in self.body]
        jsonobj["id"] = self.id
        jsonobj["name"] = self.name
        return jsonobj

    def place_at_spot(self, spot) -> None:
        if (len(self.body) > 0):
            raise ValueError("Snake already placed")
        
        self.body.append(spot)