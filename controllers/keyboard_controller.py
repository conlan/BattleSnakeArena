from snake_controller import SnakeController

# import keyboard

class KeyboardController (SnakeController):
    def __init__(self, nickname):
        super().__init__(nickname)

        self.epsilon_info = {
            "curr_step" : 0
        }

        self.model = None

    def name(self) -> str:
        return "KeyboardController"
    
    def get_epsilon_info(self) -> dict:
        return {
            "epsilon" : 0,
            "epsilon_decay" : 0,
            "epsilon_min" : 0
        }

    def act(self, data) -> dict:
        game_id = data["game"]["id"]

        move = None
        
        while (move != "w") and (move != "a") and (move != "s") and (move != "d"):
            # get keyboard input
            move = input("Enter move: ").lower()

        if (move == "w"):
            move = "up"
        elif (move == "a"):
            move = "left"
        elif (move == "s"):
            move = "down"
        else:
            move = "right"
        
        return self.store_move(game_id, move, 0, [])