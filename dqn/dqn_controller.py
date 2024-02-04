from snake_controller import SnakeController

class DQNController (SnakeController):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def act(self, data) -> dict:
        move = 'up'
        local_dir = 0

        game_id = data["game"]["id"]
        if (game_id not in self.moves_made):
            self.moves_made[game_id] = []

        self.moves_made[game_id].append(local_dir)
    
        return {
            'move': move
        }