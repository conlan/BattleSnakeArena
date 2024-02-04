import random

from snake_controller import SnakeController

class DQNController (SnakeController):
    def __init__(self, model, epsilon_info, convert_data_to_image):
        super().__init__()

        self.model = model
        self.convert_data_to_image = convert_data_to_image

        self.load_epsilon(epsilon_info)

    def load_epsilon(self, epsilon_info) -> None:
        self.epsilon = epsilon_info["epsilon"]
        self.epsilon_decay = epsilon_info["epsilon_decay"]
        self.epsilon_min = epsilon_info["epislon_min"]

    def act(self, data) -> dict:
        game_id = data["game"]["id"]
        if (game_id not in self.moves_made):
            self.moves_made[game_id] = []

        move = None
        local_dir = None

        if (data['turn'] == 0):
            move = 'up'
        else:
            you = data['you']

            if (random.uniform(0.0, 1.0) < self.epsilon):
                local_dir = random.choice([0, 1, 2])
            else:
                obs = self.convert_data_to_image(data)

                local_dir = self.model.predict(obs)

            snakeHead = you['body'][0]
            snakeHead = (snakeHead['x'], snakeHead['y'])
            
            snakeNeck = you['body'][1]
            snakeNeck = (snakeNeck['x'], snakeNeck['y'])

            move = self.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
        self.moves_made[game_id].append(local_dir)
        
        return {
            'move': move,
            'local_direction': local_dir
        }