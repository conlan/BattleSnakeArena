import random

from snake_controller import SnakeController

from dqn.dqn_model import DQNModel

class DQNController (SnakeController):
    def __init__(self, model_save_path, convert_data_to_image):
        super().__init__()

        self.convert_data_to_image = convert_data_to_image

        self.model = DQNModel()

        self.epsilon_info = self.model.load_model(model_save_path)

        self.load_epsilon(self.epsilon_info)

    def get_epsilon_info(self) -> dict:
        return {
            "epsilon" : self.epsilon,
            "epsilon_decay" : self.epsilon_decay,
            "epsilon_min" : self.epsilon_min
        }

    def load_epsilon(self, epsilon_info) -> None:
        self.epsilon = epsilon_info["epsilon"]
        self.epsilon_decay = epsilon_info["epsilon_decay"]
        self.epsilon_min = epsilon_info["epsilon_min"]

        print(f'Loaded epsilon: {self.epsilon}, decay: {self.epsilon_decay}, min: {self.epsilon_min}')

    def act(self, data) -> dict:
        move = None
        local_dir = None
        q_values = None

        if (data['turn'] == 0):
            move = random.choice(['up', 'down', 'left', 'right'])
        else:
            you = data['you']

            if (random.uniform(0.0, 1.0) < self.epsilon):
                local_dir = random.choice([0, 1, 2])
            else:
                obs = self.convert_data_to_image(data)

                local_dir, q_values = self.model.predict(obs)

            snakeHead = you['body'][0]
            snakeHead = (snakeHead['x'], snakeHead['y'])
            
            snakeNeck = you['body'][1]
            snakeNeck = (snakeNeck['x'], snakeNeck['y'])

            move = self.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
        game_id = data["game"]["id"]
        
        return self.store_move(game_id, move, local_dir, q_values)