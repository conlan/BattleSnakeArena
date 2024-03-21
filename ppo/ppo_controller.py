import random
import constants

from snake_controller import SnakeController
from ppo.ppo_model import PPOModel

class PPOController (SnakeController):
    def __init__(self, model_save_path, nickname, convert_data_to_image, should_action_mask):
        super().__init__(nickname)

        self.convert_data_to_image = convert_data_to_image
        
        self.should_action_mask = should_action_mask

        self.model = PPOModel()

    def name(self) -> str:
        return "PPOController (model=" + self.model.model_save_path + ")"

    def act(self, data) -> dict:
        move = None
        local_dir = None
        q_values = None

        if (data['turn'] == 0):
            move = random.choice(['up', 'down', 'left', 'right'])
        else:
            you = data['you']

            snakeHead = you['body'][0]
            snakeHead = (snakeHead['x'], snakeHead['y'])
            
            snakeNeck = you['body'][1]
            snakeNeck = (snakeNeck['x'], snakeNeck['y'])

            obs_data = self.convert_data_to_image(data)

            local_dir, _ = self.model.predict(obs_data)

            move = self.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck)
            
        game_id = data["game"]["id"]
        
        return self.store_move(game_id, move, local_dir, q_values)
