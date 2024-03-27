import random
import constants

from snake_controller import SnakeController
from ppo.ppo_model import PPOModel

class PPOController (SnakeController):
    def __init__(self, model_save_dir, nickname, label, convert_data_to_image, should_action_mask):
        super().__init__(nickname)

        self.convert_data_to_image = convert_data_to_image
        
        self.should_action_mask = should_action_mask

        self.model = PPOModel(label, model_save_dir)

    def name(self) -> str:
        return "PPOController (model=" + self.model.model_save_dir + ")"

    def act(self, data) -> dict:
        move = None
        local_dir = None
        action_prob = None
        value = None

        if (data['turn'] == 0):
            move = random.choice(['up', 'down', 'left', 'right'])
        else:
            you = data['you']

            snakeHead = you['body'][0]
            snakeHead = (snakeHead['x'], snakeHead['y'])
            
            snakeNeck = you['body'][1]
            snakeNeck = (snakeNeck['x'], snakeNeck['y'])

            obs_data = self.convert_data_to_image(data)

            local_dir, action_prob, value, actor_probs = self.model.predict(obs_data)

            move = self.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck)
            
        game_id = data["game"]["id"]
        
        return self.store_ppo_move(game_id, move, local_dir, action_prob, value)
