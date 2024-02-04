import constants

class Trainer():
    def __init__(self, controller) -> None:
        if controller is None:
            raise Exception("controller cannot be None")
        
        self.controller = controller

    def _lookup_reward(self, key) -> int:
        return constants.REWARD_SETS[self.controller.model.reward_set_key][key]

    def determine_reward(self, training_snake, game_results) -> int:
        total_reward = 0

        if (training_snake.is_dead):
            total_reward += self._lookup_reward(constants.REWARD_KEY_LOSE)
        else:
            total_reward += self._lookup_reward(constants.REWARD_KEY_SURVIVE)

        if (training_snake.ate_food):
            total_reward += self._lookup_reward(constants.REWARD_KEY_EAT)

        # check if training snake is winner
        if game_results is not None:
            winner = game_results["winner"]

            if (winner == training_snake):
                total_reward += self._lookup_reward(constants.REWARD_KEY_WIN)
        
        return total_reward

    def train(self, observation, next_observation, action, reward, done) -> None:
        if (action == None):
            print(f'Action is None, skipping training...')
            return
        
        state = observation["image"]
        next_state = observation["image"]
        
        print("Training: " + str(state) + " " + str(next_state) + " " + str(action) + " " + str(reward) + " " + str(done))        
    