import constants

class Trainer():
    def __init__(self, controller, reward_set) -> None:
        if controller is None:
            raise Exception("controller cannot be None")

        if reward_set is None:
            raise Exception("reward_set cannot be None")
        
        self.controller = controller
        self.reward_set = reward_set

    def determine_reward(self, training_snake, game_results) -> int:
        total_reward = 0

        if (training_snake.is_dead):
            total_reward += self.reward_set[constants.REWARD_KEY_LOSE]
        else:
            total_reward += self.reward_set[constants.REWARD_KEY_SURVIVE]

        if (training_snake.ate_food):
            total_reward += self.reward_set[constants.REWARD_KEY_EAT]

        # check if training snake is winner
        if game_results is not None:
            winner = game_results["winner"]

            if (winner == training_snake):
                total_reward += self.reward_set[constants.REWARD_KEY_WIN]
        
        return total_reward

    def train(self, observation, next_observation, action, reward, done) -> None:
        if (action == None):
            raise Exception("action cannot be None")
        
        state = observation["image"]
        next_state = observation["image"]
        
        print("Training: " + str(state) + " " + str(next_state) + " " + str(action) + " " + str(reward) + " " + str(done))
        pass
    