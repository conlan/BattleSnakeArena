import constants

class Trainer():
    def __init__(self, controller, curr_step) -> None:
        if controller is None:
            raise Exception("controller cannot be None")
        
        self.controller = controller
        self.model = controller.model

        self.curr_step = curr_step

        self.burnin = 1_00
        self.learn_every = 3
        self.save_every = 100

    def _lookup_reward(self, key) -> int:
        return constants.REWARD_SETS[self.model.reward_set_key][key]

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

        training_snake.total_collected_reward += total_reward
        
        return total_reward

    def cache(self, observation, next_observation, action, reward, done) -> None:
        if (action == None):
            # print(f'    Action is None, skipping cache...')
            return

        self.model.cache(observation["image"], next_observation["image"], action, reward, done)

        self.curr_step += 1

        # print(f'    Current step: {self.curr_step}')
    
        if (self.curr_step < self.burnin):
            return
        
        # learn every few steps
        if (self.curr_step % self.learn_every == 0):
            loss = self.model.learn()

        # save every few steps
        if (self.curr_step % self.save_every == 0):
            training_info = self.controller.get_epsilon_info()
            training_info['curr_step'] = self.curr_step

            self.model.save_model(training_info)

            print("SAVED " + str(self.curr_step) + " " + str(training_info))