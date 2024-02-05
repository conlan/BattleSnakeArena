import constants

class Trainer():
    def __init__(self, controller, curr_step) -> None:
        if controller is None:
            raise Exception("controller cannot be None")
        
        self.controller = controller
        self.model = controller.model

        self.curr_step = curr_step

        self.burnin = 10_000
        self.learn_every = 3
        self.save_every = 5_000

        self.learning_losses = {}
        self.max_food_consumed = 0
        self.max_reward_collected = -99999999999999
        self.max_turns_survived = 0

    def _lookup_reward(self, key) -> int:
        return constants.REWARD_SETS[self.model.reward_set_key][key]
    
    def finalize(self, game_results, training_snake) -> None:
        game_id = game_results["id"]

        learning_losses_for_game = self.learning_losses[game_id] if game_id in self.learning_losses else []
        
        if (len(learning_losses_for_game) == 0):
            mean_learning_loss = 0
        else:
            mean_learning_loss = sum(learning_losses_for_game) / len(learning_losses_for_game)

        if (training_snake.num_food_consumed > self.max_food_consumed):
            self.max_food_consumed = training_snake.num_food_consumed

        if (training_snake.total_collected_reward > self.max_reward_collected):
            self.max_reward_collected = training_snake.total_collected_reward

        num_turns = game_results["turns"]
        if (num_turns > self.max_turns_survived):
            self.max_turns_survived = num_turns

        game_results["training"] = {
            "curr_step" : self.curr_step,
            "curr_epsilon" : self.controller.get_epsilon_info()["epsilon"],
            "total_reward" : training_snake.total_collected_reward,
            "max_reward_collected" : self.max_reward_collected,
            "total_food_consumed" : training_snake.num_food_consumed,
            "max_food_consumed" : self.max_food_consumed,
            "num_turns" : num_turns,
            "max_turns_survived" : self.max_turns_survived,
            "death_reason" : training_snake.death_reason,
            "mean_learning_loss" : mean_learning_loss
        }

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

    def cache(self, game, observation, next_observation, action_idx, reward, done) -> None:
        if (action_idx == None):
            print(f'    Action is None, skipping cache...')
            return
        
        # convert action_idx to an action tensor
        action = [0, 0, 0]        
        action[action_idx] = 1

        self.model.cache(observation["image"], next_observation["image"], action, reward, done)

        self.curr_step += 1

        # print(f'    Current step: {self.curr_step}')
    
        if (self.curr_step < self.burnin):
            return
        
        # learn every few steps
        if (self.curr_step % self.learn_every == 0):
            loss = self.model.learn()

            if game.id not in self.learning_losses:
                self.learning_losses[game.id] = []
            
            if (loss is not None):
                self.learning_losses[game.id].append(loss)

        # save every N steps
        if (self.curr_step % self.save_every == 0):
            self.save_state()            

    def save_state(self):
        training_info = self.controller.get_epsilon_info()
        training_info['curr_step'] = self.curr_step

        self.model.save_model(training_info)

        print("\nSAVED " + str(self.curr_step) + " " + str(training_info) + "\n")