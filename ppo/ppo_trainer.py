import constants

class PPOTrainer():
    def __init__(self, controller) -> None:
        if controller is None:
            raise Exception("controller cannot be None")
        
        self.controller = controller

        self.reset()

    def reset(self):
        self.clear_game_history()
        self.curr_step = self.controller.model.curr_step if self.controller.model is not None else 0

        self.max_food_consumed = 0
        self.max_reward_collected = -99999999999999
        self.max_turns_survived = 0
        
        self.num_training_updates_made = 0        

    def clear_game_history(self):
        self.learning_losses = {}
        self.winner_counts = {}
        self.total_collected_reward = 0

    def learn(self, game):
        loss = self.controller.model.learn()

        # we didn't learn
        if (loss == 0):
            return    
        else:
            self.num_training_updates_made += self.controller.model.n_epochs

            if game.id not in self.learning_losses:
                self.learning_losses[game.id] = []

            self.learning_losses[game.id].append(loss)

    def remember(self, state, action, probs, vals, reward, done):
        if (action == None):
            return
        
        self.curr_step += 1
        
        self.controller.model.store_memory(state, action, probs, vals, reward, done)

    def _lookup_reward(self, key) -> int:
        return constants.REWARD_SETS[self.controller.model.reward_set_key][key]
    
    def calculate_win_rate(self, winner_name) -> float:
        num_games_won_by_winner = self.winner_counts[winner_name] if winner_name in self.winner_counts else 0
        total_games = sum(self.winner_counts.values())
        win_rate = num_games_won_by_winner * 1.0 / total_games * 100.0        
        win_rate = round(win_rate, 3) # format win rate to 3 decimal places

        return win_rate
    
    def print_training_result(self, game_results, game_index, num_games) -> None:
        # determine winner and win rate
        winner = game_results["winner"] if game_results["winner"] is not None else None
        winner_name = winner.name if winner is not None else "DRAW"  

        training_snake_win_rate = self.calculate_win_rate(self.controller.nickname)              
                
        num_turns = game_results["turns"]

        collected_reward = game_results["training"]["collected_reward"]
        collected_reward = round(collected_reward, 4)
        
        food_consumed = game_results["training"]["food_consumed"]
        
        mean_learning_loss = game_results["training"]["mean_learning_loss"]
        # reduce to 4 decimal places
        mean_learning_loss = round(mean_learning_loss, 8)

        training_snake_death_reason = game_results["training"]["death_reason"]

        output_string = "[{}/{}]".format(game_index + 1, num_games) # game count
        output_string += " T={}, MaxT={}".format(num_turns, self.max_turns_survived) # turns
        # winner of this game + rolling win rate for training snake
        output_string += ", W={} ({}%)".format(winner_name, training_snake_win_rate) 
        output_string += ", F={}, MaxF={}".format(food_consumed, self.max_food_consumed) # food
        output_string += ", R={}, MaxR={}".format(collected_reward, self.max_reward_collected) # reward
            
        output_string += ", Death={}".format(training_snake_death_reason)
        
        if (mean_learning_loss != 0):
            output_string += ", L={}".format(mean_learning_loss) # learning loss

        if (self.curr_step > 0):
            output_string += ", S={}".format(self.curr_step) # step

        if (self.num_training_updates_made > 0):
            output_string += ", U={}".format(self.num_training_updates_made) # updates

        print(output_string)
    
    def finalize(self, game_results, training_snake) -> dict:
        game_id = game_results["id"]

        # store winner so we can determine win rates:
        winner = game_results["winner"]        
        winner_name = winner.name if winner is not None else "DRAW"
        if winner_name not in self.winner_counts:
            self.winner_counts[winner_name] = 0

        self.winner_counts[winner_name] += 1

        # store learning losses during training
        learning_losses_for_game = self.learning_losses[game_id] if game_id in self.learning_losses else []        
        if (len(learning_losses_for_game) == 0):
            mean_learning_loss = 0
        else:
            mean_learning_loss = sum(learning_losses_for_game) / len(learning_losses_for_game)

        # track reward collected
        self.total_collected_reward += training_snake.collected_reward

        # track food consumed
        if (training_snake.num_food_consumed > self.max_food_consumed):
            self.max_food_consumed = training_snake.num_food_consumed

        if (training_snake.collected_reward > self.max_reward_collected):
            self.max_reward_collected = training_snake.collected_reward
            self.max_reward_collected = round(self.max_reward_collected, 4)

        # track turns survived
        num_turns = game_results["turns"]
        if (num_turns > self.max_turns_survived):
            self.max_turns_survived = num_turns

        game_results["training"] = {
            "curr_step" : self.curr_step,
            "collected_reward" : training_snake.collected_reward,
            "food_consumed" : training_snake.num_food_consumed,
            "num_turns" : num_turns,
            "death_reason" : training_snake.death_reason,
            "mean_learning_loss" : mean_learning_loss
        }

        return game_results

    def determine_reward(self, training_snake, game_results) -> int:
        if (self.controller.model == None):
            return 0
        
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

        training_snake.collect_reward(total_reward)
        
        return total_reward           

    def save_state(self):
        self.controller.model.save_model(self.curr_step)