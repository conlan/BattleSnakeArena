from game import GameParameters, Game

from snake import Snake

class Validator():    
    def run_validation(self, validation_config, game_config, num_validation_games) -> None:
        print("\nVALIDATING MODEL...")

        trainer = validation_config["trainer"]

        rewards_collected = []

        for i in range(num_validation_games):
            game_results = self._run_validation_round(validation_config, game_config)
            
            trainer.print_training_result(game_results, i, num_validation_games)

            total_collected_reward = game_results["training"]["total_reward"]

            rewards_collected.append(total_collected_reward)

        mean_reward = sum(rewards_collected) / len(rewards_collected)

        return mean_reward
    
    def _run_validation_round(self, validation_config, game_config) -> None:
        parameters = GameParameters(game_config)

        controllers = validation_config["controllers"]
        trainer = validation_config["trainer"]

        snakes = []

        validating_snake = None

        for i in range(len(controllers)):
            controller = controllers[i]
            snake = Snake("S-" + str(i), None, controller)    
            snakes.append(snake)

            if (controller == validation_config["controller_under_valuation"]):
                validating_snake = snake

        game = Game(parameters, snakes)

        is_done = game.reset()    

        game_results = None

        while not is_done:
            # Perform a game step
            is_done = game.step()

            if (is_done):
                # get final game results
                game_results = game.get_game_results()

            # determine reward for the controller        
            trainer.determine_reward(validating_snake, game_results)

        trainer.finalize(game_results, validating_snake)

        return game_results