from game import GameParameters, Game

from snake import Snake

class Validator():    
    def run_validation(self, validation_config, game_config, num_validation_games) -> float:
        opponent_name = validation_config["opponent"]
        print("\nVALIDATING AGAINST OPPONENT: " + opponent_name)

        validation_trainer = validation_config["trainer"]
        validation_trainer.reset()

        for i in range(num_validation_games):
            game_results = self._run_validation_round(validation_config, game_config)
            
            validation_trainer.print_training_result(game_results, i, num_validation_games)

        mean_validation_reward = validation_trainer.total_collected_reward * 1.0 / num_validation_games

        validation_results = {
            "mean_validation_reward" : mean_validation_reward
        }
        return validation_results
    
    def _run_validation_round(self, validation_config, game_config) -> dict:
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

        game_results : dict = None

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