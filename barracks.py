import time
import constants

from threading import Thread

from game import GameParameters, Game
from snake import Snake

from observer import Observer

from trainer import Trainer

from dqn.dqn_controller import DQNController
from dqn.dqn_model import DQNModel

from recorder import Recorder
from validator import Validator
from reporter import Reporter

def main() -> None:
    # ========================================================================
    # TODO pass in as args parameter
    MODEL_SAVE_PATH = "./models/snake_net_v5.chkpt"
    HISTORY_SAVE_PATH = "./models/snake_net_v5_history.json"
    DISCORD_WEBHOOK_URL = ''

    NUM_GAMES_TO_PLAY = 500_000
    NUM_GAMES_PER_VALIDATION = 1_000
    VALIDATE_EVERY_N_GAMES = 30_000

    LEARNING_RATE = 0.00025

    # ========================================================================
    
    observer = Observer()
    
    reporter = Reporter(DISCORD_WEBHOOK_URL)
    reporter.load_history(HISTORY_SAVE_PATH)

    # Load the model that we're training
    model = DQNModel(learning_rate=LEARNING_RATE)
    
    training_info = model.load_model(MODEL_SAVE_PATH)

    trainee_controller = DQNController(model, training_info, convert_data_to_image=observer.convert_data_to_image)
    trainer = Trainer(trainee_controller, training_info["curr_step"])

    training_config = {
        "speed" : 100,
        "print_board" : False,
        "colors" : [
            constants.COLORS["red"],
            constants.COLORS["blue"]
        ],
        "controllers" : [
            trainee_controller
        ],
        "observer" : observer,
        "trainer" : trainer
    }

    game_config = {
        "food_spawn_chance" : constants.DEFAULT_FOOD_SPAN_CHANCE,
        "min_food" : constants.DEFAULT_MIN_FOOD,
        "board_size" : constants.BOARD_SIZE_MEDIUM
    }

    validation_info = {
        "epsilon" : 0.05,
        "epsilon_decay" : 0,
        "epsilon_min" : 0.05
    }    
    validation_controller = DQNController(model, validation_info, convert_data_to_image=observer.convert_data_to_image)
    validator = Validator()

    validation_config = {
        "controllers" : [
            validation_controller
        ],
        "controller_under_valuation" : validation_controller,
        "trainer" : trainer
    }

    for i in range(NUM_GAMES_TO_PLAY):
        result = run_training_game(training_config, game_config)

        print_game_result(result, i, NUM_GAMES_TO_PLAY)

        # validate our model
        if ((i + 1) % VALIDATE_EVERY_N_GAMES == 0):
            mean_validation_reward = validator.run_validation(validation_config, game_config, NUM_GAMES_PER_VALIDATION)

            # report data
            reporter.report(mean_validation_reward, trainer.curr_step)
            
            # save the history for now (we can chart it later)
            reporter.save_history()

            # save the trainer so we don't get out of sync (e.g. reporter saved but trainer didn't)
            trainer.save_state()

    # recorder = Recorder()
    # recorder.record(observer.observations[game_id], "output_video.mp4")

def print_game_result(game_results, game_index, num_games) -> None:
    # game_id = game_results["id"]
    winner = game_results["winner"].name if game_results["winner"] is not None else "Draw"    
    num_turns = game_results["turns"]

    total_collected_reward = game_results["training"]["total_reward"]
    total_food_consumed = game_results["training"]["total_food_consumed"]
    mean_learning_loss = game_results["training"]["mean_learning_loss"]

    curr_step = game_results["training"]["curr_step"]

    print(f'[{game_index + 1}/{num_games}] Turns={num_turns}, Result={winner}, Food={total_food_consumed}, Reward={total_collected_reward}, Loss={mean_learning_loss}, Curr Step={curr_step}')

def run_training_game(training_config, game_config) -> dict:
    speed = training_config["speed"]
    print_board = training_config["print_board"]
    colors = training_config["colors"]
    
    controllers = training_config["controllers"]
    trainer = training_config["trainer"]
    
    observer = training_config["observer"]

    parameters = GameParameters(game_config)

    snakes = []
    
    training_snake = None

    for i in range(len(controllers)):
        controller = controllers[i]
        snake = Snake("S-" + str(i), colors[i], controller)    
        snakes.append(snake)

        if (controller == trainer.controller):
            training_snake = snake
        
    game = Game(parameters, snakes)

    is_done = game.reset()    

    game_results = None

    while not is_done:
        t1 = time.time()

        # print the board if necessary
        if (print_board): observer.print_game(game)

        # Grab the current observation
        observation = observer.observe(game.get_board_json(training_snake), True)

        # Perform a game step
        is_done = game.step()

        # Grab an observation after the step
        next_observation = observer.observe(game.get_board_json(training_snake), is_done)

        # get move made from the controller
        action = trainer.controller.get_last_move_made(game)

        if (is_done):
            # get final game results
            game_results = game.get_game_results()
        
        # determine reward for the controller        
        reward = trainer.determine_reward(training_snake, game_results)

        # cache and possibly train on results
        trainer.cache(game, observation, next_observation, action, reward, is_done)

        # delay if necessary
        if (speed < 100):
            while(time.time()-t1 <= float(100-speed)/float(100)): pass
        
    # print the final board if necessary
    if (print_board): observer.print_game(game)

    trainer.finalize(game_results, training_snake)

    return game_results

if __name__ == "__main__":
    main()

    