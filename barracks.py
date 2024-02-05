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

LEARNING_RATE = 0.00025

def main() -> None:
    observer = Observer()

    # TODO load these from disk
    model_save_path = "./models/snake_net_v5.chkpt"
    
    # Load the model that we're training
    model = DQNModel(learning_rate=LEARNING_RATE)
    
    training_info = model.load_model(model_save_path)
    
    trainee_controller = DQNController(model, training_info, convert_data_to_image=observer.convert_data_to_image)

    trainer = Trainer(trainee_controller, training_info["curr_step"])

    controllers = [
        trainee_controller
    ]

    training_config = {
        "speed" : 100,
        "print_board" : False,
        "colors" : [
            constants.COLORS["red"],
            constants.COLORS["blue"]
        ],
        "controllers" : controllers,
        "observer" : observer,
        "trainer" : trainer
    }

    num_games = 5

    for i in range(num_games):
        result = run_training_game(training_config)

        print(result["training"])

        print_game_result(result, i, num_games)

    # recorder = Recorder()
    # recorder.record(observer.observations[game_id], "output_video.mp4")

    print(f'All {num_games} games have finished')

def print_game_result(game_results, game_index, num_games) -> None:
    game_id = game_results["id"]
    winner = game_results["winner"].name if game_results["winner"] is not None else "No Winner"
    num_turns = game_results["turns"]

    print(f'{game_index + 1} / {num_games} finished in {num_turns} turns. Winner: {winner}')

def run_training_game(config) -> dict:
    speed = config["speed"]
    print_board = config["print_board"]
    colors = config["colors"]
    
    controllers = config["controllers"]
    trainer = config["trainer"]
    
    observer = config["observer"]
    
    parameters = GameParameters(constants.BOARD_SIZE_MEDIUM, constants.DEFAULT_FOOD_SPAN_CHANCE, constants.DEFAULT_MIN_FOOD)

    snakes = []
    
    training_snake = None

    for i in range(len(controllers)):
        controller = controllers[i]
        snake = Snake("Snake-" + str(i), colors[i], controller)    
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

    