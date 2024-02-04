import time
from threading import Thread

from game import GameParameters, Game
from snake import Snake

from observer import Observer

from trainer import Trainer

from dqn.dqn_controller import DQNController
from controllers.simple_controller import SimpleController

from dqn.dqn_model import DQNModel

from recorder import Recorder

import constants

REWARD_SETS = {
    "reward-set-v1" : {
        constants.REWARD_KEY_LOSE : -1000,
        constants.REWARD_KEY_WIN : 1000,
        constants.REWARD_KEY_SURVIVE : 1,
        constants.REWARD_KEY_EAT : 10
    }
}

def main() -> None:
    # TODO load these from disk
    model_save_path = "models/snake_net.chkpt"
    
    # TODO epsilon info
    epsilon_info = {
        "epsilon" : 1,
        "epsilon_decay" : 0.0000009,
        "epislon_min" : 0.1
    }
    # TODO learning rate
    learning_rate = 0.00025
    
    model = DQNModel(model_save_path=model_save_path, epsilon_info=epsilon_info, learning_rate=learning_rate)
    
    trainee_controller = DQNController(model)

    observer = Observer()
    recorder = Recorder()
    trainer = Trainer(trainee_controller, REWARD_SETS[model.reward_set_key])

    controllers = [
        trainee_controller
    ]

    colors = [
        constants.COLORS["red"],
        constants.COLORS["blue"]
    ]

    training_config = {
        "speed" : 90,
        "print_board" : True,
        "colors" : colors,
        "controllers" : controllers,
        "observer" : observer,
        "trainer" : trainer
    }
    game_results = []

    num_games = 1

    threads = []

    for i in range(num_games):
        process = Thread(target=run_training_game, args=(training_config,game_results,))
        threads.append(process)

    for process in threads:
        process.start()
        
    for process in threads:
        process.join()

    for result in game_results:
        print_game_result(result)
        game_id = result["id"]

        # recorder.record(observer.observations[game_id], "output_video.mp4")
        # print(len(observer.observations[game_id]))
        # print(observer.observations[game_id][0])

    print(f'All {num_games} games have finished')

def print_game_result(result) -> None:
    game_id = result["id"]
    winner = result["winner"].name
    print(f'{game_id} finished. Winner: {winner}')

def run_training_game(config, results) -> dict:
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

        game_results = None

        if (is_done):
            # get final game results
            game_results = game.get_game_results()
            
            results.append(game_results)
        
        # determine reward for the controller
        reward = trainer.determine_reward(training_snake, game_results)

        # train on the results
        trainer.train(observation, next_observation, action, reward, is_done)

        # delay if necessary
        if (speed < 100):
            while(time.time()-t1 <= float(100-speed)/float(100)): pass
        
    # print the final board if necessary
    if (print_board): observer.print_game(game)

if __name__ == "__main__":
    main()

    