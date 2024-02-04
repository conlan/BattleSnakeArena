import time
from threading import Thread

from arena import ArenaParameters, Arena
from snake import Snake

from observer import Observer

from controllers.simple_controller import SimpleController

from recorder import Recorder

import constants

def main() -> None:
    trainee_controller = SimpleController()

    controllers = [
        trainee_controller,

        SimpleController()
    ]

    colors = [
        constants.COLORS["red"],
        constants.COLORS["blue"]
    ]

    observer = Observer()
    recorder = Recorder()

    training_config = {
        "speed" : 100,
        "print_board" : False,
        "colors" : colors,
        "controllers" : controllers,
        "observer" : observer,
        "trainee" : trainee_controller
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
        arena_id = result["id"]

        recorder.record(observer.observations[arena_id], "output_video.mp4")
        
    # print(observer.observations)

    print(f'All {num_games} games have finished')

def print_game_result(result) -> None:
    arena_id = result["id"]
    winner = result["winner"].name
    print(f'{arena_id} finished. Winner: {winner}')

def run_training_game(config, results) -> dict:
    speed = config["speed"]
    print_board = config["print_board"]
    colors = config["colors"]
    
    controllers = config["controllers"]
    controller_being_trained = config["trainee"]
    
    observer = config["observer"]
    
    parameters = ArenaParameters(constants.BOARD_SIZE_MEDIUM, constants.DEFAULT_FOOD_SPAN_CHANCE, constants.DEFAULT_MIN_FOOD)

    snakes = []
    training_snake = None

    for i in range(len(controllers)):
        controller = controllers[i]
        snake = Snake("Snake-" + str(i), colors[i], controller)    
        snakes.append(snake)

        if (controller == controller_being_trained):
            training_snake = snake
        
    arena = Arena(parameters, snakes)

    is_done = arena.reset()    

    while not is_done:
        t1 = time.time()

        if (print_board): observer.print_arena(arena)

        observer.observe(arena.get_board_json(training_snake))

        is_done = arena.step()

        if (speed < 100):
            while(time.time()-t1 <= float(100-speed)/float(100)): pass

    if (print_board): observer.print_arena(arena)

    observer.observe(arena.get_board_json(training_snake))

    results.append(arena.get_game_results())

if __name__ == "__main__":
    main()

    