import time
import constants
import argparse
import random

from game import GameParameters, Game
from snake import Snake

from tensor_observer import TensorObserver

from trainer import Trainer

from dqn.dqn_controller import DQNController
from ddqn.ddqn_controller import DDQNController

from controllers.simple_controller import SimpleController
from controllers.strong_controller import StrongController

from recorder import Recorder
from validator import Validator
from reporter import Reporter

def main(model_save_path, history_save_path, discord_webhook_url, should_record_gameplay) -> None:
    # ========================================================================
    NUM_GAMES_TO_PLAY = 1_000_000
    NUM_GAMES_PER_VALIDATION = 2_500
    VALIDATE_EVERY_N_STEPS = 150_000
    # ========================================================================
    
    observer = TensorObserver()
    
    reporter = Reporter(discord_webhook_url)
    reporter.load_history(history_save_path)

    trainee_controller = DDQNController(model_save_path, "me", convert_data_to_image=observer.convert_data_to_tensor, should_action_mask=True)
    trainer = Trainer(trainee_controller, trainee_controller.epsilon_info["curr_step"])

    # ========================================================================
    # The opponent snakes we'll train against
    # Simple Controller
    training_opponent_0 = SimpleController("simple")
    training_opponent_1 = StrongController("strong")
    # Snapshotted DQN Controller
    # training_opponent_2 = DQNController("/content/drive/MyDrive/ColabOutput/runs/snake_v11/snake_v8.chkpt", convert_data_to_image=observer.convert_data_to_image)
    # training_opponent_2.load_epsilon(constants.EPSILON_INFO_ALWAYS_GREEDY)
    # ========================================================================

    training_opponents = [
        training_opponent_0,
        training_opponent_1, training_opponent_1, training_opponent_1, training_opponent_1
        # training_opponent_2
    ]

    training_config = {
        "speed" : 100,
        "print_board" : False,
        "should_record_gameplay" : should_record_gameplay,
        "colors" : [
            constants.COLORS["red"],
            constants.COLORS["blue"]
        ],
        "controllers" : [
            trainee_controller,
            training_opponents
        ],
        "observer" : observer,
        "trainer" : trainer
    }

    # Track when we last ran a validation (by default set it to the current step so we don't validate right away)
    last_step_validated = trainer.curr_step

    for i in range(NUM_GAMES_TO_PLAY):
        result = run_training_game(training_config, constants.DEFAULT_GAME_CONFIG)

        trainer.print_training_result(result, i, NUM_GAMES_TO_PLAY)

        # validate our model every N steps
        if (trainer.curr_step - last_step_validated >= VALIDATE_EVERY_N_STEPS):
            last_step_validated = trainer.curr_step
            
            # save the trainer state in case we end during validation round we don't lose progress
            trainer.save_state()

            # setup the validation controller
            validation_controller = DDQNController(model_save_path, "me", convert_data_to_image=observer.convert_data_to_tensor, should_action_mask=True)
            validation_controller.load_epsilon(constants.EPSILON_INFO_VALIDATION)
    
            # setup a trainer so we can determine rewards for the validation games
            validation_trainer = Trainer(validation_controller, 0)
            validator = Validator()

            # get unique opponents to validate against
            unique_opponents_to_validate_against = list(set(training_opponents))

            # run a series of validation games against each training opponent
            for opponent_controller in unique_opponents_to_validate_against:
                opponent_name = opponent_controller.name()

                validation_config = {
                    "opponent" : opponent_name,
                    "controllers" : [
                        validation_controller,
                        opponent_controller
                    ],
                    "controller_under_valuation" : validation_controller,
                    "trainer" : validation_trainer
                }

                validation_results = validator.run_validation(validation_config, constants.DEFAULT_GAME_CONFIG, NUM_GAMES_PER_VALIDATION)

                # report data
                reporter.report(validation_results, opponent_name, trainer.curr_step)
            
            # save the history for now (we can chart it later)
            reporter.save_history()

        if (should_record_gameplay):
            recorder = Recorder()
            game_id = result["id"]
            # recorder.output_to_frames(observer.observations[game_id], "output_frames")
            recorder.output_to_video(observer.observations[game_id], "output_video.mp4")
            break

def run_training_game(training_config, game_config) -> dict:
    speed = training_config["speed"]
    print_board = training_config["print_board"]
    colors = training_config["colors"]
    
    controllers = training_config["controllers"]
    trainer = training_config["trainer"]
    
    observer = training_config["observer"]
    should_record_gameplay = training_config["should_record_gameplay"]

    parameters = GameParameters(game_config)

    snakes = []
    
    training_snake = None

    for i in range(len(controllers)):
        controller_config = controllers[i]

        # check if config is a list
        if (isinstance(controller_config, list)):
            # randomly select a controller from the list
            controller = random.choice(controller_config)
        else:
            controller = controller_config

        snake = Snake(controller.nickname, colors[i], controller)
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
        observation = observer.observe(game.get_board_json(training_snake), should_record_gameplay)

        # Perform a game step
        is_done = game.step()

        # Grab an observation after the step
        next_observation = observer.observe(game.get_board_json(training_snake), (is_done and should_record_gameplay))

        # get move made from the controller
        training_snake_action = trainer.controller.get_last_local_direction(game)
        training_snake_move = trainer.controller.get_last_move(game)
        training_snake_q_values = training_snake_move["q_values"]

        # put the last move details in the initial observation        
        observation["action"] = training_snake_action
        observation["move"] = training_snake_move["move"]
        observation["q_values"] = training_snake_q_values

        if (is_done):
            # get final game results
            game_results = game.get_game_results()
        
        # determine reward for the controller        
        reward = trainer.determine_reward(training_snake, game_results)

        # cache and possibly train on results
        trainer.cache(game, observation, next_observation, training_snake_action, reward, is_done, training_snake_q_values)

        # delay if necessary
        if (speed < 100):
            while(time.time()-t1 <= float(100-speed)/float(100)): pass
        
    # print the final board if necessary
    if (print_board): observer.print_game(game)

    trainer.finalize(game_results, training_snake)

    return game_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-dis", "--discord_webhook_url", nargs='+', help="Discord webhook for reporting", type=str, default=None)
    parser.add_argument("-mod", "--model_save_path", nargs='+', help="Model save path", type=str, default=None)
    parser.add_argument("-his", "--history_save_path", nargs='+', help="History save path", type=str, default=None)

    args = parser.parse_args()
    
    model_save_path = args.model_save_path[0] if args.model_save_path is not None else None
    history_save_path = args.history_save_path[0] if args.history_save_path is not None else None
    discord_webhook_url = args.discord_webhook_url[0] if args.discord_webhook_url is not None else None    

    should_record_gameplay = False

    # exit if not all required arguments are provided
    if (model_save_path is None or history_save_path is None):
        print("Missing required arguments")
        exit(1)

    main(model_save_path, history_save_path, discord_webhook_url, should_record_gameplay)

    