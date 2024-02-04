import json
import torch
import battlesnake
import discord_utils
import tensorboard_utils

def main():
    args = battlesnake.parse_args()
    
    training_snake = args.snake_types[0]
    training_snake_name = training_snake["name"]
    
    REPORT_STEP_FREQUENCY = 30_000 # every 50k model steps
    last_report_step_count = -1

    for i in range(args.games):
        # Don't fix the epsilon for normal learning (only for validation)
        args.is_validating_training = False

        print("-----")
        game_results = battlesnake._run_game_from_args(args)

        # num_snakes = game_results["num_snakes"]

        # winner tracking
        # winner = game_results["winner"]
        # turn count tracking
        # turns = game_results["turns"]
        # game_results["training_food_consumed"]
        
        # game_results["total_accumulated_reward"]

        # game_results["training_losses"]

        training_epsilon = game_results["training_epsilon"]
        training_curr_step = game_results["training_curr_step"]

        # # Print out the accumulated rewards for each game size
        # for snake_count in running_accumulated_rewards:
        #     rewards_for_snake_count = running_accumulated_rewards[snake_count]
            
        #     print(f'    {i+1} / {args.games}')
        #     print(f'        {snake_count}-player, Reward Mean: {sum(rewards_for_snake_count) * 1.0 / len(rewards_for_snake_count):.2f}')
        #     # print turn mean
        #     turns_for_snake_count = running_turns_count[snake_count]
        #     print(f'        {snake_count}-player, Turn Mean: {sum(turns_for_snake_count) * 1.0 / len(turns_for_snake_count):.2f}')

        # Initialize last_report_step_count
        if (last_report_step_count == -1):
            last_report_step_count = training_curr_step // REPORT_STEP_FREQUENCY
        
        if (training_curr_step // REPORT_STEP_FREQUENCY > last_report_step_count):
            # Run a validation round with a fixed epsilon
            last_report_step_count = training_curr_step // REPORT_STEP_FREQUENCY
            
            # get the mean max predicted q value on held out states
            mean_max_predicted_q_value = track_mean_max_predicted_q_on_holdout_states(training_snake, "conlan_snakes/DQNConlan2024/held-out-states-solo/")

            # run a validation round with fixed epsilon
            validation_results = run_validation_round(args)
            validation_results["mean_max_predicted_q_value"] = mean_max_predicted_q_value
            validation_results["training_epsilon"] = training_epsilon

            if (args.discord_webhook_url):
                discord_utils.report_to_discord(args.discord_webhook_url, validation_results)
        
            # running_mean_max_predicted_q_values.append(mean_max_predicted_q_value)

            # report_data = {
            #     "running_turns_count" : running_turns_count,
            #     "running_training_losses" : running_training_losses,
            #     "training_epsilon" : training_epsilon,
            #     "training_food_consumed" : running_food_consumed,
            #     "running_accumulated_rewards" : running_accumulated_rewards,
            #     "running_winners" : running_winners,
            #     "training_snake_name" : training_snake_name,
            #     "mean_max_predicted_q_value" : mean_max_predicted_q_value,
            #     "running_mean_max_predicted_q_values" : running_mean_max_predicted_q_values
            # }

            # # log to tensorboard periodically
            # if (args.tensor_board_dir):
            #     tensorboard_utils.log(args.tensor_board_dir, report_data, epoch_size=2000)

def run_validation_round(args):
    NUM_GAMES_PER_VALIDATION_ROUND = 1000

    print("-----")
    print("Running validation round for " + str(NUM_GAMES_PER_VALIDATION_ROUND) + " games...")

    training_snake = args.snake_types[0]
    training_snake_name = training_snake["name"]

    running_winners = {}
    running_turns_count = {}
    running_accumulated_rewards = {}
    running_food_consumed = {}
        
    args.is_validating_training = True

    for i in range(NUM_GAMES_PER_VALIDATION_ROUND):
        print(f'    {i+1} / {NUM_GAMES_PER_VALIDATION_ROUND}')
        
        game_results = battlesnake._run_game_from_args(args)

        num_snakes = game_results["num_snakes"]

        # winner tracking
        winner = game_results["winner"]
        if (num_snakes not in running_winners):
            running_winners[num_snakes] = []
        running_winners[num_snakes].append(winner)

        # turn count tracking
        turns = game_results["turns"]
        if (num_snakes not in running_turns_count):
            running_turns_count[num_snakes] = []
        running_turns_count[num_snakes].append(turns)

        # food consumed
        if (num_snakes not in running_food_consumed):
            running_food_consumed[num_snakes] = []
        running_food_consumed[num_snakes].append(game_results["training_food_consumed"])

         # total accumulated reward
        if (num_snakes not in running_accumulated_rewards):
            running_accumulated_rewards[num_snakes] = []
        running_accumulated_rewards[num_snakes].append(game_results["total_accumulated_reward"])


    return {
        "running_winners" : running_winners,
        "running_turns_count" : running_turns_count,
        "training_snake_name" : training_snake_name,
        "running_accumulated_rewards" : running_accumulated_rewards,
        "training_food_consumed" : running_food_consumed
    }

def track_mean_max_predicted_q_on_holdout_states(training_snake, held_out_states_path):
    print("Tracking mean max predicted Q value on holdout states...")

    all_max_q_values = []

    for i in range(1000):
        # load hold out state
        json_path = held_out_states_path + "state-" + str(i) + ".json"
        with open(json_path, 'r') as json_file:
            board_data = json.load(json_file)

        # run model and get max predicted Q_value
        move = training_snake["move"](board_data, force_greedy_move=True)
        q_values = move["q_values"] # tensor of q values
        max_q_value = torch.max(q_values).item()
        
        all_max_q_values.append(max_q_value)

    # return the average of the max predicted Q_values
    return sum(all_max_q_values) * 1.0 / len(all_max_q_values)

if __name__ == "__main__":
    main()