import battlesnake
import discord_utils
import tensorboard_utils

def main():
    args = battlesnake.parse_args()

    running_turns_count = {}
    running_food_consumed = {}
    running_accumulated_rewards = {}
    running_winners = {}
    running_training_losses = []
    
    training_snake_name = args.snake_types[0]["name"]

    REPORT_FREQUENCY = 500

    for i in range(args.games):
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

        # Training losses and epsilon
        training_losses_for_game = game_results["training_losses"]
        training_loss_mean = sum(training_losses_for_game) * 1.0 / len(training_losses_for_game) if len(training_losses_for_game) > 0 else 0
        running_training_losses.append(training_loss_mean)

        training_epsilon = game_results["training_epsilon"]

        # Print out the accumulated rewards for each game size
        for snake_count in running_accumulated_rewards:
            rewards_for_snake_count = running_accumulated_rewards[snake_count]
            
            print(f'{i+1} / {args.games}) {snake_count}-player, Reward Mean: {sum(rewards_for_snake_count) * 1.0 / len(rewards_for_snake_count):.2f}')
    
        if (i + 1) % REPORT_FREQUENCY == 0:
            mean_max_predicted_q_value = track_mean_max_predicted_q_on_holdout_states()

            report_data = {
                "running_turns_count" : running_turns_count,
                "running_training_losses" : running_training_losses,
                "training_epsilon" : training_epsilon,
                "training_food_consumed" : running_food_consumed,
                "running_accumulated_rewards" : running_accumulated_rewards,
                "running_winners" : running_winners,
                "training_snake_name" : training_snake_name,
                "mean_max_predicted_q_value" : mean_max_predicted_q_value
            }
            # report to discord periodically
            if (args.discord_webhook_url):            
                discord_utils.report_to_discord(args.discord_webhook_url, report_data, epoch_size=REPORT_FREQUENCY)

            # log to tensorboard periodically
            if (args.tensor_board_dir):
                tensorboard_utils.log(args.tensor_board_dir, report_data, epoch_size=REPORT_FREQUENCY)

    for snake_count in running_winners:
        winners = running_winners[snake_count]

        for winner in set(winners):
            if (winner == battlesnake.GAME_RESULT_DRAW):
                print(f'{snake_count}-player, Games Tied: {sum([1 for s in winners if s == winner])}')
            else:
                print(f'{snake_count}-player, {winner} Won: {sum([1 for s in winners if s == winner])}')

def track_mean_max_predicted_q_on_holdout_states():
    return 0

if __name__ == "__main__":
    main()