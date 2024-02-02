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

    REPORT_TO_DISCORD_EVERY = 500
    REPORT_TO_TENSORBOARD_EVERY = 500

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
    
        for snake_count in running_accumulated_rewards:
            rewards_for_snake_count = running_accumulated_rewards[snake_count]
            
            print(f'{i+1} / {args.games}) {snake_count}-player, Reward Mean: {sum(rewards_for_snake_count) * 1.0 / len(rewards_for_snake_count):.2f}')

        training_loss_mean = sum(game_results["training_losses"]) * 1.0 / len(game_results["training_losses"]) if len(game_results["training_losses"]) > 0 else 0
        running_training_losses.append(training_loss_mean)
    
        # log to discord periodically
        if (args.discord_webhook_url):
            if (i + 1) % REPORT_TO_DISCORD_EVERY == 0:                
                discord_utils.report_to_discord(args.discord_webhook_url, {
                    "running_turns_count" : running_turns_count,
                    "running_training_losses" : running_training_losses,
                    "training_epsilon" : game_results["training_epsilon"],
                    "training_food_consumed" : running_food_consumed,
                    "running_accumulated_rewards" : running_accumulated_rewards,
                    "running_winners" : running_winners,
                    "training_snake_name" : training_snake_name
                }, epoch_size=REPORT_TO_DISCORD_EVERY)

        # log to tensorboard periodically
        if (args.tensor_board_dir):
            if (i + 1) % REPORT_TO_TENSORBOARD_EVERY == 0:
                tensorboard_utils.log(args.tensor_board_dir, {
                    "training_food_consumed" : running_food_consumed,
                    "running_accumulated_rewards" : running_accumulated_rewards,
                    "running_winners" : running_winners,
                    "training_snake_name" : training_snake_name
                }, epoch_size=REPORT_TO_TENSORBOARD_EVERY)


    for snake_count in running_winners:
        winners = running_winners[snake_count]

        for winner in set(winners):
            if (winner == battlesnake.GAME_RESULT_DRAW):
                print(f'{snake_count}-player, Games Tied: {sum([1 for s in winners if s == winner])}')
            else:
                print(f'{snake_count}-player, {winner} Won: {sum([1 for s in winners if s == winner])}')

if __name__ == "__main__":
    main()