import requests
import json
import numpy as np
import copy

def report_to_discord(discord_webhook_url, data):    
    running_winners = data['running_winners']
    
    training_snake_name = data['training_snake_name']
    
    running_turns_count = data['running_turns_count']
    running_accumulated_rewards = data['running_accumulated_rewards']
    training_food_consumed = data['training_food_consumed']

    training_epsilon = data['training_epsilon']

    # mean max predicted q value
    mean_max_predicted_q_value = data['mean_max_predicted_q_value']
    mean_max_predicted_q_value = float("{:.5f}".format(mean_max_predicted_q_value))

    stats_per_snake_count = {}
    for snake_count in running_winners:
        stats_per_snake_count[snake_count] = {}

    # Win Rate (Per Game Size)
    for snake_count in running_winners:
        winners_for_snake_count = running_winners[snake_count]
        
        stats_per_snake_count[snake_count]["num_games"] = ":video_game:   **Num Games**: " + str(len(winners_for_snake_count)) + "\n\n"

        num_games_training_snake_won = winners_for_snake_count.count(training_snake_name)

        training_snake_win_rate = num_games_training_snake_won * 1.0 / len(winners_for_snake_count)

        stats_per_snake_count[snake_count]['win_rate'] = ":trophy:   **{}-P Win Rate**: {:.5f}".format(snake_count, training_snake_win_rate) + "\n\n"

    # Epsilon    
    training_epsilon = float("{:.5f}".format(training_epsilon))

    # Turn Count (Per Game Size)    
    for snake_count in running_turns_count:
        turns_for_snake_count = running_turns_count[snake_count]

        stats_per_snake_count[snake_count]['turn_count'] = ":alarm_clock:   **{}-P Mean Turns**: {:.5f}".format(snake_count, sum(turns_for_snake_count) * 1.0 / len(turns_for_snake_count)) + "\n\n"

    # Food Consumed (Per Game Size)    
    for snake_count in training_food_consumed:
        food_consumed_for_snake_count = training_food_consumed[snake_count]

        stats_per_snake_count[snake_count]['food_consumed'] = ":pill:   **{}-P Mean Food**: {:.5f}".format(snake_count, sum(food_consumed_for_snake_count) * 1.0 / len(food_consumed_for_snake_count)) + "\n\n"

    # Accumulated Rewards (Per Game Size)
    for snake_count in running_accumulated_rewards:
        accumulated_rewards_for_snake_count = running_accumulated_rewards[snake_count]

        stats_per_snake_count[snake_count]['accumulated_rewards'] = ":moneybag:   **{}-P Mean Rewards**: {:.5f}".format(snake_count, sum(accumulated_rewards_for_snake_count) * 1.0 / len(accumulated_rewards_for_snake_count)) + "\n\n"
    
    # build the message to post to discord
    discord_message = "Validation Results\n\n"
    discord_message += ":snake:  **Training Snake**: " + training_snake_name + "\n\n"
    discord_message += ":game_die:  **Epsilon**: " + str(training_epsilon) + "\n\n"
    # discord_message += ":skull:  **Mean Loss**: " + str(mean_training_loss) + "\n\n"
    discord_message += ":regional_indicator_q:  **Mean Max Predicted Q**: " + str(mean_max_predicted_q_value) + "\n\n"
    discord_message += "------------------------------------------------------------\n\n"

    for snake_count in range(1, 99):
        if not snake_count in stats_per_snake_count:
            continue

        discord_message += stats_per_snake_count[snake_count]['num_games']
        discord_message += stats_per_snake_count[snake_count]['win_rate']
        discord_message += stats_per_snake_count[snake_count]['food_consumed']        
        discord_message += stats_per_snake_count[snake_count]['turn_count']
        discord_message += stats_per_snake_count[snake_count]['accumulated_rewards']
        discord_message += "------------------------------------------------------------\n\n"
    
    payload = {
        "content": discord_message
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(discord_webhook_url, data=json.dumps(payload), headers=headers)

    if response.status_code == 204:
        print("Message sent successfully to discord")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")