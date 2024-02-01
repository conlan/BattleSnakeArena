import requests
import json
import numpy as np

def report_to_discord(discord_webhook_url, data, epoch_size):    
    running_winners = data['running_winners']
    
    training_snake_name = data['training_snake_name']
    training_losses = data['running_training_losses']
    
    running_turns_count = data['running_turns_count']
    training_food_consumed = data['training_food_consumed']

    training_epsilon = data['training_epsilon']

    # trim out the data so we're only dealing with the latest epoch
    training_losses = training_losses[-epoch_size:]

    for snake_count in running_winners:
        running_winners[snake_count] = running_winners[snake_count][-epoch_size:]
        running_turns_count[snake_count] = running_turns_count[snake_count][-epoch_size:]
        training_food_consumed[snake_count] = training_food_consumed[snake_count][-epoch_size:]
    
    # Training Loss
    mean_training_loss = np.mean(training_losses) if len(training_losses) > 0 else 0
    mean_training_loss = float("{:.5f}".format(mean_training_loss))

    stats_per_snake_count = {}
    for snake_count in running_winners:
        stats_per_snake_count[snake_count] = {}

    # Win Rate (Per Game Size)
    for snake_count in running_winners:
        winners_for_turn_count = running_winners[snake_count]
        
        stats_per_snake_count[snake_count]["num_games"] = ":video_game:   **Num Games**: " + str(len(winners_for_turn_count)) + "\n\n"

        num_games_training_snake_won = winners_for_turn_count.count(training_snake_name)

        training_snake_win_rate = num_games_training_snake_won * 1.0 / len(winners_for_turn_count)

        stats_per_snake_count[snake_count]['win_rate'] = ":trophy:   **{}-P Win Rate**: {:.2f}".format(snake_count, training_snake_win_rate) + "\n\n"

    # Epsilon    
    training_epsilon = float("{:.5f}".format(training_epsilon))

    # Turn Count (Per Game Size)    
    for snake_count in running_turns_count:
        turns_for_snake_count = running_turns_count[snake_count]

        stats_per_snake_count[snake_count]['turn_count'] = ":alarm_clock:   **{}-P Mean Turns**: {:.2f}".format(snake_count, sum(turns_for_snake_count) * 1.0 / len(turns_for_snake_count)) + "\n\n"

    # Food Consumed (Per Game Size)    
    for snake_count in training_food_consumed:
        food_consumed_for_snake_count = training_food_consumed[snake_count]

        stats_per_snake_count[snake_count]['food_consumed'] = ":pill:   **{}-P Mean Food**: {:.2f}".format(snake_count, sum(food_consumed_for_snake_count) * 1.0 / len(food_consumed_for_snake_count)) + "\n\n"
    
    # build the message to post to discord
    discord_message = ""        
    discord_message += ":game_die:  **Epsilon**: " + str(training_epsilon) + "\n\n"
    discord_message += ":skull:  **Mean Loss**: " + str(mean_training_loss) + "\n\n"
    discord_message += "------------------------------------------------------------\n\n"

    for snake_count in range(2, 99):
        if not snake_count in stats_per_snake_count:
            continue

        discord_message += stats_per_snake_count[snake_count]['num_games']
        discord_message += stats_per_snake_count[snake_count]['win_rate']
        discord_message += stats_per_snake_count[snake_count]['food_consumed']        
        discord_message += stats_per_snake_count[snake_count]['turn_count']        
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