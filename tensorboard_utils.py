from torch.utils.tensorboard import SummaryWriter


def log(dir, data, epoch_size):
    training_food_consumed = data["training_food_consumed"]
    running_accumulated_rewards = data["running_accumulated_rewards"]

    writer = SummaryWriter(dir)

    for snake_count in range(2, 99):
        if not snake_count in training_food_consumed:
            continue

        num_food_consumed = training_food_consumed[snake_count]
        num_reward_accumulated = running_accumulated_rewards[snake_count]

        for epoch in range(1_000_000):
            food_in_epoch = num_food_consumed[epoch * epoch_size : (epoch + 1) * epoch_size]

            # don't track epochs with less than epoch_size data (otherwise can look skewed)
            if (len(food_in_epoch) < epoch_size):
                break
            
            # calculate mean of food consumed in epoch
            mean_food_consumed = sum(food_in_epoch) / len(food_in_epoch)

            writer.add_scalar('Food Consumed ({}-Player)'.format(snake_count), mean_food_consumed, epoch)

            # calculate mean of accumulated rewards in epoch
            rewards_in_epoch = num_reward_accumulated[epoch * epoch_size : (epoch + 1) * epoch_size]

            mean_accumulated_rewards = sum(rewards_in_epoch) / len(rewards_in_epoch)

            writer.add_scalar('Accumulated Rewards ({}-Player)'.format(snake_count), mean_accumulated_rewards, epoch)

    writer.close()

    print(f"Wrote tensorboard logs to {dir}")