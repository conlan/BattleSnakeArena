from torch.utils.tensorboard import SummaryWriter


def log(dir, data, epoch_size):
    training_food_consumed = data["training_food_consumed"]

    writer = SummaryWriter(dir)

    for snake_count in range(2, 99):
        if not snake_count in training_food_consumed:
            continue

        num_food_consumed = training_food_consumed[snake_count]

        for epoch in range(1_000_000):
            food_in_epoch = num_food_consumed[epoch * epoch_size : (epoch + 1) * epoch_size]

            if (len(food_in_epoch) == 0):
                break
            
            # calculate mean of food consumed in epoch
            mean_food_consumed = sum(food_in_epoch) / len(food_in_epoch)

            writer.add_scalar('Food Consumed ({}-Player)'.format(snake_count), mean_food_consumed, epoch)

    writer.close()