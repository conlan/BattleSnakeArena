import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchvision import transforms

class DQNSnakeModel():
    def __init__(self, max_snakes) -> None:        
        # use gpu if available
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100_000, 
                                                device=torch.device("cpu")))
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        print(f"Using {self.device} device")

        self.max_snakes = max_snakes

        self.network = NeuralNetwork(self.max_snakes, 3)
        self.network.to(self.device)

        self.curr_step = 0
        self.burnin = 1_000
        self.learn_every = 3
        self.learning_rate = 0.001
        self.save_every = 3_000
        self.gamma = 0.9 # discount rate
        
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999999
        self.exploration_rate_min = 0.1

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # make a random generator that we use here so the seed doesn't get overriden in the main game
        self.random = random.Random()

        # where to save the model to
        self.model_save_path = None

    @torch.inference_mode()
    def act(self, state_obj, use_greedy=False, use_action_masking=False):
        # only use epsilon greedy if we're not using greedy
        if (not use_greedy) and (self.random.random() < self.exploration_rate):
            # random move
            action_idx = self.random.randint(0, 2)
        else:
            state, state_health = self.get_state_and_health_tensors_from_state_obj(state_obj)

            # reshape health
            state_health = state_health.view(1, 1, 1, self.max_snakes)
            # # Expand dimensions to match state
            state_health = state_health.expand(-1, 67, 67, -1)

            state = torch.cat((state.unsqueeze(3), state_health), dim=3).to(self.device)
            
            results = self.network(state)

            # if we should mask out moves that are guaranteed to lose
            if (use_action_masking):
                results = self.perform_action_mask(results, state_obj['json'], state_obj['next_move_coordinates'])

            action_idx = torch.argmax(results).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
            
        return action_idx
    
    def perform_action_mask(self, action_values, board_data, next_move_coordinates):
        MIN_Q_VALUE = -99999

        # for each action, nullify value to MIN if it will guanteed lose
        # e.g. walls or snake body hits
        # If it's a possible snake head hit then don't mask out since that's dependent on opponent strategy
        possible_actions = [0, 1, 2]

        board_width = board_data['board']['width']
        board_height = board_data['board']['height']

        # compile a set of all the snake body pieces except for their tails since those
        # will move this turn (technically if they ate food it won't but for simplification
        # assume it will)
        snakes_together = set()
        for snake in board_data['board']['snakes']:
            snake_body = snake['body']
            for i in range(len(snake_body) - 1):
                snakes_together.add((snake_body[i]['x'], snake_body[i]['y']))        

        for action in possible_actions:
            action_will_lose = False 

            # check if coordinate is simply off the board
            next_move_coor = next_move_coordinates[action]
            if (next_move_coor[0] < 0) or (next_move_coor[0] >= board_width) or \
                (next_move_coor[1] < 0) or (next_move_coor[1] >= board_height):
                action_will_lose = True

            # now check if the coordinate is anywhere in the snake bodies
            if (next_move_coor in snakes_together):
                action_will_lose = True
            
            if (action_will_lose):
                action_values[0][action] = MIN_Q_VALUE

        return action_values
    
    def get_state_and_health_tensors_from_state_obj(self, state_obj):
        to_tensor = transforms.ToTensor()

        # convert state images to tensors
        state = to_tensor(state_obj['image'])
        
        state_health = torch.tensor(state_obj['health'], dtype=torch.float)
        # pad health tensor to always be max_snakes length        
        if (state_health.shape[0] < self.max_snakes):
            pad_required = self.max_snakes - state_health.shape[0]
            state_health = F.pad(state_health, pad=(0, pad_required), mode='constant', value=0.0)    
        state_health = state_health.unsqueeze(0)

        return state, state_health

    def cache(self, state_obj, next_state_obj, reward, action, done):    
        self.curr_step += 1

        # print(f'Caching State, Reward: {reward}, Action: {action}, Done: {done}')

        state, state_health = self.get_state_and_health_tensors_from_state_obj(state_obj)        
        next_state, next_state_health = self.get_state_and_health_tensors_from_state_obj(next_state_obj)        
                
        # convert reward, action, done to tensors
        reward = torch.tensor([reward], dtype=torch.float)
        action = torch.tensor([action], dtype=torch.long)
        done = torch.tensor([done])

        self.memory.add(TensorDict({
                    "state": state, 
                    "state_health" : state_health,
                    "next_state": next_state, 
                    "next_state_health" : next_state_health,
                    "action": action, 
                    "reward": reward, 
                    "done": done}, 
                batch_size=[]))
        
        results = {
            'epsilon': self.exploration_rate
        }
        
        # wait until memory builds before learning
        if (self.curr_step < self.burnin):
            return results

        # learn every few steps
        if (self.curr_step % self.learn_every == 0):
            results['loss'] = self.learn()

        # save every few steps
        if (self.curr_step % self.save_every == 0):
            self.save_network()

        return results
            
    def learn(self):
        state, next_state, action, reward, done = self.recall()

        print("learning....")

        # predict q values for this state
        pred = self.network(state)
        # clone the predictions
        target = pred.clone()
        
        # update the target q values as immediate reward + discounted future reward
        for idx in range(len(done)):
            Q_new = reward[idx].item()
            
            if not done[idx]:
                Q_new += self.gamma * torch.max(self.network(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # clear the optimizer gradients
        self.optimizer.zero_grad()
        # determine loss from our target and our predictions
        loss = self.criterion(target, pred)
        # back propagate to determine gradients
        loss.backward()        
        # take a step in the direction of the gradients
        self.optimizer.step()

        return loss.item()    

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)

        state, state_health, next_state, next_state_health, action, reward, done = (batch.get(key) for key in ("state", "state_health", "next_state", "next_state_health", "action", "reward", "done"))
        
        # reshape health
        state_health = state_health.view(self.batch_size, 1, 1, 1, self.max_snakes)        
        # Expand dimensions to match state
        state_health = state_health.expand(-1, -1, 67, 67, -1)
        # combine state and state_health
        state = torch.cat((state.unsqueeze(4), state_health), dim=4)

        # reshape next health
        next_state_health = next_state_health.view(self.batch_size, 1, 1, 1, self.max_snakes)
        # Expand dimensions to match next_state
        next_state_health = next_state_health.expand(-1, -1, 67, 67, -1)
        # combine next_state and next_state_health
        next_state = torch.cat((next_state.unsqueeze(4), next_state_health), dim=4)
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def set_model_save_path(self, path):
        self.model_save_path = path
        
        self.load_network(self.model_save_path)
    
    def save_network(self):
        # ensure that save path exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        # save model
        torch.save(
            dict(model=self.network.state_dict(), 
                 exploration_rate=self.exploration_rate,
                 curr_step=self.curr_step),
            self.model_save_path
        )
        print(f"SnakeNet saved to {self.model_save_path} at step {self.curr_step}")

    def load_network(self, path=None):
        # check if file exists first
        if (os.path.exists(path) == False):
            print(f"SnakeNet not found at {path}, skipping...")
            return
        
        saved_dict = torch.load(path, map_location=torch.device(self.device))
        
        self.network.load_state_dict(saved_dict['model'])
        self.exploration_rate = saved_dict['exploration_rate']
        self.curr_step = saved_dict['curr_step']

        print(f"Loaded SnakeNet from {path} at step {self.curr_step}, exploration rate {self.exploration_rate}")

class NeuralNetwork(nn.Module):
    def __init__(self, max_snakes, output_size):
        super(NeuralNetwork, self).__init__()

        self.max_snakes = max_snakes

        # 67 x 67 -> 16 x 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=4)      
        # 16 x 16 -> 7 x 7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 7 x 7 -> 5 x 5
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 64 x 5 x 5 -> 512
        self.fc1 = nn.Linear(1600 + max_snakes, 512) # we're going to append the snake healths after the convolution NN does it work
        self.fc2 = nn.Linear(512, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):        
        if (len(x.shape) == 5): # Batch of tensors
            # 4th dimension is normalized healths
            health_tensor = x[:, :, 0, 0, 1:] # slice out first element since that's remnant when we concated in cache()
            # slice out the health tensor
            x = x[:, :, :, :, 0]
        elif (len(x.shape) == 4): # Single tensor
            # 3rd dimension is normalized healths
            health_tensor = x[:, 0, 0, 1:]
            # slice out the health tensor now
            x = x[:, :, :, 0]
        else:
            raise ValueError("Shouldn't be here!")
        
        # 67 x 67 -> 16 x 16
        x = F.relu(self.conv1(x))        
        # 16 x 16 -> 7 x 7
        x = F.relu(self.conv2(x))
        # 7 x 7 -> 5 x 5
        x = F.relu(self.conv3(x))
        # 64 x 5 x 5 -> 1600
        x = x.view(-1, 64 * 5 * 5)
        
        # Squeeze out dimension 1 so [32, 1, 4] becomes [32, 4]
        health_tensor = health_tensor.squeeze(1)

        x = torch.cat((x, health_tensor), dim=1)        
        # 1604 -> 512
        x = F.relu(self.fc1(x))        
        # don't relu the last layer
        x = self.fc2(x)

        return x