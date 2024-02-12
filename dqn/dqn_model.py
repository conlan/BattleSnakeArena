import os
import torch

import torch.nn as nn
import torch.optim as optim

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchvision import transforms

from dqn.cnn import CNN

import constants

class DQNModel():
    def __init__(self) -> None:
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100_000, \
                                                            device=torch.device("cpu")))
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        
        print(f"Using {self.device} device")

        self.network = CNN(3)
        self.network.to(self.device)

        self.gamma = 0.9 # Discount factor

        self.optimizer = optim.Adam(self.network.parameters(), lr=constants.DEFAULT_LEARNING_RATE)
        self.criterion = nn.MSELoss()
    
        self.model_save_path : str
        self.reward_set_key = constants.DEFAULT_REWARD_SET_KEY       

    def cache(self, obs, next_obs : int, action : torch.tensor, reward_int : int, done_bool : bool):

        # print(f"    Caching: act=" + str(action) + ", reward=" + str(reward_int) + ", done=" + str(done_bool))        

        to_tensor = transforms.ToTensor()

        state = to_tensor(obs)
        next_state = to_tensor(next_obs)

        # convert reward, action, done to tensors
        reward : torch.tensor = torch.tensor([reward_int], dtype=torch.float)
        action = torch.tensor([action], dtype=torch.long)
        done : torch.tensor = torch.tensor([done_bool])

        self.memory.add(TensorDict({
                    "state": state,
                    "next_state": next_state,
                    "action": action, 
                    "reward": reward, 
                    "done": done}, 
                batch_size=[]))
        
    def learn(self):        
        if (len(self.memory) < self.batch_size):
            # print(f"    Not enough memory to learn, skipping...")
            return None
        
        state, next_state, action, reward, done = self.recall()

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

        # print(f"    Learning Loss: {loss.item()}")

        return loss.item()   

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)

        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()       

    def predict(self, observation) -> tuple:
        to_tensor = transforms.ToTensor()

        state = to_tensor(observation).to(self.device)

        q_values = self.network(state)

        action_idx = torch.argmax(q_values).item()

        # return q values as a list so we can write it to frames later
        # in video replay (but DONT'T store the tensor as this will blow up GPU memory)
        q_values_as_list = q_values[0].tolist()

        return action_idx, q_values_as_list
        
    def save_model(self, training_info) -> None:        
        data_to_save = {
            'model': self.network.state_dict(),
            'gamma': self.gamma,
            'reward_set_key': self.reward_set_key,

            'epsilon': training_info["epsilon"],
            'epsilon_decay': training_info["epsilon_decay"],
            'epsilon_min': training_info["epsilon_min"],

            'curr_step': training_info['curr_step']
        }

        torch.save(data_to_save, self.model_save_path)

        print(f"\n    SAVED model and training state {training_info} to {self.model_save_path}\n")

    def load_model(self, path) -> dict:
        self.model_save_path = path

        # check if file exists first
        if (os.path.exists(path) == False):
            print(f"Model not found at {path}, skipping...")
            
            # return defaults for epsilon
            return {
                "epsilon" : 1,
                "epsilon_decay" : 0.0000009, # 1.0 -> 0.1 in 1,000,000 steps
                "epsilon_min" : 0.1,
                "curr_step" : 0
            }
        
        saved_dict = torch.load(path, map_location=torch.device(self.device))        
        
        self.network.load_state_dict(saved_dict['model'])    
        self.gamma = saved_dict['gamma']
        self.reward_set_key = saved_dict['reward_set_key']

        training_info = {
            "epsilon" : saved_dict["epsilon"],
            "epsilon_decay" : saved_dict["epsilon_decay"],
            "epsilon_min" : saved_dict["epsilon_min"],
            "curr_step" : saved_dict["curr_step"]
        }

        print(f"\nLoaded model from {path}, epsilon: {training_info['epsilon']}, epsilon_decay: {training_info['epsilon_decay']}, epsilon_min: {training_info['epsilon_min']}, reward_set: {self.reward_set_key} curr_step: {training_info['curr_step']}")

        return training_info