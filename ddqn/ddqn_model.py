import os
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
# from torchvision import transforms
import torch

# from ddqn.cnn_leaky import CNNLeaky
from ddqn.cnn_leaky_same import CNNLeakySame

import constants

class DDQNModel():
    def __init__(self) -> None:
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(400_000, \
                                                            device=torch.device("cpu")))
        self.batch_size = 256
        self.gamma = 0.999 # Discount factor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"                
        
        print(f"Using {self.device} device")

        self.onlineNetwork = CNNLeakySame(3)
        self.onlineNetwork.to(self.device)

        self.targetNetwork = CNNLeakySame(3)
        self.targetNetwork.to(self.device)

        self.sync_Q_target()        

        # Freeze target network parameters
        for p in self.targetNetwork.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.onlineNetwork.parameters(), lr=constants.DEFAULT_LEARNING_RATE)
        self.criterion = nn.MSELoss()
    
        self.model_save_path : str
        self.reward_set_key = constants.DEFAULT_REWARD_SET_KEY       

    def cache(self, obs, next_obs : int, action : int, reward_int : int, done_bool : bool):
        # print(f"    Caching: act=" + str(action) + ", reward=" + str(reward_int) + ", done=" + str(done_bool))

        state = torch.tensor(obs, dtype=torch.float)
        next_state = torch.tensor(next_obs, dtype=torch.float)

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
        
    def td_estimate(self, state, action) -> torch.tensor:
        current_q = self.onlineNetwork(state)
        
        current_q = current_q[
            np.arange(0, self.batch_size), action
        ]

        return current_q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done) -> torch.tensor:
        next_state_Q = self.onlineNetwork(next_state)
        
        best_action = torch.argmax(next_state_Q, axis=1)

        next_Q = self.targetNetwork(next_state)[
            np.arange(0, self.batch_size), best_action
        ]

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_q_online(self, td_estimate, td_target) -> float:
        loss = self.criterion(td_estimate, td_target)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()
    
    def sync_Q_target(self):
        print(f"    Syncing target network...")
        
        self.targetNetwork.load_state_dict(self.onlineNetwork.state_dict())
        
    def learn(self):        
        if (len(self.memory) < self.batch_size):
            # print(f"    Not enough memory to learn, skipping...")
            return None
        
        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_q_online(td_est, td_tgt)

        return loss

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)

        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()       

    def predict(self, observation) -> tuple:
        # to_tensor = transforms.ToTensor()

        # state = to_tensor(observation).to(self.device)
        state = torch.tensor(observation, dtype=torch.float).to(self.device)

        q_values = self.onlineNetwork(state)

        action_idx = torch.argmax(q_values).item()

        # return q values as a list so we can write it to frames later
        # in video replay (but DONT'T store the tensor as this will blow up GPU memory)
        q_values_as_list = q_values[0].tolist()

        return action_idx, q_values_as_list
        
    def save_model(self, training_info) -> None:        
        data_to_save = {
            'online_model': self.onlineNetwork.state_dict(),

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
                "epsilon_min" : 0.01,
                "curr_step" : 0
            }
        
        saved_dict = torch.load(path, map_location=torch.device(self.device))        
        
        self.onlineNetwork.load_state_dict(saved_dict['online_model'])
        self.sync_Q_target()
        
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