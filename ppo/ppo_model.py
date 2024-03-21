import os
import torch

import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
# import numpy as np

import torch
import constants

from ppo.cnn_leaky_same3 import CNNLeakySameV3

class PPOModel():
    def __init__(self) -> None:        
        self.gamma = 0.999 # Discount factor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"                
        
        print(f"Using {self.device} device")

        self.policy = CNNLeakySameV3(3)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=constants.DEFAULT_LEARNING_RATE)
    
        self.model_save_path : str

    def predict(self, observation) -> tuple:
        tensor = observation["tensor"]
        tensor_is_mirror = observation["is_mirror"]

        state = torch.tensor(tensor, dtype=torch.float).to(self.device)

        probs = self.policy(state)
        
        m = Categorical(probs)
        
        action = m.sample()

        return action.item(), m.log_prob(action)
        
    def save_model(self, training_info) -> None:        
        pass
        # data_to_save = {
        #     'online_model': self.onlineNetwork.state_dict(),

        #     'gamma': self.gamma,
        #     'reward_set_key': self.reward_set_key,

        #     'epsilon': training_info["epsilon"],
        #     'epsilon_decay': training_info["epsilon_decay"],
        #     'epsilon_min': training_info["epsilon_min"],

        #     'curr_step': training_info['curr_step']
        # }

        # torch.save(data_to_save, self.model_save_path)

        # print(f"\n    SAVED model and training state {training_info} to {self.model_save_path}\n")

    def load_model(self, path) -> dict:
        self.model_save_path = path

        # check if file exists first
        if (os.path.exists(path) == False):
            print(f"Model not found at {path}, skipping...")
            
            # return defaults for epsilon
            return {            
            }
        
        # saved_dict = torch.load(path, map_location=torch.device(self.device))        
        
        # self.onlineNetwork.load_state_dict(saved_dict['online_model'])
        
        # self.gamma = saved_dict['gamma']