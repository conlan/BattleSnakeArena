import os
import torch

import torch.nn as nn
import torch.optim as optim

from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from dqn.cnn import CNN

class DQNModel():
    def __init__(self, model_save_path, epsilon_info, learning_rate) -> None:
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100_000, \
                                                            device=torch.device("cpu")))
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        print(f"Using {self.device} device")

        self.network = CNN(3)
        self.network.to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.reward_set_key = None

        # Load Epsilon settings
        self.load_epsilon(epsilon_info)
        # Load network file
        self.load_network(model_save_path)

    def load_epsilon(self, epsilon_info) -> None:
        self.epsilon = epsilon_info["epsilon"]
        self.epsilon_decay = epsilon_info["epsilon_decay"]
        self.epsilon_min = epsilon_info["epislon_min"]

    def load_network(self, path) -> None:
        self.model_save_path = path
        self.reward_set_key = "reward-set-v1"

        # check if file exists first
        if (os.path.exists(path) == False):
            print(f"SnakeNet not found at {path}, skipping...")
            return
        
        saved_dict = torch.load(path, map_location=torch.device(self.device))
        
        self.network.load_state_dict(saved_dict['model'])

        # TODO set reward set key
        

        print(f"Loaded network from {path}")