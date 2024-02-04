import os
import torch

import torch.nn as nn
import torch.optim as optim

from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from dqn.cnn import CNN

class DQNModel():
    def __init__(self, model_save_path, learning_rate) -> None:
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100_000, \
                                                            device=torch.device("cpu")))
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        
        print(f"Using {self.device} device")

        self.network = CNN(3)
        self.network.to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
        # Load network file
        self.load_network(model_save_path)

    def predict(self, observation) -> int:
        return 0
        # with torch.no_grad():
        #     state = torch.tensor(board, dtype=torch.float32).to(self.device)
        #     state = state.unsqueeze(0)
        #     return self.network(state).argmax().item()
        
    def save_network(self) -> None:
        torch.save({
            'model': self.network.state_dict(),
            'reward_set_key': self.reward_set_key
        }, self.model_save_path)

        print(f"Saved network to {self.model_save_path}")

    def load_network(self, path) -> None:
        self.model_save_path = path
        self.reward_set_key = "reward-set-v1"

        # check if file exists first
        if (os.path.exists(path) == False):
            print(f"SnakeNet not found at {path}, skipping...")
            return
        
        saved_dict = torch.load(path, map_location=torch.device(self.device))
        
        self.network.load_state_dict(saved_dict['model'])

        self.reward_set_key = saved_dict['reward_set_key']

        print(f"Loaded network from {path}")