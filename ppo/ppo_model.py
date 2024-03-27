import os
import torch as T
import numpy as np

import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
# import numpy as np

import torch
import constants

from ppo.ppo_actor_network_1 import PPOActorNetworkV1
from ppo.ppo_critic_network_1 import PPOCriticNetworkV1

class PPOMemory:
    def __init__(self, batch_size):
        self.clear_memory()

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)

        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches
    
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PPOModel():
    def __init__(self, label, model_save_path) -> None:        
        self.gamma = 0.999 # Discount factor
        self.batch_size = 1000
        self.n_epochs = 1
        self.policy_clip = 0.2
        self.entropy_coef = 0.01

        self.device = "cuda" if torch.cuda.is_available() else "cpu"                
        
        print(f"{label} using device {self.device}")

        self.actor = PPOActorNetworkV1(3).to(self.device)
        self.critic = PPOCriticNetworkV1().to(self.device)
        self.memory = PPOMemory(self.batch_size)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=constants.DEFAULT_ACTOR_LEARNING_RATE, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=constants.DEFAULT_CRITIC_LEARNING_RATE, eps=1e-5)
            
    
        self.reward_set_key = constants.DEFAULT_REWARD_SET_KEY

        self.load_model(model_save_path)        

    def learn(self):
        print("learning..")

        avg_loss = 0

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # calculate the advantage of each state by iterate from end to start
            for t in reversed(range(len(reward_arr))):
                return_t = reward_arr[t]

                if (dones_arr[t] == False):
                    return_t += self.gamma * advantage[t + 1]

                advantage[t] = return_t

            advantage = advantage - values

            # convert advantage and values to tensors
            advantage = T.tensor(advantage).to(self.device)
            values = T.tensor(values).to(self.device)

            for batch in batches:                
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                critic_value = T.squeeze(self.critic(states))

                entropy = dist.entropy().mean()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp() # put back into exponential form
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()                

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                avg_loss += total_loss.item()                

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss.backward()
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()
        
        return avg_loss * 1.0 / self.n_epochs

    def store_memory(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

        # print(f"Stored memory: {type(state)}, {action}, {probs}, {vals}, {reward}, {done}")

    def predict(self, observation) -> tuple:
        tensor = observation["tensor"]

        state = T.tensor(tensor, dtype=T.float).to(self.device)

        dist = self.actor(state)
        value = self.critic(state)

        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value
        
    def save_model(self, curr_step) -> None: 
        data_to_save = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'curr_step': curr_step
        }       

        T.save(data_to_save, self.model_save_path)
        print(f"\n    SAVED model to {self.model_save_path}\n")

    def load_model(self, path) -> dict:
        self.model_save_path = path
        self.curr_step = 0

        # check if file exists first
        if (os.path.exists(path) == False):
            print(f"Model not found at {path}, skipping...")
            
            return        
        
        saved_dict = torch.load(path, map_location=torch.device(self.device))        

        self.actor.load_state_dict(saved_dict['actor'])
        self.critic.load_state_dict(saved_dict['critic'])
        self.curr_step = saved_dict['curr_step']
        
        print(f"Model loaded from {path}")    