"""
DQNConlan2024

DQNConlan2024 is Conlan Rios' reinforcement learning snake for Standard 11x11
"""
import os
import random
import bottle
import rl_utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchvision import transforms

class DQNSnakeModel():
    def __init__(self) -> None:        
        # use gpu if available
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100_000, 
                                                device=torch.device("cpu")))
        self.batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        print(f"Using {self.device} device")

        self.network = NeuralNetwork(3)
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

    def act(self, stateImage, use_greedy=False):
        # only use epsilon greedy if we're not using greedy
        if (not use_greedy) and (self.random.random() < self.exploration_rate):
            # random move
            action_idx = self.random.randint(0, 2)
        else:
            to_tensor = transforms.ToTensor()

            state = to_tensor(stateImage).to(self.device)

            results = self.network(state)

            action_idx = torch.argmax(results).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
            
        return action_idx

    def cache(self, state, next_state, reward, action, done):    
        self.curr_step += 1

        to_tensor = transforms.ToTensor()

        state = to_tensor(state)        
        next_state = to_tensor(next_state)                
        reward = torch.tensor([reward], dtype=torch.float)
        action = torch.tensor([action], dtype=torch.long)
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))
        
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

        # predict q values for this state
        pred = self.network(state)
        # clone the predictions
        target = pred.clone()

        # update the target q values as immediate reward + discounted future reward
        for idx in range(len(done)):
            Q_new = reward[idx].item()
            
            if not done[idx]:
                Q_new += self.gamma * torch.max(self.network(next_state[idx]))

            # print(target[idx])
            # print(action[idx])
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            # print(Q_new)            
            # print(target[idx])

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

        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))

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
    def __init__(self, output_size):
        super(NeuralNetwork, self).__init__()

        # 67 x 67 -> 16 x 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=4)      
        # 16 x 16 -> 7 x 7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # 7 x 7 -> 5 x 5
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # 64 x 5 x 5 -> 512
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 67 x 67 -> 16 x 16
        x = F.relu(self.conv1(x))        
        # 16 x 16 -> 7 x 7
        x = F.relu(self.conv2(x))
        # 7 x 7 -> 5 x 5
        x = F.relu(self.conv3(x))
        # 64 x 5 x 5 -> 1600
        x = x.view(-1, 64 * 5 * 5)
        # 1600 -> 512
        x = F.relu(self.fc1(x))        
        # don't relu the last layer
        x = self.fc2(x)

        return x

model = DQNSnakeModel() 

@bottle.route('/')
def index():
	return "<h1>DQNConlan2024</h1>"

@bottle.route('/static/<path:path>')
def static(path):
	return bottle.static_file(path, root='static/')

@bottle.post('/ping')
def ping():
    return {}


@bottle.post('/end')
def end():
    return {}

@bottle.post('/start')
def start(data=None):
    headUrl = '%s://%s/static/head.png' % (
        bottle.request.urlparts.scheme,
        bottle.request.urlparts.netloc
    )
    
    if (model.model_save_path == None) and (data["model_save_path"]):
        model.set_model_save_path(data["model_save_path"])
    else:
        print(f'Exploration Rate: {model.exploration_rate:.5f}, Current Step: {model.curr_step}')

    return {
        'color': '#EADA50',
        'taunt': 'glhf!',
        'head_url': headUrl
    }

def cache(state, next_state, reward, action, done):    
    return model.cache(state, next_state, reward, action, done)

@bottle.post('/move')
def move(data=None):
    if not data:
        data = bottle.request.json
    
    # Move randomly the first turn since we don't have a direction
    # TODO move toward closest food instead
    if (data['turn'] == 0):
         return {
              'move' : model.random.choice(['up', 'down', 'left', 'right']),
              'local_direction' : None
         }
    
    rl_state = rl_utils.convertBoardToImage(data)
    
    # Get all the data
    you = data['you']

    snakeHead = you['body'][0]
    snakeHead = (snakeHead['x'], snakeHead['y'])
    
    snakeNeck = you['body'][1]
    snakeNeck = (snakeNeck['x'], snakeNeck['y'])

    is_training_reinforcement = data["is_training_reinforcement"] if "is_training_reinforcement" in data else False
    # if we're not training reinforcement, use greedy strategy and no epsilon
    use_greedy = False if is_training_reinforcement else True

    # get move index from move [STRAIGHT, LEFT, RIGHT]
    dir_index = model.act(rl_state, use_greedy=use_greedy)
    local_dir = rl_utils.LocalDirection(dir_index)

    return {
         'move' : rl_utils.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck),
         'local_direction' : local_dir
    }    

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug = True)