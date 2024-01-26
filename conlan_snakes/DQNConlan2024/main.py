"""
DQNConlan2024

DQNConlan2024 is Conlan Rios' reinforcement learning snake for Standard 11x11
"""
import os
import random
import bottle
import rl_utils
from conlan_snakes.DQNConlan2024.dqn_snake_model import DQNSnakeModel

model = DQNSnakeModel(4)

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

def cache(state_obj, next_state_obj, reward, action, done):    
    return model.cache(state_obj, next_state_obj, reward, action, done)

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
    
    rl_state_image, snakes_health = rl_utils.convertBoardToState(data)
    
    # Get all the data
    you = data['you']

    snakeHead = you['body'][0]
    snakeHead = (snakeHead['x'], snakeHead['y'])
    
    snakeNeck = you['body'][1]
    snakeNeck = (snakeNeck['x'], snakeNeck['y'])

    is_training_reinforcement = data["is_training_reinforcement"] if "is_training_reinforcement" in data else False
    # if we're not training reinforcement, use greedy strategy and no epsilon
    use_greedy = False if is_training_reinforcement else True

    state_obj = {
         "image" : rl_state_image,
         "health" : snakes_health
    }
    # get move index from move [STRAIGHT, LEFT, RIGHT]
    dir_index = model.act(state_obj, use_greedy=use_greedy)
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