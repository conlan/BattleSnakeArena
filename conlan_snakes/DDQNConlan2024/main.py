"""
DDQNConlan2024

DDQNConlan2024 is Conlan Rios' reinforcement learning snake for Standard 11x11
"""
import os
import random
import bottle
import rl_utils

@bottle.route('/')
def index():
	return "<h1>DDQNConlan2024</h1>"

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

    # TODO load network weights from file
    # import json
    # print(json.dumps(data))
    # exit()

    return {
        'color': '#EADA50',
        'taunt': 'glhf!',
        'head_url': headUrl
    }

@bottle.post('/move')
def move(data=None):
    if not data:
        data = bottle.request.json

    # make a random generator that we use here so the seed doesn't get overriden in the main game
    rand = random.Random()
    # rand.seed(1)
    
    # Move randomly the first turn since we don't have a direction
    # TODO move toward closest food instead
    if (data['turn'] == 0):
         return {
              'move' : rand.choice(['up', 'down', 'left', 'right'])
         }
    
    # Get all the data
    you = data['you']

    snakeHead = you['body'][0]
    snakeHead = (snakeHead['x'], snakeHead['y'])
    
    snakeNeck = you['body'][1]
    snakeNeck = (snakeNeck['x'], snakeNeck['y'])

    dir = rand.choice([rl_utils.LocalDirection.STRAIGHT, rl_utils.LocalDirection.LEFT, rl_utils.LocalDirection.RIGHT])

    return {
         'move' : rl_utils.getLocalDirectionAsMove(dir, snakeHead, snakeNeck)
    }    

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug = True)