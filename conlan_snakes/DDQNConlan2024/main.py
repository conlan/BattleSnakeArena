"""
DDQNConlan2024

DDQNConlan2024 is Conlan Rios' reinforcement learning snake for Standard 11x11
"""
import os
import random
import bottle

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
    rand = random.Random(1)
    # rand.seed(1)
    # Get all the data
    you = data['you']

    snakeHead = you['body'][0]
    snakeHead = (snakeHead['x'], snakeHead['y'])
    
    snakeNext = you['body'][0] if len(you['body']) == 1 else you['body'][1]
    snakeNext = (snakeNext['x'], snakeNext['y'])
    
    moves = ['up', 'down', 'left', 'right']
    
    # just don't move back into body
    if (snakeHead[0] == snakeNext[0] + 1):
        moves.remove('left')
    elif (snakeHead[0] == snakeNext[0] - 1):
        moves.remove('right')
    if (snakeHead[1] == snakeNext[1] + 1):
        moves.remove('up')
    elif (snakeHead[1] == snakeNext[1] - 1):
        moves.remove('down')
        
    return {
        'move': rand.choice(moves)
    }

# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug = True)