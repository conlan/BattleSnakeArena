from PIL import Image
from enum import IntEnum

import imageio
import numpy as np

ASSET_VERSION = 2
ASSET_FOLDER = "assets-v" + str(ASSET_VERSION) + "/"

GRID_IMAGE = Image.open(ASSET_FOLDER + "grid.png")
GRID_SIZE = GRID_IMAGE.size[0]

ME_HEAD_IMAGE = Image.open(ASSET_FOLDER + "me_head.png")
ME_BODY_IMAGE_STRAIGHT = Image.open(ASSET_FOLDER + "me_body_straight.png")
ME_BODY_IMAGE_CURVE = Image.open(ASSET_FOLDER + "me_body_curve.png")

ENEMY_HEAD_IMAGE = Image.open(ASSET_FOLDER + "enemy_head.png")
ENEMY_BODY_IMAGE_STRAIGHT = Image.open(ASSET_FOLDER + "enemy_body_straight.png")
ENEMY_BODY_IMAGE_CURVE = Image.open(ASSET_FOLDER + "enemy_body_curve.png")

FOOD_IMAGE = Image.open(ASSET_FOLDER + "food.png")

TRAINING_REWARD_SETS = {
    1 : {
        "REWARD_FOR_DEATH" : -1000,
        "REWARD_FOR_VICTORY" : 1000,
        "REWARD_FOR_SURVIVAL" : 1,
        "REWARD_FOR_FOOD" : 0
    },
    2 : {
        "REWARD_FOR_DEATH" : -1000,
        "REWARD_FOR_VICTORY" : 1000,
        "REWARD_FOR_SURVIVAL" : 1,
        "REWARD_FOR_FOOD" : 25
    },
    3 : {
        "REWARD_FOR_DEATH" : -1000,
        "REWARD_FOR_VICTORY" : 1000,
        "REWARD_FOR_SURVIVAL" : 1,
        "REWARD_FOR_FOOD" : 100
    }
}

class LocalDirection(IntEnum):
    STRAIGHT = 0,
    LEFT = 1,
    RIGHT = 2

def get_training_reward(training_reward_index, training_reward_key):
    if (training_reward_index not in TRAINING_REWARD_SETS):
        raise Exception("Invalid training reward index: " + str(training_reward_index))
    
    if (training_reward_key not in TRAINING_REWARD_SETS[training_reward_index]):
        raise Exception("Invalid training reward key: " + str(training_reward_key))
    
    return TRAINING_REWARD_SETS[training_reward_index][training_reward_key]

def getLocalDirectionAsCoordinate(dir, snakeHead, snakeNeck):
    head_x, head_y = snakeHead[0], snakeHead[1]

    move = getLocalDirectionAsMove(dir, snakeHead, snakeNeck)

    if (move == 'up'):
        return (head_x, head_y - 1)
    elif (move == 'down'):
        return (head_x, head_y + 1)
    elif (move == 'left'):
        return (head_x - 1, head_y)
    else: # right
        return (head_x + 1, head_y)
                
# Take head and neck coordinates and convert a local direction to 
# actual BattleSnake move
def getLocalDirectionAsMove(dir, snakeHead, snakeNeck):
    head_x, head_y = snakeHead[0], snakeHead[1]
    neck_x, neck_y = snakeNeck[0], snakeNeck[1]

    if (dir == LocalDirection.STRAIGHT):
         # Go straight         
        if (neck_y == (head_y + 1)): # snake is facing UP
            return 'up'
        elif (neck_y == (head_y - 1)): # snake is facing DOWN
             return 'down'
        elif (neck_x == (head_x - 1)): # snake is facing RIGHT
             return 'right'
        else: # snake is facing LEFT
            return 'left'
    elif (dir == LocalDirection.LEFT):
        # Turn Left
        if (neck_y == (head_y + 1)): # snake is facing UP
            return 'left'
        elif (neck_y == (head_y - 1)): # snake is facing DOWN
             return 'right'
        elif (neck_x == (head_x - 1)): # snake is facing RIGHT
             return 'up'
        else: # snake is facing LEFT
            return 'down'
    elif (dir == LocalDirection.RIGHT):
         # Turn Left
        if (neck_y == (head_y + 1)): # snake is facing UP
            return 'right'
        elif (neck_y == (head_y - 1)): # snake is facing DOWN
             return 'left'
        elif (neck_x == (head_x - 1)): # snake is facing RIGHT
             return 'down'
        else: # snake is facing LEFT
            return 'up'
    
    return 'up'

def tupleToXY(tuple):
    x = tuple['x']
    y = tuple['y']
    return (x * (GRID_SIZE - 1), y * (GRID_SIZE - 1))

# convert a board JSON object to a state and array of snake health
def convertBoardToState(json, pov_snake_id = None):        
    board_width = json['board']['width']
    board_height = json['board']['height']
    snakes = json['board']['snakes']
    food = json['board']['food']

    board_image = Image.new("RGBA", (board_width * (GRID_SIZE - 1) + 1, board_height * (GRID_SIZE - 1) + 1), (255, 255, 255, 255))
    
    x = 0
    y = 0

    for y_index in range(board_height):
        x = 0

        for x_index in range(board_width):                            
            board_image.paste(GRID_IMAGE, (x, y), GRID_IMAGE)
            
            if ({'x': x_index, 'y': y_index} in food):            
                board_image.paste(FOOD_IMAGE, (x, y), FOOD_IMAGE)

            x += GRID_SIZE - 1

        y += GRID_SIZE - 1

    # if pov snake id is not provided then get it from the 'you' parameter
    if (pov_snake_id == None) and ('you' in json):
        pov_snake_id = json['you']['id']

    for snake in snakes:
        snake_id = snake['id']
        is_pov_snake = (snake_id == pov_snake_id)

        snake_body = snake['body']

        snake_head = snake_body[0]
        snake_neck = snake_body[1] if len(snake_body) > 1 else snake_head

        head_x, head_y = tupleToXY(snake_head)
        neck_x, neck_y = tupleToXY(snake_neck)

        head_rotation = 0

        if (neck_x < head_x) and (neck_y == head_y):
            head_rotation = -90
        elif (neck_x > head_x) and (neck_y == head_y):
            head_rotation = 90
        elif (neck_x == head_x) and (neck_y < head_y):
            head_rotation = 180
        elif (neck_x == head_x) and (neck_y > head_y):
            head_rotation = 0

        # setup the images based on whether this snake is the POV or not
        head_image = ME_HEAD_IMAGE if is_pov_snake else ENEMY_HEAD_IMAGE
        straight_body_image = ME_BODY_IMAGE_STRAIGHT if is_pov_snake else ENEMY_BODY_IMAGE_STRAIGHT
        curve_body_image = ME_BODY_IMAGE_CURVE if is_pov_snake else ENEMY_BODY_IMAGE_CURVE

        rotated_head = head_image.rotate(head_rotation) if head_rotation != 0 else head_image

        board_image.paste(rotated_head, (head_x, head_y), rotated_head)

        prev_body_x, prev_body_y = head_x, head_y

        for body_index in range(1, len(snake_body)):
            body_x, body_y = tupleToXY(snake_body[body_index])

            # don't draw body if it's the last one
            if (body_x == prev_body_x) and (body_y == prev_body_y):
                continue

            image_to_use = None
            image_rotation = 0

            if (body_index < len(snake_body) - 1):
                next_body_x, next_body_y = tupleToXY(snake_body[body_index + 1])
            else:
                next_body_x, next_body_y = None, None

            # print("prev: " + str(prev_body_x) + ", " + str(prev_body_y))
            # print("body: " + str(body_x) + ", " + str(body_y))
            # print("next: " + str(next_body_x) + ", " + str(next_body_y))

            # there's no body after this segment, we're the tail
            if (next_body_x == None) or (next_body_y == None):
                image_to_use = straight_body_image

                # rotate 90 if we're on the same row ====
                if (body_y == prev_body_y):
                    image_rotation = 90
            else:
                # there's a body after this segment, determine if straight or curved
                if (prev_body_x > body_x) and (next_body_x == body_x):
                    image_to_use = curve_body_image      

                    if (prev_body_y == body_y):
                        if (prev_body_y < next_body_y):
                            image_rotation = 0
                        elif (prev_body_y > next_body_y):
                            image_rotation = 90

                elif (prev_body_x == body_x) and (next_body_x < body_x):
                    image_to_use = curve_body_image

                    if (body_y < prev_body_y):
                        image_rotation = -90
                    elif (body_y > prev_body_y):
                        image_rotation = 180

                elif (prev_body_x == body_x) and (next_body_x > body_x):
                    image_to_use = curve_body_image

                    if (next_body_y > prev_body_y):
                        image_rotation = 90
                elif (prev_body_x == body_x) and (next_body_x == body_x):
                    image_to_use = straight_body_image
                elif (prev_body_x > body_x) and (next_body_x < body_x):
                    image_to_use = straight_body_image

                    image_rotation = 90

                elif (prev_body_y > body_y) and (next_body_y < body_y):
                    image_to_use = straight_body_image

                elif (prev_body_x < body_x) and (next_body_x == body_x):
                    image_to_use = curve_body_image
                    if (next_body_y > prev_body_y):
                        image_rotation = -90
                    elif (next_body_y < prev_body_y):
                        image_rotation = 180

                elif (prev_body_x < body_x) and (next_body_x > body_x):
                    image_to_use = straight_body_image

                    image_rotation = 90
            
            if (image_rotation != 0):
                image_to_use = image_to_use.rotate(image_rotation)
            
            board_image.paste(image_to_use, (body_x, body_y), image_to_use)

            prev_body_x, prev_body_y = body_x, body_y

    # Convert to greyscale
    board_image = board_image.convert("L")

    # Put normalized snake healths in an array
    # in the order of length descending
    # POV snake is index-0 always though
    snakes_health_in_length_descending_order = []
    
    for snake in snakes:        
        snake_id = snake['id']
        snake_length = len(snake['body'])
        snake_health = snake['health']
        # normalize from 0.0 -> 1.0
        snake_health = snake_health / 100.0

        if (snake_id == pov_snake_id):
            snakes_health_in_length_descending_order.insert(0, {
                "length" : snake_length,
                "health" : snake_health,
                "is_pov" : True
            })
            continue

        did_insert = False
        for i in range(len(snakes_health_in_length_descending_order)):
            snake_obj = snakes_health_in_length_descending_order[i]
            
            # skip over the pov snake if we already inserted at index 0
            is_other_pov = snake_obj['is_pov']
            if is_other_pov:
                continue

            other_length = snake_obj['length']
            if (snake_length > other_length):                
                snakes_health_in_length_descending_order.insert(i, {
                    "length" : snake_length,
                    "health" : snake_health,
                    "is_pov" : False
                })
                did_insert = True
                break
            
        if not did_insert:
            snakes_health_in_length_descending_order.append({
                "length" : snake_length,
                "health" : snake_health,
                "is_pov" : False
            })

    snakes_health_in_length_descending_order = [obj['health'] for obj in snakes_health_in_length_descending_order]    
    
    return board_image, snakes_health_in_length_descending_order

def output_to_video(board_history_frames):   
    if (len(board_history_frames) == 0):
        print("No board history frames to output")
        return
    
    output_file = "output_video.mp4"

    with imageio.get_writer(output_file, fps=8) as writer:  # Adjust the fps as needed
        for img in board_history_frames:
            img_array = np.array(img)
            writer.append_data(img_array)

    print("Video saved to: " + output_file)