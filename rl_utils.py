from PIL import Image, ImageDraw, ImageFont
from enum import Enum

import imageio
import numpy as np

GRID_IMAGE = Image.open("assets/grid.png")
GRID_SIZE = GRID_IMAGE.size[0]

ME_HEAD_IMAGE = Image.open("assets/me_head.png")
ME_BODY_IMAGE_STRAIGHT = Image.open("assets/me_body_straight.png")
ME_BODY_IMAGE_CURVE = Image.open("assets/me_body_curve.png")

FOOD_IMAGE = Image.open("assets/food.png")

REWARD_FOR_DEATH = -500
REWARD_FOR_VICTORY = 500
REWARD_FOR_SURVIVAL = 1
REWARD_FOR_FOOD = 25

class LocalDirection(Enum):
    STRAIGHT = 0,
    LEFT = 1,
    RIGHT = 2

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
    return (tuple[0] * (GRID_SIZE - 1), tuple[1] * (GRID_SIZE - 1))

def convertBoardToImage(board_width, board_height, snakes, food, boardHistoryFrames = None):    
    board_image = Image.new("RGBA", (board_width * (GRID_SIZE - 1) + 1, board_height * (GRID_SIZE - 1) + 1), (255, 255, 255, 255))
    
    x = 0
    y = 0

    for y_index in range(board_height):
        x = 0

        for x_index in range(board_width):                            
            board_image.paste(GRID_IMAGE, (x, y), GRID_IMAGE)

            if ((x_index, y_index) in food):
                board_image.paste(FOOD_IMAGE, (x, y), FOOD_IMAGE)

            x += GRID_SIZE - 1

        y += GRID_SIZE - 1

    for snake in snakes:
        snake_head = snake.body[0]
        snake_neck = snake.body[1] if len(snake.body) > 1 else snake_head

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

        rotated_head = ME_HEAD_IMAGE.rotate(head_rotation) if head_rotation != 0 else ME_HEAD_IMAGE

        board_image.paste(rotated_head, (head_x, head_y), rotated_head)

        prev_body_x, prev_body_y = head_x, head_y

        for body_index in range(1, len(snake.body)):
            body_x, body_y = tupleToXY(snake.body[body_index])

            # don't draw body if it's the last one
            if (body_x == prev_body_x) and (body_y == prev_body_y):
                continue

            image_to_use = None
            image_rotation = 0

            if (body_index < len(snake.body) - 1):
                next_body_x, next_body_y = tupleToXY(snake.body[body_index + 1])
            else:
                next_body_x, next_body_y = None, None

            # print("prev: " + str(prev_body_x) + ", " + str(prev_body_y))
            # print("body: " + str(body_x) + ", " + str(body_y))
            # print("next: " + str(next_body_x) + ", " + str(next_body_y))

            # there's no body after this segment, we're the tail
            if (next_body_x == None) or (next_body_y == None):
                image_to_use = ME_BODY_IMAGE_STRAIGHT

                # rotate 90 if we're on the same row ====
                if (body_y == prev_body_y):
                    image_rotation = 90
            else:
                # there's a body after this segment, determine if straight or curved
                if (prev_body_x > body_x) and (next_body_x == body_x):
                    image_to_use = ME_BODY_IMAGE_CURVE      

                    if (prev_body_y == body_y):
                        if (prev_body_y < next_body_y):
                            image_rotation = 0
                        elif (prev_body_y > next_body_y):
                            image_rotation = 90

                elif (prev_body_x == body_x) and (next_body_x < body_x):
                    image_to_use = ME_BODY_IMAGE_CURVE

                    if (body_y < prev_body_y):
                        image_rotation = -90
                    elif (body_y > prev_body_y):
                        image_rotation = 180

                elif (prev_body_x == body_x) and (next_body_x > body_x):
                    image_to_use = ME_BODY_IMAGE_CURVE

                    if (next_body_y > prev_body_y):
                        image_rotation = 90
                elif (prev_body_x == body_x) and (next_body_x == body_x):
                    image_to_use = ME_BODY_IMAGE_STRAIGHT
                elif (prev_body_x > body_x) and (next_body_x < body_x):
                    image_to_use = ME_BODY_IMAGE_STRAIGHT

                    image_rotation = 90

                elif (prev_body_y > body_y) and (next_body_y < body_y):
                    image_to_use = ME_BODY_IMAGE_STRAIGHT

                elif (prev_body_x < body_x) and (next_body_x == body_x):
                    image_to_use = ME_BODY_IMAGE_CURVE
                    if (next_body_y > prev_body_y):
                        image_rotation = -90
                    elif (next_body_y < prev_body_y):
                        image_rotation = 180

                elif (prev_body_x < body_x) and (next_body_x > body_x):
                    image_to_use = ME_BODY_IMAGE_STRAIGHT

                    image_rotation = 90
            
            if (image_rotation != 0):
                image_to_use = image_to_use.rotate(image_rotation)
            
            board_image.paste(image_to_use, (body_x, body_y), image_to_use)

            prev_body_x, prev_body_y = body_x, body_y
    
    if (boardHistoryFrames != None):
        boardHistoryFrames.append(board_image)
    
    return board_image

def outputToVideo(boardHistoryFrames):   
    output_file = "output_video.mp4"

    with imageio.get_writer(output_file, fps=8) as writer:  # Adjust the fps as needed
        for img in boardHistoryFrames:
            img_array = np.array(img)
            writer.append_data(img_array)

    print("Video saved to: " + output_file)