import constants
from PIL import Image, ImageDraw

FOOD_COLOR = constants.COLORS["green"]
BORDER_COLOR = constants.COLORS["grey"]
DEFAULT_COLOR = constants.COLORS["default"]

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

class Observer():    
    def __init__(self) -> None:
        self.observations = {}

    def tupleToXY(self, tuple):
        x = tuple['x']
        y = tuple['y']
        return (x * (GRID_SIZE - 1), y * (GRID_SIZE - 1))        
    
    def convert_data_to_image(self, json) -> Image:
        board_width = json['board']['width']
        board_height = json['board']['height']
        snakes = json['board']['snakes']
        food = json['board']['food']

        board_image = Image.new("RGBA", (board_width * (GRID_SIZE - 1) + 1, board_height * (GRID_SIZE - 1) + 1), (255, 255, 255, 255))

        # draw border
        border_color = (255, 255, 0)
        border_width = 1
        draw = ImageDraw.Draw(board_image)
        width, height = board_image.size
        draw.line([(0, 0), (width, 0)], fill=border_color, width=border_width)
        draw.line([(0, height - 1), (width, height - 1)], fill=border_color, width=border_width)
        draw.line([(0, 0), (0, height)], fill=border_color, width=border_width)
        draw.line([(width - 1, 0), (width - 1, height)], fill=border_color, width=border_width)
        
        x = 0
        y = 0

        for y_index in range(board_height):
            x = 0

            for x_index in range(board_width):                            
                # board_image.paste(GRID_IMAGE, (x, y), GRID_IMAGE)
                
                if ({'x': x_index, 'y': y_index} in food):            
                    board_image.paste(FOOD_IMAGE, (x, y), FOOD_IMAGE)

                x += GRID_SIZE - 1

            y += GRID_SIZE - 1

        # if pov snake id is not provided then get it from the 'you' parameter
        # if (pov_snake_id == None) and ('you' in json):
        pov_snake_id = json['you']['id']

        for snake in snakes:
            snake_id = snake['id']
            is_pov_snake = (snake_id == pov_snake_id)

            snake_body = snake['body']

            snake_head = snake_body[0]
            snake_neck = snake_body[1] if len(snake_body) > 1 else snake_head

            head_x, head_y = self.tupleToXY(snake_head)
            neck_x, neck_y = self.tupleToXY(snake_neck)

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
                body_x, body_y = self.tupleToXY(snake_body[body_index])

                # don't draw body if it's the last one
                if (body_x == prev_body_x) and (body_y == prev_body_y):
                    continue

                image_to_use = None
                image_rotation = 0

                if (body_index < len(snake_body) - 1):
                    next_body_x, next_body_y = self.tupleToXY(snake_body[body_index + 1])
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

        return board_image
        # # Put normalized snake healths in an array
        # # in the order of length descending
        # # POV snake is index-0 always though
        # snakes_health_in_length_descending_order = []
        
        # for snake in snakes:        
        #     snake_id = snake['id']
        #     snake_length = len(snake['body'])
        #     snake_health = snake['health']
        #     # normalize from 0.0 -> 1.0
        #     snake_health = snake_health / 100.0

        #     if (snake_id == pov_snake_id):
        #         snakes_health_in_length_descending_order.insert(0, {
        #             "length" : snake_length,
        #             "health" : snake_health,
        #             "is_pov" : True
        #         })
        #         continue

        #     did_insert = False
        #     for i in range(len(snakes_health_in_length_descending_order)):
        #         snake_obj = snakes_health_in_length_descending_order[i]
                
        #         # skip over the pov snake if we already inserted at index 0
        #         is_other_pov = snake_obj['is_pov']
        #         if is_other_pov:
        #             continue

        #         other_length = snake_obj['length']
        #         if (snake_length > other_length):                
        #             snakes_health_in_length_descending_order.insert(i, {
        #                 "length" : snake_length,
        #                 "health" : snake_health,
        #                 "is_pov" : False
        #             })
        #             did_insert = True
        #             break
                
        #     if not did_insert:
        #         snakes_health_in_length_descending_order.append({
        #             "length" : snake_length,
        #             "health" : snake_health,
        #             "is_pov" : False
        #         })

        # snakes_health_in_length_descending_order = [obj['health'] for obj in snakes_health_in_length_descending_order]    

    def observe(self, data, should_store_observation) -> None:
        image = self.convert_data_to_image(data)

        game_id = data["game"]["id"]

        if (game_id not in self.observations):
            self.observations[game_id] = []

        observation = {
            "image" : image
        }

        if (should_store_observation):
            self.observations[game_id].append(observation)

        return observation# snakes_health_in_length_descending_order

    def print_game(self, game) -> None:
        width = game.width()
        height = game.height()

        ywall = " " * 2 * width + "  "

        print(f"{BORDER_COLOR}{ywall}{DEFAULT_COLOR}") # Y Border
        
        for j in range(height):
            print(f"{BORDER_COLOR} {DEFAULT_COLOR}", end="") # X Border
            for i in range(width):
                if (i, j) in game.food:
                    print(f"{FOOD_COLOR}  {DEFAULT_COLOR}", end="") # Food
                else:
                    no_snake = True
                    for ind, s in enumerate(game.live_snakes):                        
                        if (i, j) in s.body:
                            if s.body[0] == (i, j):
                                print(f"{game.live_snakes[ind].color}OO{DEFAULT_COLOR}", end="") # Head
                            else:
                                print(f"{game.live_snakes[ind].color}  {DEFAULT_COLOR}", end="") # Body
                            no_snake = False
                    if no_snake:
                        print(f"{DEFAULT_COLOR}  {DEFAULT_COLOR}", end="") # Empty
        
            print(f"{BORDER_COLOR} {DEFAULT_COLOR}") # X Border
        
        print(f"{BORDER_COLOR}{ywall}{DEFAULT_COLOR}") # Y Border