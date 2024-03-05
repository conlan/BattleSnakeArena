import numpy as np
import constants

from observer import Observer

class TensorObserver(Observer):
    def __init__(self):
        super().__init__()

    # translates our XY (x is left to right, y is top to bottom) to the tensor's XY
    def assign_tensor_val(self, tensor, layer, x, y, val):
        tensor[layer][y][x] = val

    def convert_data_to_tensor(self, data) -> np.ndarray:
        # initialize a 13 x 11 x 11 ndarray        

        # layer0: snake health on heads {0,...,100}
        # layer1: snake bodies {0,1}        
        # layer2: body segment numbers {0,...,255}
        # layer3: snake length >= player {0,1}        
        # layer4: food {0,1}
        # layer5: head_mask {0,1}       
        # layer6: double_tail_mask {0,1}        
        # layer7: snake bodies >= us {0,1}
        # layer8: snake bodies < us {0,1}
        # layer9-12: alive count mask

        snakes = data["board"]["snakes"]

        pov_snake = data["you"]
        pov_snake_id = pov_snake["id"]

        food = data["board"]["food"]

        player_size = len(pov_snake["body"])

        # print(snakes)
        # print(pov_snake)
        # print(data)

        board_tensor = np.zeros((13, 11, 11), dtype=float)
        
        # which direction the snake is pointing in
        pov_snake_orientation = None

        # check if there's a snake in the snakes list with pov snake's id
        pov_snake_is_alive = (pov_snake_id in [s["id"] for s in snakes])
        
        # if the pov snake is dead, we don't need to do anything else
        # Terminal state is an all-zero tensor
        if not pov_snake_is_alive:
            return board_tensor

        for s in snakes:
            snake_id = s["id"]
            health = s ["health"]
            body = s["body"]
            head_x, head_y = body[0]['x'], body[0]['y']            

            snake_size = len(body)

            # layer0: snake health on heads {0,...,100}
            self.assign_tensor_val(board_tensor, 0, head_x, head_y, health * 1.0 / constants.MAX_SNAKE_HEALTH)

            if (snake_id == pov_snake_id):
                # layer5: head_mask {0,1}
                self.assign_tensor_val(board_tensor, 5, head_x, head_y, 1)
                
                neck_x, neck_y = body[1]['x'], body[1]['y']
                if (head_x == neck_x):
                    if (head_y < neck_y):
                        pov_snake_orientation = 'up'
                    elif (head_y > neck_y):
                        pov_snake_orientation = 'down'
                elif (head_y == neck_y):
                    if (head_x < neck_x):
                        pov_snake_orientation = 'left'
                    elif (head_x > neck_x):
                        pov_snake_orientation = 'right'

            # snake body
            tail_1 = None
            tail_2 = None
            for i in reversed(range(1, len(body))):
                # print(i)
                segment = body[i]
                segment_x, segment_y = segment['x'], segment['y']

                if (tail_1 == None):
                    tail_1 = segment
                elif (tail_2 == None):
                    tail_2 = segment
                    
                    if (tail_1 == tail_2):
                        # layer6: double_tail_mask {0,1}
                        self.assign_tensor_val(board_tensor, 6, segment_x, segment_y, 1)

                # layer1: snake bodies {0,1}
                self.assign_tensor_val(board_tensor, 1, segment_x, segment_y, 1)

                # layer2: body segment numbers {0,...,255}
                self.assign_tensor_val(board_tensor, 2, segment_x, segment_y, i * 0.001)

                if (snake_id != pov_snake_id):
                    if (snake_size >= player_size):
                        # layer7: snake bodies >= us {0,1}
                        self.assign_tensor_val(board_tensor, 7, segment_x, segment_y, 1)# + snake_size - player_size)
                    else:
                        # layer8: snake bodies < us {0,1}
                        self.assign_tensor_val(board_tensor, 8, segment_x, segment_y, 1)#player_size - snake_size)

            # layer3: snake length >= player {0,1}
            if (snake_id != pov_snake_id):
                if (snake_size >= player_size):
                    self.assign_tensor_val(board_tensor, 3, head_x, head_y, 1)


        for f in food:
            food_x, food_y = f['x'], f['y']

            # layer4: food {0,1}
            self.assign_tensor_val(board_tensor, 4, food_x, food_y, 1)

        board_width = data["board"]["width"]
        board_height = data["board"]["height"]

        num_snakes_alive = len(snakes)

        if (num_snakes_alive > 0):
            for x in range(board_width):
                for y in range(board_height):
                    # layer9-12: alive count mask
                    self.assign_tensor_val(board_tensor, 8 + num_snakes_alive, x, y, 1)

        # np.set_printoptions(threshold=np.inf)
        # print(board_tensor)
            
        # rotate the board so that the POV snake is always facing UP
        if (pov_snake_orientation == 'right'):
            # rotate once to the left
            board_tensor = np.rot90(board_tensor, 1, axes=(1, 2)).copy()
        elif (pov_snake_orientation == 'left'):
            # rotate thrice to the left
            board_tensor = np.rot90(board_tensor, 3, axes=(1, 2)).copy()
        elif (pov_snake_orientation == 'down'):
            # rotate twice so it's flipped upside down
            board_tensor = np.rot90(board_tensor, 2, axes=(1, 2)).copy()

        # print('-------------------')
        # np.set_printoptions(threshold=np.inf)
        # print(board_tensor)
        
        return board_tensor        

    def observe(self, data : dict, should_store_observation : bool) -> dict:
        tensor = self.convert_data_to_tensor(data)

        game_id = data["game"]["id"]

        if (game_id not in self.observations):
            self.observations[game_id] = []

        observation = {
            "tensor" : tensor
        }

        if (should_store_observation):
            observation["image"] = self.convert_data_to_image(data)
            
            self.observations[game_id].append(observation)

        return observation