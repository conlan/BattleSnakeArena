import numpy as np

from observer import Observer

class TensorObserver(Observer):
    def __init__(self):
        super().__init__()

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

        tensor = np.zeros((13, 11, 11), dtype=int)

        for s in snakes:
            snake_id = s["id"]
            health = s ["health"]
            body = s["body"]
            head_x, head_y = body[0]['x'], body[0]['y']
            snake_size = len(body)

            # layer0: snake health on heads {0,...,100}
            tensor[0][head_x][head_y] = health

            if (snake_id == pov_snake_id):
                # layer5: head_mask {0,1}
                tensor[5][head_x][head_y] = 1

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
                        tensor[6][segment_x][segment_y] = 1

                # layer1: snake bodies {0,1}
                tensor[1][segment_x][segment_y] = 1

                # layer2: body segment numbers {0,...,255}
                tensor[2][segment_x][segment_y] = i

                if (snake_id != pov_snake_id):
                    if (snake_size >= player_size):
                        # layer7: snake bodies >= us {0,1}
                        tensor[7][segment_x][segment_y] = 1 + snake_size - player_size
                    else:
                        # layer8: snake bodies < us {0,1}
                        tensor[8][segment_x][segment_y] = player_size - snake_size

            # layer3: snake length >= player {0,1}
            if (snake_id != pov_snake_id):
                if (snake_size >= player_size):
                    tensor[3][head_x][head_y] = 1


        for f in food:
            food_x, food_y = f['x'], f['y']

            # layer4: food {0,1}
            tensor[4][food_x][food_y] = 1

        board_width = data["board"]["width"]
        board_height = data["board"]["height"]

        num_snakes_alive = len(snakes)

        for x in range(board_width):
            for y in range(board_height):
                # layer9-12: alive count mask
                tensor[9 + num_snakes_alive][x][y] = 1

        # np.set_printoptions(threshold=np.inf)
        # print(tensor)
        
        return tensor
        

    def observe(self, data : dict, should_store_observation : bool) -> dict:
        tensor = self.convert_data_to_tensor(data)

        game_id = data["game"]["id"]

        if (game_id not in self.observations):
            self.observations[game_id] = []

        observation = {
            "tensor" : tensor
        }

        if (should_store_observation):
            self.observations[game_id].append(observation)

        return observation