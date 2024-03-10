import random
import constants

from snake_controller import SnakeController

from ddqn.ddqn_model import DDQNModel

class DDQNController (SnakeController):
    def __init__(self, model_save_path, nickname, convert_data_to_image, should_action_mask):
        super().__init__(nickname)

        self.convert_data_to_image = convert_data_to_image
        
        self.should_action_mask = should_action_mask

        self.model = DDQNModel()

        self.epsilon_info = self.model.load_model(model_save_path)

        self.load_epsilon(self.epsilon_info)

    def name(self) -> str:
        return "DDQNController (model=" + self.model.model_save_path + ", epsilon=" + str(self.epsilon) + ")"

    def get_epsilon_info(self) -> dict:
        return {
            "epsilon" : self.epsilon,
            "epsilon_decay" : self.epsilon_decay,
            "epsilon_min" : self.epsilon_min
        }

    def load_epsilon(self, epsilon_info) -> None:
        self.epsilon = epsilon_info["epsilon"]
        self.epsilon_decay = epsilon_info["epsilon_decay"]
        self.epsilon_min = epsilon_info["epsilon_min"]

        print(f'Loaded epsilon: {self.epsilon}, decay: {self.epsilon_decay}, min: {self.epsilon_min}')

    def act(self, data) -> dict:
        move = None
        local_dir = None
        q_values = None

        if (data['turn'] == 0):
            move = random.choice(['up', 'down', 'left', 'right'])
        else:
            you = data['you']

            snakeHead = you['body'][0]
            snakeHead = (snakeHead['x'], snakeHead['y'])
            
            snakeNeck = you['body'][1]
            snakeNeck = (snakeNeck['x'], snakeNeck['y'])

            if (random.uniform(0.0, 1.0) < self.epsilon):                
                if (self.should_action_mask):
                    # initialize a vector of 3 random starting values
                    q_values = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                    # then apply the action mask
                    local_dir, q_values = self.apply_action_mask(q_values, data)
                else:
                    local_dir = random.choice([0, 1, 2])
            else:
                obs_data = self.convert_data_to_image(data)

                local_dir, q_values = self.model.predict(obs_data)

                if (self.should_action_mask):
                    # apply action mask (override q-value with LOSE score for guaranteed losing moving directions)
                    local_dir, q_values = self.apply_action_mask(q_values, data)

            move = self.getLocalDirectionAsMove(local_dir, snakeHead, snakeNeck)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
        game_id = data["game"]["id"]
        
        return self.store_move(game_id, move, local_dir, q_values)
    
    def apply_action_mask(self, q_values, data):
        # min float value
        MIN_Q_VALUE = -99999999999999999

        you = data['you']
        you_health = you['health']

        board_width = data['board']['width']
        board_height = data['board']['height']
        
        # convert board food to tuples
        board_food = data['board']['food']        
        board_food = [(food['x'], food['y']) for food in board_food]

        snakeHead = you['body'][0]
        snakeHead = (snakeHead['x'], snakeHead['y'])
        
        snakeNeck = you['body'][1]
        snakeNeck = (snakeNeck['x'], snakeNeck['y'])
    
        # determine the absolute moves for each local direction
        next_moves_absolute = [
            self.getLocalDirectionAsMove(constants.LocalDirection.STRAIGHT, snakeHead, snakeNeck),
            self.getLocalDirectionAsMove(constants.LocalDirection.LEFT, snakeHead, snakeNeck),
            self.getLocalDirectionAsMove(constants.LocalDirection.RIGHT, snakeHead, snakeNeck)
        ]

        next_moves_coordinates = []

        # determine the coordinates for each absolute move direction
        for move in next_moves_absolute:
            if (move == 'up'):
                next_moves_coordinates.append((snakeHead[0], snakeHead[1] - 1))
            elif (move == 'down'):
                next_moves_coordinates.append((snakeHead[0], snakeHead[1] + 1))
            elif (move == 'left'):
                next_moves_coordinates.append((snakeHead[0] - 1, snakeHead[1]))
            else:
                next_moves_coordinates.append((snakeHead[0] + 1, snakeHead[1]))

        snakes_together = self.get_snakes_together(data['board']['snakes'], False)
        
        # for each move, check if the move is out of bounds or collides with a snake body
        
        for idx in range(len(next_moves_coordinates)):
            next_move = next_moves_coordinates[idx]

            # dont hit walls
            if (next_move[0] < 0 or next_move[0] >= board_width or next_move[1] < 0 or next_move[1] >= board_height):
                q_values[idx] = MIN_Q_VALUE
            elif (next_move in snakes_together):
                # dont hit snake bodies
                q_values[idx] = MIN_Q_VALUE            
            elif (you_health <= 1):
                # if health is at 1 and next move is not a food, then don't go there
                if (next_move not in board_food):
                    q_values[idx] = MIN_Q_VALUE            

        local_dir = q_values.index(max(q_values))

        return local_dir, q_values