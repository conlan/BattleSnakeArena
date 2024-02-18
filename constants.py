from enum import IntEnum

MAX_SNAKE_HEALTH = 100

SNAKE_START_SIZE = 3

DEFAULT_MIN_FOOD = 1
DEFAULT_FOOD_SPAN_CHANCE = 0.15

BOARD_SIZE_MEDIUM = (11, 11)

COLORS = {
    "black": "\033[1;37;40m",
    "red": "\033[1;37;41m",
    "green": "\033[1;37;42m",
    "yellow": "\033[1;37;43m",
    "blue": "\033[1;37;44m",
    "purple": "\033[1;37;45m",
    "cyan": "\033[1;37;46m",
    "grey": "\033[1;37;47m",
    "default": "\033[0m"
    }

REWARD_KEY_SURVIVE = "survive"
REWARD_KEY_EAT = "eat"
REWARD_KEY_WIN = "win"
REWARD_KEY_LOSE = "lose"

REWARD_SETS = {
    "reward-set-v1" : {
        REWARD_KEY_LOSE : -1000,
        REWARD_KEY_WIN : 1000,
        REWARD_KEY_SURVIVE : 1,
        REWARD_KEY_EAT : 10
    },
    "reward-set-v2" : {
        REWARD_KEY_LOSE : -500,
        REWARD_KEY_WIN : 500,
        REWARD_KEY_SURVIVE : 1,
        REWARD_KEY_EAT : 25
    },
    "reward-set-v3" : {
        REWARD_KEY_LOSE : -1.0,
        REWARD_KEY_WIN : 1.0,
        REWARD_KEY_SURVIVE : 0.01,
        REWARD_KEY_EAT : 0.025
    }
}

DEFAULT_REWARD_SET_KEY = "reward-set-v3"
DEFAULT_LEARNING_RATE = 0.00025

DEFAULT_GAME_CONFIG = {
    "food_spawn_chance" : DEFAULT_FOOD_SPAN_CHANCE,
    "min_food" : DEFAULT_MIN_FOOD,
    "board_size" : BOARD_SIZE_MEDIUM
}

EPSILON_INFO_ALWAYS_GREEDY = {
    "epsilon" : 0,
    "epsilon_decay" : 0,
    "epsilon_min" : 0
}

EPSILON_INFO_VALIDATION = {
    "epsilon" : 0.05,
    "epsilon_decay" : 0,
    "epsilon_min" : 0.05
}

class LocalDirection(IntEnum):
    STRAIGHT = 0,
    LEFT = 1,
    RIGHT = 2