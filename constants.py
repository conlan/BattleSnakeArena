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

class LocalDirection(IntEnum):
    STRAIGHT = 0,
    LEFT = 1,
    RIGHT = 2