# import test_snakes.battleJake2020.main
import test_snakes.battleJake2019.main
import test_snakes.battleJake2018.main
import test_snakes.simpleJake.main
import test_snakes.hungryJake.main
import conlan_snakes.DQNConlan2024.main

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

"""
{
 "move": The function that responds to the /move request,
 "name": The snakes name, must be unique,
 "color": A color from the list of colors
 }
"""

SNAKES = [
    {
        "name": "DQNConlan2024",
        "start":conlan_snakes.DQNConlan2024.main.start,
        "move": conlan_snakes.DQNConlan2024.main.move,
        "cache" : conlan_snakes.DQNConlan2024.main.cache,        
        "training_reward_index" : 1,
        "model_save_path" : "content/drive/MyDrive/ColabOutput/snake_net.chkpt",
        "color": COLORS["red"]
    },
    {
        "move": test_snakes.battleJake2019.main.move,
        "name": "battleJake2019",
        "color": COLORS["purple"]
    },
    {
        "move": test_snakes.battleJake2018.main.move,
        "name": "battleJake2018",
        "color": COLORS["cyan"]
    },
    {
        "move": test_snakes.simpleJake.main.move,
        "name": "simpleJake",
        "color": COLORS["blue"]
    },
    {
        "move": test_snakes.hungryJake.main.move,
        "name": "hungryJake",
        "color": COLORS["yellow"]
    }
]
