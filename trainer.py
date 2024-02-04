from arena import ArenaParameters, Arena
from snake import Snake

from observer import Observer
from controller import Controller

import constants

def main():
    # TODO pull out even more into a config file
    food_spawn_chance = 0.15

    board_width = 11
    board_height = 11

    parameters = ArenaParameters((board_width, board_height), food_spawn_chance, constants.DEFAULT_MIN_FOOD)

    observer = Observer()

    snakes = [
        Snake("Snake1", constants.COLORS["red"], Controller())
    ]

    arena = Arena(parameters, snakes)

    is_done = arena.reset()    

    while not is_done:
        observer.print_arena(arena)

        is_done = arena.step()

        observer.print_arena(arena)

if __name__ == "__main__":
    main()

    