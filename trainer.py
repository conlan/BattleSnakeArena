from arena import ArenaParameters, Arena
from snake import Snake

from observer import Observer

def main():
    # TODO pull out even more into a config file
    food_spawn_chance = 0.15

    board_width = 11
    board_height = 11

    parameters = ArenaParameters((board_width, board_height), food_spawn_chance)

    observer = Observer()

    snakes = [
        Snake("Snake1")
    ]

    arena = Arena(parameters, snakes)

    is_done = arena.reset()    

    while not is_done:
        observer.print_arena(arena)

        is_done = arena.step()
        print(is_done)

if __name__ == "__main__":
    main()

    