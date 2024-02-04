import constants

FOOD_COLOR = constants.COLORS["green"]
BORDER_COLOR = constants.COLORS["grey"]
DEFAULT_COLOR = constants.COLORS["default"]

class Observer():
    def print_arena(self, arena):
        width = arena.width()
        height = arena.height()

        ywall = " " * 2 * width + "  "

        print(f"{BORDER_COLOR}{ywall}{DEFAULT_COLOR}") # Y Border
        
        for j in range(height):
            print(f"{BORDER_COLOR} {DEFAULT_COLOR}", end="") # X Border
            for i in range(width):
                if (i, j) in arena.food:
                    print(f"{FOOD_COLOR}  {DEFAULT_COLOR}", end="") # Food
                else:
                    no_snake = True
                    for ind, s in enumerate(arena.live_snakes):                        
                        if (i, j) in s.body:
                            if s.body[0] == (i, j):
                                print(f"{arena.live_snakes[ind].color}OO{DEFAULT_COLOR}", end="") # Head
                            else:
                                print(f"{arena.live_snakes[ind].color}  {DEFAULT_COLOR}", end="") # Body
                            no_snake = False
                    if no_snake:
                        print(f"{DEFAULT_COLOR}  {DEFAULT_COLOR}", end="") # Empty
        
            print(f"{BORDER_COLOR} {DEFAULT_COLOR}") # X Border
        
        print(f"{BORDER_COLOR}{ywall}{DEFAULT_COLOR}") # Y Border