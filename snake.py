import constants

class Snake():
    def __init__(self, name) -> None:
        self.name = name        
        self.reset()
        self.color = constants.COLORS["red"]

    def reset(self) -> None:
        self.body = []

    def place_at_spot(self, spot) -> None:
        if (len(self.body) > 0):
            raise ValueError("Snake already placed")
        
        self.body.append(spot)