import random

import pooltool as pt

from pool_player import Shot, Turn


class RandomPlayer:
    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        angle = random.uniform(0, 360)
        power = random.uniform(1, 5)
        if not turn.call_required:
            return Shot(angle, power)
        return Shot(
            angle,
            power,
            called_ball=random.choice(turn.hittable_balls),
            called_pocket=random.choice(tuple(table.table.pockets)),
        )

