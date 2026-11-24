"""Players that choose a successful shot from Pooltool simulations."""

import random

import pooltool as pt

from pool_player import Shot, Turn
from random_pool import RandomPlayer
from shot_search import ShotGrid, grid_makes, legal_aim_angles, simulate_make


class AlwaysMake:
    """Take the first make from randomized aims and speeds."""

    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        for angle in legal_aim_angles(table, turn):
            # Try a random speed from each part of the 1 to 5 m/s range.
            powers = [random.uniform(low, low + 1) for low in range(1, 5)]
            random.shuffle(powers)
            for power in powers:
                shot = Shot(angle, power)
                pots = simulate_make(table, turn, shot)
                if pots:
                    return pots[0].call(shot, turn)

        return RandomPlayer().take_turn(table, turn)


class DiscreteMake:
    """Choose uniformly among successful angle and speed pairs on a fixed grid."""

    def __init__(self, grid: ShotGrid = ShotGrid()) -> None:
        self.grid = grid

    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        makes = [pots[0].call(shot, turn) for shot, pots in grid_makes(table, turn, self.grid)]
        if makes:
            return random.choice(makes)
        return RandomPlayer().take_turn(table, turn)
