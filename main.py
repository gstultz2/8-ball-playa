import argparse

import pooltool as pt

from pool_simulator import GhostSimulator, GameResult
from random_pool import RandomPlayer


GAME_TYPES = {"8-ball": pt.GameType.EIGHTBALL, "9-ball": pt.GameType.NINEBALL}

def printGameResults(game: GameResult) -> None:
    print(f"{game.game_type} against the ghost")
    for number, played in enumerate(game.shots, start=1):
        shot = played.shot
        targets = ", ".join(played.turn.hittable_balls)
        pocketed = ", ".join(played.pocketed) or "none"
        print(f"\nShot {number}")
        print(f"Targets: {targets}")
        print(f"Angle: {shot.angle:.1f}, power: {shot.power:.1f}")
        if played.turn.call_required:
            print(f"Called: ball {shot.called_ball} in pocket {shot.called_pocket}")
        print(f"Pocketed: {pocketed}")
        print(played.result.reason)
    print(f"\nWinner: {game.winner}")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=GAME_TYPES, default="9-ball")
    args = parser.parse_args()

    printGameResults(GhostSimulator(GAME_TYPES[args.game]).run(RandomPlayer()))


if __name__ == "__main__":
    main()
