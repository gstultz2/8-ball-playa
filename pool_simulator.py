"""Run a single rack against the ghost and retain its shot history."""

from dataclasses import dataclass

import pooltool as pt

from pool_player import Player, Shot, Turn
from pool_rules import EightBallRules, GameRules, NineBallRules, Outcome, ShotResult


RULES_BY_GAME = {
    pt.GameType.EIGHTBALL: EightBallRules,
    pt.GameType.NINEBALL: NineBallRules,
}


@dataclass(frozen=True)
class PlayedShot:
    turn: Turn
    shot: Shot
    pocketed: tuple[str, ...]
    result: ShotResult
    table: pt.System


@dataclass(frozen=True)
class GameResult:
    game_type: pt.GameType
    shots: tuple[PlayedShot, ...]

    @property
    def winner(self) -> str:
        return self.shots[-1].result.outcome.value


class GhostSimulator:
    def __init__(self, game_type: pt.GameType) -> None:
        if game_type not in RULES_BY_GAME:
            raise ValueError(f"Unsupported game type: {game_type}")
        self.game_type = game_type

    def run(self, player: Player) -> GameResult:
        """Start a fresh rack and play until the player or ghost wins."""
        rules: GameRules = RULES_BY_GAME[self.game_type]()
        table = pt.Table.default()
        system = pt.System(
            table=table,
            balls=pt.get_rack(self.game_type, table),
            cue=pt.Cue(cue_ball_id="cue"),
        )

        played_shots: list[PlayedShot] = []
        while True:
            turn = rules.prepare_turn(system, is_break=not played_shots)
            shot = player.take_turn(system, turn)
            system.cue.set_state(
                V0=shot.power,
                phi=shot.angle,
                theta=shot.elevation,
                a=shot.side_spin,
                b=shot.vertical_spin,
            )
            pt.simulate(system, inplace=True)
            pocketed = tuple(pt.ruleset.get_pocketed_ball_ids_during_shot(system))
            result = rules.judge_shot(system, turn, shot, pocketed)
            played_shots.append(PlayedShot(turn, shot, pocketed, result, system))

            if result.outcome is not Outcome.CONTINUE:
                return GameResult(self.game_type, tuple(played_shots))

            system = system.copy()
            system.reset_history()
