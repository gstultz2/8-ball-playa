"""Run a single rack against the ghost and retain its shot history."""

from dataclasses import dataclass

import pooltool as pt

from pool_player import Player
from pool_rules import EightBallRules, GameRules, NineBallRules, Outcome, ShotResult
from pool_shot import Shot, Turn


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
            played = _play_shot(system, player, rules, is_break=not played_shots)
            played_shots.append(played)

            if played.result.outcome is not Outcome.CONTINUE:
                return GameResult(self.game_type, tuple(played_shots))

            system = system.copy()
            system.reset_history()


def _play_shot(
    table: pt.System, player: Player, rules: GameRules, *, is_break: bool
) -> PlayedShot:
    turn = rules.prepare_turn(table, is_break=is_break)
    shot = player.take_turn(table, turn)
    shot.apply_to(table.cue)
    pt.simulate(table, inplace=True)
    pocketed = tuple(pt.ruleset.get_pocketed_ball_ids_during_shot(table))
    result = rules.judge_shot(table, turn, shot, pocketed)
    return PlayedShot(turn, shot, pocketed, result, table)
