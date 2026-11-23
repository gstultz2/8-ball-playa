"""Shot outcomes for 8-ball and 9-ball against the ghost."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pooltool as pt
from pooltool.ruleset.eight_ball import BallGroup

from nine_ball import check_nineball_foul, get_lowest_ball_on_table
from pool_player import Shot, Turn


class Outcome(Enum):
    CONTINUE = "continue"
    PLAYER_WINS = "player"
    GHOST_WINS = "ghost"


@dataclass(frozen=True)
class ShotResult:
    outcome: Outcome
    reason: str


class GameRules(Protocol):
    def prepare_turn(self, table: pt.System, *, is_break: bool) -> Turn: ...

    def judge_shot(
        self, table: pt.System, turn: Turn, shot: Shot, pocketed: tuple[str, ...]
    ) -> ShotResult: ...


class NineBallRules:
    def prepare_turn(self, table: pt.System, *, is_break: bool) -> Turn:
        lowest = get_lowest_ball_on_table(table)
        return Turn(is_break, (lowest,) if lowest is not None else ())

    def judge_shot(
        self, table: pt.System, turn: Turn, shot: Shot, pocketed: tuple[str, ...]
    ) -> ShotResult:
        lowest = turn.hittable_balls[0]
        foul, reasons = check_nineball_foul(table, lowest)
        if foul:
            return ShotResult(Outcome.GHOST_WINS, ", ".join(reasons))
        if "9" in pocketed:
            return ShotResult(Outcome.PLAYER_WINS, "Legally pocketed the 9-ball")
        if pocketed:
            return ShotResult(Outcome.CONTINUE, "Pocketed an object ball")
        return ShotResult(Outcome.GHOST_WINS, "Missed all object balls")


class EightBallRules:
    """Called-shot 8-ball with a win for the 8-ball on a clean break."""

    def __init__(self) -> None:
        self.group = BallGroup.UNDECIDED

    def prepare_turn(self, table: pt.System, *, is_break: bool) -> Turn:
        targets = tuple(
            ball_id
            for ball_id in self.group.balls
            if table.balls[ball_id].state.s in pt.constants.on_table
        )
        return Turn(is_break, targets or ("8",), call_required=not is_break)

    def judge_shot(
        self, table: pt.System, turn: Turn, shot: Shot, pocketed: tuple[str, ...]
    ) -> ShotResult:
        if "cue" in pocketed:
            return ShotResult(Outcome.GHOST_WINS, "Cue ball scratch")

        first_hit = pt.ruleset.get_id_of_first_ball_hit(table)
        if first_hit is None:
            return ShotResult(Outcome.GHOST_WINS, "No object ball contacted")
        if first_hit not in turn.hittable_balls:
            return ShotResult(Outcome.GHOST_WINS, f"Wrong first contact: {first_hit}")

        if turn.is_break:
            if "8" in pocketed:
                return ShotResult(
                    Outcome.PLAYER_WINS, "Pocketed the 8-ball on the break"
                )
            if pocketed:
                return ShotResult(
                    Outcome.CONTINUE, "Pocketed an object ball on the break"
                )
            return ShotResult(Outcome.GHOST_WINS, "Dry break")

        called_ball = shot.called_ball
        called_pocket = shot.called_pocket
        made_call = (
            called_ball is not None
            and called_ball in turn.hittable_balls
            and called_pocket is not None
            and pt.ruleset.is_ball_pocketed_in_pocket(table, called_ball, called_pocket)
        )

        if "8" in pocketed:
            if called_ball == "8" and made_call:
                return ShotResult(Outcome.PLAYER_WINS, "Pocketed the 8-ball as called")
            return ShotResult(Outcome.GHOST_WINS, "Pocketed the 8-ball illegally")
        if not made_call:
            return ShotResult(Outcome.GHOST_WINS, "Missed the called shot")

        if self.group is BallGroup.UNDECIDED:
            self.group = (
                BallGroup.SOLIDS if called_ball in BallGroup.SOLIDS.balls else BallGroup.STRIPES
            )
        return ShotResult(Outcome.CONTINUE, "Made the called shot")
