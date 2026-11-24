"""Shot outcomes for 8-ball and 9-ball against the ghost."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pooltool as pt
from pooltool.ruleset.eight_ball import BallGroup

from pool_shot import Shot, Turn


_CUSHION_EVENTS = (pt.EventType.BALL_LINEAR_CUSHION, pt.EventType.BALL_CIRCULAR_CUSHION)


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
        available = (
            ball_id
            for ball_id, ball in table.balls.items()
            if ball_id != "cue" and ball.state.s in pt.constants.on_table
        )
        lowest = min(available, key=int, default=None)
        return Turn(is_break, (lowest,) if lowest is not None else ())

    def judge_shot(
        self, table: pt.System, turn: Turn, shot: Shot, pocketed: tuple[str, ...]
    ) -> ShotResult:
        fouls = _nine_ball_fouls(table, turn.hittable_balls[0])
        if fouls:
            return ShotResult(Outcome.GHOST_WINS, ", ".join(fouls))
        if "9" in pocketed:
            return ShotResult(Outcome.PLAYER_WINS, "Legally pocketed the 9-ball")
        if pocketed:
            return ShotResult(Outcome.CONTINUE, "Pocketed an object ball")
        return ShotResult(Outcome.GHOST_WINS, "Missed all object balls")


def _nine_ball_fouls(table: pt.System, lowest_ball_id: str) -> list[str]:
    reasons = []
    if pt.ruleset.is_ball_pocketed(table, "cue"):
        reasons.append("Cue ball scratch")

    first_hit = pt.ruleset.get_id_of_first_ball_hit(table)
    if first_hit is None:
        reasons.append("No object ball contacted")
        return reasons
    if first_hit != lowest_ball_id:
        reasons.append(f"Wrong first contact: {first_hit}, expected {lowest_ball_id}")

    first_contact = next(
        (
            event.time
            for event in table.events
            if event.event_type == pt.EventType.BALL_BALL and "cue" in event.ids
        ),
        None,
    )
    if first_contact is not None and not _has_rail_or_pot_after_contact(table, first_contact):
        reasons.append("No rail contact or object ball pocketed")
    return reasons


def _has_rail_or_pot_after_contact(table: pt.System, contact_time: float) -> bool:
    rail_after = any(
        event.event_type in _CUSHION_EVENTS and event.time > contact_time
        for event in table.events
    )
    object_pocketed = any(
        event.event_type == pt.EventType.BALL_POCKET
        and event.time >= contact_time
        and event.ids[0] != "cue"
        for event in table.events
    )
    return rail_after or object_pocketed


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
            return _judge_eight_ball_break(pocketed)
        return self._judge_called_shot(table, turn, shot, pocketed)

    def _judge_called_shot(
        self, table: pt.System, turn: Turn, shot: Shot, pocketed: tuple[str, ...]
    ) -> ShotResult:
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
                BallGroup.SOLIDS
                if called_ball in BallGroup.SOLIDS.balls
                else BallGroup.STRIPES
            )
        return ShotResult(Outcome.CONTINUE, "Made the called shot")


def _judge_eight_ball_break(pocketed: tuple[str, ...]) -> ShotResult:
    if "8" in pocketed:
        return ShotResult(Outcome.PLAYER_WINS, "Pocketed the 8-ball on the break")
    if pocketed:
        return ShotResult(Outcome.CONTINUE, "Pocketed an object ball on the break")
    return ShotResult(Outcome.GHOST_WINS, "Dry break")
