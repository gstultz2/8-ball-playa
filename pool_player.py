"""Inputs and decisions for a pool player taking one shot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pooltool as pt


@dataclass(frozen=True)
class Turn:
    is_break: bool
    hittable_balls: tuple[str, ...]
    call_required: bool = False


@dataclass(frozen=True)
class Shot:
    angle: float
    power: float
    elevation: float = 0.0
    side_spin: float = 0.0
    vertical_spin: float = 0.0
    called_ball: str | None = None
    called_pocket: str | None = None


class Player(Protocol):
    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        """Choose a shot using the current table as read-only input."""
        ...
