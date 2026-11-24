"""Strategies for choosing a shot on the current turn."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import cache
from itertools import chain
from typing import TYPE_CHECKING, Protocol

from pool_shot import Shot as Shot, Turn as Turn
from shot_search import (
    Pot,
    PotRoute,
    ShotGrid,
    grid_makes,
    legal_aim_angles,
    pocket_aim_margin,
    simulate_make,
)

if TYPE_CHECKING:
    import pooltool as pt


class Player(Protocol):
    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        """Choose a shot using the current table as read-only input."""
        ...


class RandomPlayer:
    """Choose an arbitrary aim and speed, and call a pot when required."""

    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        shot = Shot(random.uniform(0, 360), random.uniform(1, 5))
        if not turn.call_required:
            return shot
        return replace(
            shot,
            called_ball=random.choice(turn.hittable_balls),
            called_pocket=random.choice(tuple(table.table.pockets)),
        )


class AlwaysMake:
    """Take the first make from randomized aims and speeds."""

    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        shot = self._first_make(table, turn)
        if shot is None:
            return RandomPlayer().take_turn(table, turn)
        return shot

    def _first_make(self, table: pt.System, turn: Turn) -> Shot | None:
        for shot in self._randomized_shots(table, turn):
            pots = simulate_make(table, turn, shot)
            if pots:
                return pots[0].call(shot, turn)
        return None

    def _randomized_shots(self, table: pt.System, turn: Turn) -> Iterator[Shot]:
        for angle in legal_aim_angles(table, turn):
            # Try a random speed from each part of the 1 to 5 m/s range.
            powers = [random.uniform(low, low + 1) for low in range(1, 5)]
            random.shuffle(powers)
            for power in powers:
                yield Shot(angle, power)


class DiscreteMake:
    """Choose uniformly among successful angle and speed pairs on a fixed grid."""

    def __init__(self, grid: ShotGrid = ShotGrid()) -> None:
        self.grid = grid

    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        makes = [
            pots[0].call(shot, turn) for shot, pots in grid_makes(table, turn, self.grid)
        ]
        if not makes:
            return RandomPlayer().take_turn(table, turn)
        return random.choice(makes)


@dataclass(frozen=True)
class DifficultySettings:
    """Tune the replay budget, execution errors, and difficulty penalties."""

    finalist_count: int = 5
    aim_error_degrees: float = 0.1
    speed_error_fraction: float = 0.05
    pocket_opening_fraction: float = 0.9
    aim_margin_weight: float = 20.0
    nearby_speed_bonus: float = 5.0
    bank_penalty: float = 10.0
    indirect_penalty: float = 20.0

    def __post_init__(self) -> None:
        if self.finalist_count < 1:
            raise ValueError("At least one finalist is required.")
        if not 0 < self.aim_error_degrees < 180:
            raise ValueError("Aim error must be between 0 and 180 degrees.")
        if not 0 < self.speed_error_fraction < 1:
            raise ValueError("Speed error must be between 0 and 1.")
        if not 0 < self.pocket_opening_fraction <= 1:
            raise ValueError("Pocket opening must be greater than 0 and at most 1.")
        weights = (
            self.aim_margin_weight,
            self.nearby_speed_bonus,
            self.bank_penalty,
            self.indirect_penalty,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Difficulty weights must be finite and nonnegative.")

    def route_penalty(self, route: PotRoute) -> float:
        if route is PotRoute.DIRECT:
            return 0.0
        if route is PotRoute.BANK:
            return self.bank_penalty
        return self.indirect_penalty


@dataclass(frozen=True)
class _Candidate:
    shot: Shot
    pot: Pot
    proxy_score: float


class LeastDifficult:
    """Choose a make that holds up to small errors in the player's execution."""

    def __init__(
        self, grid: ShotGrid = ShotGrid(), settings: DifficultySettings = DifficultySettings()
    ) -> None:
        self.grid = grid
        self.settings = settings

    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        if turn.is_break:
            return AlwaysMake().take_turn(table, turn)

        ranked = self._rank_candidates(table, turn)
        if not ranked:
            return RandomPlayer().take_turn(table, turn)

        winner = self._best_finalist(table, turn, ranked)
        return winner.pot.call(winner.shot, turn)

    def _rank_candidates(self, table: pt.System, turn: Turn) -> list[_Candidate]:
        makes = dict(grid_makes(table, turn, self.grid))
        powers = self.grid.powers
        neighboring_powers = {
            power: powers[max(0, index - 1) : index + 2]
            for index, power in enumerate(powers)
        }

        @cache
        def aim_margin(angle: float, pot: Pot) -> float:
            return self._aim_margin(table, pot, angle)

        candidates = []
        for shot, pots in makes.items():
            for pot in pots:
                margin = aim_margin(shot.angle, pot)
                nearby_makes = _count_neighbor_makes(
                    shot, pot, makes, neighboring_powers[shot.power]
                )
                proxy = self._proxy_score(pot, margin, nearby_makes)
                candidates.append(_Candidate(shot, pot, proxy))
        return sorted(candidates, key=lambda candidate: candidate.proxy_score)

    def _aim_margin(self, table: pt.System, pot: Pot, angle: float) -> float:
        if pot.route is not PotRoute.DIRECT:
            return 0.0
        return pocket_aim_margin(
            table.balls[table.cue.cue_ball_id],
            table.balls[pot.ball_id],
            table.table.pockets[pot.pocket_id],
            angle,
            self.settings.pocket_opening_fraction,
        )

    def _proxy_score(self, pot: Pot, aim_margin: float, nearby_makes: int) -> float:
        aim_bonus = self.settings.aim_margin_weight * (
            aim_margin / (aim_margin + self.settings.aim_error_degrees)
        )
        return (
            self.settings.route_penalty(pot.route)
            - aim_bonus
            - self.settings.nearby_speed_bonus * nearby_makes
        )

    def _best_finalist(
        self, table: pt.System, turn: Turn, ranked: list[_Candidate]
    ) -> _Candidate:
        @cache
        def replay(shot: Shot) -> tuple[Pot, ...]:
            return simulate_make(table, turn, shot)

        finalists = _finalists(ranked, self.settings.finalist_count)
        return min(finalists, key=lambda candidate: self._replay_score(candidate, replay))

    def _replay_score(
        self, candidate: _Candidate, replay: Callable[[Shot], tuple[Pot, ...]]
    ) -> tuple[float, float]:
        probes = self._execution_probes(candidate.shot)
        intended = candidate.pot.ball_id, candidate.pot.pocket_id
        hits = sum(
            any((pot.ball_id, pot.pocket_id) == intended for pot in replay(probe))
            for probe in probes
        )
        error_cost = 100 * (len(probes) - hits) / len(probes)
        difficulty = error_cost + self.settings.route_penalty(candidate.pot.route)
        return difficulty, candidate.proxy_score

    def _execution_probes(self, shot: Shot) -> tuple[Shot, ...]:
        degrees = self.settings.aim_error_degrees
        fraction = self.settings.speed_error_fraction
        return (
            replace(shot, angle=(shot.angle - degrees) % 360),
            replace(shot, angle=(shot.angle + degrees) % 360),
            replace(shot, power=shot.power * (1 - fraction)),
            replace(shot, power=shot.power * (1 + fraction)),
        )


def _count_neighbor_makes(
    shot: Shot, pot: Pot, makes: dict[Shot, tuple[Pot, ...]], powers: tuple[float, ...]
) -> int:
    return sum(
        pot in makes.get(replace(shot, power=power), ())
        for power in powers
        if power != shot.power
    )


def _finalists(ranked: list[_Candidate], count: int) -> list[_Candidate]:
    # Give an indirect shot a chance, then spread the remaining probes across pots.
    best_by_style: dict[bool, _Candidate] = {}
    best_by_pot: dict[Pot, _Candidate] = {}
    for candidate in ranked:
        best_by_style.setdefault(candidate.pot.route is PotRoute.DIRECT, candidate)
        best_by_pot.setdefault(candidate.pot, candidate)
    distinct = dict.fromkeys(chain(best_by_style.values(), best_by_pot.values(), ranked))
    return list(distinct)[:count]
