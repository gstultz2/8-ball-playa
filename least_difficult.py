"""Estimate difficulty cheaply, then replay a small set of promising makes."""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from itertools import chain

import pooltool as pt

from always_make import AlwaysMake
from pool_player import Shot, Turn
from random_pool import RandomPlayer
from shot_search import Pot, PotRoute, ShotGrid, grid_makes, pocket_aim_margin, simulate_make


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

        candidates = self._rank_candidates(table, turn)
        if not candidates:
            return RandomPlayer().take_turn(table, turn)

        @cache
        def replay(shot: Shot) -> tuple[Pot, ...]:
            return simulate_make(table, turn, shot)

        winner = min(
            _finalists(candidates, self.settings.finalist_count),
            key=lambda candidate: self._replay_score(candidate, replay),
        )
        return winner.pot.call(winner.shot, turn)

    def _rank_candidates(self, table: pt.System, turn: Turn) -> list[_Candidate]:
        makes = dict(grid_makes(table, turn, self.grid))
        powers = self.grid.powers
        neighbor_powers = {
            power: powers[max(0, index - 1):index + 2] for index, power in enumerate(powers)
        }
        aim_margins: dict[tuple[float, Pot], float] = {}
        candidates = []
        for shot, pots in makes.items():
            for pot in pots:
                margin_key = shot.angle, pot
                if margin_key not in aim_margins:
                    aim_margins[margin_key] = self._aim_margin(table, pot, shot.angle)
                margin = aim_margins[margin_key]
                speed_matches = sum(
                    pot in makes.get(replace(shot, power=power), ())
                    for power in neighbor_powers[shot.power]
                    if power != shot.power
                )
                aim_bonus = self.settings.aim_margin_weight * (
                    margin / (margin + self.settings.aim_error_degrees)
                )
                proxy = (
                    self.settings.route_penalty(pot.route)
                    - aim_bonus
                    - self.settings.nearby_speed_bonus * speed_matches
                )
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

    def _replay_score(
        self, candidate: _Candidate, replay: Callable[[Shot], tuple[Pot, ...]]
    ) -> tuple[float, float]:
        shot = candidate.shot
        degrees = self.settings.aim_error_degrees
        fraction = self.settings.speed_error_fraction
        probes = (
            replace(shot, angle=(shot.angle - degrees) % 360),
            replace(shot, angle=(shot.angle + degrees) % 360),
            replace(shot, power=shot.power * (1 - fraction)),
            replace(shot, power=shot.power * (1 + fraction)),
        )
        intended = candidate.pot.ball_id, candidate.pot.pocket_id
        hits = sum(
            any((pot.ball_id, pot.pocket_id) == intended for pot in replay(probe))
            for probe in probes
        )
        error_cost = 100 * (len(probes) - hits) / len(probes)
        difficulty = error_cost + self.settings.route_penalty(candidate.pot.route)
        return difficulty, candidate.proxy_score


def _finalists(ranked: list[_Candidate], count: int) -> list[_Candidate]:
    # Give an indirect shot a chance, then spread the remaining probes across pots.
    best_by_style: dict[bool, _Candidate] = {}
    best_by_pot: dict[Pot, _Candidate] = {}
    for candidate in ranked:
        best_by_style.setdefault(candidate.pot.route is PotRoute.DIRECT, candidate)
        best_by_pot.setdefault(candidate.pot, candidate)
    distinct = dict.fromkeys(chain(best_by_style.values(), best_by_pot.values(), ranked))
    return list(distinct)[:count]
