"""Generate center-ball shots and inspect what their simulations pocketed."""

import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from enum import Enum, auto

import numpy as np
import pooltool as pt
from pooltool.events import Event
from pooltool.objects.table.components import Pocket

from pool_shot import Shot, Turn


_CUSHION_EVENTS = (
    pt.EventType.BALL_LINEAR_CUSHION,
    pt.EventType.BALL_CIRCULAR_CUSHION,
)
_POCKET_OFFSETS = (0.0, -0.6, -0.3, 0.3, 0.6)


@dataclass(frozen=True)
class ShotGrid:
    """Aim offsets are fractions of the contact cone or pocket radius."""

    contact_offsets: tuple[float, ...] = tuple(step / 10 for step in range(-10, 11))
    pocket_offsets: tuple[float, ...] = (-0.9, *_POCKET_OFFSETS, 0.9)
    powers: tuple[float, ...] = tuple(step / 2 for step in range(2, 11))

    def __post_init__(self) -> None:
        if not self.powers or any(
            not math.isfinite(power) or power <= 0 for power in self.powers
        ):
            raise ValueError("The grid needs positive, finite cue speeds.")
        if any(left >= right for left, right in zip(self.powers, self.powers[1:])):
            raise ValueError("Grid cue speeds must be in strictly increasing order.")
        offsets = self.contact_offsets + self.pocket_offsets
        if any(not -1 <= offset <= 1 for offset in offsets):
            raise ValueError("Grid aim offsets must be between -1 and 1.")


class PotRoute(Enum):
    DIRECT = auto()
    BANK = auto()
    COMBINATION = auto()
    CAROM = auto()


@dataclass(frozen=True)
class Pot:
    ball_id: str
    pocket_id: str
    route: PotRoute

    def call(self, shot: Shot, turn: Turn) -> Shot:
        if turn.call_required:
            return replace(shot, called_ball=self.ball_id, called_pocket=self.pocket_id)
        return shot


def grid_makes(
    table: pt.System, turn: Turn, grid: ShotGrid
) -> Iterator[tuple[Shot, tuple[Pot, ...]]]:
    # Several targets or pockets can generate the same angle.
    for angle in dict.fromkeys(legal_aim_angles(table, turn, grid)):
        for power in grid.powers:
            shot = Shot(angle, power)
            pots = simulate_make(table, turn, shot)
            if pots:
                yield shot, pots


def legal_aim_angles(table: pt.System, turn: Turn, grid: ShotGrid | None = None) -> list[float]:
    """Return cue angles in degrees, with pocket aims before contact-cone samples."""
    cue = table.balls[table.cue.cue_ball_id]
    object_balls = [
        ball
        for ball in table.balls.values()
        if ball.id != cue.id and ball.state.s in pt.constants.on_table
    ]
    targets = [ball for ball in object_balls if ball.id in turn.hittable_balls]
    pockets = tuple(table.table.pockets.values())
    if not targets or not pockets:
        return []

    pocket_angles = []
    contact_angles = []
    pocket_offsets = None if grid is None else grid.pocket_offsets
    for target in targets:
        cue_to_target = target.xyz[:2] - cue.xyz[:2]
        center_angle = math.atan2(cue_to_target[1], cue_to_target[0])
        contact_radius = cue.params.R + target.params.R
        distance = max(contact_radius, math.hypot(*cue_to_target))
        half_width = math.asin(contact_radius / distance)
        if grid is None:
            contact_offsets = (0.0,) + tuple(
                random.uniform(low, low + 0.1) for low in np.linspace(-1, 0.9, 20)
            )
        else:
            contact_offsets = grid.contact_offsets
        contact_angles.extend(center_angle + half_width * offset for offset in contact_offsets)

        for pocket in pockets:
            pocket_angles.extend(
                angle
                for angle in _pocket_aims(cue, target, pocket, pocket_offsets)
                if angle is not None
            )

    if grid is None:
        random.shuffle(pocket_angles)
        random.shuffle(contact_angles)
    return _angles_with_legal_first_contact(cue, object_balls, turn, pocket_angles + contact_angles)


def pocket_aim_margin(
    cue: pt.Ball, target: pt.Ball, pocket: Pocket, angle: float, opening_fraction: float
) -> float:
    """Estimate the smaller angular miss the pocket can absorb, in degrees."""
    offsets = (-opening_fraction, 0, opening_fraction)
    left, center, right = _pocket_aims(cue, target, pocket, offsets)
    if left is None or center is None or right is None:
        return 0.0

    low, high = sorted(
        math.degrees(math.remainder(edge - center, math.tau)) for edge in (left, right)
    )
    aim = math.remainder(angle - math.degrees(center), 360)
    return max(0.0, min(aim - low, high - aim))


def pocket_aim_options(
    cue: pt.Ball, target: pt.Ball, pocket: Pocket, offsets: tuple[float, ...]
) -> Iterator[tuple[float, float]]:
    """Yield pocket offsets and direct-pot aims in degrees from table geometry."""
    for offset, angle in zip(offsets, _pocket_aims(cue, target, pocket, offsets), strict=True):
        if angle is not None:
            yield offset, math.degrees(angle) % 360


def _pocket_aims(
    cue: pt.Ball, target: pt.Ball, pocket: Pocket, offsets: tuple[float, ...] | None
) -> Iterator[float | None]:
    """Find the cue-ball center at contact for each point across a pocket."""
    to_pocket = pocket.center[:2] - target.xyz[:2]
    distance = math.hypot(*to_pocket)
    if distance == 0:
        if offsets is not None:
            yield from (None for _ in offsets)
        return

    if offsets is None:
        offsets = (*_POCKET_OFFSETS, random.uniform(-0.9, 0.9))
    perpendicular = np.array([-to_pocket[1], to_pocket[0]]) / distance
    cue_to_target = target.xyz[:2] - cue.xyz[:2]
    contact_radius = cue.params.R + target.params.R
    for offset in offsets:
        pot_direction = to_pocket + offset * pocket.radius * perpendicular
        to_contact = cue_to_target - contact_radius * pot_direction / math.hypot(*pot_direction)
        if to_contact @ pot_direction > 0:
            yield math.atan2(to_contact[1], to_contact[0])
        else:
            yield None


def _angles_with_legal_first_contact(
    cue: pt.Ball, balls: list[pt.Ball], turn: Turn, angles: list[float]
) -> list[float]:
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    offsets = np.array([ball.xyz[:2] - cue.xyz[:2] for ball in balls])
    radii = np.array([cue.params.R + ball.params.R for ball in balls])
    forward = directions @ offsets.T
    sideways = (
        directions[:, 0, None] * offsets[:, 1] - directions[:, 1, None] * offsets[:, 0]
    )
    # Intersect each ray with circles expanded by the cue ball's radius.
    discriminant = radii**2 - sideways**2
    distances = np.where(
        (forward > 0) & (discriminant >= 0),
        forward - np.sqrt(np.maximum(discriminant, 0)),
        np.inf,
    )
    return [
        math.degrees(angle) % 360
        for angle, row in zip(angles, distances)
        if np.isfinite(row.min()) and balls[row.argmin()].id in turn.hittable_balls
    ]


def simulate_make(table: pt.System, turn: Turn, shot: Shot) -> tuple[Pot, ...]:
    """Return the eligible pots from a direct-first-contact simulation."""
    trial = table.copy()
    shot.apply_to(trial.cue)
    pt.simulate(trial, inplace=True)
    pots = _pots_from_events(trial.events, trial.cue.cue_ball_id, turn)
    if not turn.call_required:
        return pots
    if "8" not in turn.hittable_balls and any(pot.ball_id == "8" for pot in pots):
        return ()
    return tuple(pot for pot in pots if pot.ball_id in turn.hittable_balls)


def _pots_from_events(events: Iterable[Event], cue_id: str, turn: Turn) -> tuple[Pot, ...]:
    legal_contact = False
    routes: dict[str, PotRoute] = {}
    cushioned_balls = set()
    pots = []
    for event in events:
        if event.event_type == pt.EventType.BALL_BALL:
            first, second = event.ids
            if cue_id in (first, second):
                target = second if first == cue_id else first
                if not legal_contact:
                    if target not in turn.hittable_balls:
                        return ()
                    legal_contact = True
                    routes[target] = PotRoute.DIRECT
                else:
                    routes[target] = PotRoute.CAROM
            else:
                # A second ball may be driven in, or the first ball may kiss off it.
                for ball_id in (first, second):
                    if routes.get(ball_id) is PotRoute.DIRECT:
                        routes[ball_id] = PotRoute.CAROM
                    else:
                        routes.setdefault(ball_id, PotRoute.COMBINATION)
        elif event.event_type in _CUSHION_EVENTS:
            ball_id, _ = event.ids
            if ball_id == cue_id and not legal_contact:
                return ()
            cushioned_balls.add(ball_id)
        elif event.event_type == pt.EventType.BALL_POCKET:
            ball_id, pocket_id = event.ids
            if ball_id == cue_id:
                return ()
            route = routes.get(ball_id, PotRoute.COMBINATION)
            if route is PotRoute.DIRECT and ball_id in cushioned_balls:
                route = PotRoute.BANK
            pots.append(Pot(ball_id, pocket_id, route))

    return tuple(pots) if legal_contact else ()
