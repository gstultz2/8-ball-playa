"""Choose a makeable shot by testing direct angles with Pooltool."""

import math
import random

import numpy as np
import pooltool as pt
from pooltool.objects.table.components import Pocket

from pool_player import Shot, Turn
from random_pool import RandomPlayer


CUSHION_EVENTS = (
    pt.EventType.BALL_LINEAR_CUSHION,
    pt.EventType.BALL_CIRCULAR_CUSHION,
)


class AlwaysMake:
    def take_turn(self, table: pt.System, turn: Turn) -> Shot:
        cue = table.balls[table.cue.cue_ball_id]
        object_balls = [
            ball
            for ball in table.balls.values()
            if ball.id != cue.id and ball.state.s in pt.constants.on_table
        ]
        targets = [ball for ball in object_balls if ball.id in turn.hittable_balls]
        if targets and table.table.pockets:
            angles = _candidate_angles(cue, targets, tuple(table.table.pockets.values()))
            for angle in _direct_angles(cue, object_balls, turn, angles):
                # Try a random speed from each part of the existing 1–5 m/s range.
                powers = [random.uniform(low, low + 1) for low in range(1, 5)]
                random.shuffle(powers)
                for power in powers:
                    shot = _try_shot(table, turn, angle, power)
                    if shot is not None:
                        return shot

        return RandomPlayer().take_turn(table, turn)


def _candidate_angles(
    cue: pt.Ball, targets: list[pt.Ball], pockets: tuple[Pocket, ...]
) -> list[float]:
    pocket_angles = []
    contact_angles = []
    for target in targets:
        cue_to_target = target.xyz[:2] - cue.xyz[:2]
        contact_radius = cue.params.R + target.params.R
        center_angle = math.atan2(cue_to_target[1], cue_to_target[0])
        distance = max(contact_radius, math.hypot(*cue_to_target))
        half_width = math.asin(contact_radius / distance)

        # Also sample the full contact cone for combinations and banks.
        contact_angles.append(center_angle)
        contact_angles.extend(
            center_angle + half_width * random.uniform(low, low + 0.1)
            for low in np.linspace(-1, 0.9, 20)
        )

        for pocket in pockets:
            to_pocket = pocket.center[:2] - target.xyz[:2]
            distance = math.hypot(*to_pocket)
            if distance == 0:
                continue
            perpendicular = np.array([-to_pocket[1], to_pocket[0]]) / distance
            for offset in (0, -0.6, -0.3, 0.3, 0.6, random.uniform(-0.9, 0.9)):
                pot_direction = to_pocket + offset * pocket.radius * perpendicular
                # Aim at the cue ball's center when it contacts the target.
                contact_offset = contact_radius * pot_direction / math.hypot(*pot_direction)
                to_contact = cue_to_target - contact_offset
                if to_contact @ pot_direction > 0:
                    pocket_angles.append(math.atan2(to_contact[1], to_contact[0]))

    random.shuffle(pocket_angles)
    random.shuffle(contact_angles)
    return pocket_angles + contact_angles


def _direct_angles(
    cue: pt.Ball, balls: list[pt.Ball], turn: Turn, angles: list[float]
) -> list[float]:
    """Keep directions whose first ball along the cue ray is a legal target."""
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    offsets = np.array([ball.xyz[:2] - cue.xyz[:2] for ball in balls])
    radii = np.array([cue.params.R + ball.params.R for ball in balls])

    forward = directions @ offsets.T
    sideways = (
        directions[:, 0, None] * offsets[:, 1] - directions[:, 1, None] * offsets[:, 0]
    )
    # Intersect the ray with circles expanded by the cue ball's radius.
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


def _try_shot(table: pt.System, turn: Turn, angle: float, power: float) -> Shot | None:
    trial = table.copy()
    trial.cue.set_state(V0=power, phi=angle, theta=0, a=0, b=0)
    pt.simulate(trial, inplace=True)

    cue_id = trial.cue.cue_ball_id
    for event in trial.events:
        if cue_id not in event.ids:
            continue
        if event.event_type in CUSHION_EVENTS:
            return None
        if event.event_type == pt.EventType.BALL_BALL:
            first_hit = next(ball_id for ball_id in event.ids if ball_id != cue_id)
            if first_hit not in turn.hittable_balls:
                return None
            break
    else:
        return None

    pocketed = {
        event.ids[0]: event.ids[1]
        for event in trial.events
        if event.event_type == pt.EventType.BALL_POCKET
    }
    if not pocketed or cue_id in pocketed:
        return None
    if not turn.call_required:
        return Shot(angle, power)
    if "8" in pocketed and "8" not in turn.hittable_balls:
        return None
    for ball_id, pocket_id in pocketed.items():
        if ball_id in turn.hittable_balls:
            return Shot(angle, power, called_ball=ball_id, called_pocket=pocket_id)
    return None

