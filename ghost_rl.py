"""Train and replay a noisy three-ball 9-ball Ghost player."""

import argparse
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pooltool as pt
from pooltool.layouts import DEFAULT_STANDARD_BALLSET

from pool_rules import NineBallRules, Outcome
from pool_shot import Shot, Turn
from pool_simulator import PlayedShot, play_shot
from shot_search import pocket_aim_options


BALL_IDS = ("cue", "7", "8", "9")
# Coordinates are fractions of table width and length, in BALL_IDS order.
LAYOUTS = {
    "side-pocket": ((0.22, 0.45), (0.55, 0.50), (0.67, 0.74), (0.26, 0.78)),
    "long-cut": ((0.26, 0.25), (0.46, 0.56), (0.73, 0.72), (0.35, 0.87)),
    "cross-table": ((0.71, 0.30), (0.46, 0.49), (0.28, 0.75), (0.80, 0.82)),
}
POCKET_OFFSETS = (0.0, -0.4, 0.4)
POCKET_IDS = ("lb", "lc", "lt", "rb", "rc", "rt")
POWERS = (0.9, 1.4, 2.0, 2.8, 3.6)
AIM_ERROR_DEGREES = 0.18
SPEED_ERROR_FRACTION = 0.08
FEATURE_COUNT = 3 * len(BALL_IDS) + 8 + len(POCKET_IDS) + len(POCKET_OFFSETS) + len(POWERS)
HIDDEN_SIZE = 48
BATCH_SIZE = 32


@dataclass(frozen=True)
class ShotOption:
    shot: Shot
    pocket_offset: float | None
    cut_degrees: float | None
    cue_distance: float
    object_distance: float | None


@dataclass(frozen=True)
class Transition:
    features: np.ndarray
    outcome: Outcome
    next_features: np.ndarray | None


@dataclass(frozen=True)
class EpisodeShot:
    option: ShotOption
    played: PlayedShot
    transition: Transition


def make_layout(name: str) -> pt.System:
    table = pt.Table.default()
    balls = {
        ball_id: pt.Ball.create(
            ball_id, xy=(x * table.w, y * table.l), ballset=DEFAULT_STANDARD_BALLSET
        )
        for ball_id, (x, y) in zip(BALL_IDS, LAYOUTS[name], strict=True)
    }
    return pt.System(table=table, balls=balls, cue=pt.Cue(cue_ball_id="cue"))


def shot_options(table: pt.System, turn: Turn) -> tuple[list[ShotOption], np.ndarray]:
    """Build and encode direct-shot options without simulating their outcomes."""
    cue = table.balls[table.cue.cue_ball_id]
    target = table.balls[turn.hittable_balls[0]]
    cue_delta = target.xyz[:2] - cue.xyz[:2]
    options = []
    for pocket_id, pocket in table.table.pockets.items():
        object_delta = pocket.center[:2] - target.xyz[:2]
        object_distance = math.hypot(*object_delta)
        if object_distance == 0:
            continue
        pocket_direction = math.degrees(math.atan2(object_delta[1], object_delta[0]))
        contact_radius = cue.params.R + target.params.R
        cue_distance = math.hypot(*(cue_delta - contact_radius * object_delta / object_distance))
        for offset, angle in pocket_aim_options(cue, target, pocket, POCKET_OFFSETS):
            cut_degrees = abs(math.remainder(pocket_direction - angle, 360))
            for speed in POWERS:
                shot = Shot(angle, speed, called_ball=target.id, called_pocket=pocket_id)
                options.append(ShotOption(shot, offset, cut_degrees, cue_distance, object_distance))

    if not options:
        # A nearly touching ball can have no direct pocket aim in the grid.
        angle = math.degrees(math.atan2(cue_delta[1], cue_delta[0])) % 360
        shot = Shot(angle, POWERS[1], called_ball=target.id)
        options.append(ShotOption(shot, None, None, math.hypot(*cue_delta), None))

    layout = []
    length = table.table.l
    width = table.table.w
    pockets = table.table.pockets
    for ball_id in BALL_IDS:
        ball = table.balls[ball_id]
        if ball.state.s in pt.constants.on_table:
            layout.extend((ball.xyz[0] / width, ball.xyz[1] / length, 1.0))
        else:
            layout.extend((0.0, 0.0, 0.0))

    features = []
    for option in options:
        shot = option.shot
        angle = math.radians(shot.angle)
        pocket = pockets.get(shot.called_pocket)
        pocket_x, pocket_y = (0.0, 0.0) if pocket is None else pocket.center[:2]
        features.append(layout + [
            math.cos(angle), math.sin(angle), shot.power / POWERS[-1],
            pocket_x / width, pocket_y / length,
            (option.cut_degrees or 0.0) / 90,
            option.cue_distance / length, (option.object_distance or 0.0) / length,
            *(shot.called_pocket == pocket_id for pocket_id in POCKET_IDS),
            *(option.pocket_offset == offset for offset in POCKET_OFFSETS),
            *(shot.power == speed for speed in POWERS),
        ])
    return options, np.asarray(features, dtype=np.float32)


def noisy_shot(shot: Shot, rng: np.random.Generator) -> Shot:
    angle = shot.angle + rng.uniform(-AIM_ERROR_DEGREES, AIM_ERROR_DEGREES)
    speed = shot.power * (1 + rng.uniform(-SPEED_ERROR_FRACTION, SPEED_ERROR_FRACTION))
    return replace(shot, angle=angle % 360, power=speed)


def play_episode(
    layout: str, noise_rng: np.random.Generator, choose: Callable[[np.ndarray], int]
) -> Iterator[EpisodeShot]:
    table = make_layout(layout)
    rules = NineBallRules()
    turn = rules.prepare_turn(table, is_break=False)
    options, features = shot_options(table, turn)
    while True:
        index = choose(features)
        option = options[index]
        played = play_shot(table, turn, noisy_shot(option.shot, noise_rng), rules)
        outcome = played.result.outcome
        if outcome is not Outcome.CONTINUE:
            yield EpisodeShot(option, played, Transition(features[index], outcome, None))
            return

        table = played.table.copy()
        table.reset_history()
        turn = rules.prepare_turn(table, is_break=False)
        options, next_features = shot_options(table, turn)
        yield EpisodeShot(option, played, Transition(features[index], outcome, next_features))
        features = next_features


class QNetwork:
    """A small NumPy network that scores one layout and intended shot per row."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.weights = (
            rng.normal(0, 1 / math.sqrt(FEATURE_COUNT), (FEATURE_COUNT, HIDDEN_SIZE)),
            np.zeros(HIDDEN_SIZE),
            rng.normal(0, 0.1 / math.sqrt(HIDDEN_SIZE), HIDDEN_SIZE),
            np.zeros(1),
        )

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w1, b1, w2, b2 = self.weights
        hidden = np.tanh(features @ w1 + b1)
        logits = np.clip(hidden @ w2 + b2, -30, 30)
        return hidden, 1 / (1 + np.exp(-logits))

    def values(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[1]

    def choose(self, features: np.ndarray) -> int:
        return int(np.argmax(self.values(features)))

    def update(self, features: np.ndarray, targets: np.ndarray) -> None:
        hidden, values = self._forward(features)
        error = (values - targets) / len(targets)
        hidden_error = error[:, None] * self.weights[2] * (1 - hidden**2)
        gradients = (
            features.T @ hidden_error, hidden_error.sum(axis=0),
            hidden.T @ error, error.sum(),
        )
        for weights, gradient in zip(self.weights, gradients, strict=True):
            weights -= 0.08 * gradient


def bellman_target(transition: Transition, network: QNetwork, target: QNetwork) -> float:
    if transition.next_features is None:
        return float(transition.outcome is Outcome.PLAYER_WINS)
    index = network.choose(transition.next_features)
    return float(target.values(transition.next_features[index:index + 1])[0])


def train(layout: str, episodes: int, seed: int) -> QNetwork:
    rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    network = QNetwork(rng)
    target = QNetwork(rng)
    target.weights = tuple(weights.copy() for weights in network.weights)
    experience: deque[Transition] = deque(maxlen=5000)
    recent_wins: deque[bool] = deque(maxlen=100)
    steps = 0

    for episode in range(1, episodes + 1):
        epsilon = float(np.interp(episode - 1, (0, max(1, episodes * 0.75)), (0.9, 0.1)))

        def choose(features: np.ndarray) -> int:
            if rng.random() < epsilon:
                return int(rng.integers(len(features)))
            return network.choose(features)

        for step in play_episode(layout, noise_rng, choose):
            experience.append(step.transition)
            steps += 1
            if len(experience) >= BATCH_SIZE:
                batch = [experience[i] for i in rng.integers(len(experience), size=BATCH_SIZE)]
                features = np.stack([sample.features for sample in batch])
                targets = np.array([bellman_target(sample, network, target) for sample in batch])
                network.update(features, targets)
            if steps % 100 == 0:
                target.weights = tuple(weights.copy() for weights in network.weights)
            if step.transition.outcome is not Outcome.CONTINUE:
                recent_wins.append(step.transition.outcome is Outcome.PLAYER_WINS)

        if episode % 100 == 0 or episode == episodes:
            print(f"Rack {episode}/{episodes} | recent wins {sum(recent_wins)}/{len(recent_wins)}",
                  flush=True)
    return network


def train_command(args: argparse.Namespace) -> None:
    network = train(args.layout, args.episodes, args.seed)
    checkpoint = NamedTemporaryFile(dir=args.checkpoint.parent, delete=False)
    temporary_path = Path(checkpoint.name)
    try:
        with checkpoint:
            np.savez(checkpoint, *network.weights, layout=args.layout)
        temporary_path.replace(args.checkpoint)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved policy to {args.checkpoint}")


def replay_command(args: argparse.Namespace) -> None:
    with np.load(args.checkpoint) as checkpoint:
        layout = str(checkpoint["layout"])
        network = QNetwork(np.random.default_rng(0))
        network.weights = tuple(checkpoint[f"arr_{i}"] for i in range(len(network.weights)))
    seeds = np.random.SeedSequence(args.seed).spawn(args.trials)
    visual_shots = pt.MultiSystem() if args.visualize else None
    for trial, seed in enumerate(seeds, start=1):
        print(f"\nTrial {trial} on {layout}")
        for number, step in enumerate(
            play_episode(layout, np.random.default_rng(seed), network.choose), start=1
        ):
            intended = step.option.shot
            actual = step.played.shot
            cue = step.played.table.balls["cue"]
            leave = f"({cue.xyz[0]:.2f}, {cue.xyz[1]:.2f}) m"
            if cue.state.s not in pt.constants.on_table:
                leave = "pocketed"
            cut = step.option.cut_degrees
            object_distance = step.option.object_distance
            geometry = ""
            if cut is not None and object_distance is not None:
                geometry = (f" | cut {cut:.0f}° | cue to contact {step.option.cue_distance:.2f} m"
                            f" | object to pocket {object_distance:.2f} m")
            print(f"  Shot {number} | aim {intended.called_ball} to {intended.called_pocket or 'open play'}"
                  f" | {intended.angle:.2f}° at {intended.power:.2f} m/s{geometry}")
            pocketed = ", ".join(step.played.pocketed) or "none"
            print(f"    Executed {actual.angle:.2f}° at {actual.power:.2f} m/s"
                  f" | pocketed {pocketed} | cue leave {leave} | {step.played.result.reason}")
            if visual_shots is not None:
                visual_shots.append(step.played.table)
            if step.transition.outcome is not Outcome.CONTINUE:
                print(f"  Winner {step.transition.outcome.value}")
    if visual_shots is not None:
        from shot_viewer import RackShotViewer

        RackShotViewer().show(visual_shots, title="Ghost policy replay | Space to play | N/P for shots")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    trainer = commands.add_parser("train", help="learn from noisy racks")
    trainer.add_argument("--layout", choices=LAYOUTS, default="side-pocket")
    trainer.add_argument("--episodes", type=positive_int, default=2000)
    trainer.add_argument("--seed", type=int, default=0)
    trainer.add_argument("--checkpoint", type=Path, default=Path("ghost_q.npz"))
    trainer.set_defaults(run=train_command)

    replay = commands.add_parser("replay", help="play a saved policy with fresh noise")
    replay.add_argument("--trials", type=positive_int, default=5)
    replay.add_argument("--seed", type=int, default=1)
    replay.add_argument("--checkpoint", type=Path, default=Path("ghost_q.npz"))
    replay.add_argument("--visualize", action="store_true", help="view the executed shots")
    replay.set_defaults(run=replay_command)

    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    try:
        args.run(args)
    except FileNotFoundError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
