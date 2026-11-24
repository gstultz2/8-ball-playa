from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pooltool as pt

from ghost_rl import (
    AIM_ERROR_DEGREES,
    FEATURE_COUNT,
    LAYOUTS,
    SPEED_ERROR_FRACTION,
    QNetwork,
    Transition,
    bellman_target,
    make_layout,
    play_episode,
    shot_options,
)
from pool_rules import NineBallRules, Outcome


class GhostRLTests(TestCase):
    def test_layouts_propose_shots_before_running_physics(self) -> None:
        with patch("pooltool.simulate") as simulate:
            for name in LAYOUTS:
                with self.subTest(layout=name):
                    table = make_layout(name)
                    turn = NineBallRules().prepare_turn(table, is_break=False)
                    options, features = shot_options(table, turn)
                    self.assertEqual(set(table.balls), {"cue", "7", "8", "9"})
                    self.assertEqual(features.shape, (len(options), FEATURE_COUNT))
                    self.assertGreater(len({option.shot.called_pocket for option in options}), 1)
                    self.assertTrue(all(option.shot.called_ball == "7" for option in options))
            simulate.assert_not_called()

    def test_one_committed_shot_trains_on_the_intended_action(self) -> None:
        table = make_layout("side-pocket")
        turn = NineBallRules().prepare_turn(table, is_break=False)
        options, features = shot_options(table, turn)
        index = next(
            index for index, option in enumerate(options)
            if option.shot.called_pocket == "rc" and option.pocket_offset == 0
        )

        with patch("pooltool.simulate", wraps=pt.simulate) as simulate:
            episode = play_episode("side-pocket", np.random.default_rng(4), lambda _: index)
            step = next(episode)
            simulate.assert_called_once()
            episode.close()

        executed = step.played.shot
        intended = options[index].shot
        self.assertEqual(step.option.shot, intended)
        np.testing.assert_array_equal(step.transition.features, features[index])
        self.assertEqual(executed.called_pocket, intended.called_pocket)
        self.assertGreater(abs(executed.angle - intended.angle), 0)
        self.assertLessEqual(abs(executed.angle - intended.angle), AIM_ERROR_DEGREES)
        self.assertLessEqual(abs(executed.power / intended.power - 1), SPEED_ERROR_FRACTION)
        self.assertIs(step.transition.outcome, Outcome.CONTINUE)
        self.assertIsNotNone(step.transition.next_features)

    def test_terminal_targets_and_update(self) -> None:
        network = QNetwork(np.random.default_rng(0))
        features = np.ones((1, FEATURE_COUNT))
        win = Transition(features[0], Outcome.PLAYER_WINS, None)
        loss = Transition(features[0], Outcome.GHOST_WINS, None)
        self.assertEqual(bellman_target(win, network, network), 1)
        self.assertEqual(bellman_target(loss, network, network), 0)
        initial = network.values(features)[0]
        network.update(features, np.ones(1))
        self.assertGreater(network.values(features)[0], initial)
