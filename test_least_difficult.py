from unittest import TestCase
from unittest.mock import patch

import pooltool as pt

from least_difficult import DifficultySettings, LeastDifficult
from pool_player import Shot, Turn
from pool_rules import NineBallRules
from shot_search import Pot, PotRoute, ShotGrid, grid_makes, simulate_make


def side_pocket_layout(blocker_offset: float) -> pt.System:
    table = pt.Table.default()
    middle = table.l / 2
    positions = {
        "cue": (0.25, middle),
        "1": (0.65, middle),
        "2": (table.w - 0.16, middle + blocker_offset),
    }
    balls = {ball_id: pt.Ball.create(ball_id, xy=xy) for ball_id, xy in positions.items()}
    return pt.System(table=table, balls=balls, cue=pt.Cue(cue_ball_id="cue"))


class LeastDifficultTests(TestCase):
    def test_prefers_direct_pot_when_a_carom_is_available(self) -> None:
        table = side_pocket_layout(blocker_offset=0.075)
        turn = NineBallRules().prepare_turn(table, is_break=False)
        grid = ShotGrid(
            contact_offsets=(-0.08, -0.04, 0, 0.04, 0.08),
            pocket_offsets=(0,),
            powers=(1.5, 2, 2.5),
        )
        makes = {pot for _, pots in grid_makes(table, turn, grid) for pot in pots}
        self.assertIn(Pot("2", "rc", PotRoute.CAROM), makes)

        with patch("least_difficult.simulate_make", wraps=simulate_make) as replays:
            shot = LeastDifficult(grid).take_turn(table, turn)

        self.assertIn(Pot("1", "rc", PotRoute.DIRECT), simulate_make(table, turn, shot))
        self.assertGreater(replays.call_count, 0)
        self.assertLessEqual(replays.call_count, 20)
        self.assertFalse(table.simulated)

    def test_uses_combination_when_the_direct_pot_is_blocked(self) -> None:
        table = side_pocket_layout(blocker_offset=0)
        turn = NineBallRules().prepare_turn(table, is_break=False)
        grid = ShotGrid(contact_offsets=(0,), pocket_offsets=(), powers=(1.5, 2, 2.5))
        shot = LeastDifficult(grid).take_turn(table, turn)
        self.assertEqual(simulate_make(table, turn, shot), (Pot("2", "rc", PotRoute.COMBINATION),))

    def test_replays_must_make_the_same_ball_in_the_same_pocket(self) -> None:
        table = side_pocket_layout(blocker_offset=0)
        turn = NineBallRules().prepare_turn(table, is_break=False)
        direct, combination = Shot(5, 2), Shot(10, 2)
        direct_pot = Pot("1", "rc", PotRoute.DIRECT)
        combination_pot = Pot("2", "rc", PotRoute.COMBINATION)
        nominal_makes = [(direct, (direct_pot,)), (combination, (combination_pot,))]

        def replay(_table: pt.System, _turn: Turn, shot: Shot) -> tuple[Pot, ...]:
            if abs(shot.angle - direct.angle) < 1:
                return Pot("1", "lc", PotRoute.DIRECT), combination_pot
            return (combination_pot,)

        settings = DifficultySettings(finalist_count=2)
        with (
            patch("least_difficult.grid_makes", return_value=nominal_makes),
            patch("least_difficult.simulate_make", side_effect=replay) as replays,
        ):
            self.assertEqual(LeastDifficult(settings=settings).take_turn(table, turn), combination)
        self.assertEqual(replays.call_count, 8)
