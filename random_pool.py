
import random
import pooltool as pt

from nine_ball import (
    get_lowest_ball_on_table,
    check_nineball_foul
)

table = pt.Table.default()
balls = pt.get_rack(pt.GameType.NINEBALL, table)
cue = pt.Cue(cue_ball_id="cue")

game = pt.System(table=table, balls=balls, cue=cue)

shots = pt.MultiSystem()
NUM_SHOTS = 20

for i in range(NUM_SHOTS):

    lowest_ball = get_lowest_ball_on_table(game)

    if lowest_ball is None:
        print("No object balls remaining!")
        break

    angle = random.uniform(0, 360)
    power = random.uniform(1, 5)

    game.cue.set_state(
        V0=power,
        phi=angle,
        theta=0,
        a=0,
        b=0
    )

    # Run simulation
    pt.simulate(game, inplace=True)

    # Check foul
    foul, reasons = check_nineball_foul(
        game,
        lowest_ball
    )

    # Print results
    print(f"\nShot {i + 1}")
    print(f"Target ball: {lowest_ball}")
    print(f"Angle: {angle:.1f}")
    print(f"Power: {power:.1f}")
    print(f"Foul: {foul}")

    for reason in reasons:
        print(f"  - {reason}")

    pocketed = pt.ruleset.get_pocketed_ball_ids_during_shot(game)
    print(f"Pocketed: {pocketed}")

    # Save simulated shot
    shots.append(game)

    # Prepare next shot
    game = game.copy()
    game.reset_history()

    # Return cue ball after scratch
    if "cue" in pocketed:
        # Simple fixed location, not full ball-in-hand logic
        x = table.w / 2
        y = table.l / 4

        pt.ruleset.respot(game, "cue", x, y)

print("\nSimulation complete.")
