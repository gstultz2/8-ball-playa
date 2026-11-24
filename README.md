# 8 Ball Playa

8 Ball Playa is a Python command-line simulator for 8-ball and 9-ball against the ghost. A player strategy breaks a fresh rack and keeps shooting from where the balls stop. The player wins by legally sinking the game ball. The ghost wins as soon as the player misses or fouls. Pooltool simulates the physics, and this project decides which shots to try and how to score them.

## Setup

Use Python 3.12 to match the tested environment. From the repository root, create a virtual environment with `python -m venv .venv` and activate it.

- Windows PowerShell uses `.\.venv\Scripts\Activate.ps1`.
- macOS and Linux use `source .venv/bin/activate`.

Install Pooltool, which also installs NumPy for the shot geometry. The distribution is imported as `pooltool` in Python. [Pooltool on PyPI](https://pypi.org/project/pooltool-billiards/0.6.0/)

```sh
python -m pip install pooltool-billiards==0.6.0
```

## Run a rack

```sh
python main.py
python main.py --game 8-ball
python main.py --game 9-ball --player always-make
python main.py --game 9-ball --player least-difficult
python main.py --game 9-ball --player discrete-make --visualize
```

The defaults are `--game 9-ball` and `--player random`. The terminal shows the legal first-contact targets, aim angle in degrees, cue speed in meters per second (displayed as `power`), any ball and pocket call, what dropped, and the reason for the outcome. It prints the winner at the end.

`--visualize` opens Pooltool's shot viewer after the rack finishes. Each shot opens paused with a yellow line showing where the cue is aimed. Press `Space` to play or pause, `R` to return to the paused aiming view, `N` for the next shot, `P` for the previous shot, and `Esc` to close. Run `python main.py --help` to see the available options.

## Train a three-ball Ghost player

```sh
python ghost_rl.py train --layout side-pocket --episodes 2000 --checkpoint ghost_q.npz
python ghost_rl.py replay --checkpoint ghost_q.npz --trials 5 --visualize
```

Choose `side-pocket`, `long-cut`, or `cross-table` to set up the cue ball and the 7, 8, and 9. Each checkpoint is trained on one starting layout. Training uses the existing Pooltool and NumPy installation. Pass `--seed` to vary or reproduce a new training run or replay.

For a long run in PowerShell, use the same checkpoint path each time. `--episodes` counts additional racks.

```powershell
.\.venv\Scripts\python.exe -u ghost_rl.py train --layout side-pocket --episodes 1000000 --save-every 500 --checkpoint ghost_q.npz 2>&1 | Tee-Object -FilePath ghost-training.log -Append
```

Training saves every 500 completed racks by default and at the end of the run. Rerunning `train` loads the saved network, experience, and random number generator state and continues from the last completed save. After an interruption, progress since that save is lost. Older policy files can resume from their saved weights and begin accumulating a new experience buffer. Use a different checkpoint path for a fresh run or for another layout.

On each turn, the network chooses a pocket, aim, and center-ball speed from a grid built from the table geometry. After the choice, the simulator adds uniform aim error up to ±0.18° and speed error up to ±8%, then plays the shot. The network learns from the resulting layout and whether the rack ends in a win. Cut angle and shot distances are logged for replay.

Replay starts from the saved layout each time and applies fresh shot noise. It prints the intended and executed shots, what dropped, and where the cue ball stopped. Use `--visualize` to step through the executed shots in the rack viewer.

## Game rules

- **9-ball** requires the cue ball to hit the lowest-numbered ball still on the table first. After that contact, a ball must hit a rail or an object ball must drop. Any legal pot keeps the rack going, and a legal pot of the 9-ball wins, including on the break. A scratch, illegal first contact, or shot that pockets nothing gives the rack to the ghost.
- **8-ball** requires a called ball and pocket after the break. A clean break that sinks the 8-ball wins. A clean break that sinks one or more solids or stripes leaves the table open for the next shot. The first successful called pot assigns solids or stripes. Once that group is cleared, the player must call and make the 8-ball. A scratch, illegal first contact, dry break, missed call, or illegally pocketed 8-ball gives the rack to the ghost.

## Player strategies

| `--player` | How it chooses a shot |
| --- | --- |
| `random` | Picks a random angle and cue speed between 1 and 5 m/s. It also chooses a random eligible ball and pocket when a call is required. |
| `always-make` | Tries randomized aims and speeds and takes the first shot that the search finds will pocket an eligible ball. |
| `discrete-make` | Tests a fixed grid of angles and cue speeds from 1 to 5 m/s in 0.5 m/s steps, then chooses uniformly among successful angle and speed pairs. |
| `least-difficult` | Uses the fixed grid and looks for a pot that still works when the aim or speed is slightly off. It uses `always-make` for the break. |

Each strategy chooses one shot at a time from the current table layout. The search-based strategies fall back to `random` when their search finds no make. The fixed-grid strategies can take longer on an open 8-ball table, where many balls are legal targets.

### How the search works

1. Generate center-ball aims using several points across each pocket opening and several angles that can contact each legal target. Keep angles whose first ball along the cue ball's path is a legal target.
2. Simulate candidate shots on copies of the table. Inspect the collision and pocket events to find makes and classify each pot as direct, bank, combination, or carom. Discard trials that scratch or send the cue ball to a cushion before it contacts an object ball. For called shots, keep only eligible pots and discard any trial that drops the 8-ball early.
3. For `least-difficult`, rank the makes using the estimated aim margin for direct pots, whether nearby speeds make the same pot, and penalties for banks and indirect routes. Replay up to five finalists with four separate changes by default, using aim offsets of ±0.1° and speed changes of ±5%. Each replay counts as a success when it pockets the intended ball in the intended pocket. The final choice minimizes the replay miss cost plus the route penalty, with the initial ranking breaking ties.

`ShotGrid` in [shot_search.py](shot_search.py) configures the aim samples and speeds. `DifficultySettings` in [pool_player.py](pool_player.py) configures the finalist count, execution errors, and scoring weights.

## Code walkthrough

| File | Role |
| --- | --- |
| [main.py](main.py) | Parses command-line options, runs a rack, prints the result, and optionally opens the replay viewer. |
| [pool_simulator.py](pool_simulator.py) | Creates the rack and runs the turn loop. It asks the rules for a `Turn`, asks the strategy for a `Shot`, simulates the shot, and tells the rules what happened. It records the shot and its Pooltool event history in `GameResult`, then copies the resulting table and clears its history before the next turn. |
| [pool_rules.py](pool_rules.py) | Supplies legal first-contact targets and call requirements, then judges each simulated shot. The 8-ball rules also track the player's ball group. |
| [pool_shot.py](pool_shot.py) | Defines the shared `Turn` and `Shot` data. `Shot.apply_to()` maps the shot's speed, angle, elevation, and spin to the Pooltool cue. |
| [pool_player.py](pool_player.py) | Defines the `Player` interface and implements the four strategies. |
| [shot_search.py](shot_search.py) | Builds candidate aims, simulates trial shots, and extracts pots from the resulting events. |

To add a strategy, implement `take_turn(table, turn) -> Shot` with the table as read-only input and register the strategy class in `PLAYERS` in [main.py](main.py).

## Tests

```sh
python -m unittest -v
```

[test_least_difficult.py](test_least_difficult.py) checks that the strategy can prefer a direct pot, find a combination when the direct pot is blocked, and score replays by the intended ball and pocket. It also checks that the replay budget is respected and searching leaves the input table untouched.

## License

[MIT](LICENSE)
