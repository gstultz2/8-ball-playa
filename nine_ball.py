import pooltool as pt

def get_lowest_ball_on_table(system):
    """Return the lowest-numbered ball currently on the table."""

    available = [
        ball_id
        for ball_id, ball in system.balls.items()
        if ball_id != "cue"
        and ball.state.s in pt.constants.on_table
    ]

    if not available:
        return None

    return min(available, key=int)


def check_nineball_foul(shot, lowest_ball_id):
    """
    Detect three common 9-ball fouls:
    - Cue ball pocketed
    - Wrong ball contacted first
    - No rail contact or object ball pocketed after contact

    Returns:
        (bool, list[str])
    """

    reasons = []
    events = shot.events

    # 1. Cue ball scratch
    if pt.ruleset.is_ball_pocketed(shot, "cue"):
        reasons.append("Cue ball scratch")

    # 2. Check first object ball contacted
    first_hit = pt.ruleset.get_id_of_first_ball_hit(shot)

    if first_hit is None:
        reasons.append("No object ball contacted")

    elif first_hit != str(lowest_ball_id):
        reasons.append(
            f"Wrong first contact: {first_hit}, "
            f"expected {lowest_ball_id}"
        )

    # 3. Check rail contact or pocket after first collision
    if first_hit is not None:
        first_contact = next(
            (
                event.time
                for event in events
                if event.event_type == pt.EventType.BALL_BALL
                and "cue" in event.ids
            ),
            None
        )

        if first_contact is not None:
            rail_types = {
                pt.EventType.BALL_LINEAR_CUSHION,
                pt.EventType.BALL_CIRCULAR_CUSHION,
            }

            rail_after = any(
                event.event_type in rail_types
                and event.time > first_contact
                for event in events
            )

            object_pocketed = any(
                event.event_type == pt.EventType.BALL_POCKET
                and event.time >= first_contact
                and event.ids[0] != "cue"
                for event in events
            )

            if not rail_after and not object_pocketed:
                reasons.append(
                    "No rail contact or object ball pocketed"
                )

    return bool(reasons), reasons
