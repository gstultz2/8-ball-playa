"""Replay rack shots with a paused preview of the cue's aim."""

import math

from panda3d.core import LineSegs, NodePath
from pooltool.ani.action import Action
from pooltool.ani.animate import ShotViewer
from pooltool.ani.globals import Global
from pooltool.ani.modes.datatypes import Mode
from pooltool.ani.modes.shot import ShotMode
from pooltool.ani.scene import PlaybackMode, visual
from pooltool.system.datatypes import multisystem


class RackShotViewer(ShotViewer):
    def __init__(self):
        super().__init__()
        Global.mode_mgr.modes[Mode.shot] = RackShotMode()


class RackShotMode(ShotMode):
    view_only = True

    def __init__(self):
        super().__init__()
        self.aim_line: NodePath | None = None

    def enter(self, build_animations=False, playback_mode: PlaybackMode | None = None):
        super().enter(build_animations=build_animations, playback_mode=playback_mode)
        if build_animations:
            self._pause_at_start()
        self._update_aim_line()

    def exit(self, key="soft"):
        self._clear_aim_line()
        super().exit(key=key)

    def shot_animation_task(self, task):
        was_parallel = visual.is_parallel_mode
        previous_system = visual.system
        if self.keymap[Action.restart_ani] and not was_parallel:
            self.keymap[Action.restart_ani] = False
            self._pause_at_start()
            result = task.cont
        else:
            result = super().shot_animation_task(task)
            if previous_system is not visual.system:
                self._clear_aim_line()
                if not was_parallel:
                    self._pause_at_start()

        self._update_aim_line()
        return result

    def _pause_at_start(self) -> None:
        visual.pause_animation()
        visual.restart_animation()

    def _update_aim_line(self) -> None:
        if (
            visual.is_parallel_mode
            or not visual.paused
            or visual.shot_animation.get_t() != 0
        ):
            if self.aim_line is not None:
                self.aim_line.hide()
            return

        if self.aim_line is None:
            system = multisystem.active
            cue_ball = system.balls[system.cue.cue_ball_id]
            x, y, z = cue_ball.history[0].rvw[0]
            radius = cue_ball.params.R
            angle = math.radians(system.cue.phi)
            dx, dy = math.cos(angle), math.sin(angle)

            # Clip the aim ray at the edge of the playing surface.
            distances = [
                ((edge if direction > 0 else 0) - origin) / direction
                for origin, direction, edge in (
                    (x, dx, system.table.w),
                    (y, dy, system.table.l),
                )
                if abs(direction) > 1e-9
            ]
            distance = min(distances)
            if distance <= radius:
                return

            line = LineSegs("cue-aim-line")
            line.set_color(1.0, 0.85, 0.25, 1.0)
            line.set_thickness(3)
            line.move_to(x + radius * dx, y + radius * dy, z)
            line.draw_to(x + distance * dx, y + distance * dy, z)
            # Pooltool reuses the scene's table node when switching shots.
            table = Global.render.find("scene/table")
            self.aim_line = table.attach_new_node(line.create())
            self.aim_line.set_light_off(1)
            self.aim_line.set_shader_off(1)

        self.aim_line.show()

    def _clear_aim_line(self) -> None:
        if self.aim_line is not None:
            self.aim_line.remove_node()
            self.aim_line = None
