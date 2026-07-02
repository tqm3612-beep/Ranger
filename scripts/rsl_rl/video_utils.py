from __future__ import annotations

import gc
import os
from collections.abc import Callable
from typing import Any, Generic, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import error, logger
from gymnasium.core import ActType, ObsType, RenderFrame


class HighQualityRecordVideo(
    gym.Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType, RenderFrame],
    gym.utils.RecordConstructorArgs,
):
    """Record videos with the same trigger behavior as Gymnasium's wrapper but higher output quality."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
        video_bitrate: str = "20000k",
        video_codec: str = "libx264",
        video_crf: int = 16,
        video_preset: str = "slow",
        output_size: tuple[int, int] | None = (3840, 2160),
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        gym.Wrapper.__init__(self, env)

        if env.render_mode in {None, "human", "ansi"}:
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo.",
                "Initialize your environment with a render_mode that returns an image, such as rgb_array.",
            )

        if episode_trigger is None and step_trigger is None:
            from gymnasium.utils.save_video import capped_cubic_video_schedule

            episode_trigger = capped_cubic_video_schedule

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.disable_logger = disable_logger
        self.gc_trigger = gc_trigger

        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        if fps is None:
            fps = self.metadata.get("render_fps", 30)
        self.frames_per_sec: int = fps
        self.name_prefix = name_prefix
        self.video_length: int = video_length if video_length != 0 else float("inf")
        self.video_bitrate = video_bitrate
        self.video_codec = video_codec
        self.video_crf = video_crf
        self.video_preset = video_preset
        self.output_size = output_size

        self._video_name: str | None = None
        self.recording = False
        self.recorded_frames: list[RenderFrame] = []
        self.render_history: list[RenderFrame] = []
        self.step_id = -1
        self.episode_id = -1

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "gymnasium[other]"`'
            ) from e

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.env.render()
        if isinstance(frame, list):
            if len(frame) == 0:
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.episode_id += 1

        if self.recording and self.video_length == float("inf"):
            self.stop_recording()

        if self.episode_trigger and self.episode_trigger(self.episode_id):
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")

        if self.recording:
            self._capture_frame()

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = super().step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

        if self.recording:
            self._capture_frame()

            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, rew, terminated, truncated, info

    def render(self):
        render_out = super().render()
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            return tmp_history + render_out
        return render_out

    def close(self):
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self, video_name: str):
        if self.recording:
            self.stop_recording()
        self.recording = True
        self._video_name = video_name

    def stop_recording(self):
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            if self.output_size is not None:
                clip = clip.resized(new_size=self.output_size)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
            clip.write_videofile(
                path,
                logger=moviepy_logger,
                audio=False,
                codec=self.video_codec,
                bitrate=self.video_bitrate,
                preset=self.video_preset,
                ffmpeg_params=["-crf", str(self.video_crf), "-pix_fmt", "yuv420p"],
            )

        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()
