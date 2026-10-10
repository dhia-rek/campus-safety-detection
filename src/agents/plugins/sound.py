"""Sound perception agent — reads the audio_z signal (PANNs CNN14).

Non-verbal acoustics: screams, impacts, breaking glass. Distinct from Speech,
which analyses the *words* — different modality, different model.
"""

from __future__ import annotations

from src.agents.base import Observation, PerceptionAgent
from src.agents.context import frame_value
from src.agents.registry import register


@register
class SoundAgent(PerceptionAgent):
    """Perception agent for the ``audio_z`` signal."""

    name        = "sound"
    label       = "Sound (PANNs)"
    tier        = "pro"
    description = "Detects screams, impacts and breaking glass from the audio waveform."
    col         = "audio_z"

    def observe(self, ctx: dict, frame: int) -> Observation:
        """Report the z-score of this agent's column at *frame*.

        Triggered when the z-score is at or above the context threshold.
        """
        z = frame_value(ctx, frame, self.col)
        return Observation(self.name, self.label, z, z >= ctx["threshold"])
