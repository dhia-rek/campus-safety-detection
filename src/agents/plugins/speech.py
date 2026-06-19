"""Speech perception agent — reads verbal_z + flagged words (Whisper transcription).

This is where HATE SPEECH lives: Whisper transcribes the spoken words and the
pipeline flags threats/insults. It surfaces the actual words so the brain can
weigh them — fenced as untrusted evidence, never as instructions.
"""

from __future__ import annotations

from src.agents.base import Observation, PerceptionAgent
from src.agents.context import frame_value
from src.agents.registry import register


@register
class SpeechAgent(PerceptionAgent):
    name        = "speech"
    label       = "Speech (Whisper)"
    tier        = "pro"
    description = "Transcribes speech and flags hate speech, threats and insults."
    col         = "verbal_z"

    def _spoken_text(self, ctx: dict) -> str:
        """Surface any flagged words so the coordinator has the verbal evidence."""
        flagged = [s for s in ctx["segments"] if s.get("bad_words")]
        if not flagged:
            return ""
        words = sorted({w for s in flagged for w in s["bad_words"]})
        return "; ".join(words)

    def observe(self, ctx: dict, frame: int) -> Observation:
        z = frame_value(ctx, frame, self.col)
        return Observation(
            self.name, self.label, z, z >= ctx["threshold"],
            detail=self._spoken_text(ctx),
        )
