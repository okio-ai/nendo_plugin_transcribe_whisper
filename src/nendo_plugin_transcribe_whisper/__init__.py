"""A nendo plugin for audio transcription using the DistilWhisper model."""
from __future__ import annotations

from .plugin import TranscribeWhisper

__version__ = "0.1.0"

__all__ = [
    "TranscribeWhisper",
]
