"""Configuration for the TranscribeWhisper plugin."""
from nendo import NendoConfig
from pydantic import Field


class TranscribeWhisperConfig(NendoConfig):
    """Configuration for the TranscribeWhisper plugin.

    Attributes:
        model (str): The name of the model to use. Defaults to "distil-whisper/distil-large-v2".

    """
    transcription_model: str = Field("openai/whisper-large-v3")
    attn_implementation: str = Field("eager")
    chunk_length: int = Field(15)
    batch_size: int = Field(16)
