"""A nendo plugin for audio transcription using the DistilWhisper model."""
from typing import Any, Optional

import gc
import torch
from nendo import Nendo, NendoAnalysisPlugin, NendoConfig, NendoTrack
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline

from .config import TranscribeWhisperConfig

settings = TranscribeWhisperConfig()


def to_timestamp(seconds: Optional[float]) -> str:
    """Convert seconds to timestamp."""
    # if no seconds found for current timestamp default to zero
    if seconds is None:
        return "00:00"
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


class TranscribeWhisper(NendoAnalysisPlugin):
    """A nendo plugin for audio transcription using the DistilWhisper model.

    Examples:
        ```python
        from nendo import Nendo, NendoConfig, NendoTrack

        nd = Nendo(
            config=NendoConfig(
                log_level="INFO",
                plugins=["nendo_plugin_transcribe_whisper"],
            ),
        )

        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        track = nd.plugins.transcribe_whisper(track=track)
        print(track.get_plugin_value("transcription"))
        ```
    """

    nendo_instance: Nendo = None
    config: NendoConfig = None
    pipe: Pipeline = None
    device: str = None

    def __init__(self, **data: Any):
        """Initialize the plugin."""
        super().__init__(**data)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            settings.transcription_model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=settings.attn_implementation,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(settings.transcription_model)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=settings.chunk_length,
            batch_size=settings.batch_size,
            torch_dtype=torch_dtype,
            device=self.device,
            return_timestamps="word",
        )

    @NendoAnalysisPlugin.run_track
    def transcribe(
            self,
            track: NendoTrack,
            return_timestamps: bool = False,
    ) -> NendoTrack:
        """Transcribe the audio in the track.

        Args:
            track (NendoTrack): The track to transcribe.
            return_timestamps (bool, optional): Whether to return the transcription with word-level timestamps. Defaults to False.

        Returns:
            NendoTrack: The track with the transcription added to its `plugin_data`.
        """
        with torch.no_grad():
            result = self.pipe(track.resource.src)

        if return_timestamps:
            transcription = "\n".join(
                [f"[{to_timestamp(chunk['timestamp'][0])}-{to_timestamp(chunk['timestamp'][1])}]:{chunk['text']}" for
                 chunk in result["chunks"]])
        else:
            transcription = result["text"]

        track = track.add_plugin_data(
            plugin_name="nendo_plugin_transcribe_whisper",
            plugin_version="0.1.0",
            key="transcription",
            value=transcription,
        )

        # manually clear memory
        gc.collect()
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return track

