from nendo import Nendo, NendoConfig, NendoTrack
import torch
import unittest
import random
import numpy as np

nd = Nendo(
    config=NendoConfig(
        log_level="INFO",
        plugins=["nendo_plugin_transcribe_whisper"],
        copy_to_library=False,
    ),
)

# fix seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


class TranscribeWhisperTests(unittest.TestCase):
    def test_run_transcribe_whisper(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        track = nd.plugins.transcribe_whisper(track=track)
        self.assertEqual(type(track), NendoTrack)
        self.assertEqual(
            track.get_plugin_value("transcription"),
            " Number 1. Number 2. Number 3. Number 4. Number 5. Number 6. Number 7. Number 8. Number 9. Number 10.",
        )

    def test_run_transcribe_whisper_with_timestamps(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.mp3")
        track = nd.plugins.transcribe_whisper(track=track, return_timestamps=True)
        self.assertEqual(type(track), NendoTrack)
        self.assertEqual(
            track.get_plugin_value("transcription"),
            "[00:00-00:00]: Number\n[00:00-00:00]: 1.\n[00:00-00:02]: Number\n[00:02-00:02]: 2.\n[00:02-00:04]: Number\n[00:04-00:04]: 3.\n[00:04-00:06]: Number\n[00:06-00:07]: 4.\n[00:07-00:08]: Number\n[00:08-00:09]: 5.\n[00:09-00:10]: Number\n[00:10-00:11]: 6.\n[00:11-00:12]: Number\n[00:12-00:13]: 7.\n[00:13-00:14]: Number\n[00:14-00:14]: 8.\n[00:14-00:16]: Number\n[00:16-00:16]: 9.\n[00:16-00:17]: Number\n[00:17-00:18]: 10.",
        )

    def test_run_transcribe_whisper_with_no_vocals(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/no_vocals.mp3")
        track = nd.plugins.transcribe_whisper(track=track, return_timestamps=True)
        self.assertEqual(type(track), NendoTrack)
        # only assert that it runs through, because transcription is gibberish


if __name__ == "__main__":
    unittest.main()
