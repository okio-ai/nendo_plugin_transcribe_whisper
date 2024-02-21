# Nendo Plugin Transcribe Whisper

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

<p align="left">
<a href="https://okio.ai" target="_blank">
    <img src="https://img.shields.io/website/https/okio.ai" alt="Website">
</a>
<a href="https://twitter.com/okio_ai" target="_blank">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai" alt="Twitter">
</a>
<a href="https://discord.gg/gaZMZKzScj" target="_blank">
    <img src="https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat" alt="Discord">
</a>
</p>

---

A nendo plugin for speech transcription, based on Whisper by OpenAI.

## Features

- Fast speech transcription with optional word-level timestamps.

## Requirements

Since we depend on `transformers`, please make sure that you fulfill their requirements.
You also need Pytorch installed on your system, please refer to the [pytorch installation instructions](https://pytorch.org/get-started/locally/).

## Installation

1. [Install Nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-transcribe-whisper`

If you have a cuda GPU on your machine you can also install `flash-attn` to get an additional speedup:

`pip install flash-attn --no-build-isolation`

Then set `ATTN_IMPLEMENTATION=flash_attention_2` in your environment variables.

## Usage
```pycon
>>> from nendo import Nendo
>>> nd = Nendo(plugins=["nendo_plugin_transcribe_whisper"])
>>> track = nd.library.add_track(file_path="path/to/file.mp3")

>>> nd.plugins.transcribe_whisper(track=track)
>>> track.get_plugin_value("transcription")
```

## Contributing
Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)

## License
Nendo: MIT License

Pretrained models: The weights are released under the Apache 2.0 license.
