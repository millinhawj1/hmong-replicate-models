# Hmong TTS — AI Voice Synthesis

Generate natural-sounding Hmong speech from text using CosyVoice, a state-of-the-art multilingual TTS model fine-tuned on native Hmong speakers.

## Voices

Four native Hmong voices — two male, two female — covering both major dialects:

| Voice ID | Name | Gender | Dialect |
|---|---|---|---|
| `Kim_tshaj` | Kim Tshaj | Male | White Hmong (Dawb) |
| `txeej_txaim` | Txeej Txaim | Male | Green Hmong (Leeg) |
| `Ntshiab_Li` | Ntshiab Li | Female | White Hmong (Dawb) |
| `Kaj_Siab` | Kaj Siab | Female | Green Hmong (Leeg) |

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | — | Hmong text to synthesize (RPA spelling) |
| `voice` | string | `Kim_tshaj` | Speaker voice ID |
| `speed` | float | `1.0` | Speaking speed (0.5–2.0) |
| `seed` | int | `0` | Random seed for reproducible output |

## Output

WAV audio file at 22050 Hz.

## Example

```python
import replicate

output = replicate.run(
    "millinhawj1/hmong-tts:latest",
    input={
        "text": "Nyob zoo, kuv lub npe hu ua Maiv Yeeb.",
        "voice": "Ntshiab_Li",
        "speed": 1.0
    }
)
```

## About

This model is part of the **Hmong Voice Company** — building the first professional-grade AI voice and music studio for the Hmong language.

🎵 Try it with a full creative studio experience at **[xuajpaj.com](https://xuajpaj.com)** — generate Hmong songs, customize vocals, and create professional audio from simple text prompts.
