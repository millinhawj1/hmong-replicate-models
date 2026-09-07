# Hmong TTS — AI Voice Synthesis

Generate natural-sounding Hmong speech from text using CosyVoice, a state-of-the-art multilingual TTS model fine-tuned on native Hmong speakers.

**9 native Hmong voices** — both dialects, male and female.

## Voices

### Hmoob Dawb (White Hmong) — 5 voices

| Voice ID | Name | Gender |
|---|---|---|
| `Kim_tshaj` | Kim Tshaj | Male |
| `Ntshiab_Li` | Ntshiab Li | Female |
| `hmong_white_male` | Hmong White Male | Male |
| `hmong_white_female` | Hmong White Female | Female |
| `vaj` | Vaj | Male |

### Moob Leeg (Green Hmong) — 4 voices

| Voice ID | Name | Gender |
|---|---|---|
| `txeej_txaim` | Txeej Txaim | Male |
| `Kaj_Siab` | Kaj Siab | Female |
| `hmong_green_male` | Hmong Green Male | Male |
| `hmong_green_female` | Hmong Green Female | Female |

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | — | Hmong text to synthesize (RPA spelling) |
| `voice` | string | `Kim_tshaj` | Speaker voice ID (see voices above) |
| `speed` | float | `1.0` | Speaking speed (0.5–2.0) |
| `seed` | int | `0` | Random seed for reproducible output |

## Output

WAV audio file.

## Example

```python
import replicate

output = replicate.run(
    "millinhawj1/hmong-tts:latest",
    input={
        "text": "Nyob zoo! Kuv yog koj tus pab cuam Ees Ai Hmoob Dawb.",
        "voice": "Ntshiab_Li",
        "speed": 1.0
    }
)
```

## Pricing

**$0.016 per 1,000 characters** — same as Google TTS. The only TTS on Replicate with native Hmong voices.

## About

Built by the **Hmong Voice Company** — the first AI voice platform built for the Hmong language.

🎵 Try all 9 voices in our full creative studio at **[xuajpaj.com](https://xuajpaj.com)**.
