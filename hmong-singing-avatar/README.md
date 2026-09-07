# Hmong Singing Avatar — Text to Singing Portrait Video

Type Hmong lyrics or text. Upload a portrait. Get back a high-quality video of that person performing it in a native Hmong AI voice.

The only model in the world that goes directly from **Hmong text → singing portrait video** in one API call.

## How It Works

1. **CosyVoice TTS** synthesizes natural Hmong speech from your text
2. **SadTalker** animates the portrait with cinematic-quality lip-sync (512px + GFPGAN face enhancement)

## Voices

Four native Hmong voices — two dialects, two genders:

| Voice ID | Name | Gender | Dialect |
|---|---|---|---|
| `Kim_tshaj` | Kim Tshaj | Male | White Hmong (Dawb) |
| `txeej_txaim` | Txeej Txaim | Male | Green Hmong (Leeg) |
| `Ntshiab_Li` | Ntshiab Li | Female | White Hmong (Dawb) |
| `Kaj_Siab` | Kaj Siab | Female | Green Hmong (Leeg) |

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | image | — | Portrait photo (jpg/png) — face must be clearly visible |
| `text` | string | — | Hmong lyrics or text to perform |
| `voice` | string | `Kim_tshaj` | Speaker voice ID |
| `speed` | float | `1.0` | Speaking speed (0.5–2.0) |
| `face_model_resolution` | int | `512` | 256 (fast) or 512 (sharper) |
| `enhancer` | bool | `true` | Apply GFPGAN face enhancer (on by default) |
| `expression_scale` | float | `0.7` | Expression intensity — lower = smoother movement |

## Difference vs. hmong-talking-avatar

| Feature | talking-avatar | singing-avatar |
|---|---|---|
| Default resolution | 256px | **512px** |
| GFPGAN enhancer | off | **on** |
| Expression scale | 1.0 | **0.7 (smoother)** |
| Best for | Conversations, dialogue | **Performances, songs, stories** |

## Output

High-quality MP4 video of the portrait performing your Hmong text.

## Example

```python
import replicate

output = replicate.run(
    "millinhawj1/hmong-singing-avatar:latest",
    input={
        "image": open("portrait.jpg", "rb"),
        "text": "Nyob zoo! Kuv yog koj tus pab cuam Ees Ai Hmoob Dawb.",
        "voice": "Ntshiab_Li",
        "expression_scale": 0.7,
    }
)
```

## About

Built by the **Hmong Voice Company** — the first AI voice platform built for the Hmong language.

🎵 Full creative studio at **[xuajpaj.com](https://xuajpaj.com)** — generate Hmong songs, animate photos, clone voices, and produce professional Hmong audio from simple text.
