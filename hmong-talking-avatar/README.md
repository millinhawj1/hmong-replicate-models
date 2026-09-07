# Hmong Talking Avatar — Text to Talking Portrait Video

Type Hmong text. Upload a portrait. Get back a video of that person speaking Hmong in a native AI voice.

The only model in the world that goes directly from **Hmong text → talking portrait video** in one API call.

## How It Works

1. **CosyVoice TTS** synthesizes natural Hmong speech from your text
2. **SadTalker** animates the portrait in perfect lip-sync with the audio
3. Returns a ready-to-use MP4 video

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
| `text` | string | — | Hmong text to speak |
| `voice` | string | `Kim_tshaj` | Speaker voice ID |
| `speed` | float | `1.0` | Speaking speed (0.5–2.0) |
| `face_model_resolution` | int | `256` | 256 (fast) or 512 (sharper) |
| `enhancer` | bool | `false` | Apply GFPGAN face enhancer |
| `still_mode` | bool | `false` | Minimize head motion |

## Output

MP4 video of the portrait speaking your Hmong text.

## Example

```python
import replicate

output = replicate.run(
    "millinhawj1/hmong-talking-avatar:latest",
    input={
        "image": open("portrait.jpg", "rb"),
        "text": "Nyob zoo! Kuv zoo siab tau ntsib koj hnub no.",
        "voice": "Ntshiab_Li",
    }
)
```

## About

Built by the **Hmong Voice Company** — the first AI voice platform built for the Hmong language.

🎵 Full creative studio at **[xuajpaj.com](https://xuajpaj.com)** — generate Hmong songs, animate photos, clone voices, and produce professional Hmong audio from simple text.
