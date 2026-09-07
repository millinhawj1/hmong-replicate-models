# Hmong Voice Clone — Zero-Shot Voice Conversion

Clone any voice into any audio. Upload a 3–10 second reference clip of the target speaker — the model transfers their vocal characteristics onto your source audio instantly. No training required.

Works with **any language, any accent, any gender**.

## What It Can Clone

This model uses **Seed-VC V2** (ByteDance), a state-of-the-art zero-shot voice conversion model. It works with virtually any voice type:

### Languages
- 🟢 **Hmoob Dawb** (White Hmong)
- 🟢 **Moob Leeg** (Green Hmong)
- 🟢 **English**
- 🟢 **Thai**
- 🟢 **Mandarin / Cantonese**
- 🟢 **Spanish, French, Vietnamese, Korean, Japanese**
- 🟢 Any other spoken language

### Voice Types
- Male voices (any age)
- Female voices (any age)
- Elderly voices
- Children's voices
- Accented speakers
- Regional dialects

### Use Cases
- Clone a family member's voice for storytelling
- Convert your Hmong TTS output into a custom voice
- Preserve an elder's voice for cultural heritage
- Dub content into another speaker's voice
- Create consistent AI spokesperson voices

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_audio` | audio | — | The speech to convert (wav/mp3) — any language |
| `reference_audio` | audio | — | 3–10 sec clip of the target voice (wav/mp3) — clean speech, no background music |
| `diffusion_steps` | int | `50` | Quality vs speed — higher = better quality, slower (10–100) |
| `length_adjust` | float | `1.0` | Output length relative to source (0.5–2.0) |

## Output

WAV audio of the source speech in the cloned target voice.

## Example

```python
import replicate

output = replicate.run(
    "millinhawj1/hmong-voice-clone:latest",
    input={
        "source_audio":    open("hmong_speech.wav", "rb"),
        "reference_audio": open("grandma_voice_clip.wav", "rb"),
        "diffusion_steps": 50,
    }
)
```

## Tips for Best Results

- **Reference clip:** 3–10 seconds of clean, clear speech — no background music, no crowd noise
- **Source audio:** Any clean speech in any language
- **Quality vs speed:** Use `diffusion_steps=75` or `100` for maximum similarity to the reference voice
- **Length:** Keep source audio under 60 seconds for fastest results

## Pricing

**$0.059 per run** — matches OpenVoice pricing. Seed-VC V2 delivers significantly better quality, especially for tonal Asian languages.

## About

Built by the **Hmong Voice Company** — the first AI voice platform built for the Hmong language.

🎵 Full creative studio at **[xuajpaj.com](https://xuajpaj.com)** — generate Hmong songs, clone voices, animate photos, and produce professional Hmong audio from simple text.
