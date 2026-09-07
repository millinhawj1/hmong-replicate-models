# Hmong STT — Speech to Text

Transcribe Hmong audio to text. The only Hmong speech recognition model on Replicate — supports both dialects.

## Powered By

**Pakorn2112/whisper-model-large-hmong-multi-speech** — the most advanced publicly available Hmong Whisper model, trained by Pakorn Archakeeree (the most prolific Hmong AI researcher).

## Dialects Supported

- 🟢 **Hmoob Dawb** (White Hmong)
- 🟢 **Moob Leeg** (Green Hmong)

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | audio | — | Hmong audio to transcribe (wav/mp3/m4a) |
| `dialect` | string | `dawb` | Dialect hint (`dawb` or `leeg`) |

## Output

Transcribed Hmong text in RPA spelling.

## Pricing

**$0.010 per run** — matches Whisper pricing. The only Hmong STT on Replicate.

## About

Built by the **Hmong Voice Company** — [xuajpaj.com](https://xuajpaj.com)
