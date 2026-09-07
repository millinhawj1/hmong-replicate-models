# Hmong STS — Speech to Speech Voice Swap

Speak Hmong in one voice — get it back in any of 9 native Hmong AI voices. Instantly.

Pipeline: **Your audio → Whisper STT → CosyVoice TTS → Output in target voice**

## How It Works

1. Your Hmong audio is transcribed using our fine-tuned Whisper model
2. The transcript is re-synthesized in your chosen target voice using CosyVoice
3. Returns clean audio in the new voice

## 9 Available Voices

| Voice ID | Name | Gender | Dialect |
|---|---|---|---|
| `Kim_tshaj` | Kim Tshaj | Male | Dawb |
| `Ntshiab_Li` | Ntshiab Li | Female | Dawb |
| `hmong_white_male` | Hmong White Male | Male | Dawb |
| `hmong_white_female` | Hmong White Female | Female | Dawb |
| `vaj` | Vaj | Male | Dawb |
| `txeej_txaim` | Txeej Txaim | Male | Leeg |
| `Kaj_Siab` | Kaj Siab | Female | Leeg |
| `hmong_green_male` | Hmong Green Male | Male | Leeg |
| `hmong_green_female` | Hmong Green Female | Female | Leeg |

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | audio | — | Source Hmong speech (wav/mp3) |
| `target_voice` | string | `Kim_tshaj` | Target voice ID |
| `speed` | float | `1.0` | Speaking speed (0.5–2.0) |

## Output

WAV audio of the same speech in the target voice.

## Pricing

**$0.08 per run** — unique on Replicate. No other model does Hmong speech-to-speech.

## About

Built by the **Hmong Voice Company** — [xuajpaj.com](https://xuajpaj.com)
