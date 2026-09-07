# Hmong Pipecat — Real-Time Hmong Voice AI Pipeline

The world's first Hmong voice AI pipeline. Speak Hmong → get an intelligent Hmong audio response back. One API call.

Pipeline: **Your audio → STT → LLM → TTS → AI voice response**

## How It Works

1. **Whisper STT** transcribes your spoken Hmong
2. **GPT-4o-mini** generates a natural Hmong response
3. **CosyVoice TTS** speaks the response in your chosen voice
4. Returns audio of the AI speaking Hmong back to you

## Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | audio | — | User's spoken Hmong (wav/mp3) |
| `voice` | string | `Ntshiab_Li` | AI assistant voice (9 choices) |
| `openai_api_key` | secret | — | Your OpenAI API key |
| `conversation_history` | string | `[]` | Previous turns as JSON for multi-turn conversation |
| `speed` | float | `1.0` | Speaking speed (0.5–2.0) |

## Output

WAV audio of the AI's spoken Hmong response.

## Multi-Turn Conversations

Pass `conversation_history` as a JSON array to maintain context:

```python
history = []

# Turn 1
output = replicate.run("millinhawj1/hmong-pipecat:latest", input={
    "audio": open("turn1.wav", "rb"),
    "voice": "Ntshiab_Li",
    "openai_api_key": "sk-...",
    "conversation_history": "[]",
})

# Turn 2 — pass history
history.append({"role": "user", "content": "Nyob zoo"})
history.append({"role": "assistant", "content": "Nyob zoo! Kuv zoo siab..."})

output = replicate.run("millinhawj1/hmong-pipecat:latest", input={
    "audio": open("turn2.wav", "rb"),
    "voice": "Ntshiab_Li",
    "openai_api_key": "sk-...",
    "conversation_history": json.dumps(history),
})
```

## Voices

All 9 native Hmong voices available — 5 Dawb, 4 Leeg. See [hmong-tts](https://replicate.com/millinhawj1/hmong-tts) for the full list.

## Pricing

**$0.12 per run** (STT + LLM + TTS) + your OpenAI usage (~$0.001/turn for GPT-4o-mini).

## About

Built by the **Hmong Voice Company** — [xuajpaj.com](https://xuajpaj.com)
