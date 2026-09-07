"""
Hmong Pipecat — Real-time Hmong voice AI pipeline.
One call: speak Hmong → get an intelligent Hmong audio response back.

Pipeline: Audio → STT (Whisper Hmong) → LLM (respond in Hmong) → TTS (CosyVoice)
"""
import os, sys, tempfile
import numpy as np
import soundfile as sf
from cog import BaseModel, Input
from pathlib import Path, Secret

COSY_APP = "/workspace/cosyvoice_app/app"
if COSY_APP not in sys.path:
    sys.path.insert(0, COSY_APP)

MODEL_DIR = COSY_APP + "/Fun-CosyVoice3-0.5B-hmong"
COSY_DIR  = COSY_APP + "/CosyVoice"
REF_AUDIO = COSY_APP + "/ref_audio"
CT2_DIR   = "/workspace/pakorn-ct2"

DAWB = "You are a helpful assistant. Please speak Hmong (White Hmong RPA).<|endofprompt|>"
LEEG = "You are a helpful assistant. Please speak Hmong (Green Hmong RPA).<|endofprompt|>"

VOICES = {
    "Kim_tshaj":         {"ref": REF_AUDIO + "/Kim_tshaj.mp3",         "instruct": DAWB},
    "Ntshiab_Li":        {"ref": REF_AUDIO + "/Ntshiab_Li.mp3",        "instruct": DAWB},
    "hmong_white_male":  {"ref": REF_AUDIO + "/hmong_wite_male.mp3",   "instruct": DAWB},
    "hmong_white_female":{"ref": REF_AUDIO + "/hmong_wite_Female.mp3", "instruct": DAWB},
    "vaj":               {"ref": REF_AUDIO + "/vaj.mp3",               "instruct": DAWB},
    "txeej_txaim":       {"ref": REF_AUDIO + "/txeej_txaim.mp3",       "instruct": LEEG},
    "Kaj_Siab":          {"ref": REF_AUDIO + "/Kaj_Siab.mp3",          "instruct": LEEG},
    "hmong_green_male":  {"ref": REF_AUDIO + "/hmong_green_male.mp3",  "instruct": LEEG},
    "hmong_green_female":{"ref": REF_AUDIO + "/hmong_green_Female.mp3","instruct": LEEG},
}

SYSTEM_PROMPT = """You are a helpful Hmong AI assistant. Always respond in Hmong (RPA spelling).
Keep responses natural, conversational, and concise (2-4 sentences max).
Do not use English unless the user speaks English to you."""


class Predictor(BaseModel):
    def setup(self):
        import ctranslate2
        from faster_whisper import WhisperModel
        from tts_core import CosyVoiceEngine

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute = "int8_float16" if device == "cuda" else "int8"

        self.stt = WhisperModel(CT2_DIR, device=device, compute_type=compute)
        self.tts = CosyVoiceEngine(
            model_dir=MODEL_DIR, cosyvoice_dir=COSY_DIR, fp16=True
        )
        print("Hmong Pipecat ready ✓ (STT + LLM + TTS)", flush=True)

    def predict(
        self,
        audio: Path = Input(description="User's spoken Hmong audio (wav/mp3)"),
        voice: str = Input(
            description="AI assistant voice",
            default="Ntshiab_Li",
            choices=["Kaj_Siab", "Kim_tshaj", "Ntshiab_Li", "hmong_green_female",
                     "hmong_green_male", "hmong_white_female", "hmong_white_male",
                     "txeej_txaim", "vaj"],
        ),
        openai_api_key: Secret = Input(description="OpenAI API key for the LLM brain"),
        conversation_history: str = Input(
            description='Previous conversation as JSON array: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]',
            default="[]",
        ),
        speed: float = Input(default=1.0, ge=0.5, le=2.0),
    ) -> Path:
        import json
        import openai

        # ── Step 1: STT ────────────────────────────────────────────────────────
        data, sr = sf.read(str(audio), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = np.interp(
                np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr)),
                np.arange(len(data)), data,
            )
        segments, _ = self.stt.transcribe(
            data, beam_size=1, language=None, task="transcribe",
            condition_on_previous_text=False, vad_filter=True,
        )
        user_text = " ".join(
            s.text.strip() for s in segments
            if s.no_speech_prob <= 0.5 and s.avg_logprob >= -1.0
        ).strip()

        if not user_text:
            raise ValueError("Could not transcribe any speech from the audio")
        print(f"User said: {user_text}", flush=True)

        # ── Step 2: LLM ────────────────────────────────────────────────────────
        history = json.loads(conversation_history) if conversation_history else []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        client = openai.OpenAI(api_key=openai_api_key.get_secret_value())
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
        )
        reply_text = response.choices[0].message.content.strip()
        print(f"AI reply: {reply_text}", flush=True)

        # ── Step 3: TTS ────────────────────────────────────────────────────────
        spec = VOICES[voice]
        sr_out, wav = self.tts.synthesize(
            tts_text=reply_text, mode="Instruct2",
            prompt_wav=spec["ref"], prompt_text="",
            instruct_text=spec["instruct"], text_frontend=False, speed=speed,
        )

        out = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(out), wav, sr_out, subtype="PCM_16")
        return out
