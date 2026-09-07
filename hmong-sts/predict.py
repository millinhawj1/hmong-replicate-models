import os, sys, tempfile
import numpy as np
import soundfile as sf
from cog import BasePredictor, Input
from pathlib import Path

# CosyVoice app
COSY_APP = "/workspace/cosyvoice_app/app"
if COSY_APP not in sys.path:
    sys.path.insert(0, COSY_APP)

MODEL_DIR = COSY_APP + "/Fun-CosyVoice3-0.5B-hmong"
COSY_DIR  = COSY_APP + "/CosyVoice"
REF_AUDIO = COSY_APP + "/ref_audio"
CT2_DIR   = "/workspace/pakorn-ct2"
RAW_DIR   = "/workspace/hmong_ears_pakorn"

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


class Predictor(BasePredictor):
    def setup(self):
        import ctranslate2
        from faster_whisper import WhisperModel
        from tts_core import CosyVoiceEngine

        # Load STT
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute = "int8_float16" if device == "cuda" else "int8"
        self.stt = WhisperModel(CT2_DIR, device=device, compute_type=compute)

        # Load TTS
        self.tts = CosyVoiceEngine(
            model_dir=MODEL_DIR, cosyvoice_dir=COSY_DIR, fp16=True
        )
        print("Hmong STS ready ✓ (STT + TTS)", flush=True)

    def run(
        self,
        audio: Path = Input(description="Hmong speech audio to convert (wav/mp3)"),
        target_voice: str = Input(
            description="Target voice to speak in",
            default="Kim_tshaj",
            choices=["Kaj_Siab", "Kim_tshaj", "Ntshiab_Li", "hmong_green_female",
                     "hmong_green_male", "hmong_white_female", "hmong_white_male",
                     "txeej_txaim", "vaj"],
        ),
        speed: float = Input(default=1.0, ge=0.5, le=2.0),
    ) -> Path:
        # Step 1 — STT: transcribe source audio
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
        text = " ".join(
            s.text.strip() for s in segments
            if s.no_speech_prob <= 0.5 and s.avg_logprob >= -1.0
        ).strip()

        if not text:
            raise ValueError("Could not transcribe any speech from the audio")

        print(f"STT: {text}", flush=True)

        # Step 2 — TTS: synthesize in target voice
        spec = VOICES[target_voice]
        sr_out, wav = self.tts.synthesize(
            tts_text=text, mode="Instruct2",
            prompt_wav=spec["ref"], prompt_text="",
            instruct_text=spec["instruct"], text_frontend=False, speed=speed,
        )

        out = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(out), wav, sr_out, subtype="PCM_16")
        return out
