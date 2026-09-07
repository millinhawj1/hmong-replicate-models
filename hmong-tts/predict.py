import sys, io, tempfile
from pathlib import Path

# CosyVoice app lives here on the pod
COSY_APP = Path("/workspace/cosyvoice_app/app")
if str(COSY_APP) not in sys.path:
    sys.path.insert(0, str(COSY_APP))

import numpy as np
import soundfile as sf
from cog import BasePredictor, Input
from tts_core import CosyVoiceEngine

MODEL_DIR    = COSY_APP / "Fun-CosyVoice3-0.5B-hmong"
COSY_DIR     = COSY_APP / "CosyVoice"
REF_AUDIO    = COSY_APP / "ref_audio"

# Mirror of VOICE_CATALOG in api.py
# 9 native Hmong voices — 5 Dawb, 4 Leeg
DAWB = "You are a helpful assistant. Please speak Hmong (White Hmong RPA).<|endofprompt|>"
LEEG = "You are a helpful assistant. Please speak Hmong (Green Hmong RPA).<|endofprompt|>"

VOICES = {
    # ── Pakorn voices (Dawb) ─────────────────────────────────────────────
    "Kim_tshaj": {
        "name":    "Kim Tshaj (Male · Dawb)",
        "ref":     str(REF_AUDIO / "Kim_tshaj.mp3"),
        "instruct": DAWB,
    },
    "Ntshiab_Li": {
        "name":    "Ntshiab Li (Female · Dawb)",
        "ref":     str(REF_AUDIO / "Ntshiab_Li.mp3"),
        "instruct": DAWB,
    },
    # ── Pakorn voices (Leeg) ─────────────────────────────────────────────
    "txeej_txaim": {
        "name":    "Txeej Txaim (Male · Leeg)",
        "ref":     str(REF_AUDIO / "txeej_txaim.mp3"),
        "instruct": LEEG,
    },
    "Kaj_Siab": {
        "name":    "Kaj Siab (Female · Leeg)",
        "ref":     str(REF_AUDIO / "Kaj_Siab.mp3"),
        "instruct": LEEG,
    },
    # ── Bundled Dawb voices ──────────────────────────────────────────────
    "hmong_white_male": {
        "name":    "Hmong White Male (Dawb)",
        "ref":     str(REF_AUDIO / "hmong_wite_male.mp3"),
        "instruct": DAWB,
    },
    "hmong_white_female": {
        "name":    "Hmong White Female (Dawb)",
        "ref":     str(REF_AUDIO / "hmong_wite_Female.mp3"),
        "instruct": DAWB,
    },
    "vaj": {
        "name":    "Vaj (Male · Dawb)",
        "ref":     str(REF_AUDIO / "vaj.mp3"),
        "instruct": DAWB,
    },
    # ── Bundled Leeg voices ──────────────────────────────────────────────
    "hmong_green_male": {
        "name":    "Hmong Green Male (Leeg)",
        "ref":     str(REF_AUDIO / "hmong_green_male.mp3"),
        "instruct": LEEG,
    },
    "hmong_green_female": {
        "name":    "Hmong Green Female (Leeg)",
        "ref":     str(REF_AUDIO / "hmong_green_Female.mp3"),
        "instruct": LEEG,
    },
}


class Predictor(BasePredictor):
    def setup(self):
        self.engine = CosyVoiceEngine(
            model_dir=MODEL_DIR,
            cosyvoice_dir=COSY_DIR,
            fp16=True,
        )
        print("CosyVoice loaded ✓", flush=True)

    def predict(
        self,
        text: str = Input(description="Hmong text to synthesize"),
        voice: str = Input(
            description="Speaker voice",
            default="Kim_tshaj",
            choices=["Kaj_Siab","Kim_tshaj","Ntshiab_Li","hmong_green_female","hmong_green_male","hmong_white_female","hmong_white_male","txeej_txaim","vaj"],
        ),
        speed: float = Input(
            description="Speaking speed (0.5–2.0)", default=1.0, ge=0.5, le=2.0
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 = random)",
            default=0, ge=-1, le=2_147_483_647,
        ),
    ) -> Path:
        text = text.strip()
        if not text:
            raise ValueError("text is required")

        spec = VOICES[voice]

        sample_rate, wav = self.engine.synthesize(
            tts_text=text,
            mode="Instruct2",
            prompt_wav=spec["ref"],
            prompt_text="",
            instruct_text=spec["instruct"],
            text_frontend=False,
            speed=speed,
            seed=seed,
        )

        out = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(out), wav, sample_rate, subtype="PCM_16")
        return out
