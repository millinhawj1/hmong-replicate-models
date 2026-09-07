import sys, io, subprocess, tempfile
from pathlib import Path as SysPath

# CosyVoice app
COSY_APP = SysPath("/workspace/cosyvoice_app/app")
if str(COSY_APP) not in sys.path:
    sys.path.insert(0, str(COSY_APP))

import soundfile as sf
from cog import BaseModel, Input
from pathlib import Path, Secret
from tts_core import CosyVoiceEngine

MODEL_DIR = COSY_APP / "Fun-CosyVoice3-0.5B-hmong"
COSY_DIR  = COSY_APP / "CosyVoice"
REF_AUDIO = COSY_APP / "ref_audio"
SAD       = SysPath("/workspace/SadTalker")

# 4 Pakorn voices: 2 Dawb + 2 Leeg
VOICES = {
    "Kim_tshaj": {
        "name":    "Kim Tshaj (Male · Dawb)",
        "ref":     str(REF_AUDIO / "Kim_tshaj.mp3"),
        "instruct":"You are a helpful assistant. Please speak Hmong (White Hmong RPA).<|endofprompt|>",
    },
    "txeej_txaim": {
        "name":    "Txeej Txaim (Male · Leeg)",
        "ref":     str(REF_AUDIO / "txeej_txaim.mp3"),
        "instruct":"You are a helpful assistant. Please speak Hmong (Green Hmong RPA).<|endofprompt|>",
    },
    "Ntshiab_Li": {
        "name":    "Ntshiab Li (Female · Dawb)",
        "ref":     str(REF_AUDIO / "Ntshiab_Li.mp3"),
        "instruct":"You are a helpful assistant. Please speak Hmong (White Hmong RPA).<|endofprompt|>",
    },
    "Kaj_Siab": {
        "name":    "Kaj Siab (Female · Leeg)",
        "ref":     str(REF_AUDIO / "Kaj_Siab.mp3"),
        "instruct":"You are a helpful assistant. Please speak Hmong (Green Hmong RPA).<|endofprompt|>",
    },
}


class Predictor(BaseModel):
    def setup(self):
        self.engine = CosyVoiceEngine(
            model_dir=MODEL_DIR,
            cosyvoice_dir=COSY_DIR,
            fp16=True,
        )
        print("CosyVoice loaded ✓", flush=True)

    def predict(
        self,
        image: Path = Input(description="Portrait photo (jpg/png) — face must be clearly visible"),
        text: str = Input(description="Hmong text to speak"),
        voice: str = Input(
            description="Speaker voice",
            default="Kim_tshaj",
            choices=list(VOICES.keys()),
        ),
        speed: float = Input(
            description="Speaking speed (0.5–2.0)", default=1.0, ge=0.5, le=2.0
        ),
        face_model_resolution: int = Input(
            description="Face model resolution — 256 (fast) or 512 (sharper)",
            default=256,
            choices=[256, 512],
        ),
        enhancer: bool = Input(
            description="Apply GFPGAN face enhancer for sharper output",
            default=False,
        ),
        still_mode: bool = Input(
            description="Minimize head motion — mainly lips move",
            default=False,
        ),
    ) -> Path:
        text = text.strip()
        if not text:
            raise ValueError("text is required")

        spec = VOICES[voice]
        work_dir = SysPath(tempfile.mkdtemp(prefix="hmong_avatar_"))

        # ── Step 1: CosyVoice TTS ──────────────────────────────────────────
        sample_rate, wav = self.engine.synthesize(
            tts_text=text,
            mode="Instruct2",
            prompt_wav=spec["ref"],
            prompt_text="",
            instruct_text=spec["instruct"],
            text_frontend=False,
            speed=speed,
        )
        audio_path = work_dir / "speech.wav"
        sf.write(str(audio_path), wav, sample_rate, subtype="PCM_16")
        print(f"TTS done → {audio_path}", flush=True)

        # ── Step 2: SadTalker lip-sync ─────────────────────────────────────
        result_dir = work_dir / "result"
        result_dir.mkdir()

        cmd = [
            "python3", "inference.py",
            "--driven_audio",  str(audio_path),
            "--source_image",  str(image),
            "--result_dir",    str(result_dir),
            "--pose_style",    "0",
            "--preprocess",    "crop",
            "--size",          str(face_model_resolution),
            "--batch_size",    "2",
        ]
        if still_mode:
            cmd.append("--still")
        if enhancer:
            cmd.extend(["--enhancer", "gfpgan"])

        subprocess.run(cmd, cwd=str(SAD), check=True, timeout=600)

        mp4s = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("SadTalker did not produce a video")

        out = Path(tempfile.mktemp(suffix=".mp4"))
        SysPath(out).write_bytes(mp4s[0].read_bytes())
        return out
