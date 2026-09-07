import os, base64, tempfile
import numpy as np
import soundfile as sf
from cog import BasePredictor, Input
from pathlib import Path

MODEL_DIR = "/workspace/hmong_ears_pakorn"
CT2_DIR   = "/workspace/pakorn-ct2"


class Predictor(BasePredictor):
    def setup(self):
        from huggingface_hub import snapshot_download
        import ctranslate2

        # Download Pakorn's Hmong Whisper model if not present
        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            print("Downloading Hmong Whisper model...", flush=True)
            snapshot_download(
                repo_id="Pakorn2112/whisper-model-large-hmong-multi-speech",
                local_dir=MODEL_DIR,
            )

        # Convert to CTranslate2 format if needed
        if not os.path.exists(os.path.join(CT2_DIR, "model.bin")):
            print("Converting to fast format...", flush=True)
            from ctranslate2.converters import TransformersConverter
            conv = TransformersConverter(
                MODEL_DIR, copy_files=["preprocessor_config.json"]
            )
            conv.convert(CT2_DIR, quantization="int8", force=True)
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(MODEL_DIR).save_pretrained(CT2_DIR)

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute = "int8_float16" if device == "cuda" else "int8"

        from faster_whisper import WhisperModel
        self.model = WhisperModel(CT2_DIR, device=device, compute_type=compute)
        print(f"Hmong STT ready on {device} ✓", flush=True)

    def run(
        self,
        audio: CogPath = Input(description="Hmong audio to transcribe (wav/mp3/m4a)"),
        dialect: str = Input(
            description="Dialect hint (both models are the same Pakorn multi-speech model)",
            default="dawb",
            choices=["dawb", "leeg"],
        ),
    ) -> str:
        # Load audio
        data, sr = sf.read(str(audio), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            target_len = int(len(data) * 16000 / sr)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            )

        # Skip near-silent
        if float(np.sqrt(np.mean(data**2))) < 0.008:
            return ""

        CHUNK = 15 * 16000
        pieces = []
        for start in range(0, len(data), CHUNK):
            part = data[start:start + CHUNK]
            if len(part) < 1600:
                continue
            segments, _ = self.model.transcribe(
                part,
                beam_size=1,
                language=None,
                task="transcribe",
                condition_on_previous_text=False,
                vad_filter=True,
            )
            kept = [
                s.text.strip() for s in segments
                if s.no_speech_prob <= 0.5
                and s.compression_ratio <= 2.2
                and s.avg_logprob >= -1.0
            ]
            pieces.append(" ".join(kept).strip())

        return " ".join(p for p in pieces if p).strip()
