import os, sys, uuid, tempfile
import numpy as np
import torch
import soundfile as sf
from cog import BasePredictor, Input, Path as CogPath
from pathlib import Path as SysPath

SEED_VC_DIR = "/workspace/seed-vc"
os.chdir(SEED_VC_DIR)
sys.path.insert(0, SEED_VC_DIR)

DEV = torch.device("cuda")
DT  = torch.float16


class Predictor(BasePredictor):
    def setup(self):
        from hydra.utils import instantiate
        from omegaconf import DictConfig
        import yaml

        cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml")))
        self.vc = instantiate(cfg)
        self.vc.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        self.vc.to(DEV)
        self.vc.eval()
        self.vc.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=DT, device=DEV)
        print("Seed-VC loaded ✓", flush=True)

    def predict(
        self,
        source_audio: CogPath = Input(
            description="Audio to convert — the speech you want cloned into the target voice (wav/mp3)"
        ),
        reference_audio: CogPath = Input(
            description="3–10 second reference clip of the target voice (wav/mp3)"
        ),
        diffusion_steps: int = Input(
            description="Quality vs speed — higher = better quality, slower (10–100)",
            default=50, ge=10, le=100,
        ),
        length_adjust: float = Input(
            description="Output length relative to source — 1.0 = same length",
            default=1.0, ge=0.5, le=2.0,
        ),
    ) -> CogPath:
        r = self.vc.convert_voice_with_streaming(
            str(source_audio),
            str(reference_audio),
            diffusion_steps=diffusion_steps,
            length_adjust=length_adjust,
            stream_output=False,
            device=DEV,
            dtype=DT,
        )

        # Handle generator or tuple output
        if hasattr(r, "__iter__") and not isinstance(r, np.ndarray):
            r = list(r)[-1]
        if isinstance(r, tuple):
            r = r[-1]

        wav = np.asarray(r)
        out = CogPath(tempfile.mktemp(suffix=".wav"))
        sf.write(str(out), wav, self.vc.sr)
        return out
