import subprocess, tempfile
from pathlib import Path
from cog import BasePredictor, Input


SAD = Path("/workspace/SadTalker")


class Predictor(BasePredictor):
    def setup(self):
        print("SadTalker ready", flush=True)

    def predict(
        self,
        image: Path = Input(description="Portrait photo (jpg/png)"),
        audio: Path = Input(description="Driving audio (wav/mp3)"),
        pose_style: int = Input(default=0, ge=0, le=45),
        face_model_resolution: int = Input(default=256, choices=[256, 512]),
        preprocess: str = Input(default="crop", choices=["crop", "resize", "full"]),
        enhancer: bool = Input(default=False),
        still_mode: bool = Input(default=False),
    ) -> Path:
        job_dir = Path(tempfile.mkdtemp())
        cmd = [
            "python3", "inference.py",
            "--driven_audio", str(audio),
            "--source_image", str(image),
            "--result_dir", str(job_dir),
            "--pose_style", str(pose_style),
            "--preprocess", preprocess,
            "--size", str(face_model_resolution),
            "--batch_size", "2",
        ]
        if still_mode:
            cmd.append("--still")
        if enhancer:
            cmd.extend(["--enhancer", "gfpgan"])
        subprocess.run(cmd, cwd=str(SAD), check=True, timeout=600)
        mp4s = sorted(job_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("SadTalker did not produce a video")
        out = Path(tempfile.mktemp(suffix=".mp4"))
        out.write_bytes(mp4s[0].read_bytes())
        return out
