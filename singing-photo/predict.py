import subprocess, tempfile
from pathlib import Path
from cog import BasePredictor, Input


SAD = Path("/workspace/SadTalker")


def extract_vocals(source: Path, work_dir: Path) -> Path:
    try:
        demucs_out = work_dir / "demucs"
        subprocess.run(
            ["python3", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
             "-o", str(demucs_out), str(source)],
            check=True, capture_output=True, timeout=600,
        )
        vocals = list(demucs_out.rglob("vocals.wav"))
        if vocals:
            return vocals[0]
    except Exception as e:
        print(f"Demucs failed, using original: {e}", flush=True)
    return source


class Predictor(BasePredictor):
    def setup(self):
        print("SadTalker (singing) ready", flush=True)

    def predict(
        self,
        image: Path = Input(description="Portrait photo (jpg/png)"),
        audio: Path = Input(description="Song audio (wav/mp3)"),
        separate_vocals: bool = Input(default=True),
        pose_style: int = Input(default=0, ge=0, le=45),
        face_model_resolution: int = Input(default=512, choices=[256, 512]),
        enhancer: bool = Input(default=True),
        expression_scale: float = Input(default=0.7, ge=0.1, le=2.0),
    ) -> Path:
        job_dir = Path(tempfile.mkdtemp())
        source = Path(str(audio))
        sync_audio = extract_vocals(source, job_dir) if separate_vocals else source

        cmd = [
            "python3", "inference.py",
            "--driven_audio", str(sync_audio),
            "--source_image", str(image),
            "--result_dir", str(job_dir),
            "--pose_style", str(pose_style),
            "--preprocess", "crop",
            "--size", str(face_model_resolution),
            "--batch_size", "2",
            "--expression_scale", str(expression_scale),
        ]
        if enhancer:
            cmd.extend(["--enhancer", "gfpgan"])

        subprocess.run(cmd, cwd=str(SAD), check=True, timeout=600)

        mp4s = sorted(job_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("SadTalker did not produce a video")

        # Swap in full original song audio
        merged = mp4s[0]
        out_tmp = job_dir / "singing_photo.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(merged), "-i", str(source),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-shortest", str(out_tmp)],
            check=True, capture_output=True, timeout=300,
        )
        out = Path(tempfile.mktemp(suffix=".mp4"))
        out.write_bytes(out_tmp.read_bytes())
        return out
