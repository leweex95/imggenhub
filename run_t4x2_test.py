"""
Run T4x2 test only.
"""
import subprocess
import sys
import time
from pathlib import Path

python = sys.executable

cmd = [
    python, "-m", "imggenhub.kaggle.main",
    "--model_id", "black-forest-labs/FLUX.1-schnell",
    "--accelerator", "nvidia-t4-x2",
    "--gpu",
    "--precision", "bf16",
    "--guidance", "1.5",
    "--steps", "4",
    "--img_width", "1024",
    "--img_height", "1024",
    "--prompt", "portrait of a woman, cinematic lighting, photorealistic",
    "--dest", "t4x2_dual_gpu_test2",
]

print("Running T4x2 dual-GPU test...")
start = time.time()
result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
elapsed = time.time() - start
if result.returncode == 0:
    print(f"\nTest 'T4x2_dual_gpu' PASSED in {elapsed:.0f}s")
else:
    print(f"\nTest 'T4x2_dual_gpu' FAILED (exit {result.returncode}) after {elapsed:.0f}s")
