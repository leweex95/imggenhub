"""
Run enhance_photorealism test and T4x2 test sequentially.
Usage: python run_tests.py
"""
import subprocess
import sys
import time
from pathlib import Path

python = sys.executable

tests = [
    {
        "name": "enhance_photorealism",
        "cmd": [
            python, "-m", "imggenhub.kaggle.main",
            "--model_id", "black-forest-labs/FLUX.1-schnell",
            "--accelerator", "nvidia-p100",
            "--gpu",
            "--precision", "bf16",
            "--guidance", "1.5",
            "--steps", "4",
            "--img_width", "1024",
            "--img_height", "1024",
            "--enhance_photorealism",
            "--prompt", "a woman walking in a park",
            "--dest", "enhance_photorealism_test",
        ],
    },
    {
        "name": "T4x2_dual_gpu",
        "cmd": [
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
            "--dest", "t4x2_dual_gpu_test",
        ],
    },
]

for test in tests:
    print(f"\n{'='*60}")
    print(f"Running test: {test['name']}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(test["cmd"], cwd=str(Path(__file__).parent))
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n✓ Test '{test['name']}' PASSED in {elapsed:.0f}s")
    else:
        print(f"\n✗ Test '{test['name']}' FAILED (exit {result.returncode}) in {elapsed:.0f}s")
        print("Continuing with next test...")

print("\nAll tests complete.")
