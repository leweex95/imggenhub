import subprocess
import sys

try:
    result = subprocess.run(
        ["poetry", "run", "kaggle", "kernels", "status", "leventecsibi/stable-diffusion-batch-generator"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd="c:\\Users\\csibi\\Desktop\\imggenhub"
    )
    print(result.stdout.strip())
except subprocess.TimeoutExpired:
    print("TIMEOUT")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
