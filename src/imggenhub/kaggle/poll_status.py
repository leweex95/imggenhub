import json
import subprocess
import time

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"
POLL_INTERVAL = 10  # seconds

while True:
    # Run `kaggle kernels status` and capture output
    result = subprocess.run(
        ["python", "-m", "kaggle.cli", "kernels", "status", KERNEL_ID],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error fetching status:", result.stderr)
        break

    # Parse status from CLI output instead of JSON
    import re
    match = re.search(r'has status "(.*)"', result.stdout)
    status = match.group(1) if match else "unknown"
    print(f"Kernel status: {status}")

    if status.lower() in ["kernelworkerstatus.complete", "kernelworkerstatus.error"]:
        print("Kernel finished with status:", status)
        break

    time.sleep(POLL_INTERVAL)
