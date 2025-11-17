#!/usr/bin/env python3
"""Poll kernel status until complete or error."""
import subprocess
import re
import time
import sys

def check_status():
    try:
        result = subprocess.run(
            ["poetry", "run", "kaggle", "kernels", "status", "leventecsibi/stable-diffusion-batch-generator"],
            capture_output=True,
            text=True,
            cwd="c:\\Users\\csibi\\Desktop\\imggenhub",
            timeout=30
        )
        if result.returncode == 0:
            match = re.search(r'has status "([^"]+)"', result.stdout)
            return match.group(1) if match else None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    return None

start = time.time()
while True:
    status = check_status()
    elapsed = int(time.time() - start)
    
    if status:
        print(f"[{elapsed}s] {status}")
        if "COMPLETE" in status.upper() or "ERROR" in status.upper():
            sys.exit(0)
    
    if elapsed > 1800:  # 30 min timeout
        print("TIMEOUT", file=sys.stderr)
        sys.exit(1)
    
    time.sleep(60)
