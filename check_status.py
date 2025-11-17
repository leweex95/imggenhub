#!/usr/bin/env python3
"""Simple one-shot status checker for Kaggle kernel."""
import subprocess
import re
import sys

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
        status = match.group(1) if match else "UNKNOWN"
        print(f"Status: {status}")
        sys.exit(0)
    else:
        print(f"Error: {result.stderr}")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
