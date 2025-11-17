import subprocess
import time
import sys
from datetime import datetime

start = time.time()
while True:
    result = subprocess.run(
        ["poetry", "run", "kaggle", "kernels", "status", "leventecsibi/stable-diffusion-batch-generator"],
        capture_output=True,
        text=True,
        cwd="c:\\Users\\csibi\\Desktop\\imggenhub"
    )
    
    elapsed = int(time.time() - start)
    timestamp = datetime.now().isoformat()
    
    if "RUNNING" in result.stdout:
        status = "RUNNING"
    elif "COMPLETE" in result.stdout:
        status = "COMPLETE"
        print(f"[{timestamp}] [{elapsed}s] ✓ COMPLETE")
        with open("c:\\Users\\csibi\\Desktop\\imggenhub\\kernel_status.txt", "w") as f:
            f.write(f"COMPLETE after {elapsed}s\n")
        sys.exit(0)
    elif "ERROR" in result.stdout:
        status = "ERROR"
        print(f"[{timestamp}] [{elapsed}s] ✗ ERROR")
        with open("c:\\Users\\csibi\\Desktop\\imggenhub\\kernel_status.txt", "w") as f:
            f.write(f"ERROR after {elapsed}s\n")
        sys.exit(1)
    else:
        status = "UNKNOWN"
    
    print(f"[{timestamp}] [{elapsed}s] {status}")
    
    if elapsed > 1800:
        print(f"[{timestamp}] TIMEOUT")
        sys.exit(2)
    
    time.sleep(90)  # Check every 90 seconds
