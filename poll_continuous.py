import subprocess
import re
import time
import logging
from datetime import datetime

KERNEL_ID = "leventecsibi/stable-diffusion-batch-generator"
POLL_INTERVAL = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def check_status():
    try:
        result = subprocess.run(
            ["poetry", "run", "kaggle", "kernels", "status", KERNEL_ID],
            capture_output=True,
            text=True,
            cwd="c:\\Users\\csibi\\Desktop\\imggenhub",
            timeout=30
        )
        if result.returncode == 0:
            match = re.search(r'has status "([^"]+)"', result.stdout)
            return match.group(1) if match else None
    except Exception as e:
        logging.error(f"Error: {e}")
    return None

logging.info(f"Polling kernel {KERNEL_ID}")
start = time.time()

while True:
    status = check_status()
    elapsed = int(time.time() - start)
    
    if status:
        logging.info(f"[{elapsed}s] Status: {status}")
        
        if "COMPLETE" in status.upper():
            logging.info("✓ Kernel COMPLETED successfully!")
            break
        elif "ERROR" in status.upper():
            logging.error("✗ Kernel FAILED!")
            break
    else:
        logging.warning(f"[{elapsed}s] Could not determine status")
    
    if elapsed > 1800:  # 30 minute timeout
        logging.error("Polling timeout exceeded")
        break
    
    time.sleep(POLL_INTERVAL)

logging.info("Polling finished")
