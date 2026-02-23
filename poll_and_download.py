import sys
import time

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

from kaggle_connector import JobManager, SelectiveDownloader
from pathlib import Path

kernel_id = 'leventecsibi/imggenhub-generator'

jm = JobManager(kernel_id)

while True:
    status = jm.get_status()
    ts = time.strftime('%H:%M:%S')
    print(f'{ts} status: {status}', flush=True)
    if 'complete' in status or 'error' in status:
        break
    time.sleep(60)

print(f'Kernel finished with status: {status}', flush=True)

if 'complete' in status:
    dest = Path('outputs/p100_bf16_lora_result')
    dest.mkdir(parents=True, exist_ok=True)
    downloader = SelectiveDownloader(kernel_id)
    downloader.download_images(
        dest_path=str(dest),
        expected_image_count=1,
        stable_count_patience=4,
        polling_interval=60
    )
    print(f'Download done. Files in {dest}:', flush=True)
    for f in dest.rglob('*'):
        if f.is_file():
            print(f'  {f.name}: {f.stat().st_size} bytes', flush=True)
