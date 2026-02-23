import json
from kaggle_connector.utils.cli import run_kaggle_cli
import os

# Download kernel output
r = run_kaggle_cli(['kernels', 'output', 'leventecsibi/imggenhub-generator', '--path', 'C:/temp/t4x2_log2'])
print('Download result:', r.stdout[:500])

log_path = 'C:/temp/t4x2_log2/imggenhub-generator.log'
if os.path.exists(log_path):
    with open(log_path, 'rb') as f:
        raw = f.read()
    print(f'Log size: {len(raw)} bytes')
    if raw:
        data = json.loads(raw)
        print(f'Log entries: {len(data)}')
        for entry in data:
            d = entry.get('data', '')
            if d.strip():
                print(f"t={entry.get('time',0):.2f} [{entry.get('stream_name','')}]: {d}", end='')
                print()
    else:
        print('Log is empty')
else:
    print('No log file found')
    for f in os.listdir('C:/temp/t4x2_log2'):
        print(f'  Found: {f}')
