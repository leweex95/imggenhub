import json

with open('src/imggenhub/kaggle/notebooks/kaggle-flux-dual-t4.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = cell.get('source', '')
    code = ''.join(src) if isinstance(src, list) else src
    ctype = cell.get('cell_type', '')
    print(f'=== Cell {i} ({ctype}), {len(code)} chars ===')
    print(code)
    print()
