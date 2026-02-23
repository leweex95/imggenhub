import json
nb = json.load(open('src/imggenhub/kaggle/notebooks/kaggle-flux-schnell-bf16.ipynb'))
print('Total cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    cid = cell.get('id', '?')
    print(f'=== Cell {i} (id={cid}) len={len(src)} ===')
    print(src)
    print()
