import json
with open('HW11/Part_A/HW11-ICA-PartA_Adam_Nelson-Archer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    print(f"\n--- Cell {i} ({cell.get('cell_type')}) ---")
    print(''.join(cell.get('source', [])))