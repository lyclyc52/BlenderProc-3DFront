import json
import os
import pdb


def save_room_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = data['scene']['room'][0]
    with open(json_path.replace('data_info.json', 'room_info.json'), 'w') as f:
        json.dump(data, f, indent=4)

def extract_room_info(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    floors, ceilings = [], []
    for obj in data['mesh']:
        if 'floor' in obj['type'].lower():
            floors.append(obj)
        if 'ceiling' in obj['type'].lower():
            ceilings.append(obj)
    pdb.set_trace()
    x_min, y_min, z_min, x_max, y_max, z_max = 0, 0, 0, 0, 0, 0
    for floor in floors:
        for i in range(0, len(floor['xyz']), 3):
            x, y, z = floor['xyz'][i], floor['xyz'][i+1], floor['xyz'][i+2]
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            z_min = min(z_min, z)
            x_max = max(x_max, x)
            z_max = max(z_max, z)
    for ceiling in ceilings:
        for i in range(0, len(ceiling['xyz']), 3):
            x, y, z = ceiling['xyz'][i], ceiling['xyz'][i+1], ceiling['xyz'][i+2]
            y_min = min(y_min, y)
            y_max = max(y_max, y)
    return x_min, y_min, z_min, x_max, y_max, z_max

if __name__ == "__main__":
    json_path = "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/renderings/6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9_room_01/data_info.json"
    extract_room_info(json_path)