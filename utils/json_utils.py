import json

from cmd_utils import ModifyRoomJson

MODEL_INFO_PATH = "examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model"


def save_room_json(json_path, extra_data = None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    data = data['scene']['room'][0]
    if extra_data is not None:
        data.update(extra_data)
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
    x, y, z = (x_max - x_min), (y_max - y_min), (z_max - z_min)
    print(f"room size: {x}, {y}, {z}")
    print(f"range: x: [{x_min}, {x_max}], y: [{y_min}, {y_max}], z: [{z_min}, {z_max}]")
    return x, y, z


