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

class ModifyRoomJson:
    def change(self, data, command_args):
        # delete the space in the command_args
        obj_name = command_args[0].replace(' ', '')
        pos = command_args[1].replace(' ', '').split(',')
        rot = command_args[2].replace(' ', '').split(',')
        for obj in data['scene']['room'][0]['children']:
            if obj['ref'] == obj_name:
                obj['pos'] = [float(p) for p in pos]
                obj['rot'] = [float(r) for r in rot]
        return data

def modify_room_json(json_path, command_line: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    command_args = command_line.split('|')
    action = command_args[0].replace(' ', '')
    modify_room_json = ModifyRoomJson()
    # modify the data according to the action
    method = getattr(modify_room_json, action)
    data = method(data, command_args[1:])
    new_json_path = json_path.replace('data_info.json', 'data_info_modified.json')
    with open(new_json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    json_path = "examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/data_info.json"
    save_room_json(json_path)
    x, y, z = extract_room_info(json_path)
    print(x, y, z)
    # modify_room_json(json_path, "change | 6920/model | -0.2114, 0.0, -0.3726 | 0, 0, 0, 1")