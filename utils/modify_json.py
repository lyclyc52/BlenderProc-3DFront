import argparse
import json
import os
from cmd_utils import ModifyRoomJson

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/data_info.json")
    parser.add_argument("--save_path", type=str, default="FRONT3D_render/0aa95eb8-4c86-4696-8022-7708bab5448e/0.josn")
    parser.add_argument("--command_line", "-c", type=str, default="add | nightstand | -1.6, 0.0, -0.5 | 0, 0, 0, 1")
    return parser.parse_args() 


def main():
    args = parse_args()
    json_path = args.json_path
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    command_line = args.command_line
    if args.command_line == "":
        os.system(f"cp {json_path} {args.save_path}")
        return
    print('command_line', command_line)
    with open(json_path, 'r') as f:
        data = json.load(f)
    command_args = command_line.split('|')
    action = command_args[0].replace(' ', '')
    modify_room_json = ModifyRoomJson()
    # modify the data according to the action
    method = getattr(modify_room_json, action)
    data = method(data, command_args[1:])
    with open(args.save_path, 'w') as f:
        json.dump(data, f, indent=4)
    print('save_path', args.save_path)

if __name__ == "__main__":
    # generate_prompt()
    main()
    