import argparse
import json

from cmd_utils import ModifyRoomJson

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/data_info.json")
    parser.add_argument("--command_line", "-c", type=str, default="add | nightstand | -1.6, 0.0, -0.5 | 0, 0, 0, 1")
    return parser.parse_args() 


def main():
    args = parse_args()
    json_path = args.json_path
    command_line = args.command_line
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
    # generate_prompt()
    main()
    