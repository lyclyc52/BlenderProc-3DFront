import argparse
import json
import os

from json_utils import save_room_json, extract_room_info
from cmd_utils import ModifyRoomJson
from prompt_config import get_llm_prompt


def parse_args_generate_prompt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/data_info.json")
    return parser.parse_args()


def generate_prompt():
    args = parse_args_generate_prompt()
    json_path = args.json_path
    x, y, z = extract_room_info(json_path)
    extra_data = {
        "room_size": [x, y, z]
    }
    print(f"room size: x: {x}, y: {y}, z: {z}")
    save_room_json(json_path, extra_data)
    room_json = json_path.replace('data_info.json', 'room_info.json')
    data_generated_by_llm = get_llm_prompt(room_json, [x, y, z])
    print(data_generated_by_llm)
    prompt_path = json_path.replace('data_info.json', 'prompt.txt')
    with open(prompt_path, 'w') as f:
        f.write(data_generated_by_llm)
    
    
def parse_args_modify_room_json():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/data_info.json")
    parser.add_argument("--command_line", "-c", type=str, default="add | nightstand | -1.6, 0.0, -0.5 | 0, 0, 0, 1")
    return parser.parse_args() 


def modify_room_json(args):
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
    generate_prompt()
    # modify_room_json()
    