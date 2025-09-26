import argparse
from ast import List
import json
import subprocess
import os
import pdb
SCENE_DATA_ROOT = "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT"
ROOM_PROMPT = [
"""# Room Design Modification Prompt

## Task Overview
You are an expert interior designer specializing in room modifications. Your task is to analyze a JSON representation of a well-designed master bedroom and suggest specific modifications using standardized commands. The room dimensions are {x}m (x) × {y}m (y) × {z}m (z). The json file is shown below:
```json
{room_json}
```""",

"""## Available Commands
You must use one of the following command formats:

1. **Add an object**
   ```
   add | {object_category} | {x, y, z_position} | {rotation_values}
   ```
   Example: `add | table | -1.31, 1.3, -0.7319 | 0, -0.70711, 0, 0.70711`

2. **Delete an object**
   ```
   delete | {instanceid}
   ```
   Example: `delete | furniture/56`

3. **Swap two objects**
   ```
   swap | {instanceid_1} | {instanceid_2}
   ```
   Example: `swap | furniture/256 | furniture/250`

4. **Reposition/rotate an object**
   ```
   change | {instanceid} | {new_x, new_y, new_z_position} | {new_rotation_values}
   ```
   Example: `change | furniture/30 | -1.7457, 2.3932, -2.7761 | 0, 0, 0, 1`

5. **Replace an object**
   ```
   replace | {instanceid} | {new_object_category}
   ```
   Example: `replace | furniture/56 | table`

## Design Guidelines
- Only modify furniture objects; leave structural elements (walls, floors, ceilings) unchanged
- Maintain realistic room arrangement (avoid object overlaps and placing items outside room boundaries)
- Consider typical master bedroom functionality and aesthetics

Your response should contain only the command in the specified format, with no additional explanations or commentary. Now start to generate five commands."""
]


# OUTPUT_PROMPT = """
# The room is modified according to the prompt. The original json file is shown below:
# ```json
# {room_json}
# ```
# The modification command is "{command}". Now the image of the room after modification is given. The camera of the image is positioned at {camera_position} and looking at {camera_target}.

# Now you need to provide instructions on how to convert the current room back to the original room. The instructions should be natural language. Your response should contain only the instructions, with no additional explanations or commentary. 

# Here are some examples of the instructions:
# - Add a table in the center of the room.
# - Delete the abnormal chair.
# - Swap the table and the chair.
# - Move bed to the left
# - Place the sofa at [1.0, 1.0, 1.0]

# Your response should contain only the instructions, with no additional explanations or commentary.
# """



OUTPUT_PROMPT = [
"""This room is modified from the following json file:
```json
{room_json}
```
The modification command is "{command}". Now the image of the room after modification is given. The camera of the image is positioned at {camera_position} and looking at {camera_target}.
Now you need to provide detailed instructions on how to convert the current room back to the original room. The instructions should be natural language. Your response should contain only the instructions, with no additional explanations or commentary.""",


"""# Room Restoration Task

## Context
You are given:
1. A JSON representation of the modified room state: 
```json
{room_json}
```
2. A JSON representation of the original room state:
```json
{original_room_json}
```
3. A modification command that was applied to this room (`{command}`)
4. An image showing the room after modification
5. The camera position (`{camera_position}`) and target (`{camera_target}`) for the image

## Your Task
Analyze the original room JSON and the modified room image to determine what changes were made. Then provide clear, natural language instructions that would revert the modified room back to its original state.

## Requirements
- Your instructions must be specific and actionable
- Provide only the restoration instructions with no explanations or commentary
- Use natural language commands that describe the necessary actions
- Focus on the differences between the original and modified rooms
- Avoid absolute positions and orientation if possible. Try to use relative positions and orientation.
- Avoid specifying the specific object name. Use the object category instead.

## Examples of Valid Instructions
- "Add a wooden table in the center of the room."
- "Remove the red chair from the corner."
- "Swap the positions of the bookshelf and the desk."
- "Move the bed 2 meters to the left."
- "Rotate the sofa to face the window."
- "Place the lamp at coordinates [1.0, 0.5, 2.0]."

## Output Format
Provide only the restoration instructions as plain text, with no additional commentary, explanations, or reasoning."""
]
def render_room_dataset_improved_mat_of_single_scene(scene_id:str):
    with open(os.path.join(SCENE_DATA_ROOT, f"{scene_id}.json"), "r") as f:
        scene_data = json.load(f)
    num_rooms = len(scene_data["scene"]["room"])
    for room_id in range(num_rooms):
        # room_data = scene_data["scene"]["room"][room_id]
        command = [
            "blenderproc run",
            "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/render_room_dataset_improved_mat.py",
            "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT",
            "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model",
            "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture",
            f"{scene_id}.json",
            "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/resources/cctextures/",
            "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/renderings",
            "--save_scene_as_blend",
            "--room_id",f"{room_id}"
        ]
        print(" ".join(command))
        command = " ".join(command)
        env = os.environ.copy()
        subprocess.run(command, shell=True, env=env)


def get_llm_json_command_prompt(room_json):
   with open(room_json, "r") as f:
      room_json = json.load(f)
   x,y,z = room_json["room_size"]
   prompt = ROOM_PROMPT[0].format(room_json=room_json, x=x, y=y, z=z) +'\n' + ROOM_PROMPT[1]
   print(prompt)
   return prompt

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--modified_json_path", type=str, default="FRONT3D_render/0aa95eb8-4c86-4696-8022-7708bab5448e/data_info.json")
   parser.add_argument("--original_json_path", type=str, default="FRONT3D_render/0aa95eb8-4c86-4696-8022-7708bab5448e/data_info.json")
   parser.add_argument("--command", type=str, default="add | nightstand | -1.6, 0.0, -0.5 | 0, 0, 0, 1")
   parser.add_argument("--pose_json", type=str, default="FRONT3D_render/0aa95eb8-4c86-4696-8022-7708bab5448e/camera_dict.json")
   parser.add_argument("--save_path", type=str, default="prompt.txt")
   parser.add_argument("--view_id", type=int, default=0)
   return parser.parse_args()

def get_llm_final_instruction_prompt():
   args = parse_args()
   with open(args.pose_json, "r") as f:
      pose_dict = json.load(f)
   camera_position = pose_dict["camera_pos"][args.view_id]
   camera_position = [camera_position[0], camera_position[2], camera_position[1]]
   camera_target = pose_dict["camera_lookat"][args.view_id]
   camera_target = [camera_target[0], camera_target[2], camera_target[1]]
   with open(args.modified_json_path, "r") as f:
      room_json = json.load(f)
   with open(args.original_json_path, "r") as f:
      original_room_json = json.load(f)
   prompt = OUTPUT_PROMPT[1].format(room_json=room_json["scene"]["room"][0],
                                    original_room_json=original_room_json["scene"]["room"][0],
                                    command=args.command,
                                    camera_position=camera_position,
                                    camera_target=camera_target)
   print(prompt)
   with open(args.save_path, "w") as f:
      f.write(prompt)
   return prompt

if __name__ == "__main__":
   get_llm_final_instruction_prompt()
   # room_json = "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/room_info.json"
   # prompt = get_llm_json_command_prompt(room_json)
   # with open("command_prompt.txt", "w") as f:
   #    f.write(prompt)
   
   # room_json = "FRONT3D_render/finished/0aa95eb8-4c86-4696-8022-7708bab5448e_04/data_info.json"
   # command = "add | nightstand | -1.6, 0.0, -0.5 | 0, 0, 0, 1"
   # pose_json = "FRONT3D_render/finished/0aa95eb8-4c86-4696-8022-7708bab5448e_modified_00/camera_dict.json"
   # with open(pose_json, "r") as f:
   #    pose_dict = json.load(f)
   # camera_position = pose_dict["camera_pos"][1]
   # camera_position = [camera_position[0], camera_position[2], camera_position[1]]
   # camera_target = pose_dict["camera_lookat"][1]
   # camera_target = [camera_target[0], camera_target[2], camera_target[1]]
   # prompt = get_llm_final_instruction_prompt(room_json, command, camera_position, camera_target)
   # # save the prompt to a file
   # with open("prompt.txt", "w") as f:
   #    f.write(prompt)
   