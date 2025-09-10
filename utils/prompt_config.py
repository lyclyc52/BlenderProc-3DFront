import argparse
from ast import List
import json
import subprocess
import os
import pdb
SCENE_DATA_ROOT = "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT"
ROOM_PROMPT = [
"""
# Room Design Modification Prompt

## Task Overview
You are an expert interior designer specializing in room modifications. Your task is to analyze a JSON representation of a well-designed master bedroom and suggest specific modifications using standardized commands. The room dimensions are {x}m (x) × {y}m (y) × {z}m (z). The json file is shown below:
```json
{room_json}
```
""",
"""
## Available Commands
You must use one of the following command formats:

1. **Add an object**
   ```
   add | {object_category} | {x, y, z_position} | {rotation_values}
   ```
   Example: `add | table | -1.31, 1.3, -0.7319 | 0, -0.70711, 0, 0.70711`

2. **Delete an object**
   ```
   delete | {object_reference_id}
   ```
   Example: `delete | 6920/model`

3. **Swap two objects**
   ```
   swap | {object_reference_id_1} | {object_reference_id_2}
   ```
   Example: `swap | 7882/model | 26791533265579077/1`

4. **Reposition/rotate an object**
   ```
   change | {object_reference_id} | {new_x, new_y, new_z_position} | {new_rotation_values}
   ```
   Example: `change | 7a699d4f-6614-42ba-9477-a75c8f4f03dd/28805838 | -1.7457, 2.3932, -2.7761 | 0, 0, 0, 1`

5. **Replace an object**
   ```
   replace | {object_reference_id} | {new_object_category}
   ```
   Example: `replace | 7857/model | table`

## Design Guidelines
- Only modify furniture objects; leave structural elements (walls, floors, ceilings) unchanged
- Maintain realistic room arrangement (avoid object overlaps and placing items outside room boundaries)
- Consider typical master bedroom functionality and aesthetics

Your response should contain only the command in the specified format, with no additional explanations or commentary.
"""
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


def get_llm_prompt(room_json, room_size):
   x,y,z = room_size
   with open(room_json, "r") as f:
      room_json = json.load(f)
   prompt = ROOM_PROMPT[0].format(room_json=room_json, x=x, y=y, z=z) + ROOM_PROMPT[1]
   print(prompt)
   return prompt

if __name__ == "__main__":
   room_json = "/mnt/afs/liuyichen/repo/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/renderings/0aa95eb8-4c86-4696-8022-7708bab5448e_room_04/room_info.json"
   room_size = [10, 10, 10]
   get_llm_prompt(room_json, room_size)