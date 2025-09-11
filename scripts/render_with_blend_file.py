# python cli.py run ./scripts/utils.py 
import blenderproc as bproc

"""
    Example commands:

        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode plan
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode overview 
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -ppo 10 -gd 0.15
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -ppo 0 -gp 5
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -nc

"""

from random import shuffle
import shutil
import sys
# sys.path.append('/home/jhuangce/miniconda3/lib/python3.9/site-packages')
# sys.path.append('/home/yliugu/BlenderProc/scripts')
sys.path.append('./scripts')
import cv2
import os
from os.path import join
import numpy as np

import imageio
import sys
# sys.path.append('/data/jhuangce/BlenderProc/scripts')
# sys.path.append('/data2/jhuangce/BlenderProc/scripts')
from floor_plan import *
from load_helper import *
from render_configs import *
from json_utils import extract_room_info, save_room_json
from utils import build_and_save_scene_cache
from bbox_proj import get_aabb_coords, project_aabb_to_image, project_obb_to_image
import json
from typing import List
from os.path import join
import glob
import argparse
from mathutils import Vector, Matrix

import pandas as pd
import pdb
# from seg import build_segmentation_map, build_metadata


pi = np.pi
cos = np.cos
sin = np.sin
COMPUTE_DEVICE_TYPE = "CUDA"


def construct_scene_list():
    """ Construct a list of scenes and save to SCENE_LIST global variable. """
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        SCENE_LIST.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(SCENE_LIST)} scenes in total")

def check_pos_valid(pos, room_objs_dict, room_bbox):
    """ Check if the position is in the room, not too close to walls and not conflicting with other objects. """
    room_bbox_small = [[item+0.3 for item in room_bbox[0]], [room_bbox[1][0]-0.3, room_bbox[1][1]-0.3, room_bbox[1][2]-0.3]] # ceiling is lower
    if not pos_in_bbox(pos, room_bbox_small):
        print("Position is not in the room")
        return False
    for obj_dict in room_objs_dict['objects']:
        obj_bbox = obj_dict['aabb']
        if pos_in_bbox(pos, obj_bbox):
            print("Position is conflicting with other objects")
            return False

    return True

############################## poses generation ##################################

def normalize(x, axis=-1, order=2):
    l2 = np.linalg.norm(x, order, axis)
    l2 = np.expand_dims(l2, axis)
    l2[l2 == 0] = 1
    return x / l2,

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = np.zeros_like(camera_position)
    else:
        at = np.array(at)
    if up is None:
        up = np.zeros_like(camera_position)
        up[2] = -1
    else:
        up = np.array(up)
    
    z_axis = normalize(camera_position - at)[0]
    x_axis = normalize(np.cross(up, z_axis))[0]
    y_axis = normalize(np.cross(z_axis, x_axis))[0]

    R = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def c2w_from_loc_and_at(cam_pos, at, up=(0, 0, 1)):
    """ Convert camera location and direction to camera2world matrix. """
    c2w = np.eye(4)
    cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
    c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
    return c2w

def generate_four_corner_poses(room_bbox_min, room_bbox_max, room_objs_dict, room_bbox):
    """ Return a list of matrices of 4 corner views in the room. """
    bbox_xy = [[room_bbox_min[0], room_bbox_min[2]], [room_bbox_max[0], room_bbox_max[2]]]
    corners = [[i+0.5 for i in bbox_xy[0]], [i-0.5 for i in bbox_xy[1]]]
    x1, y1, x2, y2 = corners[0][0], corners[0][1], corners[1][0], corners[1][1]
    at = [(x1+x2)/2, (y1+y2)/2, 1.2]
    locs = [[x1, y1, 2], [x1, y2, 2], [x2, y1, 2], [x2, y2, 2]]
    camera_dict = {
        "c2w": [],
        "camera_pos": [],
        "camera_lookat": []
    }
    for pos in locs:
        cam2world_matrix = c2w_from_loc_and_at(pos, at)
        count = 0
        while not check_pos_valid(pos, room_objs_dict, room_bbox):
            pdb.set_trace()
            pos = [pos[0] + 0.1, pos[1] + 0.1, pos[2]]
            cam2world_matrix = c2w_from_loc_and_at(pos, at)
            count += 1
            if count >= 10:
                break
        if count < 10:
            camera_dict["c2w"].append(cam2world_matrix)
            camera_dict["camera_pos"].append(pos)
            camera_dict["camera_lookat"].append(at)
    return camera_dict

def pos_in_bbox(pos, bbox):
    """
    Check if a point is inside a bounding box.
    Input:
        pos: 3 x 1
        bbox: 2 x 3
    Output:
        True or False
    """
    return  pos[0] >= bbox[0][0] and pos[0] <= bbox[1][0] and \
            pos[1] >= bbox[0][1] and pos[1] <= bbox[1][1] and \
            pos[2] >= bbox[0][2] and pos[2] <= bbox[1][2]

# For metadata and 2D/3D mask generation
def filter_room_objects(scene_idx, room_idx, room_objs):
    for obj in room_objs:
        obj.set_cp('instance_name', obj.get_name())
        obj.set_cp('instance_id', 0)

    if 'merge_list' in ROOM_CONFIG[scene_idx][room_idx]:
        merge_dict = ROOM_CONFIG[scene_idx][room_idx]['merge_list']
        for merged_label, merge_items in merge_dict.items():
            # select objs to be merged
            objs_to_be_merged = [obj for obj in room_objs if obj.get_name() in merge_items]
            for obj in objs_to_be_merged:
                obj.set_cp('instance_name', merged_label)

    result_objects = []
    for obj in room_objs:
        obj_name = obj.get_cp('instance_name')
        flag_use = True
        # check global OBJ_BAN_LIST
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                flag_use=False
        # check keyword_ban_list
        if 'keyword_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for ban_word in ROOM_CONFIG[scene_idx][room_idx]['keyword_ban_list']:
                if ban_word in obj_name:
                    flag_use=False
        # check fullname_ban_list
        if 'fullname_ban_list' in ROOM_CONFIG[scene_idx][room_idx].keys():
            for fullname in ROOM_CONFIG[scene_idx][room_idx]['fullname_ban_list']:
                if fullname == obj_name.strip():
                    flag_use=False
        
        if flag_use:
            result_objects.append(obj)

    id_map = {}
    for obj in result_objects:
        obj_name = obj.get_cp('instance_name')
        if obj_name not in id_map:
            id_map[obj_name] = len(id_map) + 1
        obj.set_cp('instance_id', id_map[obj_name])
    
    return result_objects, id_map


def render_poses(poses, temp_dir=RENDER_TEMP_DIR) -> List:
    """ Render a scene with a list of poses. 
        No room idx is needed because the poses can be anywhere in the room. """

    # add camera poses to render queue
    for cam2world_matrix in poses:
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    # render
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200, transmission_bounces=200, transparent_max_bounces=200)
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)
    data = bproc.renderer.render(output_dir=temp_dir)
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data['colors']]

    return imgs

###########################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-j', '--json_path', type=str, default=None)
    parser.add_argument('-s', '--scene_idx', type=int, default=None)
    parser.add_argument('-r', '--room_idx', type=int, default=None)
    parser.add_argument('-n', '--save_name', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['plan', 'overview', 'render', 'bbox', 'seg', 'depth'], 
                        help="plan: Generate the floor plan of the scene. \
                              overview:Generate 4 corner overviews with bbox projected. \
                              render: Render images in the scene. \
                              bbox: Overwrite bboxes by regenerating transforms.json."
                              "\nseg: Create 3D semantic/instance segmentation map.")
    parser.add_argument('-ppo', '--pos_per_obj', type=int, default=15, help='Number of close-up poses for each object.')
    parser.add_argument('-gp', '--max_global_pos', type=int, default=150, help='Max number of global poses.')
    parser.add_argument('-gd', '--global_density', type=float, default=0.15, help='The radius interval of global poses. Smaller global_density -> more global views')
    parser.add_argument('-nc', '--no_check', action='store_true', default=False, help='Do not check the poses. Render directly.')
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--relabel', action='store_true', help='Relabel the objects in the scene by rewriting transforms.json.')
    parser.add_argument('--rotation', action='store_true', help = 'output rotation bounding boxes if it is true.')
    parser.add_argument('--bbox_type', type=str, default="aabb", choices=['aabb', 'obb'], help='Output aabb or obb')
    parser.add_argument('--render_root', type=str, default='./FRONT3D_render', help='Output directory. If not specified, use the default directory.')

    parser.add_argument('--seg_res', type=int, default=256, help='The max grid resolution for 3D segmentation map.')
    parser.add_argument('--pose_dir', type=str, default='', 
                        help='The directory containing the poses (transforms.json) for 2D mask rendering.')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



    return parser.parse_args()

def check_args(args):
    if args.json_path is not None or (args.scene_idx is not None or args.room_idx is not None):
        return
    else:
        raise ValueError("Either json_path or scene_idx and room_idx must be provided")

def get_scene_bbox(loaded_objects=None, scene_objs_dict=None):
    """ Return the bounding box of the scene. """
    bbox_mins = []
    bbox_maxs = []
    if loaded_objects!=None:
        for i, object in enumerate(loaded_objects):
            bbox = object.get_bound_box()
            bbox_mins.append(np.min(bbox, axis=0))
            bbox_maxs.append(np.max(bbox, axis=0))
            scene_min = np.min(bbox_mins, axis=0)
            scene_max = np.max(bbox_maxs, axis=0)
            return scene_min, scene_max
    elif scene_objs_dict!=None:
        return scene_objs_dict['bbox']
    else:
        raise ValueError('Either loaded_objects or scene_objs_dict should be provided.')


def get_room_bbox(json, scene_objects=None, scene_objs_dict=None):
    """ Return the bounding box of the room. """
    # get global height
    scene_min, scene_max = get_scene_bbox(scene_objects, scene_objs_dict)
    room_config = ROOM_CONFIG[scene_idx][room_idx]
    # overwrite width and length with room config
    scene_min[:2] = room_config['bbox'][0]
    scene_max[:2] = room_config['bbox'][1]

    return [scene_min, scene_max]

def bbox_contained(bbox_a, bbox_b):
    """ Return whether the bbox_a is contained in bbox_b. """
    return bbox_a[0][0]>=bbox_b[0][0] and bbox_a[0][1]>=bbox_b[0][1] and bbox_a[0][2]>=bbox_b[0][2] and \
           bbox_a[1][0]<=bbox_b[1][0] and bbox_a[1][1]<=bbox_b[1][1] and bbox_a[1][2]<=bbox_b[1][2]

def get_room_objs_dict(room_bbox, scene_objs_dict):
    """ Get the room object dictionary containing all the objects in the room. """
    room_objects = []
    scene_objects = scene_objs_dict['objects']
    for obj_dict in scene_objects:
        if bbox_contained(obj_dict['aabb'], room_bbox):
            room_objects.append(obj_dict)

    room_objs_dict = {}
    room_objs_dict['bbox'] = np.array(room_bbox)
    room_objs_dict['objects'] = room_objects
    
    return room_objs_dict

def merge_bbox_in_dict(json_path, room_objs_dict):
    """ Merge the bounding box of the room. Operate on the object dictionary with obb """
    if 'merge_list' in ROOM_CONFIG[scene_idx][room_idx]:
        merge_dict = ROOM_CONFIG[scene_idx][room_idx]['merge_list']
        objects = room_objs_dict['objects']
        for merged_label, merge_items in merge_dict.items():
            # select objs to be merged
            result_objects = [obj for obj in objects if obj['name'] not in merge_items]
            objs_to_be_merged = [obj for obj in objects if obj['name'] in merge_items]

            # find the largest object
            largest_obj = None
            largest_vol = 0
            for obj in objs_to_be_merged:
                if obj['volume'] > largest_vol:
                    largest_vol = obj['volume']
                    largest_obj = obj
            
            # extend the largest bbox to include all the other bbox
            local2world = Matrix(largest_obj['l2w'])
            local_maxs, local_mins = np.max(largest_obj['coords_local'], axis=0), np.min(largest_obj['coords_local'], axis=0)
            local_cent = (local_maxs + local_mins) / 2
            global_cent = local2world @ Vector(local_cent)
            h_diag = (local_maxs - local_mins) / 2
            local_vecs = np.array([[h_diag[0], 0, 0], [0, h_diag[1], 0], [0, 0, h_diag[2]]]) + local_cent  # (3, 3)
            global_vecs = [(local2world @ Vector(vec) - local2world @ Vector(local_cent)).normalized() for vec in local_vecs] # (3, 3)
            global_norms = [vec for vec in global_vecs] # (3, 3)
            local_offsets = np.array([-h_diag, h_diag]) # [[x-, y-, z-], [x+, y+, z+]]

            for obj in objs_to_be_merged:
                update = [[0, 0, 0], [0, 0, 0]]
                for point in obj['coords']:
                    for i in range(3):
                        offset = (Vector(point) - global_cent) @ global_norms[i]
                        if offset < local_offsets[0][i]:
                            local_offsets[0][i] = offset
                            update[0][i] = 1
                        elif offset > local_offsets[1][i]:
                            local_offsets[1][i] = offset
                            update[1][i] = 1
            
            # TODO: update: coords, aabb, volume, coords_local
            # TODO: Compute real aabb by creating parent object
            merged_local_mins, merged_local_maxs = local_offsets + local_cent
            merged_coords_local = get_aabb_coords(np.concatenate([merged_local_mins, merged_local_maxs], axis=0))[:, :3]
            merged_coords = np.array([local2world @ Vector(cord) for cord in merged_coords_local])
            merged_aabb_mins, merged_aabb_maxs = np.min(merged_coords, axis=0), np.max(merged_coords, axis=0)
            merged_aabb = np.array([merged_aabb_mins, merged_aabb_maxs])
            merged_diag_local = merged_local_maxs - merged_local_mins
            merged_volume = merged_diag_local[0] * merged_diag_local[1] * merged_diag_local[2]


            merged_object = {'name': merged_label,
                             'coords': merged_coords,
                             'aabb': merged_aabb,
                             'volume': merged_volume,
                             'l2w': largest_obj['l2w'],
                             'coords_local': merged_coords_local,}
            result_objects.append(merged_object)
            objects = result_objects

        room_objs_dict['objects'] = objects

    return room_objs_dict

def filter_objs_in_dict(json_path, room_objs_dict):
    """ Clean up objects according to merge_list, global OBJ_BAN_LIST, keyword_ban_list, and fullname_ban_list. """

    # check merge_list

    ori_objects = room_objs_dict['objects']
    result_objects = []
    for obj_dict in ori_objects:
        obj_name = obj_dict['name']
        flag_use = True
        # check global OBJ_BAN_LIST
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                flag_use=False
        
        if flag_use:
            result_objects.append(obj_dict)
    
    room_objs_dict['objects'] = result_objects

    return room_objs_dict


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dst_dir = join(args.render_root, f"{args.save_name}")
    os.makedirs(dst_dir, exist_ok=True)

    construct_scene_list()
    check_args(args)

    cache_dir = f'./cached/{args.scene_idx}'

    # load objects
    bproc.init(compute_device='cuda:0', compute_device_type=COMPUTE_DEVICE_TYPE)
    json_path = SCENE_LIST[args.scene_idx] if args.scene_idx is not None else args.json_path
    scene_objects, data_info = load_scene_objects_with_improved_mat(json_path, args.room_idx)
    room_bbox, room_bbox_min, room_bbox_max = extract_room_info(data_info)
    extra_data = {
        "room_size": room_bbox,
        "room_bbox_min": room_bbox_min,
        "room_bbox_max": room_bbox_max
    }
    data_info['scene']['room'][0].update(extra_data)

    scene_objs_dict = build_and_save_scene_cache(scene_objects = scene_objects)

    room_bbox = [[room_bbox_min[0], room_bbox_min[2], room_bbox_min[1]], [room_bbox_max[0], room_bbox_max[2], room_bbox_max[1]]]
    room_objs_dict = get_room_objs_dict(room_bbox, scene_objs_dict)
    room_objs_dict = filter_objs_in_dict(json_path, room_objs_dict)

    


    overview_dir = os.path.join(dst_dir, 'overview')
    os.makedirs(overview_dir, exist_ok=True)
    camera_dict = generate_four_corner_poses(room_bbox_min, room_bbox_max, room_objs_dict, room_bbox)
    poses = camera_dict["c2w"]
    camera_dict["c2w"] = [i.tolist() for i in camera_dict["c2w"]]

    cache_dir = join(dst_dir, 'overview/raw')
    cached_img_paths = glob.glob(cache_dir+'/*')
    imgs = []
    # convert camera_dict to json
    if not os.path.exists(join(dst_dir, 'data_info.json')):
        with open(join(dst_dir, 'data_info.json'), 'w') as f:
            json.dump(data_info, f, indent=4)
    with open(join(dst_dir, 'camera_dict.json'), 'w') as f:
        json.dump(camera_dict, f, indent=4)

    if len(cached_img_paths) > 0 and True:
        # use cached overview images
        for img_path in sorted(cached_img_paths):
            imgs.append(cv2.imread(img_path))
    else:
        # render overview images
        imgs = render_poses(poses, overview_dir)
        os.makedirs(cache_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            cv2.imwrite(join(cache_dir, f'raw_{i}.jpg'), img)

if __name__ == '__main__':
    main()
    print("Success.")