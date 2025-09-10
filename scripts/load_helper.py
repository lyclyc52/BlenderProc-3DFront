import numpy as np
import os

import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
import re
import random
random.seed(0)
import bpy
import json

# LAYOUT_DIR = '/data/yliugu/3D-FRONT'
# TEXTURE_DIR = '/data/yliugu/3D-FRONT-texture'
# MODEL_DIR = '/data/yliugu/3D-FUTURE-model'

LAYOUT_DIR = 'examples/datasets/front_3d_with_improved_mat/3D-FRONT'
TEXTURE_DIR = 'examples/datasets/front_3d_with_improved_mat/3D-FRONT-texture'
MODEL_DIR = 'examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model'
CC_MATERIAL_DIR = 'resources/cctextures'

RENDER_TEMP_DIR = './FRONT3D_render/temp'
SCENE_LIST = []


def check_cache_dir(scene_idx):
    if not os.path.isdir(f'./cached/{scene_idx}'):
        os.makedirs(f'./cached/{scene_idx}')
        
        
        
def get_scene_rot_bbox_meta(scene_idx, overwrite=False):
    """ Get the bounding box meta data of a scene. 
        [(name1, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), (name2, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), ...]
    """
    check_cache_dir(scene_idx)
    if os.path.isfile('./cached/%d/bboxes.npy' % scene_idx) and overwrite==False:
        print(f'Found cached information for scene {scene_idx}.')
        names = np.load(f'./cached/{scene_idx}/names.npy')
        # with open('./cached/{scene_idx}/bbox.json', 'r') as f:
        #     bbox = json.load(f)
        bboxes = np.load(f'./cached/{scene_idx}/bboxes.npy')

    else:
        loaded_objects = load_scene_objects(scene_idx, overwrite)
        names = []
        bboxes = []
        
        for i in range(len(loaded_objects)):
            object = loaded_objects[i]
            name = object.get_name()
            bbox = object.get_bound_box()

            names.append(name)
            bboxes.append(bbox)


        np.save(f'./cached/{scene_idx}/names.npy', names)
        np.save(f'./cached/{scene_idx}/bboxes.npy', bboxes)


    return names, bboxes




def get_scene_bbox_meta(scene_idx, overwrite=False):
    """ Get the bounding box meta data of a scene. 
        [(name1, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), (name2, [[xmin, ymin, zmin], [xmax, ymax, zmax]]), ...]
    """
    check_cache_dir(scene_idx)
    if os.path.isfile('./cached/%d/bbox_mins.npy' % scene_idx) and overwrite==False:
        print(f'Found cached information for scene {scene_idx}.')
        names = np.load(f'./cached/{scene_idx}/names.npy')
        bbox_mins = np.load(f'./cached/{scene_idx}/bbox_mins.npy')
        bbox_maxs = np.load(f'./cached/{scene_idx}/bbox_maxs.npy')
    else:
        loaded_objects = load_scene_objects_with_improved_mat(scene_idx, overwrite)
        names = []
        bbox_mins = []
        bbox_maxs = []
        for i in range(len(loaded_objects)):
            object = loaded_objects[i]
            name = object.get_name()
            bbox = object.get_bound_box()
            
            bbox_min = np.min(bbox, axis=0)
            bbox_max = np.max(bbox, axis=0)
            names.append(name)
            bbox_mins.append(bbox_min)
            bbox_maxs.append(bbox_max)

        np.save(f'./cached/{scene_idx}/names.npy', names)
        np.save(f'./cached/{scene_idx}/bbox_mins.npy', bbox_mins)
        np.save(f'./cached/{scene_idx}/bbox_maxs.npy', bbox_maxs)

    return names, bbox_mins, bbox_maxs




def add_texture(obj:MeshObject, tex_path):
    """ Add a texture to an object. """
    obj.clear_materials()
    mat = obj.new_material('my_material')
    bsdf = mat.nodes["Principled BSDF"]
    texImage = mat.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(tex_path)
    mat.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

# TODO: read config file
def load_scene_objects(scene_idx, overwrite=False):
    check_cache_dir(scene_idx)
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    
    loaded_objects = bproc.loader.load_front3d(
        json_path=SCENE_LIST[scene_idx],
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30
    )

    # add texture to wall and floor. Otherwise they will be white.
    for obj in loaded_objects:
        name = obj.get_name()
        if 'wall' in name.lower():
            add_texture(obj, TEXTURE_DIR+"/1b57700d-f41b-4ac7-a31a-870544c3d608/texture.png")
        elif 'floor' in name.lower():
            add_texture(obj, TEXTURE_DIR+"/0b48b46d-4f0b-418d-bde6-30ca302288e6/texture.png")
        # elif 'ceil' in name.lower():


    return loaded_objects

# TODO: read config file
def load_scene_objects_with_improved_mat(json_path, room_idx, overwrite=False):

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    with open(os.path.join(MODEL_DIR, 'model_info_revised.json'), 'r') as f:
            model_info_data = json.load(f)
    model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' 
                         for m in model_info_data}
    
    loaded_objects, data_info = bproc.loader.load_front3d_with_collection(
        json_path=json_path,
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        model_id_to_label=model_id_to_label,
        room_id=room_idx,
        return_data_info=True)

    # -------------------------------------------------------------------------
    #          Sample materials
    # -------------------------------------------------------------------------
    cc_materials = bproc.loader.load_ccmaterials(CC_MATERIAL_DIR, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    for floor in floors:
        # For each material of the object
        for i in range(len(floor.get_materials())):
            floor.set_material(i, random.choice(cc_materials))

    baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
    wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
    for obj in baseboards_and_doors:
        # For each material of the object
        for i in range(len(obj.get_materials())):
            # Replace the material with a random one
            obj.set_material(i, random.choice(wood_floor_materials))

    walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
    marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
    for wall in walls:
        # For each material of the object
        for i in range(len(wall.get_materials())):
            wall.set_material(i, random.choice(marble_materials))
    


    return loaded_objects, data_info