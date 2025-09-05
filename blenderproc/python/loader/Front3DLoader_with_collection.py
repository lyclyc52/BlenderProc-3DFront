import collections
import json
import os
import warnings
from math import radians
from typing import List, Mapping
import bpy
import mathutils
import numpy as np
from urllib.request import urlretrieve

from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.types.MeshObjectUtility import MeshObject, create_with_empty_mesh
from blenderproc.python.utility.Utility import resolve_path
from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.loader.TextureLoader import load_texture
import pdb

def load_front3d_with_collection(json_path: str, future_model_path: str, front_3D_texture_path: str, label_mapping: LabelIdMapping,
                 model_id_to_label: dict, ceiling_light_strength: float = 0.8, lamp_light_strength: float = 7.0, 
                 room_id = None, return_data_info: bool = False) -> List[MeshObject]:
    """ Loads the 3D-Front scene specified by the given json file.

    :param json_path: Path to the json file, where the house information is stored.
    :param future_model_path: Path to the models used in the 3D-Front dataset.
    :param front_3D_texture_path: Path to the 3D-FRONT-texture folder.
    :param label_mapping: A dict which maps the names of the objects to ids.
    :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
    :param lamp_light_strength: Strength of the emission shader used in each lamp.
    :return: The list of loaded mesh objects.
    """
    json_path = resolve_path(json_path)
    future_model_path = resolve_path(future_model_path)
    front_3D_texture_path = resolve_path(front_3D_texture_path)

    if not os.path.exists(json_path):
        raise Exception("The given path does not exists: {}".format(json_path))
    if not json_path.endswith(".json"):
        raise Exception("The given path does not point to a .json file: {}".format(json_path))
    if not os.path.exists(future_model_path):
        raise Exception("The 3D future model path does not exist: {}".format(future_model_path))

    # load data from json file
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    if "scene" not in data:
        raise Exception("There is no scene data in this json file: {}".format(json_path))
    # only generate the objects in the first room
    if room_id is not None:
        if len(data["scene"]["room"]) <= room_id:
            raise Exception("The room id is out of range: {}".format(room_id))
        all_uid_ids = []
        for child in data['scene']['room'][room_id]['children']:
            all_uid_ids.append(child['ref'])
        room_item_uid = []
        for room in data['scene']['room']:
            for child in room['children']:
                room_item_uid.append(child['ref'])
        # delete the objects that are not in the first room
        label_dict = {}
        new_furniture = []
        new_mesh = []
        for obj in data['mesh']:
            if obj['uid'] in all_uid_ids:
                if obj['uid'] not in label_dict:
                    label_dict[obj['uid']] = obj['type']
                else:
                    if label_dict[obj['uid']] != obj['type']:
                        raise Exception("The type of the object is not consistent: {}".format(obj['uid']))
                new_mesh.append(obj)
            if obj['uid'] not in room_item_uid:
                new_mesh.append(obj)
        for obj in data['furniture']:
            if obj['uid'] in all_uid_ids:
                if obj['uid'] not in label_dict:
                    if 'category' in obj:
                        label_dict[obj['uid']] = obj['category']
                    else:
                        if obj['jid'] in model_id_to_label:
                            label_dict[obj['uid']] = model_id_to_label[obj['jid']]
                        else:
                            label_dict[obj['uid']] = 'Other'
                new_furniture.append(obj)
            if obj['uid'] not in room_item_uid:
                new_furniture.append(obj)

        data['furniture'] = new_furniture
        data['mesh'] = new_mesh
                
        data["scene"]["room"] = data["scene"]["room"][room_id:room_id+1]
        data['mesh'] = [obj for obj in data['mesh'] if obj['uid'] in room_item_uid]
        data['furniture'] = [obj for obj in data['furniture'] if obj['uid'] in room_item_uid]


    created_objects = Front3DLoader._create_mesh_objects_from_file(data, front_3D_texture_path,
                                                                   ceiling_light_strength, label_mapping, json_path)

    all_loaded_furniture = Front3DLoader._load_furniture_objs(data, future_model_path, lamp_light_strength,
                                                              label_mapping, model_id_to_label)
    created_objects += Front3DLoader._move_and_duplicate_furniture(data, all_loaded_furniture)
    # Update the category_id of the objects in data
    if room_id is not None:
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
                y_max = max(y_max, y)
                z_max = max(z_max, z)
        # flip the y and z coordinates
        center_of_room = np.array([(x_min + x_max) / 2,  (z_min + z_max) / 2, (y_min + y_max) / 2])
        # move all the objects to the origin
        for obj in created_objects:
            obj.set_location(obj.get_location() - center_of_room)
        
        center_of_room = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2,  (z_min + z_max) / 2])
        for obj in data['scene']['room'][0]['children']:
            obj['category_id'] = label_dict[obj['ref']] if obj['ref'] in label_dict else 'Other'
            obj['pos'] = (np.array(obj['pos']) - center_of_room).tolist()
        for obj in data['mesh']:
            for i in range(0, len(obj['xyz']), 3):
                obj['xyz'][i], obj['xyz'][i+1], obj['xyz'][i+2] = obj['xyz'][i] - center_of_room[0], \
                    obj['xyz'][i+1] - center_of_room[1], obj['xyz'][i+2] - center_of_room[2]


    if return_data_info:
        return created_objects, data.copy()
    else:
        return created_objects

class Front3DLoader:
    """ Loads the 3D-Front dataset.

    https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset

    Each object gets the name based on the category/type, on top of that you can use a mapping specified in the
    resources/front_3D folder.

    The dataset already supports semantic segmentation with either the 3D-Front classes or the nyu classes.
    As we have created this mapping ourselves it might be faulty.

    The Front3DLoader creates automatically lights in the scene, by adding emission shaders to the ceiling and lamps.
    """

    @staticmethod
    def _extract_hash_nr_for_texture(given_url: str, front_3D_texture_path: str) -> str:
        """
        Constructs the path of the hash folder and checks if the texture is available if not it is downloaded

        :param given_url: The url of the texture
        :param front_3D_texture_path: The path to where the texture are saved
        :return: The hash id, which is used in the url
        """
        # extract the hash nr from the given url
        hash_nr = given_url.split("/")[-2]
        hash_folder = os.path.join(front_3D_texture_path, hash_nr)
        if not os.path.exists(hash_folder):
            # download the file
            os.makedirs(hash_folder)
            warnings.warn(f"This texture: {hash_nr} could not be found it will be downloaded.")
            # replace https with http as ssl connection out of blender are difficult
            urlretrieve(given_url.replace("https://", "http://"), os.path.join(hash_folder, "texture.png"))
            if not os.path.exists(os.path.join(hash_folder, "texture.png")):
                raise Exception(f"The texture could not be found, the following url was used: "
                                f"{front_3D_texture_path}, this is the extracted hash: {hash_nr}, "
                                f"given url: {given_url}")
        return hash_folder

    @staticmethod
    def _get_used_image(hash_folder_path: str, saved_image_dict: Mapping[str, bpy.types.Texture]) -> bpy.types.Texture:
        """
        Returns a texture object for the given hash_folder_path, the textures are stored in the saved_image_dict,
        to avoid that texture are loaded multiple times

        :param hash_folder_path: Path to the hash folder
        :param saved_image_dict: Dict which maps the hash_folder_paths to bpy.types.Texture
        :return: The loaded texture bpy.types.Texture
        """
        if hash_folder_path in saved_image_dict:
            ret_used_image = saved_image_dict[hash_folder_path]
        else:
            textures = load_texture(hash_folder_path)
            if len(textures) != 1:
                raise Exception(f"There is not just one texture: {len(textures)}")
            ret_used_image = textures[0].image
            saved_image_dict[hash_folder_path] = ret_used_image
        return ret_used_image

    @staticmethod
    def _create_mesh_objects_from_file(data: dict, front_3D_texture_path: str, ceiling_light_strength: float,
                                       label_mapping: LabelIdMapping, json_path: str) -> List[MeshObject]:
        """
        This creates for a given data json block all defined meshes and assigns the correct materials.
        This means that the json file contains some mesh, like walls and floors, which have to built up manually.

        It also already adds the lighting for the ceiling

        :param data: json data dir. Must contain "material" and "mesh"
        :param front_3D_texture_path: Path to the 3D-FRONT-texture folder.
        :param ceiling_light_strength: Strength of the emission shader used in the ceiling.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :param json_path: Path to the json file, where the house information is stored.
        :return: The list of loaded mesh objects.
        """
        # extract all used materials -> there are more materials defined than used
        used_materials = []
        for mat in data["material"]:
            used_materials.append({"uid": mat["uid"], "texture": mat["texture"],
                                   "normaltexture": mat["normaltexture"], "color": mat["color"]})

        created_objects = []
        # maps loaded images from image file path to bpy.type.image
        saved_images = {}
        saved_normal_images = {}
        # materials based on colors to avoid recreating the same material over and over
        used_materials_based_on_color = {}
        # materials based on texture to avoid recreating the same material over and over
        used_materials_based_on_texture = {}

        mesh_id = -1
        for mesh_data in data["mesh"]:
            # extract the obj name, which also is used as the category_id name
            used_obj_name = mesh_data["type"].strip()
            if used_obj_name == "":
                used_obj_name = "void"
            if "material" not in mesh_data:
                warnings.warn(f"Material is not defined for {used_obj_name} in this file: {json_path}")
                continue
            
            # create a new mesh
            obj = create_with_empty_mesh(used_obj_name, used_obj_name + "_mesh")
            created_objects.append(obj)

            # set two custom properties, first that it is a 3D_future object and second the category_id
            mesh_id += 1
            obj.set_cp("uid", mesh_data["uid"])
            obj.set_cp("jid", mesh_data["jid"])
            obj.set_cp("inst_mark", "layout_" + str(mesh_id))
            obj.set_cp("is_3D_future", True)
            obj.set_cp("category_id", label_mapping.id_from_label(used_obj_name.lower()))

            # get the material uid of the current mesh data
            current_mat = mesh_data["material"]
            used_mat = None
            # search in the used materials after this uid
            for u_mat in used_materials:
                if u_mat["uid"] == current_mat:
                    used_mat = u_mat
                    break
            # If there should be a material used
            if used_mat:
                if used_mat["texture"]:
                    # extract the has folder is from the url and download it if necessary
                    hash_folder = Front3DLoader._extract_hash_nr_for_texture(used_mat["texture"], front_3D_texture_path)
                    if hash_folder in used_materials_based_on_texture and "ceiling" not in used_obj_name.lower():
                        mat = used_materials_based_on_texture[hash_folder]
                        obj.add_material(mat)
                    else:
                        # Create a new material
                        mat = MaterialLoaderUtility.create(name=used_obj_name + "_material")
                        principled_node = mat.get_the_one_node_with_type("BsdfPrincipled")
                        if used_mat["color"]:
                            principled_node.inputs["Base Color"].default_value = mathutils.Vector(used_mat["color"]) / 255.0

                        used_image = Front3DLoader._get_used_image(hash_folder, saved_images)
                        mat.set_principled_shader_value("Base Color", used_image)

                        if "ceiling" in used_obj_name.lower():
                            mat.make_emissive(ceiling_light_strength, emission_color=mathutils.Vector(used_mat["color"]) / 255.0)

                        if used_mat["normaltexture"]:
                            # get the used image based on the normal texture path
                            # extract the has folder is from the url and download it if necessary
                            hash_folder = Front3DLoader._extract_hash_nr_for_texture(used_mat["normaltexture"],
                                                                                     front_3D_texture_path)
                            used_image = Front3DLoader._get_used_image(hash_folder, saved_normal_images)

                            # create normal texture
                            normal_texture = MaterialLoaderUtility.create_image_node(mat.nodes, used_image, True)
                            normal_map = mat.nodes.new("ShaderNodeNormalMap")
                            normal_map.inputs["Strength"].default_value = 1.0
                            mat.links.new(normal_texture.outputs["Color"], normal_map.inputs["Color"])
                            # connect normal texture to principled shader
                            mat.set_principled_shader_value("Normal", normal_map.outputs["Normal"])

                        obj.add_material(mat)
                        used_materials_based_on_texture[hash_folder] = mat
                # if there is a normal color used
                elif used_mat["color"]:
                    used_hash = tuple(used_mat["color"])
                    if used_hash in used_materials_based_on_color and "ceiling" not in used_obj_name.lower():
                        mat = used_materials_based_on_color[used_hash]
                    else:
                        # Create a new material
                        mat = MaterialLoaderUtility.create(name=used_obj_name + "_material")
                        # create a principled node and set the default color
                        principled_node = mat.get_the_one_node_with_type("BsdfPrincipled")
                        principled_node.inputs["Base Color"].default_value = mathutils.Vector(used_mat["color"]) / 255.0
                        # if the object is a ceiling add some light output
                        if "ceiling" in used_obj_name.lower():
                            mat.make_emissive(ceiling_light_strength, emission_color=mathutils.Vector(used_mat["color"]) / 255.0)
                        else:
                            used_materials_based_on_color[used_hash] = mat

                    # as this material was just created the material is just append it to the empty list
                    obj.add_material(mat)

            # extract the vertices from the mesh_data
            vert = [float(ele) for ele in mesh_data["xyz"]]
            # extract the faces from the mesh_data
            faces = mesh_data["faces"]
            # extract the normals from the mesh_data
            normal = [float(ele) for ele in mesh_data["normal"]]

            # map those to the blender coordinate system
            num_vertices = int(len(vert) / 3)
            vertices = np.reshape(np.array(vert), [num_vertices, 3])
            normal = np.reshape(np.array(normal), [num_vertices, 3])
            # flip the first and second value
            vertices[:, 1], vertices[:, 2] = vertices[:, 2], vertices[:, 1].copy()
            normal[:, 1], normal[:, 2] = normal[:, 2], normal[:, 1].copy()
            # reshape back to a long list
            vertices = np.reshape(vertices, [num_vertices * 3])
            normal = np.reshape(normal, [num_vertices * 3])

            # add this new data to the mesh object
            mesh = obj.get_mesh()
            mesh.vertices.add(num_vertices)
            mesh.vertices.foreach_set("co", vertices)
            mesh.vertices.foreach_set("normal", normal)

            # link the faces as vertex indices
            num_vertex_indicies = len(faces)
            mesh.loops.add(num_vertex_indicies)
            mesh.loops.foreach_set("vertex_index", faces)

            # the loops are set based on how the faces are a ranged
            num_loops = int(num_vertex_indicies / 3)
            mesh.polygons.add(num_loops)
            # always 3 vertices form one triangle
            loop_start = np.arange(0, num_vertex_indicies, 3)
            # the total size of each triangle is therefore 3
            loop_total = [3] * num_loops
            mesh.polygons.foreach_set("loop_start", loop_start)
            mesh.polygons.foreach_set("loop_total", loop_total)

            # the uv coordinates are reshaped then the face coords are extracted
            uv_mesh_data = [float(ele) for ele in mesh_data["uv"] if ele is not None]
            # bb1737bf-dae6-4215-bccf-fab6f584046b.json includes one mesh which only has no UV mapping
            if uv_mesh_data:
                uv = np.reshape(np.array(uv_mesh_data), [num_vertices, 2])
                used_uvs = uv[faces, :]
                # and again reshaped back to the long list
                used_uvs = np.reshape(used_uvs, [2 * num_vertex_indicies])

                mesh.uv_layers.new(name="new_uv_layer")
                mesh.uv_layers[-1].data.foreach_set("uv", used_uvs)
            else:
                warnings.warn(f"This mesh {obj.name} does not have a specified uv map!")

            # this update converts the upper data into a mesh
            mesh.update()

            # the generation might fail if the data does not line up
            # this is not used as even if the data does not line up it is still able to render the objects
            # We assume that not all meshes in the dataset do conform with the mesh standards set in blender
            #result = mesh.validate(verbose=False)
            #if result:
            #    raise Exception("The generation of the mesh: {} failed!".format(used_obj_name))

        return created_objects

    @staticmethod
    def _load_furniture_objs(data: dict, future_model_path: str, lamp_light_strength: float,
                             label_mapping: LabelIdMapping, model_id_to_label: dict) -> List[MeshObject]:
        """
        Load all furniture objects specified in the json file, these objects are stored as "raw_model.obj" in the
        3D_future_model_path. For lamp the lamp_light_strength value can be changed via the config.

        :param data: json data dir. Should contain "furniture"
        :param future_model_path: Path to the models used in the 3D-Front dataset.
        :param lamp_light_strength: Strength of the emission shader used in each lamp.
        :param label_mapping: A dict which maps the names of the objects to ids.
        :return: The list of loaded mesh objects.
        """
        # collect all loaded furniture objects
        all_objs = []
        # for each furniture element
        for ele in data["furniture"]:
            if "valid" not in ele or not ele["valid"]:
                continue
            # get furniture category as obj name
            used_obj_name = model_id_to_label[ele['jid']]
            # create the paths based on the "jid"
            folder_path = os.path.join(future_model_path, ele["jid"])
            obj_file = os.path.join(folder_path, "raw_model.obj")
            # if the object exists load it -> a lot of object do not exist
            # we are unsure why this is -> we assume that not all objects have been made public
            if os.path.exists(obj_file) and not "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                # load all objects from this .obj file
                objs = load_obj(filepath=obj_file)
                for obj in objs:
                    obj.set_name(used_obj_name)
                    # add some custom properties
                    obj.set_cp("uid", ele["uid"])
                    obj.set_cp("jid", ele["jid"])
                    # this custom property determines if the object was used before
                    # is needed to only clone the second appearance of this object
                    obj.set_cp("is_used", False)
                    obj.set_cp("is_3D_future", True)
                    obj.set_cp("type", "Non-Object")  # is an non object used for the interesting score
                    # set the category id based on the used obj name
                    obj.set_cp("category_id", label_mapping.id_from_label(used_obj_name))
                    # walk over all materials
                    for mat in obj.get_materials():
                        if mat is None:
                            continue
                        principled_node = mat.get_nodes_with_type("BsdfPrincipled")
                        if "bed" in used_obj_name.lower() or "sofa" in used_obj_name.lower():
                            if len(principled_node) == 1:
                                principled_node[0].inputs["Roughness"].default_value = 0.5
                        is_lamp = "lamp" in used_obj_name.lower()
                        if len(principled_node) == 0 and is_lamp:
                            # this material has already been transformed
                            continue
                        elif len(principled_node) == 1:
                            principled_node = principled_node[0]
                        else:
                            raise Exception("The amount of principle nodes can not be more than 1, "
                                            "for obj: {}!".format(obj.get_name()))

                        # Front3d .mtl files contain emission color which make the object mistakenly emissive
                        # => Reset the emission color
                        principled_node.inputs["Emission"].default_value[:3] = [0, 0, 0]

                        # For each a texture node
                        image_node = mat.new_node('ShaderNodeTexImage')
                        # and load the texture.png
                        base_image_path = os.path.join(folder_path, "texture.png")
                        image_node.image = bpy.data.images.load(base_image_path, check_existing=True)
                        mat.link(image_node.outputs['Color'], principled_node.inputs['Base Color'])
                        # if the object is a lamp, do the same as for the ceiling and add an emission shader
                        if is_lamp:
                            mat.make_emissive(lamp_light_strength)
                all_objs.extend(objs)
            elif "7e101ef3-7722-4af8-90d5-7c562834fabd" in obj_file:
                warnings.warn(f"This file {obj_file} was skipped as it can not be read by blender.")
        return all_objs

    @staticmethod
    def _move_and_duplicate_furniture(data: dict, all_loaded_furniture: list) -> List[MeshObject]:
        """
        Move and duplicate the furniture depending on the data in the data json dir.
        After loading each object gets a location based on the data in the json file. Some objects are used more than
        once these are duplicated and then placed.

        :param data: json data dir. Should contain "scene", which should contain "room"
        :param all_loaded_furniture: all objects which have been loaded in _load_furniture_objs
        :return: The list of loaded mesh objects.
        """
        # this rotation matrix rotates the given quaternion into the blender coordinate system
        blender_rot_mat = mathutils.Matrix.Rotation(radians(-90), 4, 'X')
        created_objects = []
        # for each room
        mesh_id = -1
        for room_id, room in enumerate(data["scene"]["room"]):
            # for each object in that room
            all_collections = collections.defaultdict(list)
            for child in room["children"]:
                # create a dict of list
                if "furniture" in child["instanceid"]:
                    # find the object where the uid matches the child ref id
                    mesh_id += 1
                    for obj in all_loaded_furniture:
                        if obj.get_cp("uid") == child["ref"]:
                            # if the object was used before, duplicate the object and move that duplicated obj
                            if obj.get_cp("is_used"):
                                new_obj = obj.duplicate()
                            else:
                                # if it is the first time use the object directly
                                new_obj = obj
                            # gather the mesh of each object
                            all_collections[obj.get_cp("uid")].append(new_obj)
                            # created_objects.append(new_obj)
                            # new_obj.set_cp("inst_mark", 'furniture_' + str(mesh_id))
                            # new_obj.set_cp("is_used", True)
                            # new_obj.set_cp("room_id", room['instanceid'])
                            # new_obj.set_cp("type", "Object")  # is an object used for the interesting score
                            # new_obj.set_cp("coarse_grained_class", new_obj.get_cp("category_id"))
                            # # this flips the y and z coordinate to bring it to the blender coordinate system
                            # new_obj.set_location(mathutils.Vector(child["pos"]).xzy)
                            # new_obj.set_scale(child["scale"])
                            # new_obj.blender_obj.scale.x = -1 * new_obj.blender_obj.scale.x
                            # # extract the quaternion and convert it to a rotation matrix
                            # rotation_mat = mathutils.Quaternion(child["rot"]).to_euler().to_matrix().to_4x4()
                            # # transform it into the blender coordinate system and then to an euler
                            # new_obj.set_rotation_euler((blender_rot_mat @ rotation_mat).to_euler())
            for child in room["children"]:
                if "furniture" in child["instanceid"]:
                    if child["ref"] in all_collections and len(all_collections[child["ref"]]) > 0:
                        
                        new_obj = all_collections[child["ref"]][0]
                        new_obj_name = new_obj.get_name() if len(all_collections[child["ref"]]) == 1 else new_obj.get_name()[:-4]
                        if len(all_collections[child["ref"]]) > 1:
                            bpy.ops.object.select_all(action='DESELECT')
                            for obj in all_collections[child["ref"]]:
                                obj.blender_obj.select_set(True)
                            bpy.context.view_layer.objects.active = new_obj.blender_obj
                            bpy.ops.object.join()
                            bpy.context.view_layer.objects.active.name = new_obj_name
                        created_objects.append(new_obj)
                        new_obj.set_cp("inst_mark", 'furniture_' + str(mesh_id))
                        new_obj.set_cp("room_id", room['instanceid'])
                        new_obj.set_cp("type", "Object")  # is an object used for the interesting score
                        new_obj.set_cp("coarse_grained_class", new_obj.get_cp("category_id"))
                        # this flips the y and z coordinate to bring it to the blender coordinate system
                        new_obj.set_location(mathutils.Vector(child["pos"]).xzy)
                        new_obj.set_scale(child["scale"])
                        new_obj.blender_obj.scale.x = -1 * new_obj.blender_obj.scale.x
                        # extract the quaternion and convert it to a rotation matrix
                        rotation_mat = mathutils.Quaternion(child["rot"]).to_euler().to_matrix().to_4x4()
                        # transform it into the blender coordinate system and then to an euler
                        new_obj.set_rotation_euler((blender_rot_mat @ rotation_mat).to_euler())

        return created_objects




# def collection():
#     import bpy

#     # Deselect all objects first to ensure a clean selection
#     bpy.ops.object.select_all(action='DESELECT')

#     # Define the objects you want to group
#     object_names = ["Cube", "Sphere", "Cylinder"] 

#     # Select the objects you want to add to the collection
#     for obj_name in object_names:
#         if obj_name in bpy.data.objects:
#             bpy.data.objects[obj_name].select_set(True)

#     # Create a new collection and move the selected objects into it
#     collection_name = "My_New_Collection"
#     bpy.ops.object.move_to_collection(collection_name=collection_name)

#     # Optional: Deselect the objects after grouping
#     bpy.ops.object.select_all(action='DESELECT')

#     print(f"Objects grouped into collection: '{collection_name}'")

# def move_collection(collection_name: str):
#     import bpy

#     # Get the collection by its name
#     collection_name = "MyCollection"  # Replace with your collection's name
#     my_collection = bpy.data.collections.get(collection_name)

#     if my_collection:
#         # Define the new location
#         new_location = (5.0, 2.0, 1.0)  # X, Y, Z coordinates

#         # Iterate through each object in the collection and set its location
#         for obj in my_collection.objects:
#             obj.location = new_location
#     else:
#         print(f"Collection '{collection_name}' not found.")