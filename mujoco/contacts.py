import random
import colorsys
import pdb

import mujoco_py as mjc

from mujoco.XML import XML
import utils as utils

def sample_settled_fixed(asset_path, num_objects, bounds):
    xml = XML(asset_path)
    num_set = max(num_objects)
    names = []

    ## hard-coded polygon instances, to change later
    polygon_instances = [
        {'name':1, 'category':'cube','rgba':[1,0,0,1]},
        {'name':2, 'category':'cube','rgba':[0,1,0,1]},
        {'name':3, 'category':'cube','rgba':[0,0,1,1]},
        {'name':4, 'category':'rectangle','rgba':[1,0,0,1]},
        {'name':5, 'category':'rectangle','rgba':[0,1,0,1]},
        {'name':6, 'category':'rectangle','rgba':[0,0,1,1]},
        {'name':7, 'category':'cylinder','rgba':[1,0,0,1]},
        {'name':8, 'category':'cylinder','rgba':[0,1,0,1]},
        {'name':9, 'category':'horizontal_rectangle','rgba':[0,1,0,1]},
        {'name':0, 'category':'horizontal_rectangle','rgba':[0,0,1,1]}
    ]
    
    # routin to add meshes to xml file
    for obj_num in range(num_set):
        random.shuffle(polygon_instances)

        # hard-coded, to drop 10 objects on the preparing table
        pos = [4 * obj_num - 18, 30, 1]

        if 'horizontal' in polygon_instances[obj_num]['category']:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle  = utils.random_axangle(axis = axis)
        scale = utils.uniform(*bounds['scale'])

        name = xml.add_mesh(polygon_instances[obj_num]['category'], pos = pos, axangle = axangle, scale = scale, rgba = polygon_instances[obj_num]['rgba'])
        print(name)
        names.append(name)

    xml_str = xml.instantiate()
    model = mjc.load_model_from_xml(xml_str)
    sim = mjc.MjSim(model)
    return sim, xml, names

def sample_settled(asset_path, num_objects, polygons, bounds, spacing = 1):
    xml = XML(asset_path)
    num_set = max(num_objects)
    names = []
    
    # routin to add meshes to xml file
    for obj_num in range(num_set):
        ply = random.choice(polygons)

        # hard-coded, to drop 10 objects on the preparing table
        pos = [4 * obj_num - 18, 30, 1]
        # utils.uniform(*bounds['pos'])
        # pos[-1] = spacing * (obj_num)

        if 'horizontal' in ply:
            axis = [1, 0, 0]
        else:
            axis = [0, 0, 1]
        axangle  = utils.random_axangle(axis = axis)
        scale = utils.uniform(*bounds['scale'])
        rgba = sample_rgba_from_hsv(*bounds['hsv'])

        name = xml.add_mesh(ply, pos = pos, axangle = axangle, scale = scale, rgba = rgba)
        names.append(name)

    xml_str = xml.instantiate()
    model = mjc.load_model_from_xml(xml_str)
    sim = mjc.MjSim(model)
    return sim, xml, names

def sample_rgba_from_hsv(*hsv_bounds):
    hsv = utils.uniform(*hsv_bounds)
    rgba = list(colorsys.hsv_to_rgb(*hsv)) + [1]
    return rgba

def is_overlapping(sim, name = None):
    sim.forward()
    ncon = sim.data.ncon
    for contact_ind in range(ncon):
        contact = sim.data.contact[contact_ind]
        geom1 = sim.model._geom_id2name[contact.geom1]
        geom2 = sim.model._geom_id2name[contact.geom2]
        relevant_name = name is None or (geom1 == name or geom2 == name)

        if contact.dist < 0 and relevant_name:
            return True
    return False







