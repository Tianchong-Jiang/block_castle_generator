import random
import colorsys
import pdb

import mujoco_py as mjc

from mujoco.XML import XML
import utils as utils

def sample_settled(asset_path, num_objects, polygons, bounds, spacing = 1):
    xml = XML(asset_path)
    num_set = max(num_objects)
    names = []

    for obj_num in range(num_set):
        ply = random.choice(polygons)

        pos = utils.uniform(*bounds['pos'])
        pos[-1] = spacing * (obj_num)

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







