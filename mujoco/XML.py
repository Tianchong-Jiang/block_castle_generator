import sys
import os

import utils

class XML:

  def __init__(self, asset_path = 'assets/stl/', timestep = 0.002):
    self.timestep = timestep
    self.asset_path = asset_path
    self.names = set()
    self.assets_mesh = []
    self.assets_material = [{'name': 'wall_visible', 'rgba': '.9 .9 .9 1'}, {'name': 'wall_invisible', 'rgba': '.9 .9 .9 0'}]
    self.meshes = []
    self.base = '''
<mujoco>
    <asset>
       {}
       {}
    </asset>

    <compiler angle='radian'/>

    <option timestep='{}'>
        <flag override='enable'/>
    </option>

    <visual>
      <map znear="0.001"/>
    </visual>

   <worldbody>
      <camera name='front' pos='0 10 3' euler='-1.57 0 0' fovy = '57'/>
      <camera name='left' pos='10 0 3' euler='0 1.57 -1.57'/>
      <camera name='right' pos='-10 0 3' euler='0 -1.57 1.57'/>
      <camera name='above' pos='0 0 100' euler='0 0 0'/>
      <light diffuse='1.5 1.5 1.5' pos='0 4 4' dir='0 -1 0'/>  
      <light diffuse='1.5 1.5 1.5' pos='0 4 1' dir='0 -1 0'/>  
      
      <geom name='preparing_table'  type='plane' pos='0 30 0' euler='0 0 0' size='20 5 0.1' material='wall_visible'/>
      <geom name='wall_floor'  type='plane' pos='0 0 0' euler='0 0 0' size='5 5 0.1' material='wall_visible'/>
      {}

   </worldbody>
</mujoco>
'''

  def get_unique_name(self, polygon):
    i = 0
    while '{}_{}'.format(polygon, i) in self.names:
      i += 1
    name = '{}_{}'.format(polygon, i)
    self.names.add(name)
    return name

  def add_asset(self, name, polygon, scale):
    self.assets.append( {'name': name, 'polygon': polygon, 'scale': scale} )

  def add_mesh(self, polygon, scale = 1, pos = [0, 0, 0], axangle = [1, 0, 0], rgba = [1, 1, 1, 1], force = [0, 0, 0], name = None):
    if name is None:
      name = self.get_unique_name(polygon)
    else:
      self.names.add(name)
    scale_rep = self.__rep_vec([scale, scale, scale])
    pos_rep = self.__rep_vec(pos)

    quat = utils.axangle_to_quat(axangle)
    quat_rep = self.__rep_vec(quat)

    rgba_rep = self.__rep_vec(rgba)
    self.assets_mesh.append( {'name': name, 'polygon': polygon, 'scale': scale_rep} )
    self.assets_material.append( {'name': name, 'rgba': rgba_rep} )
    self.meshes.append( {'name': name, 'polygon': polygon, 'pos': pos_rep, 'quat': quat_rep, 'force': force, 'xscale': scale, 'xrgba': rgba, 'material': name} )

    return name

  def __rep_vec(self, vec):
    vec = [str(v) for v in vec]
    return ' '.join(vec)

  def get_body_str(self):
      body_base = '''
        <body name='{}' pos='{}' quat='{}'>
          <joint type='free' name='{}'/>
          <geom name='{}' type='mesh' mesh='{}' pos='0 0 0' quat='1 0 0 0' material='{}'/>
        </body>
      '''

      body_list = [body_base.format( \
         m['name'], m['pos'], m['quat'], m['name'], m['name'], m['name'], m['material'] )
        for m in self.meshes]

      body_str = '\n'.join(body_list)
      return body_str

  def get_asset_mesh_str(self):
    asset_base = '<mesh name="{}" scale="{}" file="{}"/>'

    asset_list = [asset_base.format( \
        a['name'], a['scale'],
        os.path.join(self.asset_path, a['polygon'] + '.stl') ) 
        for a in self.assets_mesh]

    asset_str = '\n'.join(asset_list)
    return asset_str

  def get_asset_material_str(self):
    asset_base = '<material name="{}" rgba="{}" specular="0" shininess="0" emission="0.25"/>'

    asset_list = [asset_base.format( \
        a['name'], a['rgba'] ) 
        for a in self.assets_material]

    asset_str = '\n'.join(asset_list)
    return asset_str

  def instantiate(self):
    xml_str = self.base.format( self.get_asset_mesh_str(), self.get_asset_material_str(), self.timestep, self.get_body_str() )
    return xml_str

  def apply_forces(self, sim):
    for mesh in self.meshes:
      mesh_name = mesh['name']
      force = mesh['force']
      mesh_ind = sim.model._body_name2id[mesh_name]
      sim.data.xfrc_applied[mesh_ind, :3] = force
    sim.step()
    sim.data.xfrc_applied.fill(0)

