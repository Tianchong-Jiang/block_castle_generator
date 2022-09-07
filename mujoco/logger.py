import random
import numpy as np
import scipy.misc
import pdb

from mujoco.XML import XML
import mujoco.contacts as contacts
import utils

# modified line
import numpy as np
from PIL import Image


class Logger:

  def __init__(self, xml, sim, steps = 100, image_width = 64, image_height = 64, albedo = False, cameras = []):
    self.sim = sim
    self.steps = steps
    self.image_width = image_width
    self.image_height = image_height
    self.albedo_flag = albedo
    self.render_width = self.image_width * 2 
    self.render_height = self.image_height * 2
    self.meshes = {}
    self.masks = {}
    for mesh in xml.meshes:
      mesh_name = mesh['name']
      mesh_log = {}
      mesh_log['xpos']  = np.zeros( (steps, 3) )
      mesh_log['xaxangle'] = np.zeros( (steps, 4) )
      mesh_log['xvelp'] = np.zeros( (steps, 3) )
      mesh_log['xvelr'] = np.zeros( (steps, 3) )

      mesh_log['xscale'] = np.zeros( (steps, 1) )
      mesh_log['xrgba'] = np.zeros( (steps, 3) )
      mesh_log['xscale'][:,:] = mesh['xscale'] 
      mesh_log['xrgba'][:,:] = mesh['xrgba'][:3]
      mesh_log['ply'] = mesh_name[:-2]
      self.meshes[mesh_name] = mesh_log
      self.masks[mesh_name] = np.zeros( (steps, len(cameras), self.image_height, self.image_width, 3) )

  ## ======== UTILS ========

  def log_step(self, step):
    self.sim.forward()
    for mesh_name in self.meshes.keys():
      self.log_mesh(mesh_name, step)

  ## log pos and vel data of a mesh
  def log_mesh(self, mesh_name, step):
    xpos, xaxangle, xvelp, xvelr = self.get_body_data(mesh_name)
    self.meshes[mesh_name]['xpos'][step]  = xpos
    self.meshes[mesh_name]['xaxangle'][step] = xaxangle
    self.meshes[mesh_name]['xvelp'][step] = xvelp
    self.meshes[mesh_name]['xvelr'][step] = xvelr

  def get_body_data(self, mesh_name):
    ## position
    xpos  = self.sim.data.get_body_xpos(mesh_name)
    ## quaternion
    xquat = self.sim.data.get_body_xquat(mesh_name)
    ## positional velocity
    xvelp = self.sim.data.get_body_xvelp(mesh_name)
    ## rotational velocity
    xvelr = self.sim.data.get_body_xvelr(mesh_name)

    xaxangle = utils.quat_to_axangle(xquat)

    return xpos, xaxangle, xvelp, xvelr

  def log_image(self, step, transparent = [], cameras = ['front','left','right','above']):
    for i in range(len(cameras)):
      self.make_transparent(transparent)
      image = self.sim.render(self.render_width, self.render_height, camera_name = cameras[i])
      self.undo_transparent()
      if self.image_width != self.render_width or self.image_height != self.render_height:
        image = np.array(Image.fromarray(image).resize(size = (self.image_width, self.image_height))).astype(np.uint8)

      if 'images' not in dir(self):
        M, N, C = image.shape
        self.images = np.zeros( (self.steps, len(cameras), M, N, C) )
        self.albedo = np.zeros( (self.steps, len(cameras), M, N, C) )
      self.images[step][i] = image

  def log_albedo(self, step, camera = 'fixed'):
    rgba = self.sim.model.mat_rgba.copy()
    spec = self.sim.model.mat_specular.copy()
    emit = self.sim.model.mat_emission.copy()
    shin = self.sim.model.mat_shininess.copy()

    self.sim.model.mat_specular[:] = 0
    self.sim.model.mat_shininess[:] = 0

    mesh_names = self.masks.keys()
    for mesh in mesh_names:
      mesh_ind = self.sim.model._geom_name2id[mesh]
      mat_ind  = self.sim.model.geom_matid[mesh_ind]

      self.sim.model.mat_emission[mat_ind] = 1

    image = self.sim.render(self.render_width, self.image_height, camera_name = camera)
    if self.image_width != self.render_width or self.image_height != self.render_height:
      image = np.array(Image.fromarray(image).resize(size = (self.image_width, self.image_height)))

    self.albedo[step] = image

    self.sim.model.mat_rgba[:] = rgba
    self.sim.model.mat_specular[:] = spec
    self.sim.model.mat_emission[:] = emit
    self.sim.model.mat_shininess[:] = shin

  def change_rgba(self, obj_name, rgba):
    mesh_ind = self.sim.model._geom_name2id[obj_name]
    mat_ind = self.sim.model.geom_matid[mesh_ind]

    old_rgba = self.sim.model.mat_rgba[mat_ind].copy()
    self.sim.model.mat_rgba[mat_ind] = rgba
    return old_rgba

  def log_masks(self, step, cameras = ['front','left','right','above']):
    rgba = self.sim.model.mat_rgba.copy()
    spec = self.sim.model.mat_specular.copy()
    emit = self.sim.model.mat_emission.copy()
    shin = self.sim.model.mat_shininess.copy()

    self.sim.model.mat_specular[:] = 0
    self.sim.model.mat_shininess[:] = 0

    for i in range(len(cameras)):
      mesh_names = self.masks.keys()
      for mesh in mesh_names:
        mesh_ind = self.sim.model._geom_name2id[mesh]
        mat_ind  = self.sim.model.geom_matid[mesh_ind]

        self.sim.model.mat_rgba[:] = [0, 0, 0, 1]
        self.sim.model.mat_rgba[mat_ind] = [1, 1, 1, 1]
        self.sim.model.mat_emission[mat_ind] = 1
      

        image = self.sim.render(self.render_width, self.render_height, camera_name = cameras[i])
        if self.image_width != self.render_width or self.image_height != self.render_height:
          image = np.array(Image.fromarray(image).resize(size = (self.image_width, self.image_height)))

        self.masks[mesh][step][i] = image # > 0.5

    self.sim.model.mat_rgba[:] = rgba
    self.sim.model.mat_specular[:] = spec
    self.sim.model.mat_emission[:] = emit
    self.sim.model.mat_shininess[:] = shin

  def export_xml(self, xml):
    xml_new = XML()
    for mesh in xml.meshes:
      mesh_name = mesh['name']
      polygon = mesh['polygon']
      pos, axangle, _, _ = self.get_body_data(mesh_name)

      rgba = mesh['xrgba']
      scale = mesh['xscale']

      xml_new.add_mesh(polygon, scale = scale, pos = pos, axangle = axangle, rgba = rgba, name = mesh_name)
    return xml_new

  def log(self, step, transparent=[]):
    self.log_step(step)
    self.log_image(step, transparent=transparent)
    self.log_masks(step)
    if self.albedo_flag:
      self.log_albedo(step)

  def make_transparent(self, names):
    self._transparent_dict = {}
    for name in names:
      rgba = self.change_rgba(name, [0,0,0,0])
      self._transparent_dict[name] = rgba

  def undo_transparent(self):
    for name, rgba in self._transparent_dict.items():
      self.change_rgba(name, rgba)
    
  def position_body(self, name, pos, axangle):
    joint_ind = self.sim.model._joint_name2id[name]
    quat = utils.axangle_to_quat(axangle)

    qpos_start_ind = joint_ind * 7
    qpos_end_ind   = (joint_ind+1) * 7

    qfrc_start_ind = joint_ind * 6
    qfrc_end_ind   = (joint_ind+1) * 6

    qpos = self.sim.data.qpos[qpos_start_ind:qpos_end_ind]
    qpos[:3] = pos
    qpos[3:] = quat

    self.sim.data.qfrc_constraint[qpos_start_ind:qpos_end_ind] = 0

    self.sim.forward()

  ## step function that renders at a given interval of steps
  def step_render(self, render_freq, step):
    if step % render_freq == 0:
      self.log(step//render_freq)
    ## simulate one timestep
    self.sim.step()

  def get_obj_pos(self, name):
    joint_ind = self.sim.model._joint_name2id[name]
    qpos_start_ind = joint_ind * 7
    qpos_end_ind   = (joint_ind+1) * 7
    return self.sim.data.qpos[qpos_start_ind:qpos_end_ind]

  def set_obj_pose(self, name, pos):
    joint_ind = self.sim.model._joint_name2id[name]
    qpos_start_ind = joint_ind * 7
    qpos_end_ind   = (joint_ind+1) * 7
    self.sim.data.qpos[qpos_start_ind:qpos_end_ind] = pos

  def get_obj_vel(self, name):
    joint_ind = self.sim.model._joint_name2id[name]
    qvel_start_ind = joint_ind * 6
    qvel_end_ind   = (joint_ind+1) * 6
    return self.sim.data.qvel[qvel_start_ind:qvel_end_ind]

  def set_obj_vel(self, name, vel):
    joint_ind = self.sim.model._joint_name2id[name]
    qvel_start_ind = joint_ind * 6
    qvel_end_ind   = (joint_ind+1) * 6
    self.sim.data.qvel[qvel_start_ind:qvel_end_ind] = vel

  def check_stability(self, name, init_pos):
    joint_ind = self.sim.model._joint_name2id[name]
    qpos_start_ind = joint_ind * 7
    qpos_end_ind   = qpos_start_ind + 3

    qpos = self.sim.data.qpos[qpos_start_ind:qpos_end_ind]

    if np.max( np.abs(qpos - init_pos) ) < 0.5:
      return True
    else:
      return False
    
  def get_logs(self, step = None):
    if step is None:
      if self.albedo_flag:
        return self.meshes, self.images, self.masks, self.albedo
      else:
        return self.meshes, self.images, self.masks
    else:
      step_meshes = {}
      step_masks  = {}
      for mesh_name, mesh_log in self.meshes.items():
        step_log = {}
        for op, param in mesh_log.items():
          step_log[op] = param[step][np.newaxis,:]
        step_meshes[mesh_name] = step_log
        step_masks[mesh_name] = self.masks[mesh_name][step]
      step_image = self.images[step]
      return step_meshes, step_image, step_meshes

  ## ======== PIPELINE FUNCTIONS ========
  ## wait till the block on the floor are settled
  def settle_sim(self, min_steps, max_steps, step, render_freq, vel_threshold = 0.1):
    start_step = step

    for _ in range(min_steps):
      self.step_render(render_freq, step)
      step += 1

    max_vel = np.abs(self.sim.data.qvel).max()
    while max_vel > vel_threshold:
      self.step_render(render_freq, step)
      max_vel = np.abs(self.sim.data.qvel[:,]).max()
      step += 1
      if step > max_steps + start_step:
        break
    return step

  ## drop (free-fall) one object from a given position
  def drop_obj(self, drop_name, pos, min_steps, max_steps, step, render_freq):
    self.set_obj_pose(drop_name, pos)
    self.set_obj_vel(drop_name,vel = [0] * 6)
    step = self.settle_sim(min_steps, max_steps, step, render_freq)
    return step

  def drop_obj_random(self, drop_names, min_steps, max_steps, step, render_freq):
    drop_num = len(drop_names)
    
    ## drop the first object at the center of the table
    step = self.drop_obj(drop_names[0], [0,0,5,0,0,0,0], min_steps, max_steps, step, render_freq)

    ## drop the rest of the blocks
    for i in range(1, drop_num):
      ## choose an already settled block as the center of the Gaussian
      target = random.randint(0, i)
      name = drop_names[target]
      center = self.get_obj_pos(name)
      pos = [0] * 7
      if target == i:
        pos[0] = random.gauss(0, 0.5)
        pos[1] = random.gauss(0, 0.5)
      else:
        pos[0] = random.gauss(center[0], 0.1)
        pos[1] = random.gauss(center[1], 0.1)
      pos[2] = 5
      step = self.drop_obj(drop_names[i], pos, min_steps, max_steps, step, render_freq)
    return step
    