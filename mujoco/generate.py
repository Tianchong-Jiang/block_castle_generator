import os
import argparse
import pickle
import math
import random
import tqdm
import pdb
import imageio
import numpy as np

import mujoco_py as mjc
import matplotlib.pyplot as plt

from XML import XML
from logger import Logger
import contacts
import utils

parser = argparse.ArgumentParser()
## stuff you might want to edit
parser.add_argument('--start', default=0, type=int, 
        help='starting index (useful if rendering in parallel jobs)')
parser.add_argument('--num_images', default=1, type=int,
        help='total number of scenes to render')
parser.add_argument('--img_dim', default=640, type=int,
        help='image dimension')
parser.add_argument('--output_path', default='rendered/test/', type=str,
        help='path to save images')

parser.add_argument('--drop_steps_max', default=500, type=int,
        help='max number of steps simulating dropping one object')
parser.add_argument('--total_steps_max', default=5000, type=int,
        help='max number of total steps simulating one scene')
parser.add_argument('--render_freq', default=25, type=int,
        help='frequency of image saves in drop simulation')

parser.add_argument('--min_objects', default=4, type=int,
        help='min number of objects *starting on the ground*')
parser.add_argument('--max_objects', default=4, type=int,
        help='max number of objects *starting on the ground*')

## stuff about outputting video/gif
parser.add_argument('--save_gif', default=True, type=bool,
        help='if true, saves images as gif')


## stuff you probably don't need to edit
parser.add_argument('--settle_steps_min', default=2000, type=int,
        help='min number of steps simulating ground objects to rest')
parser.add_argument('--settle_steps_max', default=2000, type=int,
        help='max number of steps simulating ground objects to rest')
parser.add_argument('--save_images', default=True, type=bool,
        help='if true, saves images as png (alongside pickle files)')
args = parser.parse_args()


polygons = ['cube', 'horizontal_rectangle', 'rectangle', 'cylinder'] 

num_objects = range(args.min_objects, args.max_objects + 1)

## bounds for objects that start on the ground plane
settle_bounds = {  
            'pos':   [ [-.5, .5], [-5.5, -4.5], [1.5, 2] ],
            'hsv': [ [0, 1], [0.5, 1], [0.5, 1] ],
            'scale': [ [0.4, 0.4] ],
            'force': [ [0, 0], [0, 0], [0, 0] ]
          }

## bounds for the object to be dropped
drop_bounds = {  
            'pos':   [ [-1.75, 1.75], [-.5, 0], [0, 3] ],
          }  

## folder with object meshes
asset_path = os.path.join(os.getcwd(), 'assets/stl/')

utils.mkdir(args.output_path)

metadata = {'polygons': polygons, 'max_steps_per_drop': args.drop_steps_max, 
            'max_total_steps': args.total_steps_max,
            'min_objects': min(num_objects), 
            'max_objects': max(num_objects)}
pickle.dump( metadata, open(os.path.join(args.output_path, 'metadata.p'), 'wb') )

# number of frames rendered in total for a scene
num_images_per_scene = math.ceil(args.settle_steps_max * (1 + args.max_objects) / args.render_freq + 1)
end = args.start + args.num_images
for img_num in tqdm.tqdm( range(args.start, end) ):

    step = 0

    ## initiate the sim and settle the blocks on the first floor
    sim, xml, names = contacts.sample_settled(asset_path, num_objects, polygons, settle_bounds)

    ## initiate the logger, which is used to log data and images
    logger = Logger(xml, sim, steps = num_images_per_scene, img_dim = args.img_dim )
    
    ## drop all objects to the preparing table
    step = logger.settle_sim(args.settle_steps_min, args.settle_steps_max, step, args.render_freq)

    logger.drop_obj_random(names, args.settle_steps_min, args.settle_steps_max, step, args.render_freq)
    
    data, images, masks = logger.get_logs()

    if args.save_images:
        for timestep in range( images.shape[0] ):
            plt.imsave( os.path.join(args.output_path, '{}_{}.png'.format(img_num, timestep)), images[timestep] / 255. )

    if args.save_gif:
        images_uint8 = images.astype(np.uint8)
        imageio.mimsave(os.path.join(args.output_path, '{}.gif'.format(img_num)), images_uint8)

    config_path  = os.path.join( args.output_path, '{}.p'.format(img_num) )

    config = {'data': data, 'images': images, 'masks': masks}

    pickle.dump( config, open(config_path, 'wb') )



