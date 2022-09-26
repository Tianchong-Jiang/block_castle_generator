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
parser.add_argument('--image_width', default=1280, type=int,
        help='image width')
parser.add_argument('--image_height', default=720, type=int,
        help='image height')
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

## stuff about outputting video/gif/dataset image
parser.add_argument('--save_gif', default=False, type=bool,
        help='if true, saves images as gif')
parser.add_argument('--save_dataset', default=True, type=bool,
        help='if true, saves only the final image and the object mask as part of a dataset')
parser.add_argument('--save_images', default=False, type=bool,
        help='if true, saves images as png (alongside pickle files)')

## stuff you probably don't need to edit
parser.add_argument('--settle_steps_min', default=2000, type=int,
        help='min number of steps simulating ground objects to rest')
parser.add_argument('--settle_steps_max', default=2000, type=int,
        help='max number of steps simulating ground objects to rest')

args = parser.parse_args()

polygons = ['cube', 'horizontal_rectangle', 'rectangle', 'cylinder'] 
cameras = ['front','left','right','above']

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
num_images_per_scene = ( args.settle_steps_max * (1 + 1 + args.max_objects) ) // args.render_freq + 1
end = args.start + args.num_images
for img_num in tqdm.tqdm( range(args.start, end) ):

    step = 0

    ## initiate the sim and settle the blocks on the prepapring table
    #sim, xml, names = contacts.sample_settled(asset_path, num_objects, polygons, settle_bounds)
    sim, xml, names = contacts.sample_settled_fixed(asset_path, num_objects, settle_bounds)

    ## initiate the logger, which is used to log data and images
    logger = Logger(xml, sim, render_steps = num_images_per_scene, image_width = args.image_width,
        image_height = args.image_height, cameras = cameras)
    
    ## drop all objects to the preparing table
    step = logger.settle_sim(args.settle_steps_min, args.settle_steps_max, step, args.render_freq)

    ## drop the objects randomly to the main table
    step = logger.drop_obj_random(names, args.settle_steps_min, args.settle_steps_max, step, args.render_freq)
    
    ## wait for all the objects to be settled
    step = logger.settle_sim(args.settle_steps_min, args.settle_steps_max, step, args.render_freq)

    data, images, masks = logger.get_logs()

    ## Save the rendered images
    if args.save_images:
        for timestep in range( step // args.render_freq):
            for i in range(len(cameras)):
                plt.imsave( os.path.join(args.output_path, '{}_{}_{}.png'.format(img_num, timestep, cameras[i])),
                    images[timestep][i] / 255. )
    
    ## same the rendered images as a gif
    if args.save_gif:
        images_uint8 = images.astype(np.uint8)
        imageio.mimsave(os.path.join(args.output_path, '{}.gif'.format(img_num)), images_uint8)

    ## save the final image and masks as part of a dataset
    if args.save_dataset:
        for i in range(len(cameras)):
            plt.imsave( os.path.join(args.output_path, '{}_{}.png'.format(img_num, cameras[i])),
                images[(step - 1) // args.render_freq][i] / 255. )
        ## save the masks of each object
            mesh_names = logger.masks.keys()
            for mesh in mesh_names:
                plt.imsave( os.path.join(args.output_path, '{}_{}_{}.png'.format(img_num, mesh, cameras[i])),
                      np.multiply(masks[mesh][(step - 1) // args.render_freq][i] / 255.,
                      images[(step - 1) // args.render_freq][i] / 255.))

    ## write object poses to .txt file
    target_file = open(os.path.join(args.output_path, '{}_poses.txt'.format(img_num)), 'x')
    for name in names:
        target_file.write(name + '\n')
        target_file.write('pos' + np.array2string(data[name]['xpos'][-1]) + '\n')
        target_file.write('angle' + np.array2string(data[name]['xaxangle'][-1]) + '\n')
    target_file.close()

    config_path  = os.path.join( args.output_path, '{}.p'.format(img_num) )

    config = {'data': data, 'images': images, 'masks': masks}

    pickle.dump( config, open(config_path, 'wb') )



