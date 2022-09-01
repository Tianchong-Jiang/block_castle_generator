# Block Castle Generator

Code used to generate an dataset of block castles

## Setup
Download MuJoCo 2.10 (https://github.com/deepmind/mujoco/releases) and run `pip install -r requirements.txt` with Python 3.8.0. 

Here is a command to render some images
```
python mujoco/generate.py --drop_steps_max 1001 --render_freq 1000 \
    --num_images 10 --img_dim 64 --min_objects 2 --max_objects 4 --output_path rendered/initial_final/
```

If you would like to add a new shape, just drop the stl file in `assets/stl` and add its filename (without the stl extension) to the list of [polygons](mujoco/generate.py#L47).

## Reference

The base code of this repo is from https://github.com/jannerm/o2p2, which is the dataset generation code used in the paper Object-Oriented Prediction and Planning.