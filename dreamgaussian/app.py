import gradio as gr
import os
from PIL import Image
import subprocess
import hashlib
from dreamgaussian.main import GUI
from dreamgaussian.main2 import GUI2
from omegaconf import OmegaConf

#os.system('pip install -e ./simple-knn')
#os.system('pip install -e ./diff-gaussian-rasterization')

_TITLE = '''DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation'''
_IMG_USER_GUIDE = "Please upload an image in the block above (or choose an example above) and click **Generate 3D**."

# check if there is a picture uploaded or selected
def check_img_input(control_image):
    if control_image is None:
        raise gr.Error("Please select or upload an input image")

def optimize(image_block: Image.Image, preprocess_chk=True, elevation_slider=0):
    stage_1_output = optimize_stage_1(image_block, preprocess_chk, elevation_slider)
    stage_2_output = optimize_stage_2(image_block, elevation_slider)
    return stage_1_output, stage_2_output


def optimize_stage_1(image_block: Image.Image, preprocess_chk: bool, elevation_slider: float):
    if not os.path.exists('tmp_data'):
        os.makedirs('tmp_data')

    img_hash = hashlib.sha256(image_block.tobytes()).hexdigest()
    if preprocess_chk:
        # save image to a designated path
        image_block.save(f'tmp_data/{img_hash}.png')

        # preprocess image
        subprocess.run([f'python dreamgaussian/process.py tmp_data/{img_hash}.png'], shell=True)
    else:
        image_block.save(f'tmp_data/{img_hash}_rgba.png')

    # stage 1
    args = 'dreamgaussian/configs/image.yaml'
    extras = f'input=tmp_data/{img_hash}_rgba.png save_path={img_hash} mesh_format=glb elevation={elevation_slider} force_cuda_rast=True'

    # override default config from cli
    opt = OmegaConf.merge(args, extras)

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)

    return f'logs/{img_hash}_mesh.glb'


def optimize_stage_2(image_block: Image.Image, elevation_slider: float):
    img_hash = hashlib.sha256(image_block.tobytes()).hexdigest()
    # stage 2
    args = 'dreamgaussian/configs/image.yaml'
    extras = f'input=tmp_data/{img_hash}_rgba.png save_path={img_hash} mesh_format=glb elevation={elevation_slider} force_cuda_rast=True'
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI2(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)


    return f'logs/{img_hash}.glb'

