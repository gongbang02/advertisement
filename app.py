import gradio as gr
import cv2
import math
import os
import torch
from glob import glob
from pathlib import Path
from typing import Optional
import requests
import io
import numpy as np
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from PIL import Image
import uuid
import random
import hashlib
import subprocess
from huggingface_hub import hf_hub_download
from moviepy.editor import VideoFileClip, AudioFileClip
from controlnet.gradio_scribble import process
from controlnet.settings import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_NUM_IMAGES,
    MAX_IMAGE_RESOLUTION,
    MAX_NUM_IMAGES,
    MAX_SEED,
)
from controlnet.utils import randomize_seed_fn
from controlnet.gradio_scribble_interactive import create_canvas, process_interactive
from audiocraft.demos.musicgen_app import predict_full
from svd.scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from svd.sgm.inference.helpers import embed_watermark
from svd.sgm.util import default, instantiate_from_config
from lavie.base.app import infer
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import SUPPORTED_LANGS
from bark_clone.app import description, default_text, AVAILABLE_PROMPTS, article, gen_tts
from ledits.app import edit, crop_image, caption_image, reconstruct
from ledits.constants import *
from ledits.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from ledits.pipeline_semantic_stable_diffusion_img2img_solver import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from ledits.utils import *
from dreamgaussian.app import _TITLE, _IMG_USER_GUIDE, check_img_input, optimize, optimize_stage_1, optimize_stage_2



hf_hub_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", filename="svd_xt.safetensors", local_dir="checkpoints") 

version = "svd_xt"
device = "cuda"
max_64_bit_int = 2**63 - 1

API_URL = "https://api-inference.huggingface.co/models/openskyml/dalle-3-xl"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter

if version == "svd_xt":
    num_frames = 25
    num_steps = 30
    model_config = "./svd/scripts/sampling/configs/svd_xt.yaml"
else:
    raise ValueError(f"Version {version} does not exist.")

model, filter = load_model(
    model_config,
    device,
    num_frames,
    num_steps,
)

def sample(
    image: Image,
    seed: Optional[int] = None,
    randomize_seed: bool = True,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    version: str = "svd_xt",
    cond_aug: float = 0.02,
    decoding_t: int = 5,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: str = "output",
    progress=gr.Progress(track_tqdm=True)
):
    if(randomize_seed):
        seed = random.randint(0, max_64_bit_int)
        
    torch.manual_seed(seed)
    
    if image.mode == "RGBA":
        image = image.convert("RGB")
    w, h = image.size

    if h % 64 != 0 or w % 64 != 0:
        width, height = map(lambda x: x - x % 64, (w, h))
        image = image.resize((width, height))
        print(
            f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
        )

    image = ToTensor()(image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)
    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {}
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames_without_noise"] = image
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    value_dict["cond_aug"] = cond_aug

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            base_count = len(glob(os.path.join(output_folder, "*.mp4")))
            video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_id + 1,
                (samples.shape[-1], samples.shape[-2]),
            )

            samples = embed_watermark(samples)
            samples = filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            for frame in vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            writer.release()
    return video_path, seed

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def resize_image(image, output_size=(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2

def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")
    

def process_upload_scribble(scribble_prompt, scribble):
    controlNetOut = process(det="Scrible_HED", input_image=scribble, prompt=scribble_prompt, a_prompt="best quality", n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality", num_samples=1, image_resolution=512, detect_resolution=512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=12345, eta=1.0)[1]
    return controlNetOut

def query(prompt, is_negative=False, steps=1, cfg_scale=6, seed=None):
    payload = {
        "inputs": prompt,
        "is_negative": is_negative,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed if seed is not None else random.randint(-1, 2147483647)
        }

    image_bytes = requests.post(API_URL, headers=headers, json=payload).content
    image = Image.open(io.BytesIO(image_bytes))
    return image


def demo():
    with gr.Blocks(analytics_enabled=False) as iface:
        gr.Markdown("<div align='center'> <h1> Ad Asset Generator </span> </h1> </div>")
        
        # Text2Image: DALL-E 3
        with gr.Tab(label='Text2Image'):
            gr.HTML(
                """
                    <div style="text-align: center; margin: 0 auto;">
                    <div
                        style="
                        display: inline-flex;
                        align-items: center;
                        gap: 0.8rem;
                        font-size: 1.75rem;
                        "
                    >
                        <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                        DALL•E 3 XL
                        </h1> 
                    </div>
                    <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                        This space demonstrates the work of the model <a style="text-decoration: underline;" href="https://huggingface.co/openskyml/dalle-3-xl">openskyml/dalle-3-xl</a>.
                    </p>
                    </div>
                """
            )

            gr.Markdown(
                """
                For high quality text-to-image generation you can also try the  [Bing Image Creator](https://www.bing.com/images/create)
                """
            )
            with gr.Row():
                image_output = gr.Image(type="pil", label="Output Image", elem_id="gallery")
                with gr.Column(elem_id="prompt-container"):
                    text_prompt = gr.Textbox(label="Prompt", placeholder="a cute cat", lines=1, elem_id="prompt-text-input")
                    text_button = gr.Button("Generate", variant='primary', elem_id="gen-button")

            with gr.Accordion("Advanced settings", open=False):
                negative_prompt = gr.Textbox(label="Negative Prompt", value="text, blurry, fuzziness", lines=1, elem_id="negative-prompt-text-input")

            text_button.click(query, inputs=[text_prompt, negative_prompt], outputs=image_output)

        # Image Editting: Ledits++
        with gr.Tab(label='Image Editting'):
                def update_counter(sega_concepts_counter, concept1, concept2, concept3):
                    if sega_concepts_counter == "":
                        sega_concepts_counter = sum(1 for concept in (concept1, concept2, concept3) if concept != '')
                    return sega_concepts_counter
                def remove_concept(sega_concepts_counter, row_triggered):
                    sega_concepts_counter -= 1
                    rows_visibility = [gr.update(visible=False) for _ in range(4)]
                
                    if(row_triggered-1 > sega_concepts_counter):
                        rows_visibility[sega_concepts_counter] = gr.update(visible=True)
                    else:
                        rows_visibility[row_triggered-1] = gr.update(visible=True)
                
                    row1_visibility, row2_visibility, row3_visibility, row4_visibility = rows_visibility

                    guidance_scale_label = "Concept Guidance Scale"
                    # enable_interactive =  gr.update(interactive=True)
                    return (gr.update(visible=False),
                            gr.update(visible=False, value="",),
                            gr.update(interactive=True, value=""),
                            gr.update(visible=False,label = guidance_scale_label),
                            gr.update(interactive=True, value =False),
                            gr.update(value=DEFAULT_WARMUP_STEPS),
                            gr.update(value=DEFAULT_THRESHOLD),
                            gr.update(visible=True),
                            gr.update(interactive=True, value="custom"),
                            row1_visibility,
                            row2_visibility,
                            row3_visibility,
                            row4_visibility,
                            sega_concepts_counter
                            ) 
                
                
                
                def update_display_concept(button_label, edit_concept, neg_guidance, sega_concepts_counter):
                    sega_concepts_counter += 1
                    guidance_scale_label = "Concept Guidance Scale"
                    if(button_label=='Remove'):
                        neg_guidance = True
                        guidance_scale_label = "Negative Guidance Scale" 
                
                    return (gr.update(visible=True), #boxn
                            gr.update(visible=True, value=edit_concept), #concept_n
                            gr.update(visible=True,label = guidance_scale_label), #guidance_scale_n
                            gr.update(value=neg_guidance),#neg_guidance_n
                            gr.update(visible=False), #row_n
                            gr.update(visible=True), #row_n+1
                            sega_concepts_counter
                            ) 


                def display_editing_options(run_button, clear_button, sega_tab):
                    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                
                def update_interactive_mode(add_button_label):
                    if add_button_label == "Clear":
                        return gr.update(interactive=False), gr.update(interactive=False)
                    else:
                        return gr.update(interactive=True), gr.update(interactive=True)
                
                def update_dropdown_parms(dropdown):
                    if dropdown == 'custom':
                        return DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD
                    elif dropdown =='style':
                        return STYLE_SEGA_CONCEPT_GUIDANCE_SCALE,STYLE_WARMUP_STEPS, STYLE_THRESHOLD
                    elif dropdown =='object':
                        return OBJECT_SEGA_CONCEPT_GUIDANCE_SCALE,OBJECT_WARMUP_STEPS, OBJECT_THRESHOLD
                    elif dropdown =='faces':
                        return FACE_SEGA_CONCEPT_GUIDANCE_SCALE,FACE_WARMUP_STEPS, FACE_THRESHOLD


                def reset_do_inversion():
                    return True

                def reset_do_reconstruction():
                    do_reconstruction = True
                    return  do_reconstruction

                def reset_image_caption():
                    return ""

                def update_inversion_progress_visibility(input_image, do_inversion):
                    if do_inversion and not input_image is None:
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)

                def update_edit_progress_visibility(input_image, do_inversion):
                    return gr.update(visible=True)

                wts = gr.State()
                zs = gr.State()
                attention_store=gr.State()
                text_cross_attention_maps = gr.State()
                reconstruction = gr.State()
                do_inversion = gr.State(value=True)
                do_reconstruction = gr.State(value=True)
                sega_concepts_counter = gr.State(0)
                image_caption = gr.State(value="")

                with gr.Row():
                    input_image = gr.Image(label="Input Image", interactive=True, elem_id="input_image")
                    ddpm_edited_image = gr.Image(label=f"Pure DDPM Inversion Image", interactive=False, visible=False)
                    sega_edited_image = gr.Image(label=f"LEDITS Edited Image", interactive=False, elem_id="output_image")

                with gr.Row():
                    with gr.Group(visible=False, elem_id="box1") as box1:
                        with gr.Row():
                            concept_1 = gr.Button(scale=3, value="")
                            remove_concept1 = gr.Button("x", scale=1, min_width=10)
                        with gr.Row():
                            guidnace_scale_1 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                                                info="How strongly the concept should modify the image",
                                                                    value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                                    step=0.5, interactive=True)
                    with gr.Group(visible=False, elem_id="box2") as box2:
                        with gr.Row():
                            concept_2 = gr.Button(scale=3, value="")
                            remove_concept2 = gr.Button("x", scale=1, min_width=10)
                        with gr.Row():
                            guidnace_scale_2 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                                                info="How strongly the concept should modify the image",
                                                                        value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                                        step=0.5, interactive=True)
                    with gr.Group(visible=False, elem_id="box3") as box3:
                        with gr.Row():
                            concept_3 = gr.Button(scale=3, value="")
                            remove_concept3 = gr.Button("x", scale=1, min_width=10)
                        with gr.Row():
                            guidnace_scale_3 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                                                info="How strongly the concept should modify the image",
                                                                        value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                                        step=0.5, interactive=True)


                with gr.Row():
                    inversion_progress = gr.Textbox(visible=False, label="Inversion progress")
                    
                with gr.Group():
                    intro_segs = gr.Markdown("Add/Remove Concepts from your Image <span style=\"font-size: 12px; color: rgb(156, 163, 175)\">with Semantic Guidance</span>")
                            # 1st SEGA concept
                    with gr.Row() as row1:
                        with gr.Column(scale=3, min_width=100):
                            with gr.Row():
                                edit_concept_1 = gr.Textbox(
                                                label="Concept",
                                                show_label=True,
                                                max_lines=1, value="",
                                                placeholder="E.g.: Sunglasses",
                                            )

                                dropdown1 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])
                

                        with gr.Column(scale=1, min_width=100, visible=False):
                                neg_guidance_1 = gr.Checkbox(label='Remove Concept?')
                        
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(): # better mobile ui
                                with gr.Column():
                                    add_1 = gr.Button('Add')
                                    remove_1 = gr.Button('Remove')
                        
                
                            # 2nd SEGA concept
                    with gr.Row(visible=False) as row2:
                        with gr.Column(scale=3, min_width=100):
                            with gr.Row(): #better mobile UI
                                edit_concept_2 = gr.Textbox(
                                                label="Concept",
                                                show_label=True,
                                                max_lines=1,
                                                placeholder="E.g.: Realistic",
                                            )
                                dropdown2 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])

                        with gr.Column(scale=1, min_width=100, visible=False):
                                neg_guidance_2 = gr.Checkbox(label='Remove Concept?')
                            
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(): # better mobile ui
                                with gr.Column():
                                    add_2 = gr.Button('Add')
                                    remove_2 = gr.Button('Remove')
                
                            # 3rd SEGA concept
                    with gr.Row(visible=False) as row3:
                        with gr.Column(scale=3, min_width=100):
                            with gr.Row(): #better mobile UI  
                                edit_concept_3 = gr.Textbox(
                                                label="Concept",
                                                show_label=True,
                                                max_lines=1,
                                                placeholder="E.g.: orange",
                                            )
                                dropdown3 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])
                        
                        with gr.Column(scale=1, min_width=100, visible=False):
                                neg_guidance_3 = gr.Checkbox(label='Remove Concept?',visible=True)
                        
                        with gr.Column(scale=1, min_width=100):
                            with gr.Row(): # better mobile ui
                                with gr.Column():
                                    add_3 = gr.Button('Add')
                                    remove_3 = gr.Button('Remove')
                
                    with gr.Row(visible=False) as row4:
                        gr.Markdown("### Max of 3 concepts reached. Remove a concept to add more")
                    
                
                with gr.Row():
                    run_button = gr.Button("Edit your image!", visible=True)
                    

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        tar_prompt = gr.Textbox(
                                        label="Describe your edited image (optional)",
                                        elem_id="target_prompt",
                                        # show_label=False,
                                        max_lines=1, value="", scale=3,
                                        placeholder="Target prompt, DDPM Inversion", info = "DPM Solver++ Inversion Prompt. Can help with global changes, modify to what you would like to see"
                                    )
                    with gr.Tabs() as tabs:

                        with gr.TabItem('General options', id=2):
                            with gr.Row():
                                with gr.Column(min_width=100):
                                    clear_button = gr.Button("Clear", visible=True)
                                    src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="")
                                    steps = gr.Number(value=50, precision=0, label="Num Diffusion Steps", interactive=True)
                                    src_cfg_scale = gr.Number(value=3.5, label=f"Source Guidance Scale", interactive=True)
                                    mask_type = gr.Radio(choices=["No mask", "Cross Attention Mask", "Intersect Mask"], value="Intersect Mask", label="Mask type")

                                with gr.Column(min_width=100):
                                    reconstruct_button = gr.Button("Show Reconstruction", visible=False)
                                    skip = gr.Slider(minimum=0, maximum=95, value=25, step=1, label="Skip Steps", interactive=True, info = "Percentage of skipped denoising steps. Bigger values increase fidelity to input image")
                                    tar_cfg_scale = gr.Slider(minimum=1, maximum=30,value=7.5, label=f"Guidance Scale", interactive=True)
                                    seed = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
                                    randomize_seed = gr.Checkbox(label='Randomize seed', value=False)

                        with gr.TabItem('SEGA options', id=3) as sega_advanced_tab:
                            # 1st SEGA concept
                            gr.Markdown("1st concept")
                            with gr.Row():
                                warmup_1 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                                    value=DEFAULT_WARMUP_STEPS,
                                                    step=1, interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                                threshold_1 = gr.Slider(label='Threshold', minimum=0, maximum=0.99,
                                                        value=DEFAULT_THRESHOLD, step=0.01, interactive=True, 
                                                        info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")

                            # 2nd SEGA concept
                            gr.Markdown("2nd concept")
                            with gr.Row() as row2_advanced:
                                warmup_2 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                                    value=DEFAULT_WARMUP_STEPS,
                                                    step=1, interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                                threshold_2 = gr.Slider(label='Threshold', minimum=0, maximum=0.99,
                                                        value=DEFAULT_THRESHOLD,
                                                        step=0.01, interactive=True,
                                                        info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")
                            # 3rd SEGA concept
                            gr.Markdown("3rd concept")
                            with gr.Row() as row3_advanced:
                                warmup_3 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                                    value=DEFAULT_WARMUP_STEPS, step=1,
                                                    interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                                threshold_3 = gr.Slider(label='Threshold', minimum=0, maximum=0.99,
                                                        value=DEFAULT_THRESHOLD, step=0.01,
                                                        interactive=True,
                                                        info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")

                add_1.click(fn=update_counter,
                            inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3],
                            outputs=sega_concepts_counter,queue=False).then(fn = update_display_concept, inputs=[add_1, edit_concept_1, neg_guidance_1, sega_concepts_counter],  outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1,row1, row2, sega_concepts_counter],queue=False)
                add_2.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(fn = update_display_concept, inputs=[add_2, edit_concept_2, neg_guidance_2, sega_concepts_counter],  outputs=[box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3, sega_concepts_counter],queue=False)
                add_3.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(fn = update_display_concept, inputs=[add_3, edit_concept_3, neg_guidance_3, sega_concepts_counter],  outputs=[box3, concept_3, guidnace_scale_3,neg_guidance_3,row3, row4, sega_concepts_counter],queue=False)
                
                remove_1.click(fn = update_display_concept, inputs=[remove_1, edit_concept_1, neg_guidance_1, sega_concepts_counter],  outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1,row1, row2, sega_concepts_counter],queue=False)
                remove_2.click(fn = update_display_concept, inputs=[remove_2, edit_concept_2, neg_guidance_2 ,sega_concepts_counter],  outputs=[box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3,sega_concepts_counter],queue=False)
                remove_3.click(fn = update_display_concept, inputs=[remove_3, edit_concept_3, neg_guidance_3, sega_concepts_counter],  outputs=[box3, concept_3, guidnace_scale_3,neg_guidance_3, row3, row4, sega_concepts_counter],queue=False)
                
                remove_concept1.click(
                    fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(
                    fn = remove_concept, inputs=[sega_concepts_counter,gr.State(1)], outputs= [box1, concept_1, edit_concept_1, guidnace_scale_1,neg_guidance_1,warmup_1, threshold_1, add_1, dropdown1, row1, row2, row3, row4, sega_concepts_counter],queue=False)
                remove_concept2.click(
                    fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(
                    fn = remove_concept,  inputs=[sega_concepts_counter,gr.State(2)], outputs=[box2, concept_2, edit_concept_2, guidnace_scale_2,neg_guidance_2, warmup_2, threshold_2, add_2 , dropdown2, row1, row2, row3, row4, sega_concepts_counter],queue=False)
                remove_concept3.click(
                    fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter,queue=False).then(
                    fn = remove_concept,inputs=[sega_concepts_counter,gr.State(3)], outputs=[box3, concept_3, edit_concept_3, guidnace_scale_3,neg_guidance_3,warmup_3, threshold_3,  add_3, dropdown3, row1, row2, row3, row4, sega_concepts_counter],queue=False)

                run_button.click(
                    fn=edit,
                    inputs=[input_image,
                            wts, zs, attention_store,
                            text_cross_attention_maps,
                            tar_prompt,
                            image_caption,
                            steps,
                            skip,
                            tar_cfg_scale,
                            edit_concept_1,edit_concept_2,edit_concept_3,
                            guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
                            warmup_1, warmup_2, warmup_3,
                            neg_guidance_1, neg_guidance_2, neg_guidance_3,
                            threshold_1, threshold_2, threshold_3, do_reconstruction, reconstruction,
                            do_inversion,
                            seed, 
                            randomize_seed,
                            src_prompt,
                            src_cfg_scale,
                            mask_type
                    ],
                    outputs=[sega_edited_image, reconstruct_button, do_reconstruction, reconstruction, wts, zs,attention_store, text_cross_attention_maps, do_inversion]
                    
                )



                input_image.change(
                    fn = reset_do_inversion,
                    outputs = [do_inversion],
                    queue=False
                ).then(
                    fn = randomize_seed_fn,
                    inputs = [seed, randomize_seed],
                    outputs = [seed],
                    queue=False
                )
                
                # Automatically start inverting upon input_image change
                input_image.upload(
                    fn = crop_image,
                    inputs = [input_image],
                    outputs = [input_image],
                    queue=False
                ).then(
                    fn = reset_do_inversion,
                    outputs = [do_inversion],
                    queue=False     
                ).then(
                    fn = randomize_seed_fn,
                    inputs = [seed, randomize_seed],
                    outputs = [seed], 
                    queue=False     
                ).then(fn = caption_image,
                    inputs = [input_image],
                    outputs = [tar_prompt, image_caption],
                    queue=False    
                )

                # Repeat inversion (and reconstruction) when these params are changed:
                src_prompt.change(
                    fn = reset_do_inversion,
                    outputs = [do_inversion],
                    queue = False
                ).then(
                    fn = reset_do_reconstruction,
                    outputs = [do_reconstruction],
                    queue = False
                )

                steps.change(
                    fn = reset_do_inversion,
                    outputs = [do_inversion],
                    queue = False
                ).then(
                    fn = reset_do_reconstruction,
                    outputs = [do_reconstruction],
                    queue = False
                )

                src_cfg_scale.change(
                    fn = reset_do_inversion,
                    outputs = [do_inversion],
                    queue = False
                ).then(
                    fn = reset_do_reconstruction,
                    outputs = [do_reconstruction],
                    queue = False
                )

                # Repeat only reconstruction these params are changed:
                tar_prompt.change(
                    fn = reset_do_reconstruction,
                    outputs = [do_reconstruction],
                    queue = False
                )

                tar_cfg_scale.change(
                    fn = reset_do_reconstruction,
                    outputs = [do_reconstruction],
                    queue = False
                )

                skip.change(
                    fn = reset_do_inversion,
                    outputs = [do_inversion],
                    queue = False
                ).then(
                    fn = reset_do_reconstruction,
                    outputs = [do_reconstruction],
                    queue = False
                )

                seed.change(
                    fn=reset_do_inversion,
                    outputs=[do_inversion],
                    queue=False
                ).then(
                    fn=reset_do_reconstruction,
                    outputs=[do_reconstruction],
                    queue=False
                )

                dropdown1.change(fn=update_dropdown_parms, inputs = [dropdown1], outputs = [guidnace_scale_1,warmup_1,  threshold_1], queue=False)
                dropdown2.change(fn=update_dropdown_parms, inputs = [dropdown2], outputs = [guidnace_scale_2,warmup_2,  threshold_2], queue=False)
                dropdown3.change(fn=update_dropdown_parms, inputs = [dropdown3], outputs = [guidnace_scale_3,warmup_3,  threshold_3], queue=False)

                clear_components = [input_image,ddpm_edited_image,ddpm_edited_image,sega_edited_image, do_inversion,
                                            src_prompt, steps, src_cfg_scale, seed,
                                            tar_prompt, skip, tar_cfg_scale, reconstruct_button,reconstruct_button,
                                            edit_concept_1, guidnace_scale_1,guidnace_scale_1,warmup_1,  threshold_1, neg_guidance_1,dropdown1, concept_1, concept_1, row1,
                                            edit_concept_2, guidnace_scale_2,guidnace_scale_2,warmup_2,  threshold_2, neg_guidance_2,dropdown2, concept_2, concept_2, row2,
                                            edit_concept_3, guidnace_scale_3,guidnace_scale_3,warmup_3,  threshold_3, neg_guidance_3,dropdown3, concept_3,concept_3, row3,
                                            row4,sega_concepts_counter, box1, box2, box3 ]

                clear_components_output_vals = [None, None,gr.update(visible=False), None, True,
                                "", DEFAULT_DIFFUSION_STEPS, DEFAULT_SOURCE_GUIDANCE_SCALE, DEFAULT_SEED,
                                "", DEFAULT_SKIP_STEPS, DEFAULT_TARGET_GUIDANCE_SCALE, gr.update(value="Show Reconstruction"),gr.update(visible=False),
                                "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,gr.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "custom","", gr.update(visible=False), gr.update(visible=True),
                                "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,gr.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "custom","", gr.update(visible=False), gr.update(visible=False),
                                "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,gr.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "custom","",gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=0),
                                    gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)]


                clear_button.click(lambda: clear_components_output_vals, outputs = clear_components)

                reconstruct_button.click(lambda: ddpm_edited_image.update(visible=True), outputs=[ddpm_edited_image]).then(fn = reconstruct,
                            inputs = [tar_prompt,
                            image_caption,
                            tar_cfg_scale,
                            skip,
                            wts, zs,
                            do_reconstruction,
                            reconstruction,
                                    reconstruct_button],
                            outputs = [ddpm_edited_image,reconstruction, ddpm_edited_image, do_reconstruction, reconstruct_button])

                randomize_seed.change(
                    fn = randomize_seed_fn,
                    inputs = [seed, randomize_seed],
                    outputs = [seed],
                    queue = False)

        # Scribble2Image: ControlNet
        with gr.Tab(label='Scribble2Image'):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            scribble_upload = gr.Image(label="Upload your scribble")
                        with gr.Row():
                            scribble_upload_prompt = gr.Text(label='Describe your scene')
                        s2i_end_btn = gr.Button("Submit")
                    with gr.Tab(label='Result'):
                        with gr.Row():
                            output_image = gr.Image(label="Output")
            s2i_end_btn.click(inputs=[scribble_upload_prompt, scribble_upload],
                            outputs=[output_image],
                            fn = process_upload_scribble
            )

        # Scribble Interactive: ControlNet
        with gr.Tab(label='Scribble Interactive'):
            with gr.Row():
                gr.Markdown("## Control Stable Diffusion with Interactive Scribbles")
            with gr.Row():
                with gr.Column():
                    canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=1)
                    canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=1)
                    create_button = gr.Button(label="Start", value='Open drawing canvas!')
                    input_image = gr.Image(source='upload', type='numpy', tool='sketch')
                    gr.Markdown(value='Do not forget to change your brush width to make it thinner. '
                                    'Just click on the small pencil icon in the upper right corner of the above block.')
                    create_button.click(fn=create_canvas, inputs=[canvas_width, canvas_height], outputs=[input_image])
                    prompt = gr.Textbox(label="Prompt")
                    run_button = gr.Button(label="Run")
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
                    with gr.Accordion("Advanced options", open=False):
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
            run_button.click(fn=process_interactive, inputs=ips, outputs=[result_gallery])

        # Image-to-3D: DreamGaussian
        with gr.Tab(label='Image-to-3D'):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown('# ' + _TITLE)

            # Image-to-3D
            with gr.Row(variant='panel'):
                left_column = gr.Column(scale=5)
                with left_column:
                    image_block = gr.Image(type='pil', image_mode='RGBA', height=290, label='Input image', tool=None)

                    elevation_slider = gr.Slider(-90, 90, value=0, step=1, label='Estimated elevation angle')
                    gr.Markdown(
                        "default to 0 (horizontal), range from [-90, 90]. If you upload a look-down image, try a value like -30")

                    preprocess_chk = gr.Checkbox(True,
                                                label='Preprocess image automatically (remove background and recenter object)')



                with gr.Column(scale=5):
                    obj3d_stage1 = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Stage 1)")
                    obj3d = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model (Final)")

                with left_column:
                    img_run_btn = gr.Button("Generate 3D")
                    img_guide_text = gr.Markdown(_IMG_USER_GUIDE, visible=True)

                # if there is an input image, continue with inference
                # else display an error message
                img_run_btn.click(check_img_input, inputs=[image_block], queue=False).success(optimize_stage_1,
                                                                                            inputs=[image_block,
                                                                                                    preprocess_chk,
                                                                                                    elevation_slider],
                                                                                            outputs=[
                                                                                                obj3d_stage1]).success(
                    optimize_stage_2, inputs=[image_block, elevation_slider], outputs=[obj3d])

        # Text2Video: LaVie
        with gr.Tab(label='Text2Video'):
            gr.Markdown("<font color=red size=10><center>LaVie: Text-to-Video generation</center></font>")
            gr.Markdown(
                """<div style="text-align:center">
                [<a href="https://arxiv.org/abs/2309.15103">Arxiv Report</a>] | [<a href="https://vchitect.github.io/LaVie-project/">Project Page</a>] | [<a href="https://github.com/Vchitect/LaVie">Github</a>]</div>
                """
            )
            with gr.Column():
                with gr.Row(elem_id="col-container"):
                    with gr.Column():
                            
                        prompt = gr.Textbox(value="a corgi walking in the park at sunrise, oil painting style", label="Prompt", placeholder="enter prompt", show_label=True, elem_id="prompt-in", min_width=200, lines=2)
                        infer_type = gr.Dropdown(['ddpm','ddim','eulerdiscrete'], label='infer_type',value='ddim')
                        ddim_steps = gr.Slider(label='Steps', minimum=50, maximum=300, value=50, step=1)
                        seed_inp = gr.Slider(value=-1,label="seed (for random generation, use -1)",show_label=True,minimum=-1,maximum=2147483647)
                        cfg = gr.Number(label="guidance_scale",value=7.5)

                    with gr.Column():
                        submit_btn = gr.Button("Generate video")
                        video_out = gr.Video(label="Video result", elem_id="video-output")

                    inputs = [prompt, seed_inp, ddim_steps, cfg, infer_type]
                    outputs = [video_out]                
            submit_btn.click(infer, inputs, outputs)

        # Image2Video: Stable Video Diffusion
        with gr.Tab(label='Image2Video'):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            image = gr.Image(label="Upload your image", type="pil")
                        generate_btn = gr.Button("Generate")
                        image.upload(fn=resize_image, inputs=image, outputs=image, queue=False)
                        with gr.Accordion("Advanced options", open=False):
                            seed = gr.Slider(label="Seed", value=42, randomize=True, minimum=0, maximum=max_64_bit_int, step=1)
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                            motion_bucket_id = gr.Slider(label="Motion bucket id", info="Controls how much motion to add/remove from the image", value=127, minimum=1, maximum=255)
                            fps_id = gr.Slider(label="Frames per second", info="The length of your video in seconds will be 25/fps", value=6, minimum=5, maximum=30)
                    with gr.Tab(label='Result'):
                        with gr.Row():
                            output_video = gr.Video(label="Ad Video", format="mp4")
            generate_btn.click(fn=sample, inputs=[image, seed, randomize_seed, motion_bucket_id, fps_id], outputs=[output_video, seed], api_name="video")
        
        # Text2Music: MusicGen
        with gr.Tab(label='Text2Music'):
            gr.Markdown(
                """
                # MusicGen
                This is your private demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
                a simple and controllable model for music generation
                presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
                """
            )
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        text = gr.Text(label="Input Text", interactive=True)
                        with gr.Column():
                            radio = gr.Radio(["file", "mic"], value="file",
                                            label="Condition on a melody (optional) File or Mic")
                            melody = gr.Audio(source="upload", type="numpy", label="File",
                                            interactive=True, elem_id="melody-input")
                    with gr.Row():
                        submit = gr.Button("Submit")
                    with gr.Row():
                        model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                        "facebook/musicgen-large"],
                                        label="Model", value="facebook/musicgen-melody", interactive=True)
                    with gr.Row():
                        decoder = gr.Radio(["Default", "MultiBand_Diffusion"],
                                        label="Decoder", value="Default", interactive=True)
                    with gr.Row():
                        duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                    with gr.Row():
                        topk = gr.Number(label="Top-k", value=250, interactive=True)
                        topp = gr.Number(label="Top-p", value=0, interactive=True)
                        temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                        cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Column():
                    output = gr.Video(label="Generated Music")
                    audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
                    diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")
                    audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')
            submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                        show_progress=False).then(predict_full, inputs=[model, decoder, text, melody, duration, topk, topp,
                                                                        temperature, cfg_coef],
                                                outputs=[output, audio_output, diffusion_output, audio_diffusion])
            radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)

        # Speech Synthesis: Bark
        with gr.Tab(label='Speech Synthesis'):
            gr.Markdown(description)
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text", lines=2, value=default_text, elem_id="input_text")
                    options = gr.Dropdown(
                        AVAILABLE_PROMPTS, value="Speaker 1 (en)", label="Acoustic Prompt", elem_id="speaker_option")
                    run_button = gr.Button(text="Generate Audio", type="button")
                with gr.Column():
                    audio_out = gr.Audio(label="Generated Audio",
                                        type="numpy", elem_id="audio_out")
            inputs = [input_text, options]
            outputs = [audio_out]
            gr.Markdown(article)
            run_button.click(fn=lambda: gr.update(visible=True), inputs=None, outputs=outputs, queue=False).then(
                fn=gen_tts, inputs=inputs, outputs=outputs, queue=True).then(
                fn=lambda: gr.update(visible=True), inputs=None, outputs=outputs, queue=False)

    return iface

if __name__ == "__main__":
    iface = demo()
    iface.queue(concurrency_count=1, max_size=10)
    iface.launch(share=True)
