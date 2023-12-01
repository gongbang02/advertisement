import gradio as gr
import cv2
import math
import os
import torch
from glob import glob
from pathlib import Path
from typing import Optional
import numpy as np
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
import PIL.Image
import uuid
import random
from huggingface_hub import hf_hub_download
from gradio.components import Textbox, Image, Video
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip, AudioFileClip
from controlnet.gradio_scribble import process
from audiocraft.demos.musicgen_app import predict_full
from svd.scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from svd.sgm.inference.helpers import embed_watermark
from svd.sgm.util import default, instantiate_from_config


hf_hub_download(repo_id="stabilityai/stable-video-diffusion-img2vid-xt", filename="svd_xt.safetensors", local_dir="checkpoints") 

version = "svd_xt"
device = "cuda"
max_64_bit_int = 2**63 - 1

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
    output_folder: str = "outputs",
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
                0x00000021,
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
            print(video_path)
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
        resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), PIL.Image.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

def predict(scribble_prompt, music_prompt, scribble):
    controlNetOut = process(det="Scrible_HED", input_image=scribble, prompt=scribble_prompt, a_prompt="best quality", n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality", num_samples=1, image_resolution=512, detect_resolution=512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=12345, eta=1.0)[1]
    # repeat controlNetOut to create a 10-second video
    imgs = []
    for i in range(0, 300):
        imgs.append(controlNetOut)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(imgs, fps=30)
    videoPath = 'video_with_music.mp4'
    clip.write_videofile(videoPath)
    clip = VideoFileClip(videoPath)
    vidDuration = clip.duration
    musicOut = predict_full(model="facebook/musicgen-medium", decoder="MultiBand_Diffusion", text=music_prompt, melody=None, duration=vidDuration, topk=250, topp=0, temperature=1.0, cfg_coef=3.0)[1]
    video_clip = VideoFileClip(videoPath)
    audio_clip = AudioFileClip(musicOut)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("video_with_music.mp4", fps=30, threads=1, codec="libx264")
    return os.path.join('./', 'video_with_music.mp4')


def demo():
    with gr.Blocks(analytics_enabled=False) as iface:
        gr.Markdown("<div align='center'> <h1> Ad Asset Generator </span> </h1> </div>")
        # Scratch2Video with ControlNet, MusicGen
        with gr.Tab(label='Scratch2Video'):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            scribble = gr.Image(label="Upload your scribble")
                        with gr.Row():
                            scribble_prompt = gr.Text(label='Describe your scene')
                        with gr.Row():
                            music_prompt = gr.Text(label='Describe your bgm')
                        i2v_end_btn = gr.Button("Submit")
                    with gr.Tab(label='Result'):
                        with gr.Row():
                            output_video = gr.Video(label="Ad Video", format="mp4")
            i2v_end_btn.click(inputs=[scribble_prompt, music_prompt, scribble],
                            outputs=[output_video],
                            fn = predict
            )
        # Image2Video with Stable Video Diffusion
        with gr.Tab(label='Image2Video'):
            gr.Markdown('''# Community demo for Stable Video Diffusion - Img2Vid - XT ([model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt), [paper](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets), [stability's ui waitlist](https://stability.ai/contact))
#### Research release ([_non-commercial_](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE)): generate `4s` vid from a single image at (`25 frames` at `6 fps`). this demo uses [🧨 diffusers for low VRAM and fast generation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd).
  ''')
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="Upload your image", type="pil")
                    generate_btn = gr.Button("Generate")
                video = gr.Video(type="filepath")
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Slider(label="Seed", value=42, randomize=True, minimum=0, maximum=max_64_bit_int, step=1)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                motion_bucket_id = gr.Slider(label="Motion bucket id", info="Controls how much motion to add/remove from the image", value=127, minimum=1, maximum=255)
                fps_id = gr.Slider(label="Frames per second", info="The length of your video in seconds will be 25/fps", value=6, minimum=5, maximum=30)
      
            image.upload(fn=resize_image, inputs=image, outputs=image, queue=False)
            generate_btn.click(fn=sample, inputs=[image, seed, randomize_seed, motion_bucket_id, fps_id], outputs=[video, seed], api_name="video")

    return iface

if __name__ == "__main__":
    iface = demo()
    iface.queue(concurrency_count=1, max_size=10)
    iface.launch(share=True)
