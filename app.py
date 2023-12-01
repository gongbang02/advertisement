import gradio as gr
import cv2
import os
import torch
from glob import glob
from pathlib import Path
from typing import Optional
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
import uuid
import random
from huggingface_hub import hf_hub_download
from gradio.components import Textbox, Image, Video
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip, AudioFileClip
from controlnet.gradio_scribble import process
from audiocraft.demos.musicgen_app import predict_full


videoDiffuser = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
videoDiffuser.to("cuda")
videoDiffuser.unet = torch.compile(videoDiffuser.unet, mode="reduce-overhead", fullgraph=True)
videoDiffuser.vae = torch.compile(videoDiffuser.vae, mode="reduce-overhead", fullgraph=True)

max_64_bit_int = 2**63 - 1

def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = True,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    version: str = "svd_xt",
    cond_aug: float = 0.02,
    decoding_t: int = 3,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: str = "outputs",
):
    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    if(randomize_seed):
        seed = random.randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)
    
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    frames = videoDiffuser(image, decode_chunk_size=decoding_t, generator=generator, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.1).frames[0]
    export_to_video(frames, video_path, fps=fps_id)
    torch.manual_seed(seed)
    
    return video_path, seed

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
        with gr.Tab(label='Image2Video'):
            gr.Markdown('''# Community demo for Stable Video Diffusion - Img2Vid - XT ([model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt), [paper](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets), [stability's ui waitlist](https://stability.ai/contact))
#### Research release ([_non-commercial_](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE)): generate `4s` vid from a single image at (`25 frames` at `6 fps`). this demo uses [🧨 diffusers for low VRAM and fast generation](https://huggingface.co/docs/diffusers/main/en/using-diffusers/svd).
  ''')
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="Upload your image", type="pil")
                    generate_btn = gr.Button("Generate")
                video = gr.Video()
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
