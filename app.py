import gradio as gr
import cv2
import os
from gradio.components import Textbox, Image, Video
import moviepy.video.io.ImageSequenceClip
from moviepy.editor import VideoFileClip, AudioFileClip
from controlnet.gradio_scribble import process
from videocrafter.i2v_test import Image2Video
from audiocraft.demos.musicgen_app import predict_full

def predict(scribble_prompt, music_prompt, scribble):
    controlNetOut = process(det="Scrible_HED", input_image=scribble, prompt=scribble_prompt, a_prompt="best quality", n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality", num_samples=1, image_resolution=512, detect_resolution=512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=12345, eta=1.0)[1]
    # repeat controlNetOut to create a 10-second video
    imgs = []
    for i in range(0, 300):
        imgs.append(controlNetOut)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(imgs, fps=30)
    videoPath = 'video_with_music.mp4'
    clip.write_videofile(videoPath)
    # crafter = Image2Video()
    # videoPath = crafter.get_image(image=controlNetOut, prompt=scribble_prompt, steps=30, cfg_scale=12.0, eta=1.0, fps=16)
    video = cv2.VideoCapture(videoPath)
    vidDuration = video.get(cv2.CAP_PROP_POS_MSEC)
    video.release()
    musicOut = predict_full(model="facebook/musicgen-medium", decoder="MultiBand_Diffusion", text=music_prompt, melody=None, duration=vidDuration, topk=250, topp=0, temperature=1.0, cfg_coef=3.0)[1]
    video_clip = VideoFileClip(videoPath)
    audio_clip = AudioFileClip(musicOut)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("video_with_music.mp4", fps=30, threads=1, codec="libx264")
    return os.path.join('./', 'video_with_music.mp4')


def demo():
    with gr.Blocks(analytics_enabled=False) as iface:
        gr.Markdown("<div align='center'> <h1> Ad Asset Generator </span> </h1> </div>")
        with gr.Tab(label='Image2Video'):
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

    return iface

if __name__ == "__main__":
    iface = demo()
    iface.queue(concurrency_count=1, max_size=10)
    iface.launch(share=True)
