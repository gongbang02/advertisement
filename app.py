import gradio as gr
import cv2
from gradio.components import Textbox, Image, Video
from moviepy.editor import VideoFileClip, AudioFileClip
from controlnet.gradio_scribble import process
from videocrafter.i2v_test import Image2Video
from audiocraft.demos.musicgen_app import predict_full

def predict(scribble_prompt, music_prompt, scribble):
    controlNetOut = process(det="Scrible_HED", input_image=scribble, prompt=scribble_prompt, a_prompt="best quality", n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality", num_samples=1, image_resolution=512, detect_resolution=512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=12345, eta=1.0)[0]
    crafter = Image2Video()
    #videoPath = crafter.get_image(image=controlNetOut, prompt=scribble_prompt, steps=30, cfg_scale=12.0, eta=1.0, fps=16)
    # video = cv2.VideoCapture(videoPath)
    # vidDuration = video.get(cv2.CAP_PROP_POS_MSEC)
    # video.release()
    # musicOut = predict_full(model="facebook/musicgen-medium", decoder="MultiBand_Diffusion", text=music_prompt, melody=None, duration=vidDuration, topk=250, topp=0, temperature=1.0, cfg_coef=3.0)[1]
    # video_clip = VideoFileClip(videoPath)
    # audio_clip = AudioFileClip(musicOut)
    # final_clip = video_clip.set_audio(audio_clip)
    # final_clip.write_videofile("./video_with_music.mp4", fps=26, threads=1, codec="libx264")
    # return "./video_with_music.mp4"
    return controlNetOut

gr.Interface(
    predict,
    inputs=[Textbox(lines=2, label="Describe your scene"), Textbox(lines=2, label="Describe your bgm"), Image(label="Upload your scribble")],
    outputs=Image(label="Advertisement Image", type="numpy"),
    #outputs=Video(label="Advertisement Video", type="filepath"),
    title="Ad Asset Generator",
    allow_flagging='never'
).launch(share=True)
