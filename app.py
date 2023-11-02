import gradio as gr
from gradio.components import Textbox, Image, Video
from controlnet.gradio_scribble import process
from videocrafter.i2v_test import get_image

def predict(scribble_prompt, scribble):
    controlNetOut = process(det="Scrible_HED", input_image=scribble, prompt=scribble_prompt, a_prompt="best quality", n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality", num_samples=1, image_resolution=512, detect_resolution=512, ddim_steps=30, guess_mode=False, strength=1.0, scale=9.0, seed=12345, eta=1.0)
    videoCrafterOut = get_image(controlNetOut, scribble_prompt, steps=30, cfg_scale=12.0, eta=1.0, fps=16)
    return videoCrafterOut

gr.Interface(
    predict,
    inputs=[Textbox(lines=2, label="Describe your scene"),Image(label="Upload your scribble", type="filepath")],
    outputs=Video(label="Advertisement Video"),
    title="Advertisement Generator",
    allow_flagging='never'
).launch(share=True)
