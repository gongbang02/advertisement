import gradio as gr
import requests
import io
import random
import os
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/openskyml/dalle-3-xl"
API_TOKEN = os.getenv("HF_READ_TOKEN") # it is free
headers = {"Authorization": f"Bearer {API_TOKEN}"}

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
