import os
import torch
import argparse
import torchvision

from lavie.base.pipelines.pipeline_videogen import VideoGenPipeline

from lavie.base.download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from lavie.base.models import get_models
import imageio

config_path = "./lavie/base/configs/sample.yaml"
args = OmegaConf.load("./lavie/base/configs/sample.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_t2v_fun(args):
    # sd_path = args.pretrained_path + "/stable-diffusion-v1-4"
    sd_path = args.pretrained_path
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model("./lavie/pretrained_models/lavie_base.pt")
    # state_dict = find_model("./pretrained_models/lavie_base.pt")
    unet.load_state_dict(state_dict)
    	
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge
    unet.eval()
    vae.eval()
    text_encoder_one.eval()
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler", beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule=args.beta_schedule)
    return VideoGenPipeline(vae=vae, text_encoder=text_encoder_one, tokenizer=tokenizer_one, scheduler=scheduler, unet=unet)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
	

