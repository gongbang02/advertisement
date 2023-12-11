# AI Toolbox for CREATE Design-a-thon

# Set up
Run the following code to set up and activate the environment:
```
conda env create -f environment.yaml
conda activate nih-ad
```
In the conda environment, run the following code to install two other modules:
```
pip install -e ./dreamgaussian/simple-knn
pip install -e ./dreamgaussian/diff-gaussian-rasterization
```
Download the "control_v11p_sd15_scribble.pth" file from [this link](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) and put it in the controlnet/models folder. <br />
Download the "v1-5-pruned.ckpt" file from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and put it in the controlnet/models folder. <br />
Download the "model.safetensors" file from [this link](https://huggingface.co/spaces/Vchitect/LaVie/tree/main/pretrained_models/safety_checker) and put it in the lavie/pretrained_models/safety_checker folder. <br />
Download the "pytorch_model.bin" file from [this link](https://huggingface.co/spaces/Vchitect/LaVie/tree/main/pretrained_models/text_encoder) and put it in the lavie/pretrained_models/text_encoder folder. <br />
Download the "diffusion_pytorch_model.bin" file from [this link](https://huggingface.co/spaces/Vchitect/LaVie/tree/main/pretrained_models/vae) and put it in the lavie/pretrained_models/vae folder. <br />
Download the "lavie_base.pt" file from [this link](https://huggingface.co/spaces/Vchitect/LaVie/tree/main/pretrained_models) and put it in the lavie/pretrained_models folder. <br />
Download the "p_head_v1.npz" and "w_head_v1.npz" file from [this link](https://huggingface.co/spaces/TheVilfer/stable-video-diffusion/tree/main/scripts/util/detection) and put it in the svd/scripts/util/detection folder. <br />
Start the gradio demo by running:
```
python app.py
```
