# This code uses a local model, not an API.

# conda install conda
# conda --version

# conda create -n env_stable transformers diffusers
# conda activate env_stable
# cpu
# conda install pytorch torchvision torchaudio cpuonly -c pytorch


# librairies
from diffusers import StableDiffusionPipeline

# model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4") # typically stored in the ~/.cache/huggingface/ directory
# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", cache_dir="./models") # if you prefer to save model closer
pipe.to("cpu")

# generate + save pic
prompt = "a majestic mountain landscape"
prompt = "a peaceful sky full of stars"

image = pipe(prompt, height=400, width=400).images[0] # default size 512x512
image.save("output.png")

image = pipe(prompt, height=400, width=400, num_inference_steps=25).images[0] # default inference steps 50 ?
image.save("output_less_steps.png")

image = pipe(prompt, height=400, width=400, num_inference_steps=40, guidance_scale=5.0).images[0] # default guidance_scale=7.5 ?
image.save("output_less_guidance.png")

