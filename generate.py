from diffusers import StableDiffusionPipeline
import torch

PROMPTS = ["resilience", "love", "hope", "curiosity", "strength"]

model_path = "model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

for p in PROMPTS:
    for i in range(20):
        image = pipe(prompt=p).images[0]
        image.save(str(i)+"_"+p+".png")