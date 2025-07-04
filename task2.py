import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("mps")  
pipe.enable_attention_slicing()

# Disabling NSFW filter
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

pipe.safety_checker = dummy_safety_checker


prompt = input("Enter your prompt: ")


image = pipe(prompt).images[0]

# Display 
plt.imshow(image)
plt.axis("off")
plt.title(f"Prompt: {prompt}")
plt.show()


image.save("generated_image.png")
