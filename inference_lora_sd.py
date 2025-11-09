import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from PIL import Image

def load_lora_unet(base_model, lora_path):
    """
    Load LoRA layers into the UNet of Stable Diffusion.
    """
    # Load LoRA processors
    lora_attn_procs = AttnProcsLayers.load(lora_path, weight_name=None)
    base_model.unet.set_attn_processor(lora_attn_procs)
    print(f"✅ Loaded LoRA weights from: {lora_path}")

def infer(
    prompt="happy",
    lora_dir="lora-output",
    model_name="CompVis/stable-diffusion-v1-4",
    out_path="output/generated.png"
):
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load LoRA weights into the UNet
    load_lora_unet(pipe, lora_dir)

    # Disable safety checker (optional)
    pipe.safety_checker = lambda images, **kwargs: (images, False)

    # Generate image
    image: Image.Image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    # Save output
    image.save(out_path)
    print(f"✅ Image saved at: {out_path}")


if __name__ == "__main__":
    infer()
