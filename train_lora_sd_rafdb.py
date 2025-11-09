import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from accelerate import Accelerator

# --------------------------
# 1. Dataset Definition
# --------------------------
class RAFDBDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.dataset_dir = os.path.join(root_dir, "DATASET", split)
        self.labels_path = os.path.join(root_dir, f"{split}_labels.csv")

        df = pd.read_csv(self.labels_path)
        df.columns = ["filename", "label"]
        self.df = df

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # emotion map
        self.emotions = {
            1: "surprise", 2: "fear", 3: "disgust",
            4: "happy", 5: "sad", 6: "anger", 7: "neutral"
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Images are organized in subdirectories by label: DATASET/{split}/{label}/{filename}
        img_path = os.path.join(self.dataset_dir, str(row.label), row.filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        emotion = self.emotions[row.label]

        # prompt = f"a portrait of a person showing {emotion} emotion"
        prompt = f"{emotion}"
        return {"pixel_values": image, "prompt": prompt}


# --------------------------
# 2. Load Stable Diffusion + LoRA layers
# --------------------------
def create_lora_attn_procs(unet, rank=4):
    attn_procs = {}
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Module) and hasattr(module, "transformer_blocks"):
            continue

    # diffusers provides ready LoRA wrapper:
    lora_attn_procs = AttnProcsLayers.from_unet(unet)
    lora_attn_procs.set_attention_processor(rank=rank)
    return lora_attn_procs


# --------------------------
# 3. Training Script
# --------------------------
def train(
    dataset_path="data/rafdb",
    model_name="CompVis/stable-diffusion-v1-4",
    output_dir="lora-output",
    train_batch_size=2,
    learning_rate=1e-4,
    max_steps=500
):
    accelerator = Accelerator(mixed_precision="fp16")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    unet = pipe.unet

    # Add LoRA layers
    lora_attn_procs = create_lora_attn_procs(unet)
    unet.set_attn_processor(lora_attn_procs)

    # Freeze everything except LoRA params
    for p in unet.parameters():
        p.requires_grad = False
    for p in lora_attn_procs.parameters():
        p.requires_grad = True

    # Dataset + dataloader
    dataset = RAFDBDataset(dataset_path, split="train")
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # Noise scheduler for training
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Optimizer
    optimizer = torch.optim.AdamW(lora_attn_procs.parameters(), lr=learning_rate)

    # Prepare for distributed training
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    global_step = 0
    unet.train()

    for epoch in range(99):  # arbitrary, we stop by step
        for batch in dataloader:

            # encode prompt text
            text_embeds = pipe.text_encoder(
                pipe.tokenizer(batch["prompt"], padding=True, return_tensors="pt").input_ids.to(accelerator.device)
            )[0]

            pixel_values = batch["pixel_values"].to(accelerator.device)
            noise = torch.randn_like(pixel_values)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=accelerator.device).long()

            # Add noise to image
            noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)

            # Pred noise with UNet
            model_pred = unet(noisy_images, timesteps, encoder_hidden_states=text_embeds).sample

            loss = torch.nn.functional.mse_loss(model_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 20 == 0:
                accelerator.print(f"Step {global_step}, loss={loss.item():.4f}")

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    # Save the LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet.save_attn_procs(output_dir)
        print(f"✅ LoRA saved to {output_dir}")


if __name__ == "__main__":
    train()
