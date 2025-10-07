import torch
import random
import numpy as np
from PIL import Image
import itertools
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import login
from .utils import create_binary_mask, load_mask


LABEL_MAP = {
    1: "car",
    2: "sky",
    3: "tree",
    4: "building",
    5: "signs",
    6: "road",
    7: "footpath",
    8: "electric light"
}

PROMPT_TEMPLATES = {
    1: lambda: f"the car should be {random.choice(['blue','gray','white'])}, only recolor, keep same shape",
    2: lambda: f"the sky should be {random.choice(['pale blue','soft gray','light pink'])}, subtle hue shift",
    3: lambda: f"the tree foliage should be {random.choice(['green','yellow'])}, only color change",
    4: lambda: f"the building should be {random.choice(['light gray','reddish brown'])}, only wall color change",
    6: lambda: "the road should be with proper lanes",
    7: lambda: f"the footpath should be {random.choice(['light beige','soft gray'])}, only recolor"
}

NEGATIVE_PROMPT = "extra objects, distorted shapes, new patterns, unrealistic textures, overpainting"


def load_pipeline(hf_token=None):
    """Load Stable Diffusion inpainting pipeline."""
    if hf_token:
        login(token=hf_token)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe


def inpaint_stage(pipe, current_image, mask_array, label_id, prompt, steps=20, guidance=4.0):
    """Apply diffusion inpainting for a single label."""
    sub_mask_array = create_binary_mask(mask_array, label_id)
    sub_mask = Image.fromarray(sub_mask_array).convert("L").resize((512, 512))
    current_image = current_image.resize((512, 512))

    output = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=current_image,
        mask_image=sub_mask,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    return output


def run_scene_editing(
    image_path="assets/scene.jpg",
    mask_path="assets/mask_0.png",
    output_dir="outputs",
    num_images=10,
    seed=42,
    hf_token=None
):
    """Main entrypoint for diffusion-based scene editing."""
    import os
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    random.seed(seed)

    mask_array = load_mask(mask_path)
    original_image = Image.open(image_path).convert("RGB")

    pipe = load_pipeline(hf_token=hf_token)
    active_labels = [1, 2, 3, 4, 6, 7] #not taking sign and electric ligt into consideration

    label_pairs = list(itertools.combinations(active_labels, 2))
    random.shuffle(label_pairs)

    for i in range(num_images):
        pair = label_pairs[i % len(label_pairs)]
        label1, label2 = pair

        print(f"\n=== Generating variation {i+1}/{num_images} ===")
        print(f"  Modifying {LABEL_MAP[label1]} and {LABEL_MAP[label2]}")

        prompt1 = PROMPT_TEMPLATES[label1]()
        intermediate = inpaint_stage(pipe, original_image, mask_array, label1, prompt1)

        prompt2 = PROMPT_TEMPLATES[label2]()
        final = inpaint_stage(pipe, intermediate, mask_array, label2, prompt2)

        out_path = Path(output_dir) / f"output_{i+1:02d}.png"
        final.save(out_path)
        print(f"  Saved {out_path}")

    print("\n Scene editing complete!")


if __name__ == "__main__":
    run_scene_editing(
        image_path="assets/scene.jpg",
        mask_path="outputs/mask_0.png",
        num_images=5
    )
