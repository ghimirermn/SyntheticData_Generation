import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from .utils import visualize_mask, save_mask

def coco_to_multiclass(json_path, output_dir="outputs", visualize=True):
    """
    Converts COCO-format segmentation annotations into multi-class masks.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    for img_info in data["images"]:
        width, height = img_info["width"], img_info["height"]
        image_id = img_info["id"]
        mask = np.zeros((height, width), dtype=np.uint8)

        annos = [a for a in data["annotations"] if a["image_id"] == image_id]

        for anno in annos:
            category_id = anno["category_id"]
            for seg in anno["segmentation"]:
                poly = np.array(seg).reshape((-1, 2))
                img = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(img)
                draw.polygon(list(map(tuple, poly)), outline=category_id, fill=category_id)
                mask_part = np.array(img)
                mask = np.maximum(mask, mask_part)

        out_path = Path(output_dir) / f"mask_{image_id}.png"
        save_mask(mask, out_path)
        if visualize:
            visualize_mask(mask, f"Multi-class Mask (Image {image_id})")

    print(f"Saved all masks to {output_dir}")

if __name__ == "__main__":
    coco_to_multiclass("assets/result_coco.json")
