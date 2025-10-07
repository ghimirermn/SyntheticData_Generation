# SyntheticData_Generation

This repository demonstrates diffusion-based scene editing for creating synthetic image variations across multiple semantic regions (such as car, building, road, and sky). It is designed to generate diverse and realistic datasets for training or augmenting vision models without the need for manual image collection.

## Overview

The project uses Stable Diffusion Inpainting to selectively modify regions in an image based on semantic masks. Each class label (for example, car, sky, building) is independently edited using text prompts to produce controlled visual variations.

Typical workflow:

1. Load an image and its multi-class segmentation mask.
2. Convert to binary masks for each object class.
3. Generate multiple inpainted variations using custom prompts.
4. Store outputs for dataset expansion or visualization.

## Repository Structure

```
SyntheticData_Generation/
├── diffusion_scene_editing/
│   ├── 01_generate_masks.py          # Extracts per-class binary masks
│   ├── 02_prompt_variations.py       # Defines semantic prompts for each label
│   ├── 03_diffusion_inpaint.py       # Runs Stable Diffusion Inpainting per region
│   └── 04_generate_synthetic_data.py # Full pipeline combining all steps
│
├── assets/
│   ├── input_image.jpg               # Example source image
│   ├── mask_0.png                    # Multi-class mask
│   └── labels.json                   # Label definitions (for example, {1: "car", 2: "sky"})
│
├── outputs/
│   ├── variations/                   # Generated image outputs
│   └── logs/                         # Prompt and seed history
│
├── requirements.txt
└── README.md
```

## Key Features

* Per-class editing to modify specific regions without affecting others
* Prompt-based control for flexible scene customization
* Batch generation of multiple dataset variants
* Fully reproducible pipeline using random seeds

## Setup

```bash
git clone https://github.com/ghimirermn/SyntheticData_Generation.git
cd SyntheticData_Generation

python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## Usage

1. Place your input image and COCO annotation in `assets/`.
2. Generate the segmentation mask:

   ```bash
   python diffusion_scene_editing/01_generate_masks.py
   ```

   The mask will be saved in `outputs/mask_0.png`.
3. Modify prompts and label maps in `02_prompt_variations.py`.
4. Run the full inpainting pipeline:

   ```bash
   python diffusion_scene_editing/04_generate_synthetic_data.py
   ```
5. Generated images will appear in `outputs/variations/`.

## Example Output

![Example synthetic variation](https://github.com/ghimirermn/SyntheticData_Generation/blob/main/assets/demo.gif)

