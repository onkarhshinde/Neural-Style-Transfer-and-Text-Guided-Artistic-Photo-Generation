

# Artistic Photo Generation using Generative AI
## Phase 1 & Phase 2 - Complete Documentation

**Project**: Neural Style Transfer & Text-Guided Artistic Photo Generation  
**Framework**: Flask Web Application with Deep Learning  
**Status**:  Phase 1 & Phase 2 Fully Implemented

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Phases](#project-phases)
3. [Architecture](#architecture)
4. [Features](#features)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Phase 1: Image-to-Image Style Transfer](#phase-1-image-to-image-style-transfer)
8. [Phase 2: Text-Guided Generation](#phase-2-text-guided-generation)
9. [File Structure](#file-structure)
10. [Usage Guide](#usage-guide)
11. [Parameters & Configuration](#parameters--configuration)
13. [Key Technologies](#key-technologies)
14. [Project Statistics](#Project-Statistics)
15. [Examples](#Examples)
---

## 📌 Overview

This project provides a complete artistic photo generation system with two complementary approaches:

- **Phase 1**: Traditional neural style transfer using VGG19 and reference art images
- **Phase 2**: Modern text-guided generation using CLIP and Stable Diffusion

Both phases are accessible through:
- 🌐 **Web Interface** (Flask)
- 💻 **Command Line** (Python scripts)
- 🔧 **Python API** (Direct imports)

---

## 🎯 Project Phases

### Phase 1: Neural Style Transfer (Image-to-Image)
Transform a photo using a reference art image's style.

**Key Features**:
- Upload content and style images
- Real-time progress tracking
- Multiple style support (style morphing)
- Color preservation option
- Customizable parameters (iterations, layers, pooling type)

**Technologies**: VGG19, L-BFGS optimization, TensorFlow

### Phase 2: Text-Guided Generation (NEW)
Transform a photo using natural language descriptions of artistic styles.

**Key Features**:
- Upload photo + describe artistic style with text
- CLIP-based multimodal alignment
- Stable Diffusion for high-quality synthesis
- Parameter tuning (steps, strength, guidance)
- Multiple variation generation
- Faster & more flexible than Phase 1

**Technologies**: CLIP, Stable Diffusion, PyTorch, Diffusers

---

## 🏗️ Architecture

### Phase 1: Neural Style Transfer Pipeline
```
User Input (Content + Style Images)
    ↓
Save Images → Preprocess (resize, normalize)
    ↓
VGG19 Feature Extraction
├─ Content Features (conv4_2)
└─ Style Features (conv1_1 → conv5_2)
    ↓
L-BFGS Optimization Loop
├─ Content Loss (minimize feature difference)
├─ Style Loss (minimize Gram matrix difference)
└─ Total Variation Loss (smoothness)
    ↓
Iterate for specified number of steps
    ↓
(Optional) Color Transfer using original image colors
    ↓
Output Stylized Image
```

### Phase 2: Text-Guided Generation Pipeline
```
User Input (Photo + Text Prompt)
    ↓
┌──────────────────────────────────────┐
│ Feature Extraction & Alignment       │
├──────────────────────────────────────┤
│ ✓ Image: CLIP Vision Encoder          │
│ ✓ Text: CLIP Language Encoder         │
│ ✓ Compute text-image similarity       │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Multimodal Stable Diffusion          │
├──────────────────────────────────────┤
│ • Encode text to embeddings          │
│ • Process image through VAE encoder  │
│ • Iterative denoising with guidance  │
│ • Classifier-free guidance           │
│ • DPM Solver for faster inference    │
└──────────────────────────────────────┘
    ↓
Generate Multiple Variations (optional)
    ↓
Output Artistic Images with Analysis
```

---

## ✨ Features

### Common Features
- ✅ Web-based user interface
- ✅ Real-time progress tracking
- ✅ Image upload and management
- ✅ Result gallery and download
- ✅ Parameter customization
- ✅ Terminal output logging
- ✅ GPU/CPU support

### Phase 1 Features
- ✅ Multiple style image support
- ✅ Intermediate output visualization
- ✅ Color preservation mode
- ✅ Adjustable style weights
- ✅ Custom layer selection (content_layer, style_layer)
- ✅ Pool type selection (max, average)
- ✅ Image size customization

### Phase 2 Features
- ✅ Text-based style description
- ✅ CLIP text-image alignment analysis
- ✅ Multiple variation generation
- ✅ Configurable diffusion steps
- ✅ Adjustable guidance scale
- ✅ Strength parameter (preserve vs. transform)
- ✅ Suggested prompt templates

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- Git (for CLIP installation)
- GPU recommended (NVIDIA CUDA for faster generation)

### Step 1: Clone or Download Repository
```bash
cd "path/to/project"
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

**For Phase 1 only**:
```bash
pip install -r requirements.txt
```

**For Phase 2**:
```bash
pip install -r requirements_phase2.txt
```

**For both phases**:
```bash
pip install -r requirements.txt
pip install -r requirements_phase2.txt
```

### Step 4: Verify Installation

**For Phase 1**:
```bash
python -c "import tensorflow as tf; print('✓ TensorFlow OK')"
```

**For Phase 2**:
```bash
python -c "import clip; import torch; import diffusers; print('✓ Phase 2 dependencies OK')"
```


---

## 🚀 Quick Start

### Option A: Phase 1 Web Interface
```bash
python app.py
# Open: http://localhost:5000
```

**Steps**:
1. Upload content image
2. Upload style image
3. Set number of intermediate images (1-10)
4. Click "Generate"
5. View progress and download results

### Option B: Phase 2 Web Interface (Recommended)
```bash
python app_phase2.py
# Open: http://localhost:5000
```

**Steps**:
1. Upload image
2. Enter text prompt: "Van Gogh painting style with swirling brushstrokes"
3. Adjust parameters (optional)
4. Click "Generate Images"
5. Download results

### Option C: Command Line (Phase 2)
```bash
python phase2_multimodal_generator.py \
    "path/to/photo.jpg" \
    "Watercolor painting style" \
    "output_folder/" \
    --num_variations 3 \
    --num_steps 50 \
    --strength 0.75 \
    --guidance_scale 7.5
```

### Option D: Python API (Phase 2)
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator

generator = MultimodalArtisticGenerator()
image, analysis = generator.generate_artistic_image(
    "photo.jpg",
    "Oil painting style",
    num_inference_steps=50,
    output_path="result.png"
)
print(f"Similarity: {analysis['image_text_similarity']:.4f}")
```

---

## Phase 1: Image-to-Image Style Transfer

### Overview
Phase 1 performs neural style transfer using a reference artwork. It combines the content of one image with the style of another using deep neural networks and optimization techniques.

### Web Interface
**File**: `app.py`

**Access**: `http://localhost:5000`

**Workflow**:
1. Upload content image (photo to stylize)
2. Upload style image (reference artwork)
3. Select number of intermediate outputs (1-10)
4. View progress with real-time update bar
5. Download all generated intermediate and final images

### Command Line Usage

#### Basic Syntax
```bash
python inetwork_tf.py "path/to/content_image" "path/to/style_image" "/path/to/result_prefix"
```

#### Example
```bash
python inetwork_tf.py "photo.jpg" "vangogh.jpg" "results/styled_"
```

#### Multiple Styles
```bash
python inetwork_tf.py "content.jpg" "style1.jpg" "style2.jpg" "results/styled_" \
    --style_weight 1.0 1.0
```

#### With Parameters
```bash
python inetwork_tf.py "content.jpg" "style.jpg" "result/" \
    --preserve_color "True" \
    --pool_type "ave" \
    --rescale_method "bicubic" \
    --content_layer "conv4_2" \
    --num_iter 10
```

### Parameters Reference

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `num_iter` | 10 | int | Number of iterations (more = better quality) |
| `image_size` | 400 | int | Gram matrix size in pixels |
| `content_weight` | 0.025 | float | Content loss weight |
| `style_weight` | 1.0 | float | Style loss weight |
| `style_masks` | None | str | Path(s) to style masks |
| `color_mask` | None | str | Path to color preservation mask |
| `init_image` | "content" | str | Initialization: "content", "noise", or "gray" |
| `pool_type` | "max" | str | Pooling method: "max" or "ave" |
| `model` | "vgg19" | str | Model: "vgg16" or "vgg19" |
| `content_layer` | "conv4_2" | str | Layer for content loss |
| `content_loss_type` | 2 | int | Loss scaling: 0, 1, or 2 |
| `preserve_color` | False | bool | Preserve source image color |
| `rescale_method` | "bilinear" | str | Rescaling: "bilinear" or "bicubic" |
| `min_improvement` | 0.0 | float | Minimum improvement threshold |

### Post-Processing: Color Transfer
```bash
python color_transfer.py "path/to/content_image" "path/to/styled_image"
```

Output: `styled_image_original_color.jpg`

### Files Used
- **`inetwork_tf.py`**: Main style transfer engine (TensorFlow)
- **`INetwork.py`**: Legacy implementation (Keras) - not recommended
- **`tf_bfgs.py`**: L-BFGS optimizer
- **`color_transfer.py`**: Color adjustment utility
- **`utils.py`**: Image handling utilities
- **`app.py`**: Flask web application
- **`templates/index.html`**: Web interface

---

## Phase 2: Text-Guided Generation

### Overview
Phase 2 uses modern generative AI to create artistic variations of photos based on text descriptions. Instead of reference images, users describe the artistic style they want using natural language.

### Web Interface
**File**: `app_phase2.py`

**Access**: `http://localhost:5000`

**Features**:
- Drag & drop image upload
- Text prompt input with suggestions
- Real-time parameter sliders
- Progress tracking
- Image gallery with download
- Terminal output viewer
- GPU/CPU detection

### Command Line Usage

#### Basic
```bash
python phase2_multimodal_generator.py "image.jpg" "prompt text" "output_dir/"
```

#### Full Example
```bash
python phase2_multimodal_generator.py \
    "photo.jpg" \
    "Van Gogh starry night painting with swirling brushstrokes" \
    "results/" \
    --num_variations 3 \
    --num_steps 50 \
    --strength 0.75 \
    --guidance_scale 7.5 \
    --device cuda
```

### Python API

#### Basic Usage
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator
import torch

generator = MultimodalArtisticGenerator(
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Single image
image, metadata = generator.generate_artistic_image(
    content_image_path="photo.jpg",
    text_prompt="Oil painting style",
    num_inference_steps=50,
    strength=0.75,
    guidance_scale=7.5,
    output_path="result.png"
)
```

#### Multiple Variations
```python
images = generator.generate_multiple_variations(
    content_image_path="photo.jpg",
    text_prompt="Watercolor painting style",
    num_variations=5,
    num_inference_steps=50,
    output_dir="results/"
)

for img, path in zip(images, paths):
    print(f"Generated: {path}")
```

#### Analyze Alignment
```python
analysis = generator.analyze_image_and_prompt(
    content_image_path="photo.jpg",
    text_prompt="Van Gogh style"
)

print(f"Image-Text Similarity: {analysis['image_text_similarity']:.4f}")
print(f"Image Features Shape: {analysis['image_features'].shape}")
print(f"Text Features Shape: {analysis['text_features'].shape}")
```

### Parameters Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `num_variations` | 1-5 | 3 | Number of different results to generate |
| `num_inference_steps` | 10-100 | 50 | Diffusion iterations (quality vs speed) |
| `strength` | 0.3-0.95 | 0.75 | How much to modify (0.3=subtle, 0.95=dramatic) |
| `guidance_scale` | 1.0-20.0 | 7.5 | Text adherence (1.0=ignore, 20.0=strict) |
| `height` | 384-576 | 512 | Output height in pixels |
| `width` | 384-576 | 512 | Output width in pixels |
| `seed` | 0-max_int | random | Random seed for reproducibility |

### Parameter Recommendations

**Fast Draft** (2-3 minutes):
```python
num_inference_steps = 30
strength = 0.7
guidance_scale = 7.5
```

**Balanced** (4-5 minutes) ⭐ RECOMMENDED:
```python
num_inference_steps = 50
strength = 0.75
guidance_scale = 7.5
```

**High Quality** (7-10 minutes):
```python
num_inference_steps = 75-100
strength = 0.8
guidance_scale = 10.0
```

### Text Prompt Engineering

#### Effective Prompt Formula
```
[Artist/Style] + [Technique] + [Colors/Mood] + [Details]
```

#### Examples

**Good Prompts**:
- "Van Gogh starry night painting with swirling brushstrokes and bright blues"
- "Watercolor landscape with soft pastel colors and gentle brushwork"
- "Oil painting impressionist style with warm earth tones"
- "Film noir cinematic photography with dramatic shadows and high contrast"
- "Abstract expressionism with bold vibrant colors and dynamic energy"

**Prompt Templates**:
```
1. "[Artist] [style] with [technique] and [color palette]"
   Example: "Picasso cubist style with geometric abstraction and earthy tones"

2. "[Art movement] [technique] with [mood] [colors]"
   Example: "Renaissance oil painting with dramatic lighting and golden tones"

3. "[Effect] [medium] style of [subject] with [details]"
   Example: "Watercolor painting of landscape with soft edges and flowing lines"

4. "[Technique] [style] with [specific details]"
   Example: "Digital art illustration with neon colors and cyberpunk aesthetic"
```

#### Suggested Prompts (Built-in)
- "Van Gogh painting style with swirling brushstrokes"
- "Watercolor painting aesthetic with soft colors"
- "Cinematic photography with dramatic lighting and moody atmosphere"
- "Oil painting style with impressionist brushwork"
- "Abstract art style with vibrant colors"
- "Pencil sketch style with fine details"
- "Surrealism artistic style with dreamlike elements"
- "Comic book art style with bold outlines"
- "Renaissance oil painting style"
- "Modern digital art style with neon colors"

### Files Used
- **`clip_text_encoder.py`**: CLIP text/image encoder
- **`diffusion_text_to_image.py`**: Stable Diffusion wrapper
- **`phase2_multimodal_generator.py`**: Main orchestrator
- **`phase2_multimodal_generator_cpu_optimized.py`**: CPU-optimized version
- **`app_phase2.py`**: Flask web application
- **`templates/index_phase2.html`**: Web interface
- **`setup_phase2.py`**: Setup and testing script
- **`utils.py`**: Shared utilities

---

## 📁 File Structure

```
project/
│
├── ==================== PHASE 1 FILES ====================
├── app.py                           # Flask app for Phase 1
├── inetwork_tf.py                   # TensorFlow style transfer (MAIN)
├── INetwork.py                      # Legacy Keras implementation
├── color_transfer.py                # Color preservation utility
├── tf_bfgs.py                       # L-BFGS optimizer
├── utils.py                         # Image utilities (shared)
├── requirements.txt                 # Phase 1 dependencies
│
├── ==================== PHASE 2 FILES ====================
├── app_phase2.py                    # Flask app for Phase 2 (RECOMMENDED)
├── clip_text_encoder.py             # CLIP multimodal encoder
├── diffusion_text_to_image.py       # Stable Diffusion wrapper
├── phase2_multimodal_generator.py   # Main orchestrator (GPU)
├── phase2_multimodal_generator_cpu_optimized.py  # CPU version
├── setup_phase2.py                  # Automated setup script
├── requirements_phase2.txt          # Phase 2 dependencies
│
├── ==================== DOCUMENTATION ====================
├── README_COMBINED.md               # THIS FILE
|
├── ==================== WEB INTERFACE ====================
├── templates/
│   ├── index.html                   # Phase 1 web UI
│   └── index_phase2.html            # Phase 2 web UI (MODERN)
│
├── static/
│   ├── uploads/                     # User uploaded images
│   ├── results/                     # Generated results
│   └── (CSS/JS auto-generated)
│
├── ==================== DATA & RESULTS ====================
├── data/
│   ├── images/                      # Sample images
│   └── styles/                      # Sample style images
│
└── result/                          # Legacy results directory
```

---

## 📖 Usage Guide

### Running Phase 1

#### Web Interface (Easiest)
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

#### Command Line
```bash
python inetwork_tf.py "content.jpg" "style.jpg" "output/"
```

#### Python API
```python
# Phase 1 is primarily CLI/web-based
# Use inetwork_tf.py for file-based processing
```

### Running Phase 2

#### Web Interface (Recommended) ⭐
```bash
python app_phase2.py
```
Then open `http://localhost:5000` in your browser.

#### Command Line
```bash
python phase2_multimodal_generator.py "image.jpg" "prompt" "output/"
```

#### Python API
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator

generator = MultimodalArtisticGenerator()
image, analysis = generator.generate_artistic_image(
    "photo.jpg",
    "Artistic style description",
    output_path="result.png"
)
```

### Batch Processing (Phase 2)

#### Command Line Batch
```bash
for img in images/*.jpg; do
    python phase2_multimodal_generator.py \
        "$img" \
        "Van Gogh style" \
        "results/"
done
```

#### Python Batch
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator
import glob
from pathlib import Path

gen = MultimodalArtisticGenerator()
for img_path in glob.glob("images/*.jpg"):
    gen.generate_artistic_image(
        img_path,
        "Style prompt",
        output_path=f"results/{Path(img_path).stem}_styled.png"
    )
```

---

## ⚙️ Parameters & Configuration

### Phase 1 Parameters

#### Style Transfer Optimization
```python
--num_iter 10              # Iterations (more = better)
--content_weight 0.025     # Content importance
--style_weight 1.0         # Style importance
--min_improvement 0.0      # Min improvement threshold
```

#### Feature Extraction
```python
--content_layer "conv4_2"  # Layer for content features
--model "vgg19"            # Model: vgg19 or vgg16
--image_size 400           # Gram matrix size
```

#### Image Processing
```python
--pool_type "max"          # Pooling: max or ave
--rescale_method "bilinear" # Rescaling method
--preserve_color True      # Keep original colors
```

### Phase 2 Parameters

#### Diffusion Steps
- **10-30**: Fast preview (not recommended)
- **40-50**: Good balance (recommended)
- **60-100**: High quality (slower)

#### Strength (0.3-0.95)
- **0.3-0.4**: Subtle, preserve content
- **0.5-0.7**: Moderate change
- **0.75**: Balanced (default)
- **0.8-0.95**: Dramatic transformation

#### Guidance Scale (1.0-20.0)
- **1.0-3.0**: Ignore text, creative freedom
- **7.5**: Balanced adherence (default)
- **10.0-15.0**: Strict text following

#### Number of Variations
- **1-3**: Single or few results
- **4-5**: Generate multiple variations

---



## 🔬 Key Technologies

### Phase 1: Classical Approach
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Feature Extraction | VGG19 CNN | Extract content & style features |
| Optimization | L-BFGS | Minimize loss function |
| Loss Functions | Content & Gram | Preserve content & style |
| Backend | TensorFlow/Keras | Neural network operations |

### Phase 2: Modern Generative AI
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Text Encoding | OpenAI CLIP | Convert text to embeddings |
| Image Encoding | CLIP Vision | Extract image features |
| Image Generation | Stable Diffusion | Text-guided synthesis |
| Diffusion | DPM Solver | Accelerated inference |
| Validation | Multimodal Similarity | Measure alignment |
| Backend | PyTorch | Tensor operations |
| Web Framework | Flask | Server & API |

### Key Papers & References

**Phase 1 (Neural Style Transfer)**:
- Gatys et al. (2015): "A Neural Algorithm of Artistic Style"

**Phase 2 (Diffusion Models)**:
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models"
- Song et al. (2020): "Score-Based Generative Modeling"
- Radford et al. (2021): "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)

---

## 📊 Project Statistics

### Models & Datasets
- **VGG19**: Pre-trained on ImageNet
- **CLIP**: Multi-modal 400M parameters
- **Stable Diffusion v1.5**: 860M parameters
- **Training Data**: Diverse artistic images

---


## 📝 Examples

### Example 1: Photo to Painting (Phase 2)
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator

gen = MultimodalArtisticGenerator()

# Transform landscape photo
image, analysis = gen.generate_artistic_image(
    content_image_path="landscape.jpg",
    text_prompt="Vincent van Gogh impressionist painting style with bold brushstrokes and vibrant colors",
    num_inference_steps=75,
    guidance_scale=10.0,
    strength=0.8,
    output_path="landscape_vangogh.png"
)

print(f"Generation complete!")
print(f"Similarity score: {analysis['image_text_similarity']:.4f}")
```

### Example 2: Batch Processing (Phase 2)
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator
import glob

gen = MultimodalArtisticGenerator()
prompts = [
    "Van Gogh starry night style",
    "Watercolor painting aesthetic",
    "Oil painting impressionist"
]

for img_path in glob.glob("photos/*.jpg"):
    for prompt in prompts:
        gen.generate_artistic_image(
            img_path,
            prompt,
            output_path=f"results/{img_path.stem}_{prompt.replace(' ', '_')}.png"
        )
```

### Example 3: Multiple Variations (Phase 2)
```python
from phase2_multimodal_generator import MultimodalArtisticGenerator

gen = MultimodalArtisticGenerator()

# Generate 5 different artistic interpretations
images = gen.generate_multiple_variations(
    content_image_path="photo.jpg",
    text_prompt="Surrealism with dreamlike floating elements",
    num_variations=5,
    num_inference_steps=60,
    output_dir="surreal_variations/"
)

print(f"Generated {len(images)} variations")
```

---


## 📄 License

This project uses open-source libraries and pre-trained models.

### Dependencies Licenses
- **PyTorch**: BSD License
- **TensorFlow**: Apache 2.0
- **CLIP**: MIT License
- **Diffusers**: Apache 2.0
- **Flask**: BSD License
- **Pillow**: PIL License

---

---

## 🎉 Conclusion

This project demonstrates the evolution of artistic image generation:

- **Phase 1** showcases classical deep learning approaches
- **Phase 2** leverages cutting-edge generative AI

Both phases are fully functional and optimized for practical use. Choose Phase 1 for traditional style transfer or Phase 2 for modern text-guided creativity!

---

**Last Updated**: 2026  
**Version**: 2.0 (Phase 1 + Phase 2 Combined)  





<!-- 
#  Neural Style Transfer Web Application

This is a Flask-based web application for performing **neural style transfer**, where a content image is combined with a style image to generate a stylized output. Users can upload their own images, choose how many intermediate images they want (for visualizing progress), and view both intermediate and final results.

---

##  Features

- Upload a **content** and **style** image
- Specify the number of **intermediate outputs** (1 to 10)
- See real-time progress via **loading bar**
- View and download **stylized result**

---

##  Project Structure

```
project/
│
├── data/                 # Stores sample data
├── static/               # Static files like CSS, JS, intermediate outputs, uploaded images
├── templates/            # HTML templates (Jinja2)
│
├── app.py                # Main Flask app
├── color_transfer.py     # Color adjustment module (optional)
├── INetwork.py           # Legacy style transfer network 
├── inetwork_tf.py        # TensorFlow-based style transfer script
├── tf_bfgs.py            # Optimization utilities
├── utils.py              # Helper utilities
│
├── requirements.txt      # Requirements for model
├── requirements_tf.txt   # Requirements for TensorFlow script
└── README.md             # Project overview
```

---

##  Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/onkarhshinde/Custom-Artistic-Neural-Style-Transfer.git
   cd neural-style-webapp
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install TensorFlow requirements (optional if not already installed):**
   ```bash
   pip install -r requirements_tf.txt
   ```

---

##  Run the Web App

Make sure all necessary dependencies are installed, then run:

```bash
python app.py
```

- Open browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)
- Upload content and style images.
- Select number of intermediate images (1 to 10).
- View progress and final output on the result page.




###  Sample Usage for webpage

1. Upload images:
   - `Content Image`: Photo you want to stylize
   - `Style Image`: Artwork style to apply
2. Set how many intermediate results to show (e.g., 5 = 25 min of training)
3. Watch progress and download results.
   
###  Web Dashboard Preview

Here are some screenshots of the interface and results:

|  Upload Section |  Stylized Results + Download Option |
|-------------------|----------------------------------------|
| ![Upload](https://github.com/user-attachments/assets/799d8206-f10c-486e-ad1a-37951d21a55c) | ![Results](https://github.com/user-attachments/assets/0953be98-dbe2-4c98-b664-494f22571d21) |

---


## Usage: Neural Style Transfer (Terminal Commands)

INetwork.py
```
python inetwork_tf.py "/path/to/content image" "path/to/style image" "/path/to/result prefix"
```

To pass multiple style images, after passing the content image path, seperate each style path with a space
```
python inetwork_tf.py "/path/to/content image" "path/to/style image 1" "path/to/style image 2" ... "/path/to/result prefix" --style_weight 1.0 1.0 ... 
```

There are various parameters discussed below which can be modified to alter the output image. Note that many parameters require the command to be enclosed in double quotes ( " " ).

Example:
```
python inetwork_tf.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix" --preserve_color "True" --pool_type "ave" --rescale_method "bicubic" --content_layer "conv4_2"
```


(For post processing of image )

To perform color preservation on an already generated image, use the `color_transform.py` as below. It will save the image in the same folder as the generated image with "_original_color" suffix.
```
python color_transfer.py "path/to/content/image" "path/to/generated/image"
```


---
# Examples
## Single Style Transfer
<img src="https://github.com/user-attachments/assets/bf227190-4a79-447b-bf50-5261427c9d21" width=49% height=300 alt="Golden_gate"> <img src="https://github.com/user-attachments/assets/b80887e2-45ce-4e8a-be9e-a4f34057aa0a" width=49% height=300 alt="blue_swirls">
<br><br> Results after 5 iterations using the inetwork_tf <br>
<img src="https://github.com/user-attachments/assets/f7027658-bbc6-43f1-aaec-5f2ff54b8218" width=98% height=450 alt="blue_swirls style transfer">
<br><br> Style Transfer with Color Preservation
<img src="https://github.com/user-attachments/assets/e1f65183-f80f-4c3f-98c7-6a100179283e" width=98% height=450 alt="blue_swirls style transfer">

## Neural Style Transfer Parameters

- **`--style_masks`**: Multiple style masks for specific regions. Number of style masks must match `style_weight` count.
- **`--color_mask`**: Defines region to preserve color.
- **`--image_size`**: Sets Gram Matrix size (default 400x400).
- **`--num_iter`**: Number of iterations (default 10).
- **`--init_image`**: Initial image type (`"content"`, `"noise"`, or `"gray"`).
- **`--pool_type`**: Pooling method (`max` or `ave`).
- **`--model`**: Model architecture (`vgg16` or `vgg19`).
- **`--content_loss_type`**: Loss scaling type (`0`, `1`, or `2`).
- **`--preserve_color`**: Preserves content image color.
- **`--min_improvement`**: Minimum required improvement to continue (default 0.0).
- **`--content_weight`**: Content weight (default 0.025).
- **`--style_weight`**: Style weight (default 1, space-separated for multiple styles).
- **`--style_scale`**: Scales style weight (default 1).
- **`--total_variation_weight`**: Regularization factor (default 8.5E-5).
- **`--rescale_image`**: Rescales to original size after each iteration.
- **`--rescale_method`**: Rescaling algorithm (`nearest`, `bilinear`, `bicubic`, or `cubic`).
- **`--maintain_aspect_ratio`**: Maintains aspect ratio during rescaling (default True).
- **`--content_layer`**: Content layer (`conv4_2` or `conv5_2`).


##  Tech Stack

- **Frontend:** HTML, CSS (Bootstrap), JavaScript
- **Backend:** Python Flask
- **ML Engine:** TensorFlow
- **Deployment-ready:** Local development mode



##  Author

Made by Onkar Shinde  
 -->
