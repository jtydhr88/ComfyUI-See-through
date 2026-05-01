# ComfyUI-See-through

A ComfyUI plugin that wraps [See-through](https://github.com/shitagaki-lab/see-through) — an AI system that decomposes a single anime illustration into manipulatable 2.5D layer-decomposed models with depth ordering, ready for Live2D workflows.

[中文说明](README_CN.md)

Paper: [arxiv:2602.03749](https://arxiv.org/abs/2602.03749) (Conditionally accepted to ACM SIGGRAPH 2026)

## Features

- **Single-Image Layer Decomposition** — Input one anime character image, get up to 24 semantic transparent layers (hair, face, eyes, clothing, accessories, etc.)
- **Depth Estimation** — Automatic depth map generation for each layer via fine-tuned Marigold, establishing correct drawing order
- **Smart Splitting** — Eyes, ears, handwear split into left/right; hair split into front/back via depth clustering
- **PSD Export** — Download layered PSD files directly from the browser (frontend ag-psd, no Python dependency)
- **Depth PSD** — Separate depth PSD export for 3D/parallax workflows
- **Preview Output** — Blended reconstruction preview as a standard ComfyUI IMAGE output
- **HuggingFace Auto-Download** — Models download automatically from HuggingFace on first use
- **VRAM Optimization** — Tag embedding caching, text encoder unloading, group offload, and configurable depth resolution for low-VRAM GPUs

## Nodes

| Node | Description |
|------|-------------|
| **SeeThrough Load LayerDiff Model** | Load the LayerDiff SDXL pipeline (layer generation) |
| **SeeThrough Load Depth Model** | Load the Marigold depth estimation pipeline |
| **SeeThrough Decompose** | Full pipeline: LayerDiff + Marigold depth + post-processing |
| **SeeThrough Save PSD** | Save layers as PNGs + metadata; download PSD via browser button |

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jtydhr88/ComfyUI-See-through.git
```

Install dependencies:

```bash
cd ComfyUI-See-through
pip install -r requirements.txt
```

Restart ComfyUI. The **SeeThrough** nodes will appear under the `SeeThrough` category.

### Dependencies

Only 4 additional Python packages beyond ComfyUI's base:

- `diffusers` — Hugging Face diffusion pipeline
- `accelerate` — Model loading acceleration
- `opencv-python` — Image processing
- `scikit-learn` — KMeans clustering for depth-based layer splitting

### Models

Models are downloaded automatically from HuggingFace on first use:

| Model | HuggingFace Repo | Purpose |
|-------|-------------------|---------|
| LayerDiff 3D | `layerdifforg/seethroughv0.0.2_layerdiff3d` | SDXL-based transparent layer generation |
| Marigold Depth | `24yearsold/seethroughv0.0.1_marigold` | Fine-tuned monocular depth for anime |

#### Manual placement

You can also download models manually and place them under `ComfyUI/models/SeeThrough/`. The loader recursively scans for valid diffusers directories (those containing `model_index.json`) up to two levels deep, so all of the following layouts are recognized and shown in the model dropdown:

```
ComfyUI/models/SeeThrough/
├── model_index.json                                        # flat layout (single model)
├── seethroughv0.0.2_layerdiff3d/                           # repo-name subfolder
│   └── model_index.json
└── layerdifforg/                                           # org/repo subfolder (matches HF naming)
    └── seethroughv0.0.2_layerdiff3d/
        └── model_index.json
```

When the loader resolves a model to a local path, it sets `local_files_only=True` on every `from_pretrained` call. This means once the model is in place, no HuggingFace requests are made — even if the upstream repo gets a new commit, the cache will not be re-fetched.

#### `auto_download` toggle

Both **Load LayerDiff Model** and **Load Depth Model** expose an `auto_download` boolean (default `true`). Set it to `false` to force local-only loading: if the model is not found on disk, the node errors out instead of contacting HuggingFace.

## Usage

### Basic Workflow

1. Add **SeeThrough Load LayerDiff Model** and **SeeThrough Load Depth Model** nodes
2. Add a **SeeThrough Decompose** node — connect both models and a **Load Image** node
3. Add **SeeThrough Save PSD** — connect the `parts` output
4. Add **Preview Image** — connect the `preview` output
5. Run the workflow
6. Click **Download PSD** button on the Save PSD node to generate and download the PSD file

### Example Workflows

Pre-made workflows are available in the `workflows/` directory:

| Workflow | Resolution | Steps | L/R Split | Description |
|----------|-----------|-------|-----------|-------------|
| `seethrough-basic.json` | 1280 | 30 | Yes | Standard quality, recommended |

Drag any `.json` file into ComfyUI to load the workflow.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `resolution` | 1280 | Processing resolution (image is center-padded to square) |
| `num_inference_steps` | 30 | Diffusion denoising steps (more = better quality, slower) |
| `tblr_split` | true | Split symmetric parts (eyes, ears, handwear) into left/right |
| `cache_tag_embeds` | true | Pre-compute and cache tag embeddings, then unload text encoders to save VRAM |
| `group_offload` | false | Enable group offload to drastically reduce peak VRAM (allocated ~0.2GB, reserved ~7GB) at cost of **2–3x slower** speed. Requires `diffusers>=0.37.0` |
| `auto_download` | true | If the model is not found locally, download from HuggingFace. Disable to force local-only and error out instead of downloading |
| `resolution_depth` | -1 | Resolution for depth inference. -1 uses the same as layers. Lower values (e.g. 720) save VRAM and speed up depth estimation |

### VRAM Optimization Guide

**For most users (12GB+ VRAM):** The default settings work well. `cache_tag_embeds=true` is already enabled and saves ~2GB VRAM with zero speed impact. No other changes needed.

**For low-VRAM users (8–12 GB):** Try the following settings in order, from least to most impact on speed:

1. **`cache_tag_embeds=true`** (default, already enabled) — Caches text embeddings and unloads text encoders, saving ~2GB VRAM with no speed penalty
2. **`resolution_depth=720`** — Run depth estimation at a lower resolution, then upscale back. Saves VRAM with minimal quality loss
3. **Lower `resolution`** — E.g. 1024 instead of 1280, reduces both VRAM and computation
4. **`group_offload=true`** — Last resort. Moves individual model blocks on/off GPU as needed, reducing peak allocated VRAM to ~0.2GB but **2–3x slower** due to frequent CPU↔GPU transfers. Requires `pip install diffusers>=0.37.0`

#### Benchmark (RTX 5090, steps=30, `cache_tag_embeds=true`)

**`group_offload` ON vs OFF (resolution=1280):**

| Stage | group_offload=OFF | group_offload=ON |
|-------|-------------------|------------------|
| UNet+VAE loaded | 7.94 GB | 0.21 GB |
| LayerDiff peak (allocated / reserved) | 7.95 GB / 13.69 GB | 0.21 GB / 7.31 GB |
| Marigold peak | 2.49 GB | 0.07 GB |
| **Total time** | **138 s** | **385 s (2.8x slower)** |

**Resolution scaling (group_offload=OFF):**

| Resolution | LayerDiff peak (allocated / reserved) | Marigold peak | Total time | Min VRAM |
|------------|---------------------------------------|---------------|------------|----------|
| 1280 | 7.95 GB / 13.69 GB | 2.49 GB | 138 s | ~16 GB |
| 2048 | 7.96 GB / 22.56 GB | 2.59 GB | 382 s | ~24 GB |

## Output Layers

The decomposition produces semantic layers including:

**Body parts:** front hair, back hair, neck, topwear, handwear, bottomwear, legwear, footwear, tail, wings, objects

**Head parts:** headwear, face, irides, eyebrow, eyewhite, eyelash, eyewear, ears, earwear, nose, mouth

Each layer is an RGBA image with transparency, positioned at its correct location in the canvas.

## Credits

This plugin wraps the [See-through](https://github.com/shitagaki-lab/see-through) research project by [shitagaki-lab](https://github.com/shitagaki-lab).

PSD generation uses [ag-psd](https://github.com/nicasiomg/ag-psd) in the browser.

## License

MIT
