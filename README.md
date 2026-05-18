# RoadFreq-GAN: Frequency-Aware GAN for Road Scene Generation

This repository contains the official implementation of **RoadFreq-GAN**, a generative adversarial network designed for road scene generation. The model is built upon the FastGAN architecture and introduces frequency-domain loss (FFT loss) and advanced augmentation strategies to improve synthesis quality.

## Table of Contents
- [1. Description](#1-description)
- [2. Requirements](#2-requirements)
- [3. Data Preparation](#3-data-preparation)
- [4. Training](#4-training)
- [5. Evaluation](#5-evaluation)
- [6. Model Architecture](#6-model-architecture)
- [7. Checkpoints](#7-checkpoints)
- [8. Citation](#8-citation)

## 1. Description
RoadFreq-GAN combines:
- A multi-scale generator with **CASEBlock** (channel and spatial attention)
- A discriminator with reconstruction branches
- **FFT-based frequency loss** to guide the generator in matching spectral characteristics
- **DiffAugment** (color, translation) for better generalization
- Evaluation metrics: Sliced Wasserstein Distance (SWD) and 1‑NN accuracy

## 2. Requirements
Create a conda/virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

`requirements.txt`
```text
torch==2.12.0
pandas==3.0.3
numpy==2.4.5
tqdm==4.67.3
scipy==1.17.1
scikit-image==0.26.0
ipdb==0.13.13
lmdb==2.2.0
opencv-python==4.13.0.92
easing-functions==1.0.4
torchvision==0.27.0
```
*Note: lpips will automatically download VGG weights on first use.*

### Installation
Clone this repository:
```bash
git clone https://github.com/Winckedy/RoadFreq-GAN.git
cd RoadFreq-GAN
```
Place your dataset (see Data Preparation) and run training.

## 3. Data Preparation
The training script accepts two data formats:

### 3.1 Image Folder (simple)
Place all training images (`.jpg`, `.png`, `.jpeg`) directly inside a folder.
Example:
```
/path/to/dataset/
    0001.jpg
    0002.png
    ...
```
No subdirectories are required – the dataset class loads every image file in the folder.

### 3.2 LMDB Database (faster I/O for large datasets)
If the path string contains `'lmdb'`, the script will use `MultiResolutionDataset`.
The LMDB must be prepared with key format `{resolution}-{index:05d}` (e.g., `256-00000`) and a key `length` storing the number of samples.
This is particularly useful for datasets like FFHQ.

**Important:** All images are automatically resized to `--im_size` (e.g., 256) and normalized to `[-1, 1]` using `mean=0.5, std=0.5`.

## 4. Training

### 4.1 Basic Command
```bash
python train.py --path /path/to/dataset --name my_experiment --iter 50000 --batch_size 8 --im_size 256
```

### 4.2 Command Line Arguments
| Argument | Default     | Description |
|----------|-------------|-------------|
| `--path` | `./dataset` | Path to dataset (image folder or LMDB). |
| `--output_path` | `./`        | Root directory for saving all results. |
| `--name` | `test`      | Experiment name – creates subfolder `train_results/{name}/`. |
| `--iter` | `25000`     | Total training iterations. |
| `--start_iter` | `0`         | Starting iteration (used when resuming from checkpoint). |
| `--batch_size` | `8`         | Batch size per GPU. Reduce if out of memory. |
| `--im_size` | `256`       | Output resolution: `256`, `512`, or `1024`. |
| `--ckpt` | `None`      | Path to a checkpoint (`.pth`) to resume training. |
| `--workers` | `8`         | Number of data loader subprocesses. |
| `--save_interval` | `1000`      | Frequency (iterations) for saving intermediate models and images. |

### 4.3 Monitoring Training
During training, the following outputs are generated in `{output_path}/train_results/{name}/`:

- `training_log.csv` – per‑iteration losses and metrics (SWD, 1‑NN) every 1000 iterations.
- `images/` – sample images (every `save_interval * 10` iters) and reconstruction images.
- `models/` – model checkpoints (every `save_interval * 50` iters).
- `args.txt` – a copy of all command‑line arguments for reproducibility.

Console output every 100 iterations:
```
GAN: loss d: 0.12345    loss g: 0.67890    loss fft: 0.04567
```

## 5. Evaluation

### 5.1 Generating Images
Use `eval.py` to generate a large batch of images from a trained generator.
```bash
python eval.py --ckpt ./train_results/my_experiment/models/50000.pth --n_sample 20000 --im_size 256
```

**Arguments for `eval.py`:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--ckpt` | *(required)* | Path to a `.pth` checkpoint (e.g., `models/10000.pth`). |
| `--artifacts` | `.` | Root directory containing `models/` subfolder. |
| `--n_sample` | `40000` | Number of images to generate. |
| `--im_size` | `1024` | Resolution of generated images (`256`/`512`/`1024`). |
| `--batch` | `16` | Batch size for generation. |
| `--dist` | `.` | Output directory for saved PNGs. |
| `--cuda` | `0` | GPU index. |

Generated images are saved as `0.png`, `1.png`, … inside `eval_{iteration}/img/`.

### 5.2 Computing Model Statistics
To compute the number of parameters and FLOPs:
```bash
python profile_models.py
```
This will print a summary and save `model_profile.txt`.

## 6. Model Architecture

### 6.1 Generator
- **Initial layer:** 4×4 transposed convolution from latent vector `z` (dimension 256) to a 4×4 feature map.
- **Progressive upsampling:** Alternating `UpBlock` (simple) and `UpBlockComp` (with noise injection and two GLU layers).
- **CASEBlocks:** Fuse features from lower scales with higher scales using channel + spatial attention (see below).
- **Output:** Two resolution outputs: high‑resolution (e.g., 256×256) and 128×128 – used for discriminator and perceptual loss.

### 6.2 Discriminator
- **Dual‑path architecture:** Processes the full‑resolution image and a 128×128 downsampled version.
- **Downsampling blocks:** `DownBlock` and `DownBlockComp` (with skip connection via average pooling).
- **Reconstruction decoders:** For real images only, the discriminator reconstructs the input at three scales (full, 128, and a random 8×8 crop) – this provides additional perceptual loss.
- **Output:** Two logits (one per scale) concatenated to produce a final score.

### 6.3 Frequency Loss (FFT Loss)
Implemented as:
```python
def fft_loss(real, fake):
    def to_fft(x):
        x_gray = x.mean(dim=1)
        spectrum = torch.fft.rfft2(x_gray)
        return torch.log(torch.abs(spectrum) + 1e-8)
    return F.mse_loss(to_fft(real), to_fft(fake))
```
This loss encourages the generator to match the amplitude spectrum of the real images in the frequency domain. It is added to the standard adversarial loss with a weight of `0.1`.

### 6.4 CASEBlock
Combines two attention mechanisms:
- **SE (Squeeze‑and‑Excitation):** Global average pooling → two conv layers → sigmoid → channel weights.
- **CA (Coordinate Attention):** Separable pooling along height and width → shared 1×1 conv → split → two dilated 3×3 convs → sigmoid → spatial weights in two directions.

Final output = `feat_big * SE(feat_small) * CA(fused)`.
This enhances the generator’s ability to retain fine spatial details.

### 6.5 DiffAugment
During training, the following augmentations are applied (policy = `'color,translation'`):
- **Color:** random brightness, saturation, contrast.
- **Translation:** random shift by up to 12.5% of image size.

These augmentations are applied to both real and fake images **only during discriminator training** (no augmentation for generator FFT loss).

## 7. Checkpoints & Resuming Training
Two types of checkpoints are saved every `save_interval * 50` iterations:
- `{iteration}.pth` – contains `'g'` and `'d'` state dicts.
- `all_{iteration}.pth` – contains `'g'`, `'d'`, `'g_ema'` (exponential moving average of generator), and optimizers state dicts.

To resume training from a full checkpoint:
```bash
python train.py --ckpt ./train_results/my_experiment/models/all_10000.pth --start_iter 10000 ...(other args)
```
The script will automatically load the EMA weights, generator/discriminator states, and optimizers.

## 8. Results & Metrics
During training, the following metrics are logged every 1000 iterations to `training_log.csv`:
- **Loss_D** – discriminator loss (Hinge loss with relativistic pairing).
- **Loss_G_Base** – generator adversarial loss `-mean(D(fake))`.
- **Loss_G_Total** – base loss + 0.1 × FFT loss.
- **SWD** – Sliced Wasserstein Distance (multiplied by 1000 for readability). Lower is better; typical values for 256×256 images range from 10 to 50.
- **1NN_Acc** – 1‑Nearest Neighbour accuracy (two‑sample test). Values near 0.5 indicate that real and fake distributions are indistinguishable.

To compute these metrics manually, use:
```python
from metrics import compute_swd, compute_1nn
swd = compute_swd(real_batch, fake_batch, device='cuda')
acc = compute_1nn(real_batch, fake_batch)
```

## License
This project is licensed under the MIT License – see the `LICENSE` file for details.

## Citation
If you use RoadFreq-GAN in your research, please cite:

```bibtex
@software{RoadFreqGAN2026,
  author = {Jia Chen, Shuyang Chen, Yun Que, Yining Chen, Jingwen Wang},
  title = {RoadFreq-GAN: Frequency-Aware GAN for Road Scene Generation},
  year = {2026},
  url = {https://github.com/Winckedy/RoadFreq-GAN},
  note = {Based on FastGAN with FFT loss and CASEBlocks}
}
```

## Contact
For questions or issues, please open an issue on GitHub or email:  
your.email@example.com

Happy generating! 🚗🛣️
