import torch
from thop import profile
from thop import clever_format
from models import Generator, Discriminator  # 确保 models.py 在同一目录下
import datetime


def count_gan_stats(im_size=256, nz=256, save_txt=True):
    print(f"\nComputing model statistics for resolution {im_size}x{im_size}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize models
    netG = Generator(ngf=64, nz=nz, nc=3, im_size=im_size).to(device)
    netD = Discriminator(ndf=64, nc=3, im_size=im_size).to(device)

    # 2. Prepare dummy inputs
    noise = torch.randn(1, nz).to(device)
    real_image = torch.randn(1, 3, im_size, im_size).to(device)
    test_part = 0

    # 3. Compute metrics
    g_flops, g_params = profile(netG, inputs=(noise,), verbose=False)
    d_flops, d_params = profile(netD, inputs=(real_image, "real", test_part), verbose=False)

    # 4. Format data
    g_flops_readable, g_params_readable = clever_format([g_flops, g_params], "%.3f")
    d_flops_readable, d_params_readable = clever_format([d_flops, d_params], "%.3f")

    total_params = g_params + d_params
    mem_mb = total_params * 4 / (1024 ** 2)

    # 5. Build output content
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_lines = [
        f"{'=' * 30} RoadFreq-GAN Model Analysis Report {'=' * 30}",
        f"Generation Time: {now}",
        f"Input Resolution: {im_size}x{im_size} | Noise Dimension: {nz}",
        f"{'-' * 75}",
        f"{'Component':<25} | {'Params':<20} | {'FLOPs':<20}",
        f"{'-' * 75}",
        f"{'Generator':<25} | {g_params_readable:<20} | {g_flops_readable:<20}",
        f"{'Discriminator':<25} | {d_params_readable:<20} | {d_flops_readable:<20}",
        f"{'-' * 75}",
        f"Total Params: {total_params / 1e6:.2f} M",
        f"Estimated VRAM: {mem_mb:.2f} MB (weights only)",
        f"{'=' * 75}\n"
    ]

    # 6. Print to screen
    content = "\n".join(output_lines)
    print(content)

    # 7. Save to text file
    if save_txt:
        file_name = "model_profile.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully saved data to text file: {file_name}")


if __name__ == "__main__":
    count_gan_stats(im_size=256, nz=256)