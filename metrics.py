import torch
import torch.nn.functional as F
import numpy as np

def compute_swd(real_images, fake_images, num_projections=128, device='cuda'):

    real_flat = real_images.view(real_images.size(0), -1)
    fake_flat = fake_images.view(fake_images.size(0), -1)

    dim = real_flat.size(1)

    projections = torch.randn((dim, num_projections), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=0, keepdim=True))

    projected_real = torch.matmul(real_flat, projections)
    projected_fake = torch.matmul(fake_flat, projections)

    sorted_real, _ = torch.sort(projected_real, dim=0)
    sorted_fake, _ = torch.sort(projected_fake, dim=0)

    diff = sorted_real - sorted_fake
    swd = torch.mean(diff ** 2)

    return swd.item() * 1000


def compute_1nn(real_images, fake_images):

    if real_images.shape[-1] > 64:
        real_images = F.interpolate(real_images, size=64, mode='bilinear', align_corners=False)
        fake_images = F.interpolate(fake_images, size=64, mode='bilinear', align_corners=False)

    real_flat = real_images.view(real_images.size(0), -1)
    fake_flat = fake_images.view(fake_images.size(0), -1)

    total_data = torch.cat([real_flat, fake_flat], dim=0)
    n_real = real_flat.size(0)
    n_total = total_data.size(0)

    dists = torch.cdist(total_data, total_data, p=2)

    dists.fill_diagonal_(float('inf'))

    _, min_indices = torch.min(dists, dim=1)

    labels = torch.cat([torch.zeros(n_real), torch.ones(n_total - n_real)]).to(real_images.device)

    predicted_labels = labels[min_indices]
    accuracy = (predicted_labels == labels).float().mean().item()

    return accuracy