# RoadFreq-GAN: Frequency-aware GAN for road scene generation
# Based on FastGAN architecture with FFT loss and DiffAugment
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import argparse
import random
from tqdm import tqdm
import csv
import os
from metrics import compute_swd, compute_1nn
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment

policy = 'color,translation'
import lpips

percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


# torch.backends.cudnn.benchmark = True

def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]


def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label == "real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean() + \
              percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
              percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
              percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


def fft_loss(real, fake):
    # Convert images to frequency domain
    def to_fft(x):
        # Convert to grayscale for simplicity, or apply FFT per channel
        x_gray = x.mean(dim=1)
        # Perform real FFT
        spectrum = torch.fft.rfft2(x_gray)
        # Take magnitude and convert to log scale for stability
        return torch.log(torch.abs(spectrum) + 1e-8)

    real_fft = to_fft(real)
    fake_fft = to_fft(fake)

    # Compute mean squared error
    return F.mse_loss(fake_fft, real_fft)


def train(args):
    print("=" * 50)
    print("Training RoadFreq-GAN")
    print("=" * 50)
    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder, saved_image_folder = get_dir(args)

    # -------------------------------------------------------------
    # CSV initialization
    # -------------------------------------------------------------
    csv_path = os.path.join(args.output_path, 'train_results', args.name, 'training_log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Record Iteration, D Loss, G Loss (Base), G Loss (Total including FFT), SWD, 1-NN
            writer.writerow(['Iteration', 'Loss_D', 'Loss_G_Base', 'Loss_G_Total', 'SWD', '1NN_Acc'])

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,
                                 pin_memory=True))

    # from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    # Initialize variables to avoid unassigned references
    err_dr = 0.0
    err_g = torch.tensor(0.0)
    total_g_loss = torch.tensor(0.0)

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)

        # -------------------------------------------------------------
        # DiffAugment
        # -------------------------------------------------------------
        real_image_aug = DiffAugment(real_image, policy=policy)
        fake_images_aug = [DiffAugment(fake, policy=policy) for fake in fake_images]

        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image_aug, label="real")
        train_d(netD, [fi.detach() for fi in fake_images_aug], label="fake")
        optimizerD.step()

        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images_aug, "fake")
        err_g = -pred_g.mean()

        # --- FFT Loss ---
        loss_fft = fft_loss(real_image_aug, fake_images_aug[0])

        # Weight factor set to 0.1
        total_g_loss = err_g + 0.1 * loss_fft
        total_g_loss.backward()

        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f    loss fft: %.5f" % (err_dr, -err_g.item(), loss_fft.item()))

        # -------------------------------------------------------------
        # SWD & 1-NN & CSV Logging
        # -------------------------------------------------------------
        if iteration % 1000 == 0:
            with torch.no_grad():
                target_eval_num = 128   # accumulate 128 images for evaluation
                accum_real = []
                accum_fake = []
                current_count = 0

                # Temporary loop to collect data
                while current_count < target_eval_num:
                    try:
                        real_eval_batch = next(dataloader)
                    except StopIteration:
                        dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                     sampler=InfiniteSamplerWrapper(dataset),
                                                     num_workers=dataloader_workers, pin_memory=True))
                        real_eval_batch = next(dataloader)

                    real_eval_batch = real_eval_batch.to(device)
                    curr_bs = real_eval_batch.size(0)

                    # Generate fake images
                    noise_eval = torch.randn(curr_bs, nz).to(device)
                    fake_eval_batch = netG(noise_eval)
                    if isinstance(fake_eval_batch, (list, tuple)):
                        fake_eval_batch = fake_eval_batch[0]  # take the high-resolution output

                    real_eval_batch = F.interpolate(real_eval_batch, size=128, mode='bilinear', align_corners=False)
                    fake_eval_batch = F.interpolate(fake_eval_batch, size=128, mode='bilinear', align_corners=False)

                    accum_real.append(real_eval_batch)
                    accum_fake.append(fake_eval_batch)
                    current_count += curr_bs

                eval_real_tensor = torch.cat(accum_real, dim=0)[:target_eval_num]
                eval_fake_tensor = torch.cat(accum_fake, dim=0)[:target_eval_num]

                val_swd = compute_swd(eval_real_tensor, eval_fake_tensor, device=device)
                val_1nn = compute_1nn(eval_real_tensor, eval_fake_tensor)

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, err_dr, -err_g.item(), total_g_loss.item(), val_swd, val_1nn])

                print(
                    f"Logged Metrics at {iteration}: Count={len(eval_real_tensor)}, SWD={val_swd:.4f}, 1NN={val_1nn:.4f}")
        # -------------------------------------------------------------

        if iteration % (save_interval * 10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration,
                                  nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='./dataset', help='path of resource dataset')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test', help='experiment name')
    parser.add_argument('--iter', type=int, default=25000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--save_interval', type=int, default=1000, help='number of iterations to save model')

    args = parser.parse_args()
    print(args)

    train(args)