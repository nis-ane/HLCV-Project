import os
import argparse
import copy
from tqdm import tqdm
import numpy as np

import utils
import models

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, datasets
from utils import CustomDataset


def exp_mov_avg(Gs, G, alpha = 0.999, global_step = 999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(generator, generator_s, discriminator, optim_g, optim_d, data_loader, device):
    # Generate fixed noise for evaluation
    fixed_noise = torch.FloatTensor(np.random.normal(0, 1, (16, args.latent_dim))).to(device)
    for step in tqdm(range(args.steps + 1)):
        # Train Discriminator
        optim_d.zero_grad()

        # Forward + Backward with real images
        r_img, synth_img = next(iter(data_loader))
        r_img, synth_img = r_img.to(device), synth_img.to(device)
        r_label = torch.ones(args.batch_size).to(device)
        # Concatenate real image with synthetic image
        r_input = torch.cat((r_img, synth_img), dim=1)
        print("r_input.shape", r_input.shape)
        r_logit = discriminator(r_input).flatten()
        lossD_real = criterion(r_logit, r_label)
        lossD_real.backward()

        # Forward + Backward with fake images
        latent_vector = torch.FloatTensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))).to(device)
        # Combine latent vector with synthetic images

        # Reshape latent vector to match the spatial dimensions of the synthetic image
        latent_vector_2 = latent_vector.view(args.batch_size, args.latent_dim, 1, 1)
        latent_vector_2 = latent_vector_2.expand(args.batch_size, args.latent_dim, synth_img.shape[2], synth_img.shape[3])


        print("latent_vector.shape, synth_img.shape", latent_vector.shape, synth_img.shape, latent_vector_2.shape)
        gen_input = torch.cat((synth_img, latent_vector_2), dim=1)
        print("gen_input.shape", gen_input.shape)
        f_img = generator(gen_input)  # Use combined input for the generator
        f_label = torch.zeros(args.batch_size).to(device)
        # Concatenate fake image with synthetic image
        f_input = torch.cat((f_img, synth_img), dim=1)
        f_logit = discriminator(f_input).flatten()
        lossD_fake = criterion(f_logit, f_label)
        lossD_fake.backward()

        optim_d.step()

        # Train Generator
        optim_g.zero_grad()
        f_img = generator(gen_input)  # Use combined input for the generator
        r_label = torch.ones(args.batch_size).to(device)
        # Concatenate fake image with synthetic image for the discriminator
        f_input = torch.cat((f_img, synth_img), dim=1)
        f_logit = discriminator(f_input).flatten()
        lossG = criterion(f_logit, r_label)
        lossG.backward()
        optim_g.step()

        exp_mov_avg(generator_s, generator, global_step=step)

        # Save samples at intervals
        if step % args.sample_interval == 0:
            generator.eval()
            # Use fixed noise and first 16 synthetic images for evaluation
            vis_input = torch.cat((synth_img[:16], fixed_noise), dim=1)
            vis = generator(vis_input).detach().cpu()
            vis = make_grid(vis, nrow=4, padding=5, normalize=True)
            vis = T.ToPILImage()(vis)
            vis.save('samples/vis{:05d}.jpg'.format(step))
            generator.train()
            print("Save sample to samples/vis{:05d}.jpg".format(step))

        # Save model checkpoints
        if (step + 1) % args.sample_interval == 0 or step == 0:
            torch.save(generator.state_dict(), 'weights/Generator.pth')
            torch.save(generator_s.state_dict(), 'weights/Generator_ema.pth')
            torch.save(discriminator.state_dict(), 'weights/Discriminator.pth')
            print("Save model state.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type = int, default = 100000,
                        help = "Number of steps for training (Default: 100000)")
    parser.add_argument("--batch-size", type = int, default = 128,
                        help = "Size of each batches (Default: 128)")
    parser.add_argument("--lr", type = float, default = 0.002,
                        help = "Learning rate (Default: 0.002)")
    parser.add_argument("--beta1", type = float, default = 0.0,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--beta2", type = float, default = 0.99,
                        help = "Coefficients used for computing running averages of gradient and its square")
    parser.add_argument("--latent-dim", type = int, default = 1024,
                        help = "Dimension of the latent vector")
    parser.add_argument("--data-dir", type = str, default = "crypko_data/faces/",
                        help = "Data root dir of your training data")
    parser.add_argument("--sample-interval", type = int, default = 1000,
                        help = "Interval for sampling image from generator")
    parser.add_argument("--gpu-id", type = int, default = 1,
                        help = "Select the specific gpu to training")
    args = parser.parse_args()

    # Device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # taking care of the dataset
    IMG_SHAPE = (3, 368, 544)
    BATCH_SIZE = args.batch_size
    real_train_folder = "/home/hevra/Desktop/hlcv/term_project/code/data/data/real/train/images"
    synt_train_folder = "/home/hevra/Desktop/hlcv/term_project/code/data/data/synthetic/train/images"
    real_val_folder = "/home/hevra/Desktop/hlcv/term_project/code/data/data/real/val/images"
    synt_val_folder = "/home/hevra/Desktop/hlcv/term_project/code/data/data/synthetic/val/images"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SHAPE[1], IMG_SHAPE[2]))
    ])

    train_dataset = CustomDataset(real_train_folder, synt_train_folder, transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomDataset(real_val_folder, synt_val_folder, transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create the log folder
    os.makedirs("weights", exist_ok = True)
    os.makedirs("samples", exist_ok = True)
    # Initialize Generator and Discriminator
    netG = models.Generator().to(device)
    print("hevv2")
    netG_s = copy.deepcopy(netG)
    netD = models.Discriminator().to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer and lr_scheduler
    optimizer_g = torch.optim.Adam(netG.parameters(), lr = args.lr,
        betas = (args.beta1, args.beta2)
    )
    optimizer_d = torch.optim.Adam(netD.parameters(), lr = args.lr,
        betas = (args.beta1, args.beta2)
    )


    # Start Training
    train(netG, netG_s, netD, optimizer_g, optimizer_d, train_dataloader, device)