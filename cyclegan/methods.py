import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from traitlets.traitlets import List
from .discriminator import Discriminator
from .generator import Generator
from torchvision.utils import save_image

def training(gen_G: Generator,
             gen_F: Generator,
             disc_DX: Discriminator,
             disc_DY: Discriminator,
             train_loader: DataLoader,
             device: torch.device,
             epochs: int,
             opt_gen: Optimizer,
             opt_disc: Optimizer,
             criterion_0: nn.Module = nn.MSELoss(),
             criterion_1: nn.Module = nn.L1Loss(),
             val_loader: DataLoader=None,
             lambda_cycle: float = 10.0,
             use_amp: bool=True):
    """to train the models for one epoch"""

    disc_DX.to(device)
    disc_DY.to(device)
    gen_G.to(device)
    gen_F.to(device)

    ## training mode
    disc_DX.train()
    disc_DY.train()
    gen_G.train()
    gen_F.train()

    g_scaler = GradScaler(enabled=use_amp)
    d_scaler = GradScaler(enabled=use_amp)

    ## Create a progress bar for the training loop
    loop = tqdm(train_loader,leave=True, desc=f'training [epoch {epochs}]')

    best_val = float('inf')  # Initialize best validation loss to infinity
    best_epoch = 0           # Initialize best epoch
    
    epoch_disc_losses, epoch_gen_losses = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    val_disc_losses, val_gen_losses = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    for idx, batch in enumerate(loop):

        satellite_imgs = batch['satellite_imgs'].to(device, non_blocking=True)
        maps_imgs = batch['maps_imgs'].to(device, non_blocking=True)

        ## ===========================
        ## Train Discriminators (DX & DY)
        ## ===========================
        with autocast(device, enabled=use_amp):
            ## Generate fake map images (X -> Y)
            fake_maps = gen_G(satellite_imgs)

            ## Discriminator DY: Evaluate real and fake map images
            D_G_real = disc_DY(maps_imgs)
            D_G_fake = disc_DY(fake_maps)
            
            D_G_ones_like = torch.ones_like(D_G_real, device=device)
            D_G_zeros_like = torch.zeros_like(D_G_fake, device=device)

            D_G_real_loss = criterion_0(D_G_real, D_G_ones_like)
            D_G_fake_loss = criterion_0(D_G_fake, D_G_zeros_like)
            D_G_loss = D_G_real_loss + D_G_fake_loss

            ## Generate fake satellite images (Y -> X)
            fake_satellite = gen_F(maps_imgs)

            ## Discriminator DX: Evaluate real and fake satellite images
            D_F_real = disc_DX(satellite_imgs)
            D_F_fake = disc_DX(fake_satellite)

            D_F_ones_like = torch.ones_like(D_F_real, device=device)
            D_F_zeros_like = torch.zeros_like(D_F_fake, device=device)

            D_F_real_loss = criterion_0(D_F_real, D_F_ones_like)
            D_F_fake_loss = criterion_0(D_F_fake, D_F_zeros_like)
            D_F_loss = D_F_real_loss + D_F_fake_loss

            ## Combined discriminator loss
            D_loss = (D_G_loss + D_F_loss) / 2

        ## Backpropagation and optimization for the discriminators
        opt_disc.zero_grad(set_to_None=True)

        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        ## =====================
        ## Train Generators (G & F)
        ## =====================
        with autocast(device, enabled=use_amp):
            ## Adversarial loss for generators
            D_Y_fake = disc_DY(fake_maps)
            D_X_fake = disc_DX(fake_satellite)

            D_Y_ones_like = torch.ones_like(D_Y_fake, device=device)
            D_X_ones_like = torch.ones_like(D_X_fake, device=device)

            loss_G_Y = criterion_0(D_Y_fake, D_Y_ones_like)
            loss_G_X = criterion_0(D_X_fake, D_X_ones_like)

            ## Cycle consistency loss
            cycle_maps = gen_F(fake_maps)
            cycle_satellite = gen_G(fake_satellite)
            cycle_satellite_loss = criterion_1(satellite_imgs, cycle_maps)
            cycle_maps_loss = criterion_1(maps_imgs, cycle_satellite)

            ## Total generator loss
            G_loss = (
                loss_G_X
                + loss_G_Y
                + cycle_satellite_loss * lambda_cycle
                + cycle_maps_loss * lambda_cycle
            )

        ## Backpropagation and optimization for the generators
        opt_gen.zero_grad(set_to_None=True)

        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        epoch_disc_losses += D_loss.detach()
        epoch_gen_losses += G_loss.detach()

    if val_loader is not None:
        valid_disc_loss, valid_gen_loss = valid_epoch(
            val_loader=val_loader,
            epoch=idx+1,
            device=device,
            disc_DX=disc_DX,
            disc_DY=disc_DY,
            gen_G=gen_G,
            gen_F=gen_F,
            criterion_0=criterion_0,
            criterion_1=criterion_1,
            lambda_cycle=lambda_cycle
        )

        ## save best models
        if valid_gen_loss < best_val:
            best_epoch = idx + 1
            best_val = valid_gen_loss

            best_gen_G = gen_G.state_dict()
            best_gen_F = gen_F.state_dict()
            best_disc_DX = disc_DX.state_dict()
            best_disc_DY = disc_DY.state_dict()

        val_disc_losses += valid_disc_loss.detach()
        val_gen_losses += valid_gen_loss.detach()

    # Average the losses for the epoch
    epoch_disc_losses /= len(train_loader)
    epoch_gen_losses /= len(train_loader)

    output = {
        "epoch_disc_losses": epoch_disc_losses,
        "epoch_gen_losses": epoch_gen_losses,
        "val_disc_losses": val_disc_losses,
        "val_gen_losses": val_gen_losses,
        "best_gen_G": best_gen_G,
        "best_gen_F": best_gen_F,
        "best_disc_DX": best_disc_DX,
        "best_disc_DY": best_disc_DY,
    }

    return output


def valid_epoch(val_loader: DataLoader, 
                epoch: int, 
                device: torch.device, 
                disc_DX: Discriminator, 
                disc_DY: Discriminator, 
                gen_G: Generator, 
                gen_F: Generator, 
                criterion_0: nn.Module= nn.MSELoss(), 
                criterion_1: nn.Module= nn.L1Loss(), 
                lambda_cycle: float= 10.0):
    """to validate the models for one epoch"""

    ## evaluation mode
    disc_DX.eval()
    disc_DY.eval()
    gen_G.eval()
    gen_F.eval()

    ## Create a progress bar for the training loop
    loop = tqdm(val_loader,leave=True, desc='validation [epoch {}]'.format(epoch))
    epoch_disc_losses, epoch_gen_losses = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    ## no need to compute gradients (optimize time)
    with torch.no_grad():

        for idx, batch in enumerate(loop):

            satellite_imgs = batch['satellite_imgs'].to(device, non_blocking=True)
            maps_imgs = batch['maps_imgs'].to(device, non_blocking=True)

            ## ===========================
            ## evaluate Discriminators (DX & DY)
            ## ===========================
            ## Generate fake map images (X -> Y)
            fake_maps = gen_G(satellite_imgs)

            ## Discriminator DY: Evaluate real and fake map images
            D_G_real = disc_DY(maps_imgs)
            D_G_fake = disc_DY(fake_maps)

            D_G_ones_like = torch.ones_like(D_G_real, device=device)
            D_G_zeros_like = torch.zeros_like(D_G_fake, device=device)

            D_G_real_loss = criterion_0(D_G_real, D_G_ones_like)
            D_G_fake_loss = criterion_0(D_G_fake, D_G_zeros_like)
            D_G_loss = D_G_real_loss + D_G_fake_loss

            ## Generate fake satellite images (Y -> X)
            fake_satellite = gen_F(maps_imgs)

            ## Discriminator DX: Evaluate real and fake satellite images
            D_F_real = disc_DX(satellite_imgs)
            D_F_fake = disc_DX(fake_satellite)

            D_F_ones_like = torch.ones_like(D_F_real, device=device)
            D_F_zeros_like = torch.zeros_like(D_F_fake, device=device)

            D_F_real_loss = criterion_0(D_F_real, D_F_ones_like)
            D_F_fake_loss = criterion_0(D_F_fake, D_F_zeros_like)
            D_F_loss = D_F_real_loss + D_F_fake_loss

            ## Combined discriminator loss
            D_loss = (D_G_loss + D_F_loss) / 2

            ## =====================
            ## evaluate Generators (G & F)
            ## =====================
            ## Adversarial loss for generators
            D_Y_fake = disc_DY(fake_maps)
            D_X_fake = disc_DX(fake_satellite)

            D_Y_ones_like = torch.ones_like(D_Y_fake, device=device)
            D_X_ones_like = torch.ones_like(D_X_fake, device=device)
            
            loss_G_Y = criterion_0(D_Y_fake, D_Y_ones_like)
            loss_G_X = criterion_0(D_X_fake, D_X_ones_like)

            ## Cycle consistency loss
            cycle_maps = gen_F(fake_maps)
            cycle_satellite = gen_G(fake_satellite)
            cycle_satellite_loss = criterion_1(satellite_imgs, cycle_maps)
            cycle_maps_loss = criterion_1(maps_imgs, cycle_satellite)

            ## Total generator loss
            G_loss = (
                loss_G_X
                + loss_G_Y
                + cycle_satellite_loss * lambda_cycle
                + cycle_maps_loss * lambda_cycle
            )

            epoch_disc_losses += D_loss.detach()
            epoch_gen_losses += G_loss.detach()

    # Average the losses for the epoch
    epoch_disc_losses /= len(val_loader)
    epoch_gen_losses /= len(val_loader)

    ## Save images for visualization
    # os.makedirs("./saved_images", exist_ok=True)
    save_image(maps_imgs*0.5+0.5, f"./figures/cyclegans/saved_images/maps_{epoch}.png")
    save_image(fake_maps*0.5+0.5, f"./figures/cyclegans/saved_images/fake_maps_{epoch}.png")
    save_image(satellite_imgs*0.5+0.5, f"./figures/cyclegans/saved_images/satellite_{epoch}.png")
    save_image(fake_satellite*0.5+0.5, f"./figures/cyclegans/saved_images/fake_satellite_{epoch}.png")

    return epoch_disc_losses, epoch_gen_losses

import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def evaluate_models(gen_G, gen_F, val_loader, cycle_criterion, device):
    ## Initialize accumulators for metrics
    total_rmse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_loss = 0.0
    num_batches = 0

    gen_G.to(device)
    gen_F.to(device)

    ## Switch models to evaluation mode
    gen_G.eval()
    gen_F.eval()

    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in tqdm(val_loader, leave=True, desc="Evaluation...."):
            ## Get satellite and map images
            satellite_imgs = batch['satellite_imgs'].to(device, non_blocking=True)
            maps_imgs = batch['maps_imgs'].to(device, non_blocking=True)

            ## Generate fake images
            fake_maps = gen_G(satellite_imgs)
            fake_satellite = gen_F(maps_imgs)

            ## Compute losses
            cycle_maps = gen_F(fake_maps)
            cycle_satellite = gen_G(fake_satellite)
            cycle_satellite_loss = cycle_criterion(satellite_imgs, cycle_maps)
            cycle_maps_loss = cycle_criterion(maps_imgs, cycle_satellite)
            loss = cycle_satellite_loss + cycle_maps_loss
            total_loss += loss.item()

            ## Compute metrics per batch
            for real, fake in [(maps_imgs, fake_maps), (satellite_imgs, fake_satellite)]:
                ## Denormalize images from [-1, 1] to [0, 1]
                real_np = ((real * 0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)
                fake_np = ((fake * 0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)

                ## Reshape tensors to (B, H, W, C) for SSIM and PSNR calculations
                real_np = real_np.transpose(0, 2, 3, 1)
                fake_np = fake_np.transpose(0, 2, 3, 1)

                for r, f in zip(real_np, fake_np):
                    ## RMSE
                    rmse = np.sqrt(np.mean((r - f) ** 2))
                    total_rmse += rmse

                    ## SSIM
                    ssim_val = ssim(r, f, multichannel=True, win_size=3)
                    total_ssim += ssim_val

                    ## PSNR
                    mse_val = np.mean((r - f) ** 2)
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse_val))
                    total_psnr += psnr

            num_batches += real.size(0)

    ## Compute average metrics
    avg_loss = total_loss / len(val_loader)
    avg_rmse = total_rmse / (num_batches * 2)  # Two image types (maps and satellite)
    avg_ssim = total_ssim / (num_batches * 2)
    avg_psnr = total_psnr / (num_batches * 2)

    ## Print metrics
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation RMSE: {avg_rmse:.4f}")
    print(f"Evaluation SSIM: {avg_ssim:.4f}")
    print(f"Evaluation PSNR: {avg_psnr:.4f}")

    return avg_loss, avg_rmse, avg_ssim, avg_psnr