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

def training(generatorG: Generator,
             generatorF: Generator,
             discriminatorX: Discriminator,
             discriminatorY: Discriminator,
             train_loader: DataLoader,
             device: torch.device,
             epochs: int,
             optimizerG: Optimizer,
             optimizerD: Optimizer,
            #  batch_size: int=64,
             criterion: List[nn.Module] = [nn.MSELoss(), nn.L1Loss()],
             val_loader: DataLoader=None,
             lambda_cycle: float = 10.0,
             use_amp: bool=True):
    """to train the models for one epoch"""

    ## put models in training mode
    disc_DX.train()
    disc_DY.train()
    gen_G.train()
    gen_F.train()

    ## Create a progress bar for the training loop
    loop = tqdm(loader,leave=True, desc='training [epoch {}]'.format(epoch))
    epoch_disc_loss, epoch_gen_loss = 0.0, 0.0

    for idx, batch in enumerate(loop):

        satellite_imgs = batch['satellite_imgs'].to(device)
        maps_imgs = batch['maps_imgs'].to(device)

        ## ===========================
        ## Train Discriminators (DX & DY)
        ## ===========================
        with torch.amp.autocast(device):
            ## Generate fake map images (X -> Y)
            fake_maps = gen_G(satellite_imgs)

            ## Discriminator DY: Evaluate real and fake map images
            D_G_real = disc_DY(maps_imgs)
            D_G_fake = disc_DY(fake_maps.detach())
            D_G_real_loss = mse(D_G_real, torch.ones_like(D_G_real))
            D_G_fake_loss = mse(D_G_fake, torch.zeros_like(D_G_fake))
            D_G_loss = D_G_real_loss + D_G_fake_loss

            ## Generate fake satellite images (Y -> X)
            fake_satellite = gen_F(maps_imgs)

            ## Discriminator DX: Evaluate real and fake satellite images
            D_F_real = disc_DX(satellite_imgs)
            D_F_fake = disc_DX(fake_satellite.detach())
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            ## Combined discriminator loss
            D_loss = (D_G_loss + D_F_loss) / 2

        ## Backpropagation and optimization for the discriminators
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        ## =====================
        ## Train Generators (G & F)
        ## =====================
        with torch.amp.autocast(device):
            ## Adversarial loss for generators
            D_Y_fake = disc_DY(fake_maps)
            D_X_fake = disc_DX(fake_satellite)
            loss_G_Y = mse(D_Y_fake,torch.ones_like(D_Y_fake))
            loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))

            ## Cycle consistency loss
            cycle_maps = gen_F(fake_maps)
            cycle_satellite = gen_G(fake_satellite)
            cycle_satellite_loss = L1(satellite_imgs, cycle_maps)
            cycle_maps_loss = L1(maps_imgs, cycle_satellite)

            ## Total generator loss
            G_loss = (
                loss_G_X
                + loss_G_Y
                + cycle_satellite_loss * lambda_cycle
                + cycle_maps_loss * lambda_cycle
            )

        ## Backpropagation and optimization for the generators
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        epoch_disc_loss += D_loss.item()
        epoch_gen_loss += G_loss.item()

    # Average the losses for the epoch
    epoch_disc_loss /= len(loader)
    epoch_gen_loss /= len(loader)

    return epoch_disc_loss, epoch_gen_loss


def valid_epoch(epoch, device, disc_DX, disc_DY, gen_G, gen_F, mse, L1, loader, lambda_cycle):
    """to validate the models for one epoch"""

    ## put models in training mode
    disc_DX.eval()
    disc_DY.eval()
    gen_G.eval()
    gen_F.eval()

    ## Create a progress bar for the training loop
    loop = tqdm(loader,leave=True, desc='validation [epoch {}]'.format(epoch))
    epoch_disc_loss, epoch_gen_loss = 0.0, 0.0

    ## no need to compute gradients (optimize time)
    with torch.no_grad():

        for idx, batch in enumerate(loop):

            satellite_imgs = batch['satellite_imgs'].to(device)
            maps_imgs = batch['maps_imgs'].to(device)

            ## ===========================
            ## evaluate Discriminators (DX & DY)
            ## ===========================
            ## Generate fake map images (X -> Y)
            fake_maps = gen_G(satellite_imgs)

            ## Discriminator DY: Evaluate real and fake map images
            D_G_real = disc_DY(maps_imgs)
            D_G_fake = disc_DY(fake_maps.detach())
            D_G_real_loss = mse(D_G_real, torch.ones_like(D_G_real))
            D_G_fake_loss = mse(D_G_fake, torch.zeros_like(D_G_fake))
            D_G_loss = D_G_real_loss + D_G_fake_loss

            ## Generate fake satellite images (Y -> X)
            fake_satellite = gen_F(maps_imgs)

            ## Discriminator DX: Evaluate real and fake satellite images
            D_F_real = disc_DX(satellite_imgs)
            D_F_fake = disc_DX(fake_satellite.detach())
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            ## Combined discriminator loss
            D_loss = (D_G_loss + D_F_loss) / 2

            ## =====================
            ## evaluate Generators (G & F)
            ## =====================
            ## Adversarial loss for generators
            D_Y_fake = disc_DY(fake_maps)
            D_X_fake = disc_DX(fake_satellite)
            loss_G_Y = mse(D_Y_fake,torch.ones_like(D_Y_fake))
            loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))

            ## Cycle consistency loss
            cycle_maps = gen_F(fake_maps)
            cycle_satellite = gen_G(fake_satellite)
            cycle_satellite_loss = L1(satellite_imgs, cycle_maps)
            cycle_maps_loss = L1(maps_imgs, cycle_satellite)

            ## Total generator loss
            G_loss = (
                loss_G_X
                + loss_G_Y
                + cycle_satellite_loss * lambda_cycle
                + cycle_maps_loss * lambda_cycle
            )

            epoch_disc_loss += D_loss.item()
            epoch_gen_loss += G_loss.item()

    # Average the losses for the epoch
    epoch_disc_loss /= len(loader)
    epoch_gen_loss /= len(loader)

    ## Save images for visualization
    os.makedirs("./saved_images", exist_ok=True)
    save_image(maps_imgs*0.5+0.5, f"./saved_images/maps_{epoch}.png")
    save_image(fake_maps*0.5+0.5, f"./saved_images/fake_maps_{epoch}.png")
    save_image(satellite_imgs*0.5+0.5, f"./saved_images/satellite_{epoch}.png")
    save_image(fake_satellite*0.5+0.5, f"./saved_images/fake_satellite_{epoch}.png")

    return epoch_disc_loss, epoch_gen_loss


def training(generator: Generator,
             discriminator: Discriminator,
             train_loader: DataLoader,
             device: torch.device,
             epochs: int,
             optimizerG: Optimizer,
             optimizerD: Optimizer,
            #  batch_size: int=64,
             criterion: nn.Module = nn.BCELoss(),
             val_loader: DataLoader=None,
             use_amp: bool=True):
    """
    Train a PyTorch Generative Adversarial Network (GAN) model with optional Automatic Mixed Precision.

    Parameters
    ----------
    generator : Generator
        The generator model to be trained.
    discriminator : Discriminator
        The discriminator model to be trained.
    train_loader : DataLoader
        DataLoader providing the training dataset.
    criterion : nn.Module
        Loss function used to compute training loss.
    optimizerG : torch.optim.Optimizer
        Optimizer used to update generator model parameters.
    optimizerD : torch.optim.Optimizer
        Optimizer used to update discriminator model parameters.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    batch_size : int, Optional
        Batch size for training. Default is 64.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    D_loss : list of float
        Average discriminator training loss for each epoch.
    G_loss : list of float
        Average generator training loss for each epoch.
    """

    generator.to(device)
    discriminator.to(device)

    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # noise_z = torch.randn(batch_size, generator.latent_dim, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    D_loss, G_loss = [], []

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        running_D_loss = 0.0
        running_G_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for id, (batch, _) in enumerate(loop):
            ############################
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # train with real image
            real_img = batch.to(device)
            # assert real_img.size(0) == batch_size, "Batch size mismatch."
            batch_size = real_img.size(0)
            assert not torch.isnan(real_img).any(), "NaN detected in real images"
            assert not torch.isinf(real_img).any(), "Inf detected in real images"
            real_labels = torch.full((batch_size,), real_label, dtype=real_img.dtype, device=device)

            # train with fake image
            noise = torch.randn(batch_size, generator.latent_dim, 1, 1, device=device)
            fake_img = generator(noise)
            assert not torch.isnan(fake_img).any(), "NaN detected in fake images"
            assert not torch.isinf(fake_img).any(), "Inf detected in fake images"
            fake_labels = torch.full((batch_size,), fake_label, dtype=real_img.dtype, device=device)

            optimizerD.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                output_real = discriminator(real_img)
                loss_real = criterion(output_real, real_labels)

                output_fake = discriminator(fake_img.detach())
                loss_fake = criterion(output_fake, fake_labels)
            
                assert not torch.isnan(loss_real), "NaN detected in Discriminator real loss"
                assert not torch.isnan(loss_fake), "NaN detected in Discriminator fake loss"

                errD = loss_real + loss_fake

            scaler.scale(errD).backward()
            scaler.unscale_(optimizerD)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            scaler.step(optimizerD)
            scaler.update()

            D_x = output_real.mean().item()
            D_G_z1 = output_fake.mean().item()

            running_D_loss += errD.item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            optimizerG.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                output = discriminator(fake_img)
                errG = criterion(output, real_labels)
            
            scaler.scale(errG).backward()
            scaler.unscale_(optimizerG)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optimizerG)
            scaler.update()

            D_G_z2 = output.mean().item()
            running_G_loss += errG.item()

            loop.set_postfix(loss_D=errD.item(), loss_G=errG.item(), D_x=D_x, D_G_z1=D_G_z1, D_G_z2=D_G_z2)

            if id % 100 == 0:
                vutils.save_image(real_img,'./GANs/outputs/real_samples.png', normalize=True)
                vutils.save_image(fake_img.detach(), './GANs/outputs/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

        torch.cuda.empty_cache()

        epoch_D_loss = running_D_loss / len(train_loader.dataset)
        D_loss.append(epoch_D_loss)
        epoch_G_loss = running_G_loss / len(train_loader.dataset)
        G_loss.append(epoch_G_loss)

        epoch_tqdm.set_postfix(train_D_loss=epoch_D_loss, train_G_loss=epoch_G_loss)

        # torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % ('./GANs/saved', epoch))
        # torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % ('./GANs/saved', epoch))
    return D_loss, G_loss