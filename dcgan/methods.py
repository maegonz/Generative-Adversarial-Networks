import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from .discriminator import Discriminator
from .generator import Generator

def weight_init(m):
    """
    Initialize the weights of the model layers from a zero-centered Normal
    distribution with a small standard deviation of 0.02.
    
    Parameters:
    m (nn.Module): A layer/module of the neural network.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)


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