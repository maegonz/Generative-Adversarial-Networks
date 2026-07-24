import matplotlib.pyplot as plt

def display_satellite_map_pairs(satellite_images, map_images, suptitle=None):
    """
    Displays a set of satellite images with their corresponding map images in a dynamic grid.
    The first row shows satellite images, and the second row shows map images.

    Parameters:
    - satellite_images (torch.Tensor): Batch of satellite images of shape (B, C, W, H).
    - map_images (torch.Tensor): Batch of corresponding map images of shape (B, C, W, H).
    """
    ## Ensure the batch sizes of satellite and map images are the same
    assert satellite_images.shape[0] == map_images.shape[0], "Batch sizes must match!"

    ## denormalize image from (-1, 1) to (0, 255)
    satellite_images = ((satellite_images + 1.0) / 2) * 255
    map_images = ((map_images + 1.0) / 2) * 255

    ## Number of images in the batch
    batch_size = satellite_images.shape[0]

    ## Create subplots: two rows (satellite & maps), and a variable number of columns based on batch size
    fig, axes = plt.subplots(2, batch_size, figsize=(15, 5))

    for i in range(batch_size):
        # Convert tensors to numpy arrays for plotting
        sat_img = satellite_images[i].permute(1, 2, 0).cpu().numpy().astype("uint8")
        map_img = map_images[i].permute(1, 2, 0).cpu().numpy().astype("uint8")

        # Display satellite image in the first row
        axes[0, i].imshow(sat_img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Satellite {i+1}")

        # Display map image in the second row
        axes[1, i].imshow(map_img)
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Map {i+1}")

    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


import torch

def plot_predictions(gen_G, gen_F, dataloader, device, num_samples=5):
    """
    Plot original images and their corresponding predicted images using the trained generators.

    Args:
        gen_G (nn.Module): Generator G (domain X -> Y).
        gen_F (nn.Module): Generator F (domain Y -> X).
        dataloader (torch.utils.data.DataLoader): Dataloader containing the test data.
        device (str): Device to run the models on ("cpu" or "cuda").
        num_samples (int): Number of samples to plot.
    """

    gen_G.to(device)
    gen_F.to(device)

    gen_G.eval()  # Set generator G to evaluation mode
    gen_F.eval()  # Set generator F to evaluation mode

    data_iter = iter(dataloader)
    fig, axes = plt.subplots(num_samples, 4, figsize=(15, num_samples * 4))
    fig.suptitle("Ground Truth vs. Predicted Images", fontsize=16)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i in range(num_samples):
            # Fetch a batch of data
            batch = next(data_iter)
            satellite_imgs = batch['satellite_imgs'].to(device, non_blocking=True)  # Domain X
            maps_imgs = batch['maps_imgs'].to(device, non_blocking=True)            # Domain Y

            # Generate predictions
            fake_maps = gen_G(satellite_imgs)  # X -> Y
            fake_satellite = gen_F(maps_imgs)  # Y -> X

            # Normalize images from [-1, 1] to [0, 1] for visualization
            satellite_imgs = (satellite_imgs + 1) / 2
            maps_imgs = (maps_imgs + 1) / 2
            fake_maps = (fake_maps + 1) / 2
            fake_satellite = (fake_satellite + 1) / 2

            # Plot the results
            axes[i, 0].imshow(satellite_imgs[0].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title("Original Satellite (X)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(fake_maps[0].permute(1, 2, 0).cpu().numpy())
            axes[i, 1].set_title("Predicted Map (X -> Y)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(maps_imgs[0].permute(1, 2, 0).cpu().numpy())
            axes[i, 2].set_title("Original Map (Y)")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(fake_satellite[0].permute(1, 2, 0).cpu().numpy())
            axes[i, 3].set_title("Predicted Satellite (Y -> X)")
            axes[i, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()