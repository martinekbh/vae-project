import numpy as np
import matplotlib.pyplot as plt
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_latent_space2d(model, data, dims=[0,1], n_batches=100, device=DEVICE):
    with torch.no_grad():
        for i, (x, labels) in enumerate(data):
            batch_size = x.size(0)
            x = x.view(batch_size, x.size(-2)*x.size(-1))
            z = model.get_latents(x.to(DEVICE))
            z = z.cpu().detach().numpy()
            plt.scatter(z[:, dims[0]], z[:, dims[1]], c = labels, cmap = 'tab20')

            if i+1 >= n_batches:
                break
    cbar = plt.colorbar(ticks = data.dataset.dataset.labels)
    cbar.ax.set_yticklabels(["{:d}".format(i) for i in data.dataset.dataset.labels]) # add the labels
    #cbar.set_ticklabels(data.dataset.dataset.labels)

def plot_reconstructed(model, ax0=(-5, 5), ax1=(-5,5), n=12, img_size = (28,28), figsize=(8,8)):
    """ Sample uniformly from the latent space, and see how the decoder reconstructs inputs from arbitrary latents. """

    for i, y in enumerate(np.linspace(*ax0, n)):
        for j, x in enumerate(np.linspace(*ax1, n)):
            z = torch.Tensor([[x, y]]).to(DEVICE)
            x_hat = model.Decoder(z)
            x_hat = x_hat.view(x_hat.shape[0], img_size[0], img_size[1])
            x_hat = x_hat.cpu().detach().numpy()

            if i == 0 and j == 0:
                images = np.zeros((n*img_size[0], n*img_size[1]))
            images[ (n-1-i)*img_size[0] : (n-1-i+1)*img_size[0], j*img_size[1] : (j+1)*img_size[1]] = x_hat
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(images, extent=[*ax0, *ax1])
    ax.set_aspect('auto')
    plt.show()