import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_reconstructed(model, ax0=(-5, 5), ax1=(-5,5), n=12, dims=[0,1], img_size = (28,28), figsize=(8,8)):
    """ Sample uniformly from the latent space, and see how the decoder reconstructs inputs from arbitrary latents. """

    for i, y in enumerate(np.linspace(*ax0, n)):
        for j, x in enumerate(np.linspace(*ax1, n)):
            z = [0]*model.latent_dim
            z[dims[0]] = x
            z[dims[1]] = y
            z = torch.Tensor([z]).to(DEVICE)
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


def plot_latent_space2d(model, data, dims=[0,1], n_batches=100, device=DEVICE):
    with torch.no_grad():
        for i, (x, labels) in enumerate(data):
            batch_size = x.size(0)
            #x = x.view(batch_size, x.size(-2)*x.size(-1))   # Flattening the image
            z = model.get_latents(x.to(DEVICE))
            z = z.cpu().detach().numpy()
            plt.scatter(z[:, dims[0]], z[:, dims[1]], c = labels, cmap = 'tab20')

            if i+1 >= n_batches:
                break
    cbar = plt.colorbar(ticks = data.dataset.dataset.labels)
    cbar.ax.set_yticklabels(["{:d}".format(i) for i in data.dataset.dataset.labels]) # add the labels
    #cbar.set_ticklabels(data.dataset.dataset.labels)


def plot_latent_space_tsne(model, data, n_batches=100, device=DEVICE):
    with torch.no_grad():
        for i, (x, l) in enumerate(data):
            batch_size = x.size(0)
            #x = x.view(batch_size, x.size(-2)*x.size(-1))  # Flattening the image
            z = model.get_latents(x.to(DEVICE))
            z = z.cpu().detach().numpy()

            if i == 0:
                latents = z
                labels = l
            else:
                latents = np.vstack((latents, z))
                labels = np.hstack((labels, l))

            if i+1 >= n_batches:
                break

    tsne = TSNE(n_components=2, verbose=1)
    z = tsne.fit_transform(latents)

    plt.scatter(z[:, 0], z[:, 1], c = labels, cmap = 'tab20')
    cbar = plt.colorbar(ticks = data.dataset.dataset.labels)
    cbar.ax.set_yticklabels(["{:d}".format(i) for i in data.dataset.dataset.labels]) # add the labels


def plot_latent_space_pca(model, data, n_batches=100, device=DEVICE):
    with torch.no_grad():
        for i, (x, l) in enumerate(data):
            batch_size = x.size(0)
            #x = x.view(batch_size, x.size(-2)*x.size(-1)) # Flattening the image
            z = model.get_latents(x.to(DEVICE))
            z = z.cpu().detach().numpy()

            if i == 0:
                latents = z
                labels = l
            else:
                latents = np.vstack((latents, z))
                labels = np.hstack((labels, l))

            if i+1 >= n_batches:
                break

    pca = PCA(n_components=2)
    z = pca.fit_transform(latents)

    plt.scatter(z[:, 0], z[:, 1], c = labels, cmap = 'tab20')
    cbar = plt.colorbar(ticks = data.dataset.dataset.labels)
    cbar.ax.set_yticklabels(["{:d}".format(i) for i in data.dataset.dataset.labels]) # add the labels