import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import silhouette_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_reconstructed(model, ax0=(-5, 5), ax1=(-5,5), n=12, dims=[0,1], img_size = (28,28), figsize=(8,8)):
    """ Sample uniformly from the latent space, and see how the decoder reconstructs inputs from arbitrary latents. """

    for i, y in enumerate(np.linspace(*ax0, n)):
        for j, x in enumerate(np.linspace(*ax1, n)):
            z = [0]*model.latent_dim
            z[dims[0]] = x
            z[dims[1]] = y
            z = torch.Tensor([z]).to(DEVICE)
            #x_hat = model.Decoder(z)
            x_hat = model.decoder(z)
            x_hat = x_hat.view(x_hat.shape[0], img_size[0], img_size[1])
            x_hat = x_hat.cpu().detach().numpy()

            if i == 0 and j == 0:
                images = np.zeros((n*img_size[0], n*img_size[1]))
            images[ (n-1-i)*img_size[0] : (n-1-i+1)*img_size[0], j*img_size[1] : (j+1)*img_size[1]] = x_hat
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(images, extent=[*ax0, *ax1], cmap=plt.get_cmap('gray'))
    ax.set_aspect('auto')
    plt.show()


def plot_latent_space2d(model, data, dims=[0,1], n_batches=1, device=DEVICE):
    with torch.no_grad():
        for i, (x, labels) in enumerate(data):
            z = model.get_latents(x.to(DEVICE))
            z = z.cpu().detach().numpy()
            plt.scatter(z[:, dims[0]], z[:, dims[1]], c = labels, cmap = 'tab20')

            if i+1 >= n_batches:
                break
    cbar = plt.colorbar(ticks = data.dataset.dataset.labels)
    cbar.ax.set_yticklabels(["{:d}".format(i) for i in data.dataset.dataset.labels]) # add the labels
    #cbar.set_ticklabels(data.dataset.dataset.labels)


def plot_latent_space_tsne(model, data, n_batches=1, device=DEVICE):
    with torch.no_grad():
        for i, (x, l) in enumerate(data):
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


def plot_data_space_tsne(data_loader, n_batches=1, device=DEVICE):
    with torch.no_grad():
        for i, (x, l) in enumerate(data_loader):
            if i == 0:
                img_shape = x.shape[1:]
                X = x.reshape([-1,np.prod(img_shape)])
                labels = l
            else:
                X = np.vstack((X, x.reshape([-1,np.prod(img_shape)])))
                labels = np.hstack((labels, l))

            if i+1 >= n_batches:
                break
    
    tsne = TSNE(n_components=2, verbose=1)
    X_tsne = tsne.fit_transform(X)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = labels, cmap = 'tab20')
    cbar = plt.colorbar(ticks = data_loader.dataset.dataset.labels)
    cbar.ax.set_yticklabels(["{:d}".format(i) for i in data_loader.dataset.dataset.labels]) # add the labels


def plot_latent_space_pca(model, data, n_batches=1, device=DEVICE):
    with torch.no_grad():
        for i, (x, l) in enumerate(data):
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


def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape


def visualizeDataset(X):
    """ Show every image. Good for picking interplation candidates. """
    for i, (image, _) in enumerate(X):
        print(image.squeeze().shape)
        cv2.imshow(str(i),image.squeeze().cpu().detach().numpy())
        cv2.waitKey()
        cv2.destroyAllWindows()


def imscatter(x, y, ax, imageData, zoom):
    """ Scatter with images instead of points. """
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]#*255.
        img = img.squeeze().cpu().detach().numpy()
        #img = img.astype(np.uint8).reshape([img_size,img_size])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def computeTSNEProjectionOfPixelSpace(data_loader, display=True, n_batches=1):
    """ Show dataset images with T-sne projection of pixel space. """
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")

    with torch.no_grad():
        for i, (x, l) in enumerate(data_loader):
            if i == 0:
                X = x
                labels = l
                img_shape = x.shape[1:]
            else:
                X = np.vstack((X, x))
                labels = np.hstack((labels, l))

            if i+1 >= n_batches:
                break

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1,np.prod(img_shape)]))
    #X_tsne = tsne.fit_transform(X)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        zoom = 0.1
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=zoom)
        plt.show()
    else:
        return X_tsne


def computeTSNEProjectionOfLatentSpace(data_loader, model, display=True, n_batches=1):
    """ Show dataset images with T-sne projection of latent space encoding. """

    # Compute latent space representation
    print("Computing latent space projection...")

    with torch.no_grad():
        for i, (x, l) in enumerate(data_loader):
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

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    z_tsne = tsne.fit_transform(z)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        zoom = 0.1
        imscatter(z_tsne[:, 0], z_tsne[:, 1], imageData=x, ax=ax, zoom=zoom)
        plt.show()
    else:
        return z_tsne


def silhouette_score_in_data(data_loader, n_batches = 100):
    """ Computes the silhouette scores of the clusters in the latent space """

    with torch.no_grad():
        for i, (x, l) in enumerate(data_loader):
            if i == 0:
                img_shape = x.shape[1:]
                X = x.reshape([-1,np.prod(img_shape)])
                labels = l
            else:
                X = np.vstack((X, x.reshape([-1,np.prod(img_shape)])))
                labels = np.hstack((labels, l))

            if i+1 >= n_batches:
                break

    sil_score = silhouette_score(X, labels)
    return sil_score

def silhouette_score_in_latents(data_loader, model, n_batches = 100):
    """ Computes the silhouette scores of the clusters in the latent space """
    model.eval()
    with torch.no_grad():
        for i, (x, l) in enumerate(data_loader):
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

    sil_score = silhouette_score(latents, labels)
    return sil_score