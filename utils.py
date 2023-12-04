import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

def plot_2d_visualization(X_2d, y, figsize=(12, 12), dpi=100, title=None, legend=True, show=True, save=False):
    """
    Plot 2D visualization of the data
    """
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.gist_rainbow, s=10)
    plt.clim(-0.5, 9.5)
    plt.axis('off')

    if legend:
        norm = Normalize(vmin=0, vmax=9)
        legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(i),
                                markerfacecolor=plt.cm.gist_rainbow(norm(i)), markersize=5) for i in range(10)]
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    if save:
        plt.savefig(f"out/{title}.png")
    if show:
        plt.show()

def plot_digits_on_points(X_orig, X_2d, y, figsize=(12, 12), dpi=100, legend=True, color=True, show=True):
    """
    Plot the digits on the 2D visualization using gist_rainbow colormap
    """
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='gist_rainbow', s=10, alpha=0.0)

    for i in range(X_orig.shape[0]):
        img = X_orig[i].reshape(28, 28)  
        inverted_digit = 1 - (img > 0.5).astype(float)  # invert colors
        alpha_channel = (img > 0.5).astype(float)  # set alpha channel based on the digit

        # give number the color of the point
        if color: 
            color = scatter.to_rgba(y[i]) # get color of point
            inverted_digit = np.stack((inverted_digit,)*3, axis=-1)
            inverted_digit[:,:,0] = inverted_digit[:,:,0] * color[0]
            inverted_digit[:,:,1] = inverted_digit[:,:,1] * color[1]
            inverted_digit[:,:,2] = inverted_digit[:,:,2] * color[2]

            im = OffsetImage(inverted_digit, zoom=0.15, resample=True, alpha=alpha_channel)
        else: 
            im = OffsetImage(inverted_digit, zoom=0.2, cmap='gray', resample=True, alpha=alpha_channel)

        ab = AnnotationBbox(im, (X_2d[i, 0], X_2d[i, 1]), xycoords='data', frameon=False)
        ax.add_artist(ab)

        if legend:
            norm = Normalize(vmin=0, vmax=9)
            legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(i),
                                    markerfacecolor=plt.cm.gist_rainbow(norm(i)), markersize=5) for i in range(10)]
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

    ax.axis('off')
    if show:
        plt.show()

def plot_multiple_2d_visualizations(X_2d_list, y, title_list, plot_size, figsize=(20, 10), dpi=100, legend=True, show=True, save=False):
    """
    Plot multiple 2D visualizations of the data
    """
    fig, axs = plt.subplots(plot_size[0], plot_size[1], figsize=figsize, dpi=dpi)
    axs = axs.flatten()

    for i, (X_2d, title) in enumerate(zip(X_2d_list, title_list)):
        axs[i].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.gist_rainbow, s=10)
        axs[i].set_title(title)
        axs[i].axis('off')

        if legend:
            norm = Normalize(vmin=0, vmax=9)
            legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(i),
                                    markerfacecolor=plt.cm.gist_rainbow(norm(i)), markersize=5) for i in range(10)]
            axs[i].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    if save:
        plt.savefig(f"out/{title}.png")
    if show:
        plt.show()