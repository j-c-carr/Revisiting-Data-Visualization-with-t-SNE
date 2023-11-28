import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

def plot_2d_visualization(X_2d, y, title, figsize=(12, 12), legend=True, show=True, save=False):
    """
    Plot 2D visualization of the data
    """
    plt.figure(figsize=figsize)
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