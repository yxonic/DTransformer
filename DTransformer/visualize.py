import numpy as np


def heat_map(ax, alpha, cmap="hot", xticks=None, yticks=None):
    im = ax.pcolormesh(alpha, edgecolors="grey", linewidth=0.4, cmap=cmap)

    ax.invert_yaxis()
    if xticks is None:
        ax.set_xticks(np.arange(0, alpha.shape[1] + 1, 5))
    else:
        ax.set_xticks(xticks)

    if yticks is None:
        ax.set_yticks(np.arange(0, alpha.shape[0] + 1, 5))
    else:
        ax.set_yticks(yticks)

    return im
