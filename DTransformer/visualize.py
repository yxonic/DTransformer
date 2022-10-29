import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


def trace_map(y, q, s, span=range(30), k_color=None, text_label=False, figsize=(22, 3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.pcolormesh(
        y[:, span].numpy(), edgecolors="w", linewidth=0.4, cmap="RdYlGn", clim=(0, 1)
    )
    ax.invert_yaxis()
    ax.set_yticks(np.arange(0, y.shape[0] + 1, 5))

    plt.colorbar(im, ax=ax, location="right")

    # circle label
    if k_color is None:
        knows = list(set(q[span].tolist()))
        cmap = matplotlib.cm.get_cmap("tab20")
        k_color = {k: cmap(i) for i, k in enumerate(knows)}

    x_offset = 0.5
    y_offset = -0.6
    for x, i in enumerate(span):
        if i == 0:
            continue
        q_ = q[i - 1].item()
        s_ = s[i - 1].item()
        ax.add_patch(
            plt.Circle((x + x_offset, y_offset), 0.4, color=k_color[q_], clip_on=False)
        )
        if s_ == 1:
            ax.add_patch(
                plt.Circle(
                    (x + x_offset, y_offset), 0.2, color="w", zorder=100, clip_on=False
                )
            )

    # text label
    if text_label:
        label = []
        for i in span:
            if i == 0:
                label.append("-")
            else:
                label.append(f"{q[i-1].item()}-{s[i-1].item()}")
        ax.set_xticks(np.arange(0.5, len(span)), label)

    return fig
