# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import division, print_function, absolute_import
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.preprocessing import OrdinalEncoder
from ._configure import color_hex
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from itertools import combinations

def regu_path_plot(zeta_range, R_lambdas, all_comb, average_gain_all, mid_x_all):
    ix_set, ix_idx = np.unique(R_lambdas.argmax(0), return_index=True)
    enc = OrdinalEncoder()
    enc.fit_transform(np.sort(ix_set)[:, None]).squeeze().astype(int)
    y_value = enc.transform(R_lambdas.argmax(0)[:, None]).squeeze().astype(int)
    ylabel = np.array([('\{%s\}' % ', '.join([str(a) for a in aaa+1]))
                       for aaa in all_comb[np.sort(ix_set)]])

    iii = np.argsort(ix_idx)
    ylabel_ = ylabel[iii]


    width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)

    c1, c2 = color_hex(4), color_hex(2)

    plt.plot(zeta_range, y_value, '.', c=c1)
    ax = plt.gca()
    plt.yticks(np.arange(ylabel.shape[0]), ylabel)
    ax.tick_params('y', colors=c1)

    ax.yaxis.grid(False)

    ax2 = ax.twinx()
    ax2.boxplot(-average_gain_all.T, showfliers=True, showmeans=True, positions=mid_x_all,
                widths=width(mid_x_all, 0.2))
    ax2.tick_params('y', colors=c2)
    ax2.set_ylabel('$\\mathfrak D(S, i)$', color=c2)


    plt.xlim(np.percentile(zeta_range, [0, 100]))
    ax.set_xlabel('$\\zeta$')

    plt.xscale('log')


def left_bottom_spines_only():
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def remove_gray_background():
    ax = plt.gca()

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')


def visualize_2d_embedding(emb, colorings, colorings_title, axis,
                           label_prefix='\\phi'):
    ncols = len(colorings)
    for ii, (color, title) in enumerate(zip(colorings, colorings_title)):
        plt.subplot(1, ncols, ii+1)
        plt.scatter(*emb[:, axis].T, s=3, edgecolors='none', c=color,
                    cmap='plasma')

        plt.xlabel('$%s_{%d}$' % (label_prefix, axis[0]+1))
        plt.ylabel('$%s_{%d}$' % (label_prefix, axis[1]+1))
        ax = plt.gca()
        ax.grid(False)
        left_bottom_spines_only()
        ax.set_title(title)


def visualize_3d_embedding(emb, colorings, colorings_title,
                           axis, label_prefix='\\phi'):
    ncols = len(colorings)
    for ii, (color, title) in enumerate(zip(colorings, colorings_title)):
        ax = plt.subplot(1, ncols, ii+1, projection='3d')
        ax.scatter(*emb[:, axis].T, c=color, s=1, cmap='plasma')
        ax.set_xlabel('$%s_{%d}$' % (label_prefix, axis[0]+1), labelpad=20)
        ax.set_ylabel('$%s_{%d}$' % (label_prefix, axis[1]+1), labelpad=20)
        ax.set_zlabel('$%s_{%d}$' % (label_prefix, axis[2]+1), labelpad=20)
        ax.set_title(title)
        remove_gray_background()
        ax.grid(False)


def visualize_4d_embedding(emb, colorings, axis, label_prefix='\\phi'):
    ii = 0
    nrows = len(colorings)
    for color in colorings:
        for xy in combinations(axis, 2):
            plt.subplot(nrows, 6, ii+1)
            plt.scatter(*emb[:, xy].T, c=color, cmap='plasma',
                        s=.5, edgecolors='none')
            plt.xlabel('$%s_{%d}$' % (label_prefix, xy[0]+1))
            plt.ylabel('$%s_{%d}$' % (label_prefix, xy[1]+1))
            left_bottom_spines_only()
            plt.gca().grid(False)
            ii += 1
    plt.tight_layout()


def discretize_x_ticks(ax_str='x'):
    ax = plt.gca()
    if 'x' in ax_str:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if 'y' in ax_str:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
