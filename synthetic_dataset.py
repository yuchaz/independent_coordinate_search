# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.spatial import procrustes
import os

from ies_manifold import setup_color_palettes, tqdm
from ies_manifold import projected_volume, zeta_search
from ies_manifold import compute_radius_embeddings, compute_tangent_plane
from ies_manifold import regu_path_plot, data_loader, discretize_x_ticks
from ies_manifold import (visualize_2d_embedding, visualize_3d_embedding,
                          visualize_4d_embedding)

import matplotlib.pyplot as plt
setup_color_palettes()
from mpl_toolkits.mplot3d import axes3d, Axes3D


def calc_m2_score(clean_data, emb, ranked_axes, max_rank=1000):
    return np.array([procrustes(clean_data[:, :axis.shape[0]],
                                emb[:, axis])[-1]
                     for _, axis in zip(range(max_rank), ranked_axes)])


def visualize_embedding(data, colorings, coloring_labels, axis, label_prefix,
                        savename=None, dpi=300):
    embedding_dim = len(axis)
    if embedding_dim == 2:
        plt.figure(figsize=(16, 6))
        visualize_2d_embedding(
            data, colorings, coloring_labels, axis, label_prefix)
    elif embedding_dim == 3:
        plt.figure(figsize=(16, 6))
        visualize_3d_embedding(
            data, colorings, coloring_labels, axis, label_prefix)
    elif embedding_dim == 4:
        plt.figure(figsize=(16, 8))
        visualize_4d_embedding(data, colorings, axis, label_prefix)
    else:
        raise NotImplementedError('embedding_dim > 4 not yet implemented.')
    if savename is not None:
        plt.savefig(savename, dpi=dpi, bbox_inches='tight')
    plt.close()


figdir = 'output'
if not os.path.exists(figdir):
    os.makedirs(figdir)


def run_exp(data_alias):
    data_dict = data_loader(data_alias, seed=11)
    data = data_dict['data']
    colorings = data_dict['colorings']
    bw = data_dict['bw']
    coloring_labels = data_dict['coloring_labels']
    intrinsic_dim = data_dict['intrinsic_dim']
    embedding_dim = data_dict['embedding_dim']
    name = data_dict['name']
    clean_data = data_dict['clean_data']
    data_dim = data_dict['data_dim']

    visualize_embedding(data, colorings, coloring_labels, range(data_dim),
                        '\\vect{X}', os.path.join(
                            figdir, 'd%d_orig_data.png' % data_alias))

    emb, lambdas, geom = compute_radius_embeddings(
        data, bandwidth=bw, verbose=False)
    evects = compute_tangent_plane(emb, geom)

    zet_chosen, plotter_dict = zeta_search(
        evects, lambdas, intrinsic_dim, embedding_dim)
    proj_volume, all_comb = projected_volume(
        evects, intrinsic_dim, embedding_dim, lambdas, zet_chosen)
    chosen_axis = all_comb[proj_volume.mean(1).argmax()]

    plt.figure()
    regu_path_plot(**plotter_dict)
    plt.savefig(os.path.join(figdir, 'd%d_regu_path.pdf' % data_alias),
                bbox_inches='tight')

    plt.close()

    visualize_embedding(emb, colorings, coloring_labels, chosen_axis, '\\phi',
                        os.path.join(
                            figdir, 'd%d_emb_chosen.png' % data_alias))
    visualize_embedding(emb, colorings, coloring_labels,
                        range(chosen_axis.shape[0]), '\\phi',
                        os.path.join(
                            figdir, 'd%d_emb_first_two.png' % data_alias))

    proc_scores = calc_m2_score(
        clean_data, emb, all_comb[proj_volume.mean(1).argsort()[::-1]])
    if embedding_dim == 2:
        figsize = (8, 6)
    elif embedding_dim == 3:
        figsize = (10, 6)
    else:
        figsize = (12, 6)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(1, proc_scores.shape[0]+1), proc_scores, '*--')
    discretize_x_ticks()
    plt.xlabel('Rank')
    plt.ylabel('Disparity -- $M^2$')
    plt.savefig(os.path.join(figdir, 'd%d_m2_score.pdf' % data_alias),
                bbox_inches='tight')
    plt.close()


def main():
    for i in tqdm(range(1, 14)):
        run_exp(i)

if __name__ == '__main__':
    main()
