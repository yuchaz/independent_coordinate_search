# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.spatial.distance import pdist
from megaman.geometry import Geometry, RiemannMetric
from megaman.embedding import SpectralEmbedding
from ._configure import time_counter


def compute_radius_embeddings(data, bandwidth, adjacency_method=None,
                              cyflann_index_type='kdtrees', radius_bw_ratio=3,
                              num_trees=10, laplacian_method='geometric',
                              eigen_solver='amg', n_components=20, verbose=False,
                              num_checks=60):
    if adjacency_method is None:
        adjacency_method = 'brute' if data.shape[0] < 20000 else 'cyflann'
    rad = bandwidth * radius_bw_ratio
    adjacency_kwds = dict(radius=rad)
    if adjacency_method == 'cyflann':
        cyflann_kwds = dict(index_type=cyflann_index_type,
                            num_trees=num_trees, num_checks=num_checks)
        adjacency_kwds['cyflann_kwds'] = cyflann_kwds
    affinity_method = 'gaussian'
    affinity_kwds = dict(radius=bandwidth)
    laplacian_kwds = dict(scaling_epps=bandwidth)
    geom = Geometry(
        adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
        affinity_method=affinity_method, affinity_kwds=affinity_kwds,
        laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
    geom.set_data_matrix(data)
    start = time_counter()
    affi = geom.compute_affinity_matrix()
    time_kernel = time_counter() - start
    if verbose:
        print('Time to build {} radius graph: {:.2f}s'.format(
            adjacency_method, time_kernel))

    start = time_counter()
    spectral = SpectralEmbedding(n_components=n_components, geom=geom,
                                 eigen_solver=eigen_solver)
    embedding = spectral.fit_transform(data)
    time_embedding = time_counter() - start
    if verbose:
        print('Time to eigendecomp using {} radius graph: {:.2f}s'.format(
            adjacency_method, time_embedding))
    return embedding, spectral.eigenvalues_, geom


def compute_tangent_plane(embedding, geom):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    L = geom.laplacian_matrix
    rmetric = RiemannMetric(embedding, L)
    rmetric.get_dual_rmetric()
    HH = rmetric.H
    evalues, evects = map(np.array, zip(*[eigsorted(HHi) for HHi in HH]))
    return evects
