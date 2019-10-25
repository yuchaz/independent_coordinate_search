# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import absolute_import, print_function, division
import numpy as np
from collections import defaultdict
import collections

def data_loader(alias_data, seed=None, **kwargs):
    """
    Input:

    Output:
        data dictionary, with keys:
            data: np.ndarray, dim = (N, d)
            colorings: List of coloring
            coloring_labels: List of string
            bw: bandwidth parameter for SpectralEmbedding
            intrinsic_dim: intrinsic dimension of manifold
            embedding_dim: embedding dimension of manifold

    """
    if isinstance(alias_data, int):
        alias_data = 'D%d' % alias_data
    elif isinstance(alias_data, str):
        alias_data = alias_data.upper()
    else:
        raise NotImplementedError('Please input int or str data')

    _loader_dict = _get_loader_dict()
    return _loader_dict[alias_data](seed=seed, **kwargs)


def _get_loader_dict():
    def default_loader():
        raise ValueError('Key not accepted')
    loader_dict = defaultdict(lambda: default_loader)

    _update = dict(
        D1=generate_2d_strip, D2=generate_2d_strip_with_cavity,
        D3=generate_swiss_roll, D4=generate_swiss_roll_with_cavity,
        D5=generate_gaussian_manifold, D6=generate_3d_cubes,
        D7=generate_high_torus, D8=generate_wide_torus,
        D9=generate_high_az_torus, D10=generate_high_ax_torus,
        D11=generate_wide_az_torus, D12=generate_wide_ax_torus,
        D13=generate_three_torus)

    loader_dict.update(_update)
    return loader_dict

# Noise helpers

def genereate_noises(sigmas, size, dimensions, seed=None, rdn_state=None):
    if rdn_state is None:
        rdn_state = np.random.RandomState(seed=seed)
    is_array_like = isinstance(sigmas, (collections.Sequence, np.ndarray))
    if is_array_like:
        assert len(sigmas) == dimensions, \
            'The size of sigmas should be the same as noises dimensions'
        return rdn_state.multivariate_normal(np.zeros(dimensions),
                                             np.diag(sigmas), size)
    else:
        return rdn_state.normal(0,sigmas,[size,dimensions])

def add_noises_on_primary_dimensions(data, sigmas=0.1, seed=None,
                                     rdn_state=None):
    size, dim = data.shape
    noises = genereate_noises(sigmas, size, dim, seed, rdn_state)
    return data + noises

def add_noises_on_additional_dimensions(data, addition_dims, sigmas=1,
                                        seed=None, rdn_state=None):
    noises = genereate_noises(sigmas, data.shape[0], addition_dims, seed,
                              rdn_state)
    return np.hstack((data,noises))


# Generate synthetic datasets

def generate_2d_strip(N_total=10000, height=2, ratio=2*np.pi, sigma_z=0.05,
                      seed=None, rdn_state=None):
    if rdn_state is None:
        rdn_state = np.random.RandomState(seed=seed)
    width = height * ratio
    x = rdn_state.uniform(-width, width, N_total)
    y = rdn_state.uniform(-height, height, N_total)
    z = np.zeros_like(x)
    data = np.vstack([x, y, z]).T
    clean_data = np.vstack([x, y]).T
    data += rdn_state.normal(0, sigma_z, size=data.shape)
    return dict(data=data, colorings=[data[:, 0], data[:, 1]], bw=.25,
                coloring_labels=['w', 'h'], intrinsic_dim=2, embedding_dim=2,
                name='2D long stripe (ratio $w/h = %.2f$)' % ratio,
                clean_data=clean_data, data_dim=2)

def generate_2d_strip_with_cavity(N_total=10000, height=2, ratio=2*np.pi,
                                  sigma_z=0.05, cavity_ratio=3, seed=None):
    data_dict = generate_2d_strip(N_total, height, ratio, sigma_z, seed)
    data_full = data_dict['data']
    clean_data = data_dict['clean_data']
    with_cavity_ix = np.logical_or(
        np.abs(data_full[:, 0]) > (ratio / cavity_ratio),
        np.abs(data_full[:, 1]) > (height / cavity_ratio))

    data = data_full[with_cavity_ix]
    clean_data = clean_data[with_cavity_ix]
    return dict(data=data, colorings=[data[:, 0], data[:, 1]], bw=.25,
                coloring_labels=['w', 'h'], intrinsic_dim=2, embedding_dim=2,
                name='2D long stripe with cavipy (ratio $w/h = %.2f$)' % ratio,
                clean_data=clean_data, data_dim=2)


def generate_swiss_roll(N_total=10000, height=2, ratio=3, sigma_z=0.05,
                        seed=None):
    data_dict = generate_2d_strip(N_total, height, ratio, sigma_z, seed)
    data_full = data_dict['data']
    clean_data = data_dict['clean_data']
    x, y, z = data_full.T
    x = x - x.min()
    data = np.array([0.5 * x * np.cos(x), y, 0.5 * x * np.sin(x)]).T
    return dict(data=data, colorings=[x, y], bw=1, coloring_labels=['w', 'h'],
                intrinsic_dim=2, embedding_dim=2,
                name='swiss roll (ratio $w/h = %.2f$)' % ratio,
                clean_data=clean_data, data_dim=3)


def generate_swiss_roll_with_cavity(N_total=10000, height=2, ratio=3,
                                    sigma_z=0.05, seed=None):

    data_dict = generate_2d_strip(N_total, height, ratio, sigma_z, seed)
    data_full = data_dict['data']
    clean_data = data_dict['clean_data']

    with_cavity_ix = np.logical_or(np.abs(data_full[:, 0]) > 1,
                                   np.abs(data_full[:, 1]) > 1/3)
    x, y, z = data_full[with_cavity_ix].T
    x = x - x.min()
    data = np.array([0.5 * x * np.cos(x), y, 0.5 * x * np.sin(x)]).T
    data_flatten = data_full[with_cavity_ix, :]
    clean_data = clean_data[with_cavity_ix]
    return dict(data=data, colorings=[data_flatten[:, 0], data_flatten[:, 1]],
                bw=1, coloring_labels=['w', 'h'], intrinsic_dim=2,
                embedding_dim=2,
                name='swiss roll with cavity (ratio $w/h = %.2f$)' % ratio,
                clean_data=clean_data, data_dim=3)


def generate_gaussian_manifold(N_total=13000, height=2, ratio=2*np.pi,
                               sigma_z=0.05, seed=None):
    rdn_state = np.random.RandomState(seed=seed)
    data_dict = generate_2d_strip(N_total, height, ratio, sigma_z,
                                  rdn_state=rdn_state)
    data_orig = data_dict['data']
    clean_data = data_dict['clean_data']

    circle_ix = ((data_orig[:, 0] / ratio) ** 2 + data_orig[:, 1] ** 2) < 4
    cicle_data = data_orig[circle_ix, :2]
    clean_data = clean_data[circle_ix]
    z_data = np.exp(-((cicle_data[:, 0] / ratio) ** 2 + (cicle_data[:, 1]) ** 2) / 2)
    data_clean = np.hstack([cicle_data, z_data[:, None]])
    data = add_noises_on_primary_dimensions(data_clean, rdn_state=rdn_state)

    return dict(data=data, colorings=[data[:, ix] for ix in range(3)],
                bw=.5, coloring_labels=['w', 'h', 'z'], intrinsic_dim=2,
                embedding_dim=2,
                name='gaussian manifold (ratio $w/h = %.2f$)' % ratio,
                clean_data=clean_data, data_dim=3)

def generate_3d_cubes(N_total=10000, ratios=(2, 4), sigma_w=0.05, seed=None):
    rdn_state = np.random.RandomState(seed=seed)
    width = 1
    length = width * ratios[0]
    hight = width * ratios[1]
    x = rdn_state.uniform(-width, width, N_total)
    y = rdn_state.uniform(-length, length, N_total)
    z = rdn_state.uniform(-hight, hight, N_total)

    w_noise = rdn_state.normal(0, sigma_w, x.shape[0])
    data = np.vstack([x, y, z, w_noise]).T
    return dict(data=data, colorings=[data[:, ix] for ix in range(3)],
                bw=.5, coloring_labels=['x', 'y', 'z'], intrinsic_dim=3,
                embedding_dim=3,
                name='3D cube (aspect ratio $(x, y, z) = (%.1f, %.1f, %.1f)$)' \
                     % ((1,) + ratios),
                clean_data=data[:, :3], data_dim=3)


def generate_torus(N, outer_rad, inner_rad, height, primary_sigma=0,
                   additional_dims=10, additional_sigma=0.5, seed=None):
    rdn_state = np.random.RandomState(seed=seed)
    theta = rdn_state.uniform(0, 2*np.pi, N)
    phi = rdn_state.uniform(0, 2*np.pi, N)

    x = (outer_rad + inner_rad*np.cos(theta)) * np.cos(phi)
    y = (outer_rad + inner_rad*np.cos(theta)) * np.sin(phi)
    z = inner_rad * np.sin(theta) * height
    torus_clean  = np.vstack([x, y, z]).T
    torus_noisy = add_noises_on_primary_dimensions(
        torus_clean, primary_sigma, rdn_state=rdn_state)
    torus_noisy = add_noises_on_additional_dimensions(
        torus_noisy, addition_dims=additional_dims, sigmas=additional_sigma,
        rdn_state=rdn_state)

    return torus_noisy, torus_clean, dict(theta=theta, phi=phi)


def generate_high_torus(N=10000, outer_rad=3, inner_rad=2, height=8,
                        primary_sigma=.1, additional_dims=10,
                        additional_sigma=.5, seed=None):
    data, data_clean, label_df = generate_torus(N, outer_rad, inner_rad, height,
                                                primary_sigma, additional_dims,
                                                additional_sigma, seed)
    return dict(data=data, colorings=[label_df['theta'], label_df['phi']],
                bw=1.5, coloring_labels=['$\\theta$', '$\\phi$'],
                intrinsic_dim=2, embedding_dim=3, name='high torus',
                clean_data=data_clean, data_dim=3)


def generate_wide_torus(N=10000, outer_rad=10, inner_rad=2, height=2,
                        primary_sigma=.1, additional_dims=10,
                        additional_sigma=.5, seed=None):
    data, data_clean, label_df = generate_torus(N, outer_rad, inner_rad, height,
                                                primary_sigma, additional_dims,
                                                additional_sigma, seed)
    return dict(data=data, colorings=[label_df['theta'], label_df['phi']],
                bw=1.5, coloring_labels=['$\\theta$', '$\\phi$'],
                intrinsic_dim=2, embedding_dim=3, name='wide torus',
                clean_data=data_clean, data_dim=3)

def generate_high_az_torus(N=10000, outer_rad=3, inner_rad=2, height=8,
                           primary_sigma=0, additional_dims=10,
                           additional_sigma=0, seed=None):
    data, torus_true, label_df = generate_torus(N, outer_rad, inner_rad, height,
                                                primary_sigma, additional_dims,
                                                additional_sigma, seed)

    def _transform(data):
        data[:, 2] = data[:, 2] - data[:, 2].min()
        data[:, 2] = data[:, 2] ** 3 / 1500
        data[:, 2] = data[:, 2] - data[:, 2].mean()
        return data

    data = _transform(data)
    torus_true = _transform(torus_true)
    return dict(data=data, colorings=[label_df['theta'], label_df['phi']],
                bw=1, coloring_labels=['$\\theta$', '$\\phi$'],
                intrinsic_dim=2, embedding_dim=3,
                name='z-asymmetrized high torus',
                clean_data=torus_true, data_dim=3)


def generate_high_ax_torus(N=10000, outer_rad=3, inner_rad=2, height=8,
                           primary_sigma=0, additional_dims=10,
                           additional_sigma=0, seed=None):
    data, torus_true, label_df = generate_torus(N, outer_rad, inner_rad, height,
                                                primary_sigma, additional_dims,
                                                additional_sigma, seed)

    def _transform(data):
        data[:, 0] = data[:, 0] - data[:, 0].min()
        data[:, 0] = data[:, 0] ** 2 / 10
        data[:, 0] = data[:, 0] - data[:, 0].mean()
        return data

    data = _transform(data)
    torus_true = _transform(torus_true)
    return dict(data=data, colorings=[label_df['theta'], label_df['phi']],
                bw=1, coloring_labels=['$\\theta$', '$\\phi$'],
                intrinsic_dim=2, embedding_dim=3,
                name='x-asymmetrized high torus',
                clean_data=torus_true, data_dim=3)


def generate_wide_az_torus(N=10000, outer_rad=10, inner_rad=2, height=2,
                           primary_sigma=.1, additional_dims=10,
                           additional_sigma=.5, seed=None):
    data, torus_true, label_df = generate_torus(N, outer_rad, inner_rad, height,
                                                primary_sigma, additional_dims,
                                                additional_sigma, seed)

    def _transform(data):
        data[:, 2] = data[:, 2] - data[:, 2].min()
        data[:, 2] = data[:, 2] ** 3 / 50
        data[:, 2] = data[:, 2] - data[:, 2].mean()
        return data

    data = _transform(data)
    torus_true = _transform(torus_true)
    return dict(data=data, colorings=[label_df['theta'], label_df['phi']],
                bw=2, coloring_labels=['$\\theta$', '$\\phi$'],
                intrinsic_dim=2, embedding_dim=3,
                name='z-asymmetrized wide torus',
                clean_data=torus_true, data_dim=3)


def generate_wide_ax_torus(N=10000, outer_rad=10, inner_rad=2, height=2,
                           primary_sigma=.1, additional_dims=10,
                           additional_sigma=.5, seed=None):
    data, torus_true, label_df = generate_torus(N, outer_rad, inner_rad, height,
                                                primary_sigma, additional_dims,
                                                additional_sigma, seed)

    def _transform(data):
        data[:, 0] = data[:, 0] - data[:, 0].min()
        data[:, 0] = data[:, 0] ** 3 / 1000
        data[:, 0] = data[:, 0] - data[:, 0].mean()
        return data

    data = _transform(data)
    torus_true = _transform(torus_true)
    return dict(data=data, colorings=[label_df['theta'], label_df['phi']],
                bw=2, coloring_labels=['$\\theta$', '$\\phi$'],
                intrinsic_dim=2, embedding_dim=3,
                name='x-asymmetrized wide torus',
                clean_data=torus_true, data_dim=3)


def generate_three_torus(seed=None):
    rdn_state = np.random.RandomState(seed=seed)
    theta = rdn_state.uniform(0, 2*np.pi, 50000)
    phi = rdn_state.uniform(0, 2*np.pi, 50000)
    varphi = rdn_state.uniform(0, 2*np.pi, 50000)

    w = (8 + (2 + np.cos(theta))*np.cos(phi))*np.cos(varphi)
    x = (8 + (2 + np.cos(theta))*np.cos(phi))*np.sin(varphi)
    y = (2 + np.cos(theta))*np.sin(phi)
    z = np.sin(theta)

    torus_clean = np.vstack([w, x, y, z]).T

    data = add_noises_on_additional_dimensions(
        add_noises_on_primary_dimensions(torus_clean, 0.1, rdn_state=rdn_state),
        10, 0.5, rdn_state=rdn_state)

    return dict(data=data, colorings=[theta, phi, varphi],
                bw=4, coloring_labels=['$\\theta$', '$\\phi$', '$\\varphi$'],
                intrinsic_dim=3, embedding_dim=4, name='three torus',
                clean_data=torus_clean, data_dim=4)
