# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import division, print_function, absolute_import
import numpy as np
from itertools import combinations
from ._configure import tqdm
import warnings

def _comp_projected_volume(principal_space, proj_axis, intrinsic_dim,
                           embedding_dim, eigen_values=None, zeta=1):
    basis = principal_space[:, proj_axis, :min(intrinsic_dim, embedding_dim)]
    basis = basis / np.linalg.norm(basis, axis=1)[:, None, :]
    try:
        vol_sq = np.linalg.det(np.einsum(
            'ijk,ijl->ikl', basis, basis))
        parallelepipe_vol = np.sqrt(vol_sq)
    except Exception as e:
        print(vol_sq[vol_sq<0])
        parallelepipe_vol = np.sqrt(np.abs(vol_sq))

    regu_term = _calc_regularizer(eigen_values, proj_axis, zeta)
    return np.log(parallelepipe_vol) - regu_term


def _calc_regularizer(eigen_values, proj_axis, zeta=1):
    if eigen_values is None:
        return 0
    eigen_values = np.abs(eigen_values[proj_axis])
    regu_term = np.sum(eigen_values) * zeta
    return regu_term


def projected_volume(principal_space, intrinsic_dim, embedding_dim=None,
                     eigen_values=None, zeta=1):
    candidate_dim = principal_space.shape[1]
    embedding_dim = intrinsic_dim if embedding_dim is None else embedding_dim

    all_axes = np.array(list(combinations(
        range(1, candidate_dim), embedding_dim-1)))
    all_axes = np.hstack([
        np.zeros((all_axes.shape[0], 1), dtype=all_axes.dtype), all_axes])

    proj_volume = []
    for proj_axis in all_axes:
        proj_vol = _comp_projected_volume(principal_space, proj_axis,
                                          intrinsic_dim, embedding_dim,
                                          eigen_values, zeta)
        proj_volume.append(proj_vol)

    proj_volume = np.array(proj_volume)
    return proj_volume, all_axes


def greedy_coordinate_search(principal_space, intrinsic_dim, eigen_values=None,
                             zeta=1, return_records=False):
    candidate_dim = principal_space.shape[1]
    proj_vol, all_comb = projected_volume(
        principal_space, intrinsic_dim, eigen_values, zeta)

    argmax_proj_vol = proj_vol.mean(1).argmax()
    opt_proj_axis = list(all_comb[argmax_proj_vol])
    remaining_axes = [ii for ii in range(candidate_dim)
                      if ii not in opt_proj_axis]

    ratio_records = [proj_vol]
    remaining_axes_records = [np.array(remaining_axes)]
    for embedding_dim in tqdm(range(intrinsic_dim+1, candidate_dim+1)):
        proj_vols = np.array([_comp_projected_volume(
            principal_space, np.array(opt_proj_axis + [k]),
            intrinsic_dim, embedding_dim, eigen_values, zeta)
            for k in remaining_axes])

        if return_records:
            ratio_records.append(proj_vols)
            remaining_axes_records.append(np.array(remaining_axes))

        k_opt_ix_ = np.argmax(proj_vols.mean(1))
        k_opt = remaining_axes[k_opt_ix_]
        opt_proj_axis.append(k_opt)
        remaining_axes.pop(k_opt_ix_)

    if return_records:
        return opt_proj_axis, ratio_records, remaining_axes_records
    else:
        return opt_proj_axis
