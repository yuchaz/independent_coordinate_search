# Author: Yu-Chia Chen <yuchaz@uw.edu>
# LICENSE: Simplified BSD

from __future__ import print_function, division, absolute_import
import numpy as np
from .coord_search import projected_volume
from ._configure import tqdm


def ies_range_search(principal_space, intrinsic_dim, embedding_dim,
                     eigen_values, zeta_range):
    proj_volume_loss_no_regu, all_comb = projected_volume(
        principal_space, intrinsic_dim, embedding_dim)
    lambdas_comb = (np.abs(eigen_values)[all_comb]).sum(1)

    R_mean = proj_volume_loss_no_regu.mean(1)
    return R_mean[:, None] - lambdas_comb[:, None] * zeta_range[None, :], all_comb


def average_no_regu_gain(proj_vol, candidate_set_ix):
    average_gain_all = []
    n_points = proj_vol.shape[1]
    for iset in candidate_set_ix:
        sum_all = proj_vol.sum(1)
        average_gain = []
        for ii in range(n_points):
            test_val = proj_vol[:, ii]
            residual_val = (sum_all - test_val) / (n_points - 1)
            test_max_set = test_val.argmax()
            average_gain.append(residual_val[iset] - residual_val[test_max_set])
        average_gain_all.append(average_gain)

    return np.array(average_gain_all)


def zeta_search(evects, lambdas, intrinsic_dim, embedding_dim, alpha=75,
                low=1e-2, high=1e5, sep=300):
    zeta_range = np.logspace(np.log10(low), np.log10(high), sep)
    R_lambdas, all_comb = ies_range_search(
        evects, intrinsic_dim, embedding_dim, lambdas, zeta_range)
    rloss, __ = projected_volume(evects, intrinsic_dim, embedding_dim)

    all_candidate_set_ix = R_lambdas.argmax(0)
    mid_x_all = []
    candiate_set = []
    last_zeta_same_set = zeta_range[0]
    for idx_, (start_ix, end_ix, start_zeta, end_zeta) in enumerate(zip(
        all_candidate_set_ix[:-1], all_candidate_set_ix[1:],
        zeta_range[:-1], zeta_range[1:])):

        if start_ix != end_ix:
            mid_x_all.append(np.exp(0.5 * (np.log(last_zeta_same_set) + np.log(start_zeta))))
            last_zeta_same_set = end_zeta
            candiate_set.append(start_ix)
        elif idx_ == zeta_range.shape[0]-2:
            mid_x_all.append(np.exp(0.5 * (np.log(last_zeta_same_set) + np.log(end_zeta))))
            candiate_set.append(start_ix)
    candiate_set, mid_x_all = map(np.array, [candiate_set, mid_x_all])

    average_gain_all = average_no_regu_gain(rloss, candiate_set)
    ptile = np.percentile(-average_gain_all, alpha, axis=1)
    last_ix = np.where(ptile <= 1e-10)[0][-1]
    zeta_chosen = mid_x_all[last_ix]

    plotting_dicts=dict(zeta_range=zeta_range, R_lambdas=R_lambdas,
                        average_gain_all=average_gain_all,
                        all_comb=all_comb, mid_x_all=mid_x_all)

    return zeta_chosen, plotting_dicts
