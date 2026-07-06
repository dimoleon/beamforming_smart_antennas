import numpy as np
import scipy.signal as ssg

import consts
import utils


def nsb_weights(theta_list, sigma_sq=0.0):
    A = utils.steering_matrix(theta_list)
    AhA = A.conj().T @ A

    basis_vec = np.zeros(len(theta_list))
    basis_vec[0] = 1.0

    chad = AhA + sigma_sq * np.eye(AhA.shape[0])
    # cond_num_init = np.linalg.cond(AhA)
    # cond_num_reg = np.linalg.cond(chad)
    # print(
    #     f"init matrix condition number {cond_num_init}, reg matrix condition number {cond_num_reg}"
    # )
    weights = A @ np.linalg.solve(chad, basis_vec)
    return weights


def iterative_sll_optim(
    theta_list, theta_grid, sll=consts.SLL_MAX, fake_nulls=consts.FAKE_MAX, sigma_sq=0.0
):
    wise_angles = list(theta_list)
    curr_weights = nsb_weights(wise_angles, sigma_sq=sigma_sq)
    slls = []
    weights_list = []
    for _ in range(fake_nulls):
        af_abs = np.abs(utils.array_factor(curr_weights, theta_grid))
        af_max = np.max(af_abs)
        af_db = utils.dB(af_abs / af_max)

        # main_lobe_idx = np.argmax(af_db)
        lobs_idxs, _ = ssg.find_peaks(af_db)
        sorted_idxs = lobs_idxs[np.argsort(-af_db[lobs_idxs])]

        if len(sorted_idxs) <= 1:
            break

        biggest_side_lobe_idx = sorted_idxs[1]
        curr_sll = af_db[biggest_side_lobe_idx]
        if curr_sll <= sll:
            return wise_angles, curr_weights
        slls.append(curr_sll)
        weights_list.append(curr_weights)

        wise_angles.append(theta_grid[biggest_side_lobe_idx])
        curr_weights = nsb_weights(wise_angles, sigma_sq=sigma_sq)

    optimal_idx = np.argmin(slls)
    optimal_weights = weights_list[optimal_idx]
    wise_angles_slice = wise_angles[: (consts.NULLS + 1 + optimal_idx)]
    return wise_angles_slice, optimal_weights
