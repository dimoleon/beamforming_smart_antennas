import numpy as np
import scipy.signal as ssg

import consts


def dB(num, field=True):
    coef = 20.0 if field else 10.0
    return coef * np.log10(num + 1e-10)


def inv_dB(num_dB, field=True):
    coef = 20 if field else 10.0
    return 10 ** (num_dB / coef)


def angular_wise_council(delta_phase):
    start = 30.0
    wise_council = []

    while True:
        wisest = np.array([start + i * delta_phase for i in range(consts.NULLS + 1)])
        if wisest[-1] > 150:
            break

        for i in range(consts.NULLS + 1):
            kid = np.zeros(consts.NULLS + 1)
            kid[0] = wisest[i]
            j = 1
            for curr in range(consts.NULLS + 1):
                if curr == i:
                    continue
                kid[j] = wisest[curr]
                j += 1
            wise_council.append(kid)
        start += 1.0

    return wise_council


def steering_vector(theta_deg, M=consts.M, bd=consts.BD):
    m = np.arange(M)
    theta_rad = np.deg2rad(theta_deg)
    return np.exp(1j * bd * np.cos(theta_rad) * m)


def steering_matrix(theta_list):
    A = np.column_stack([steering_vector(theta) for theta in theta_list])
    return A


def array_factor(weights, theta_grid):
    A = np.array([steering_vector(theta) for theta in theta_grid])  # might be buggy
    af = A @ weights.conj()
    return af


def calc_sigma_sq(snr_db, sig_pow=consts.P_SIG):
    snr = inv_dB(snr_db, field=False)
    noise_var = sig_pow / snr
    return noise_var


def calc_sinr(snr_db, theta_list, weights, sig_pow=consts.P_SIG):
    theta_0 = theta_list[0]
    thetas_i = theta_list[1 : consts.NULLS + 1]
    noise_var = calc_sigma_sq(snr_db, sig_pow=sig_pow)

    a0 = steering_vector(theta_0)
    power_desired = np.real(np.abs(weights.conj() @ a0) ** 2 * sig_pow)

    int_power_sum = 0.0
    for theta in thetas_i:
        a = steering_vector(theta)
        int_power_sum += np.abs(weights.conj() @ a) ** 2 * sig_pow

    noise_power = np.real((weights.conj().T @ weights)) * noise_var

    sinr = power_desired / (int_power_sum + noise_power)
    return dB(sinr, field=False)


def calc_divergences(weights, theta_list, theta_grid):
    af_abs = np.abs(array_factor(weights, theta_grid))
    af_max = np.max(af_abs)
    af_db = dB(af_abs / af_max)

    theta_0 = theta_list[0]
    theta_nulls = theta_list[1 : consts.NULLS + 1]

    # main lobe
    main_lobe_idx = np.argmax(af_db)
    theta_0_est = theta_grid[main_lobe_idx]
    dtheta_0 = theta_0 - theta_0_est

    # interference nulls
    nulls_idx, _ = ssg.find_peaks(-af_abs)
    dtheta_nulls = []
    for theta in theta_nulls:
        est_null_idx = nulls_idx[np.argmin(np.abs(theta_grid[nulls_idx] - theta))]
        dtheta_nulls.append(theta_grid[est_null_idx] - theta)

    # sll
    lobs_idxs, _ = ssg.find_peaks(af_db)
    # print(lobs_idxs)
    sorted_idxs = lobs_idxs[np.argsort(-af_db[lobs_idxs])]

    sll = -np.inf
    if len(sorted_idxs) > 1:
        biggest_side_lobe_idx = sorted_idxs[1]
        sll = af_db[biggest_side_lobe_idx]

    return dtheta_0, dtheta_nulls, sll
