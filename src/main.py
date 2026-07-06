# %%
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import scipy.signal as ssg
# import consts
import nsb
import utils

delta_phase = 14.0

theta_lists = utils.angular_wise_council(delta_phase)

# %%
# theta_list = theta_lists[50]
# print(theta_list)

# theta_grid = np.arange(0, 180.1, 0.1)
# weights = nsb.nsb_weights(theta_list)
# af_dB = utils.dB(utils.array_factor(weights, theta_grid))
# plt.plot(theta_grid, af_dB)
# peaks, _ = ssg.find_peaks(af_dB)
# plt.scatter(theta_grid[peaks], af_dB[peaks])


# %%
def run_sim(snr_db, delta_phase, output="AoAdev_SINR_SLL.txt"):
    print(f"snr_db: {snr_db}, delta_phase: {delta_phase}")
    theta_grid = np.arange(0, 180.01, 0.1)
    theta_wise_council = utils.angular_wise_council(delta_phase)

    entries = []
    for theta_list in theta_wise_council:
        sigma_sq = utils.calc_sigma_sq(snr_db)
        theta_extended, weights = nsb.iterative_sll_optim(
            theta_list, theta_grid, sigma_sq=sigma_sq
        )
        dtheta_0, dtheta_nulls, sll = utils.calc_divergences(
            weights, theta_list, theta_grid
        )
        # print("dtheta", dtheta_0)
        # print("dnulls", dtheta_nulls)
        # print("sll", sll)
        sinr_db = utils.calc_sinr(snr_db, theta_list, weights)
        # print("sinr_db", sinr_db)
        entry = list(theta_list) + [dtheta_0] + list(dtheta_nulls) + [sinr_db, sll]
        entries.append(entry)

    entries = np.array(entries)
    np.savetxt(output, entries)
    # return entries


def load_and_stats(snr, delta, input="AoAdev_SINR_SLL.txt"):
    all_data = np.loadtxt(input)
    dtheta_0 = all_data[:, 6]
    dtheta_nulls = all_data[:, 7:12].flatten()
    sinr = all_data[:, 12]
    sll = all_data[:, 13]

    def stats(x):
        return (
            round(np.min(x), 3),
            round(np.max(x), 3),
            round(np.mean(x), 3),
            round(np.std(x), 3),
        )

    df = pd.DataFrame(
        {
            "snr": [snr],
            "delta": [delta],
            "d_theta0_min": [stats(dtheta_0)[0]],
            "d_theta0_max": [stats(dtheta_0)[1]],
            "d_theta0_mean": [stats(dtheta_0)[2]],
            "d_theta0_std": [stats(dtheta_0)[3]],
            "d_nulls_min": [stats(dtheta_nulls)[0]],
            "d_nulls_max": [stats(dtheta_nulls)[1]],
            "d_nulls_mean": [stats(dtheta_nulls)[2]],
            "d_nulls_std": [stats(dtheta_nulls)[3]],
            "sinr_min": [stats(sinr)[0]],
            "sinr_max": [stats(sinr)[1]],
            "sinr_mean": [stats(sinr)[2]],
            "sinr_std": [stats(sinr)[3]],
            "sll_min": [stats(sll)[0]],
            "sll_max": [stats(sll)[1]],
            "sll_mean": [stats(sll)[2]],
            "sll_std": [stats(sll)[3]],
        }
    )
    return df


SNR_list = [-10, 0, 10, 20]
delta_list = [6, 8, 10, 12, 14, 16]

df_list = []
for snr in SNR_list:
    for delta in delta_list:
        filename = f"AoAdev_SINR_SLL_SNR{snr}_d{delta}.txt"
        run_sim(snr, delta, output=filename)
        stats_df = load_and_stats(snr, delta, filename)
        df_list.append(stats_df)

results_df = pd.concat(df_list, ignore_index=True)
results_df.to_csv("results.csv")
print(results_df)
