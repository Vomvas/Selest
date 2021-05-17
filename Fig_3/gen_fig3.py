""" This scripts generates figure 3 of the paper """
import numpy as np
import os
import glob
import struct
import matplotlib.pyplot as plt
import cmath

samples_dir = "../data/real_doa90"
all_angles = range(0, 180, 1)
ang_avg_fact = 18  # Steering vector averaging factor

# Number of signals in transmission
n_antennas = 4
ant_dist = 0.5      # Distance between antennas in antenna array
n_signal_vec = 1
n_noise_vec = n_antennas - n_signal_vec

# Helper functions
def load_bytes_from_fd(fd, start=None, end=None):
    """
    Reads `batch` number of samples from a file descriptor into a tuple and returns the tuple
    """
    if start:
        fd.seek(start)

    if end and start:
        batch = end - start
    elif end:
        batch = end
    else:
        batch = -1

    binary = fd.read(batch)
    syntax = str(int(len(binary) / 4)) + "f"
    try:
        data = struct.unpack(syntax, binary)
        return data
    except struct.error:        # not enough bytes to unpack, end of binary
        return None


def load_matrix_from_raw_samples(samples_dir, scaling_factor=100):
    """
    Loads all complex samples files in the given directory. The number of files denotes the number of players. The
    minimum sample size among all files is considered as input length and averaging factor.
    """
    if not os.path.isdir(os.path.abspath(samples_dir)):
        print("Directory not found")
        return None

    sample_files = sorted(glob.glob(os.path.join(os.path.abspath(samples_dir), "*.32fc")))

    if not sample_files:
        print(f"No raw samples found in {samples_dir}")
        return None

    all_antenna_inputs = []
    for samp_file in sample_files:

        with open(samp_file, "rb") as rf:
            data = load_bytes_from_fd(rf)

        data = [scaling_factor * complex(data[i], data[i + 1]) for i in range(0, len(data), 2)]
        all_antenna_inputs.append(data)

    return all_antenna_inputs


def get_cov_mat(samples_dir):
    """ Loads the received data from all antennas and computes the covariance matrix """
    if not os.path.isdir(samples_dir):
        print("Samples directory not found: ", samples_dir)
        exit()

    player_inputs = np.array((load_matrix_from_raw_samples(samples_dir, scaling_factor=100),)).T
    n_players = player_inputs.shape[1]
    aggr_cov_mat = np.zeros((n_players, n_players), dtype="complex128")
    for n_avg in range(player_inputs.shape[0]):
        aggr_cov_mat += player_inputs[n_avg] @ player_inputs[n_avg].conjugate().T

    cov_mat = aggr_cov_mat / player_inputs.shape[0]
    return cov_mat


def get_steering_vector():
    """ Returns the original steering vector, averaged steering vector (used in Selest) and a range of avged angles """

    # Distance between elements in wavelengths
    players_distances = [i * ant_dist for i in range(n_antennas)]

    # Angle function (steering vector)
    ang_fun = np.empty(shape=(len(all_angles), n_antennas), dtype="complex128")
    for th in all_angles:
        ang_fun[th] = [cmath.exp(-1j * 2 * cmath.pi * dist * cmath.cos(th * cmath.pi / 180)) for dist in
                       players_distances]

    # Perform averaging of the steering vector to compare results with the averaged pseudospectrum
    ang_fun_avg = np.empty(shape=(int(len(all_angles) / ang_avg_fact), n_antennas), dtype="complex128")
    avged_ang_range = [x + ang_avg_fact / 2 for x in range(0, len(all_angles), ang_avg_fact)]

    # Avg angles over some averaging factor
    for th in range(0, len(all_angles), ang_avg_fact):
        avg_th = int(th / ang_avg_fact)
        avgd_angle = sum([x for x in ang_fun[th:th + ang_avg_fact]]) / ang_avg_fact
        ang_fun_avg[avg_th] = avgd_angle

    return ang_fun, ang_fun_avg, avged_ang_range


def plot_figure(pwr, pwr_angle_avg, subtitle, figname):
    """ Plot figure on axes """

    ang_fun, ang_fun_avg, avged_ang_range = get_steering_vector()

    # Calculate average of power using same avg factor as before (for steering vector)
    pwr_avg = []
    for p in range(0, len(pwr), ang_avg_fact):
        avg_p = int(p / ang_avg_fact)
        avged_power = sum([x for x in pwr[p:p + ang_avg_fact]]) / ang_avg_fact
        pwr_avg.append(avged_power)

    # Plot pseudospectrum
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid('on')
    ax.set_xticks(np.linspace(0, 180, 10))
    ax.set_yticks([round(x, 1) for x in np.linspace(min(pwr), max(pwr) + 1, 15)])
    ax.plot(all_angles, pwr, "-*", linewidth=2, label="1-degree step search")
    ax.plot(avged_ang_range, pwr_avg, "-s", linewidth=2, label=f"Avg pseudospectrum (f={ang_avg_fact})")
    ax.plot(avged_ang_range, pwr_angle_avg, "-^", linewidth=3, label=f"SELEST (f={ang_avg_fact})")
    ax.set_title(subtitle, fontsize=36)
    ax.legend(loc="lower left", fontsize=24)
    ax.set_xlabel("Angle (degrees)", fontsize=28)
    ax.set_ylabel("dB", fontsize=28)

    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1)

    plt.savefig(figname)


def plot_music(eigvects, n_noise_vec):
    """ Plot for figure a: traditional music """
    # Using noise subspace
    # Noise eigenvectors
    En = eigvects[:, 0:n_noise_vec]

    ang_fun, ang_fun_avg, avged_ang_range = get_steering_vector()

    # Calculate power based on 1-180 discretized steering vector
    pwr_n = []
    for angle in range(ang_fun.shape[0]):
        pwr_n.append(10 * np.log10(abs(1 / (ang_fun[angle].conjugate() @ En @ En.conjugate().T @ ang_fun[angle]))))

    # Calculate power based on averaged steering vector
    pwr_n_angle_avg = []
    for angle in range(ang_fun_avg.shape[0]):
        pwr_n_angle_avg.append(
            10 * np.log10(abs(1 / (ang_fun_avg[angle].conjugate() @ En @ En.conjugate().T @ ang_fun_avg[angle]))))

    subtitle = "Standard MUSIC"
    figname = "a.pdf"

    plot_figure(pwr_n, pwr_n_angle_avg, subtitle, figname)


def plot_opt_music(eigvects, n_noise_vec, n_signal_vec):
    """ Plot for figure b: Opt-music (this paper) """
    # Using signal subspace
    # Signal eigenvectors
    Es = eigvects[:, n_noise_vec:n_noise_vec + n_signal_vec]

    ang_fun, ang_fun_avg, avged_ang_range = get_steering_vector()

    # Calculate power based on 1-180 discretized steering vector
    pwr_s = []
    for angle in range(ang_fun.shape[0]):
        pwr_s.append(10 * np.log10(abs(ang_fun[angle].conjugate() @ Es @ Es.conjugate().T @ ang_fun[angle])))

    # Calculate power based on averaged steering vector
    pwr_s_angle_avg = []
    for angle in range(ang_fun_avg.shape[0]):
        pwr_s_angle_avg.append(
            10 * np.log10(abs((ang_fun_avg[angle].conjugate() @ Es @ Es.conjugate().T @ ang_fun_avg[angle]))))

    subtitle = "This paper (Opt-MUSIC)"
    figname = "b.pdf"

    plot_figure(pwr_s, pwr_s_angle_avg, subtitle, figname)


def plot_selest(cov_mat):
    """ Plot for figure c: Selest (this paper) """

    ang_fun, ang_fun_avg, avged_ang_range = get_steering_vector()

    # Form a random linear combination of columns of the covariance matrix
    # In its simplest form, this would just be a single column of the covariance matrix
    v1 = np.array((cov_mat[0],)).T
    v2 = np.array((cov_mat[1],)).T
    v3 = np.array((cov_mat[2],)).T
    v4 = np.array((cov_mat[3],)).T

    a1 = np.random.rand()
    a2 = np.random.rand()
    a3 = np.random.rand()
    a4 = np.random.rand()

    rand_lin_comb = a1 * v1 + a2 * v2 + a3*v3 + a4*v4

    # Using covariance matrix vector combination
    # Calculate power based on 1-180 discretized steering vector
    pwr_v = []
    for angle in range(ang_fun.shape[0]):
        pwr_v.append(10 * np.log10(
            abs(1 / ang_fun[angle].conjugate() @ rand_lin_comb @ rand_lin_comb.conjugate().T @ ang_fun[angle])))

    # Calculate power based on averaged steering vector
    pwr_v_angle_avg = []
    for angle in range(ang_fun_avg.shape[0]):
        pwr_v_angle_avg.append(10 * np.log10(
            abs((ang_fun_avg[angle].conjugate() @ rand_lin_comb @ rand_lin_comb.conjugate().T @ ang_fun_avg[angle]))))

    subtitle = "This paper (SELEST)"
    figname = "c.pdf"

    plot_figure(pwr_v, pwr_v_angle_avg, subtitle, figname)


def main():
    """ Process figure plots """

    cov_mat = get_cov_mat(samples_dir)

    # Eigendecomposition of covariance matrix
    eigvals, eigvects = np.linalg.eigh(cov_mat)

    # Figure a
    plot_music(eigvects, n_noise_vec)

    # Figure b
    plot_opt_music(eigvects, n_noise_vec, n_signal_vec)

    # Figure c
    plot_selest(cov_mat)


if __name__ == "__main__":
    main()
