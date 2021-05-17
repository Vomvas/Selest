""" This scripts generates figure 4 of the paper """
import numpy as np
import os
import glob
import struct
import matplotlib.pyplot as plt
import cmath
import subprocess

# samples_dir = "../data/real_doa90"
all_angles = range(0, 90, 1)
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

#     all_antenna_inputs = [["{rp} {ip}".format(rp=x.real, ip=x.imag) for x in y] for y in all_antenna_inputs]

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


mpc_results = {
    # Figure a
    "real_doa60": {
        "opt_music_cmd": "python3 ../secure_detection.py --raw-samples-dir ../data/real_doa60/ --detection-mode opt_music --scaling-factor 34 --print-result-only",
        "opt_music": [],
        "opt_music_calibr": 1 / 0.5,
        "selest_cmd": "python3 ../secure_detection.py --raw-samples-dir ../data/real_doa60/ --scaling-factor 25 --print-result-only",
        "selest": [],
        "selest_calibr": 0.4,
        "savefile": "a.pdf",
        "actual_angle": 68,
        "leg_loc": "upper left"
    },
    # Figure b
    "drone_doa45": {
        "opt_music_cmd": "python3 ../secure_detection.py --raw-samples-dir ../data/drone_doa45/ --detection-mode opt_music --scaling-factor 16 --print-result-only",
        "opt_music": [],
        "opt_music_calibr": 1 / 1.2,
        "selest_cmd": "python3 ../secure_detection.py --raw-samples-dir ../data/drone_doa45/ --scaling-factor 10 --print-result-only",
        "selest": [],
        "selest_calibr": 0.55,
        "savefile": "b.pdf",
        "actual_angle": 40,
        "leg_loc": "lower left"
    }
}


def main():

    print("Generating figure 4...")

    for raw_samples in ["real_doa60", "drone_doa45"]:
        samples_dir = "../data/" + raw_samples

        cov_mat = get_cov_mat(samples_dir)

        # Eigendecomposition of covariance matrix
        eigvals, eigvects = np.linalg.eigh(cov_mat)

        En = eigvects[:, 0:n_noise_vec]
        ang_fun, _, avged_ang_range = get_steering_vector()

        # Calculate power based on 1-180 discretized steering vector
        pwr_n = []
        for angle in range(ang_fun.shape[0]):
            pwr_n.append(10 * np.log10(abs(1 / (ang_fun[angle].conjugate() @ En @ En.conjugate().T @ ang_fun[angle]))))

        # Run the MPC Protocols to get results
        print("Running MPC executions to get results, this will take a few minutes...")
        os.chdir("../MP_SPDZ_online")
        for det_mode in ["opt_music", "selest"]:
            mpc_output = subprocess.run(mpc_results[raw_samples][f"{det_mode}_cmd"].split(), capture_output=True, encoding="utf-8")
            ps = [0, 0, 0, 0, 0]
            for line in mpc_output.stdout.split("\n"):
                if "Pseudospectrum: " in line:
                    ps = line.split("Pseudospectrum: ")[1].strip(" []").split(", ")
            mpc_results[raw_samples][det_mode] = [mpc_results[raw_samples][f"{det_mode}_calibr"] * float(x) for x in ps]
        os.chdir("../Fig_4")
        res_dict = mpc_results[raw_samples]
        actual_angle = res_dict["actual_angle"]
        savefile = res_dict["savefile"]
        leg_loc = res_dict["leg_loc"]

        pwr_opt_music = [10 * np.log10(x) for x in res_dict["opt_music"]]
        pwr_selest = [10 * np.log10(x) for x in res_dict["selest"]]
        mins = [min(pwr_n), min(pwr_opt_music), min(pwr_selest)]
        maxs = [max(pwr_n), max(pwr_opt_music), max(pwr_selest)]

        # Plot pseudospectrum
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.grid('on')
        ax.set_xticks(np.linspace(0, 180, 10))
        ax.set_yticks([round(x, 1) for x in np.linspace(min(mins), max(maxs) + 1, 15)])
        ax.plot(all_angles, pwr_n, "-*", linewidth=2, label="Standard MUSIC")
        ax.plot(avged_ang_range, pwr_opt_music, "-^", linewidth=2, label=f"This paper (Opt-MUSIC)")
        ax.plot(avged_ang_range, pwr_selest, "-s", linewidth=3, label=f"This paper (SELEST)")
        ax.set_title(f"Angle of arrival: {actual_angle} degrees", fontsize=36)
        ax.set_xlabel("Angle (degrees)", fontsize=28)
        ax.set_ylabel("dB", fontsize=28)
        ax.legend(loc=leg_loc, fontsize=26)

        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.1)

        plt.savefig(savefile)


if __name__ == "__main__":
    main()
