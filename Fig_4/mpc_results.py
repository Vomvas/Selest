"""
This dictionary contains results obtained from running the MPC circuit on the raw data.
"""

mpc_results = {
    # Figure a
    "real_doa60": {
        # python3 ../secure_detection.py --raw-samples-dir ../data/real_60/ --scaling-factor 16
        "opt_music": [1 / (0.5*x) for x in [6.7333, 6.9084, 4.4169, 0.58382, 0.53709]],
        # python3 ../secure_detection.py --raw-samples-dir ../data/real_60/ --detection-mode sec_music --scaling-factor 16
        "selest": [0.4*x for x in [0.59344, 0.41284, 2.6585, 9.1196, 5.7232]],
        "savefile": "a.pdf",
        "actual_angle": 68,
        "leg_loc": "upper left"
    },
    # Figure b
    "drone_doa45": {
        # python3 ../secure_detection.py --raw-samples-dir ../data/drone_doa45/ --scaling-factor 16
        "opt_music": [1 / (1.2*x) for x in [2.3849, 2.2819, 1.4005, 2.5656, 5.2067]],
        # python3 ../secure_detection.py --raw-samples-dir ../data/drone_doa45/ --detection-mode sec_music --scaling-factor 16
        "selest": [0.55*x for x in [0.79828, 1.0455, 1.0404, 0.21618, 0.39391]],
        "savefile": "b.pdf",
        "actual_angle": 40,
        "leg_loc": "lower left"
    }
}
