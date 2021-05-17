""" ... constants ... """
import os

USER_DIR = os.path.expanduser('~')

SEC_COMP_ROOT = [
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
]

MP_SPDZ_ROOT = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "MP-SPDZ"),
]

DETECTION_MODES = {
    "opt_music": 0,
    "selest": 1,
}

FIELD = {
    0: "prime",
    1: "power_two",
    2: "binary",
}

FIELD_COMPILATION = {
    "prime": "-F",
    "power_two": "-R",
    "binary": "-B"
}

COMP_DOMAIN = {
    0: "arithmetic",
    1: "binary",
}

ADVERS_TYPES = {
    0: "semi-honest",
    1: "malicious",
}

MAJORITY = {
    0: "honest",
    1: "dishonest",
    2: "n/a",
}

# List of parameters that require circuit compilation to take effect
REQUIRE_RECOMPILE = ["comp_domain", "comp_field", "detection_mode", "cov_mat_avg", "single_output"]

OUTPUT_STATS_HELP = {
    "Time": "Overall execution time (s) as reported by MP-SPDZ",
    "Time#": "User defined timer (s) from MP-SPDZ circuit",
    "N_mults": "Number of field element multiplications as defined by each protocol",
    "Comm_size": "Size of communication (MB) (Data sent field from MP-SPDZ output",
    "Global_comm_size": "Size of total communication (MB) (Global data sent field from MP-SPDZ output",
    "Pseudospectrum": "List of power levels output by the circuit if \"single output\" is not used.",
    "DoA": "Single Direction of Arrival angle output by the circuit if \"single output\" is used.",
}

REG_EXPS = {
    "Pseudospectrum": "Music spectrum: (.[\[\]0-9., e+-]*)",
    "DoA": "Peak angle: (.[0-9.]*)",
}

BENCHMARK_PARAMS = {
    "selest": {
        "net_lat": int,
        "preprocessing": bool,
        "n_parties": int,
        "comp_field": str,
        "protocol": str,
        "net_top": str,
        "parallel": int,
        "single_output": bool,
        "detection_mode": str,
        "scaling_factor": int,
    },
    "benchmark_operations": {
        "protocol": str,
        "n_ops": int,
        "value_type": str,
        "op_type": str,
        "loop_type": str,
        "net_top": str,
    }
}

AVG_SEPARATOR = "\n\n" + 25*"#" + " AVG " + 25*"#" + "\n\n"
PARALL_SEPARATOR = "\n\n" + 25*"#" + " PARALL " + 25*"#" + "\n\n"

PARALL_TIMEOUT = {
    "online": 15,
    "offline": 1800,
}

el_range = range(90)

# Indicative cost in cents/gb transferred in a network (from AWS cross region costs))
NETWORK_COST = 0.01
