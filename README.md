# musical-guide

## SELEST: Secure Elevation Estimation of Drones using Multiparty Computation

### Requirements and setup

- Python3.8: To run the SELEST high level API Python3.8 is required.


- Clone and install MP-SPDZ, mpir and requirements according to the installation instructions:

`git clone https://github.com/data61/MP-SPDZ.git --recurse-submodules`

- Copy `CONFIG.mine`, libraries and circuit files into the appropriate directories.

```
cp mp_spdz_files/CONFIG.mine ./MP-SPDZ
cp mp_spdz_files/selest.conf ./MP-SPDZ
cp mp_spdz_files/types.py ./MP-SPDZ/Compiler
cp mp_spdz_files/complex.py ./MP-SPDZ/Compiler
cp mp_spdz_files/qr_decomposition.py ./MP-SPDZ/Compiler
cp mp_spdz_files/selest.mpc ./MP-SPDZ/Programs/Source
```

- Compile MP-SPDZ protocols.

Protocols are compiled according to MP-SPDZ instructions (https://github.com/data61/MP-SPDZ.git)

For convenience we provide a high level API to do so.

### SELEST high level API

We provide a core module that handles the MP-SPDZ preparation, and execution.

- Compile MP-SPDZ protocols

This builds the virtual machine for various MP-SPDZ protocols. For example, for mascot and shamir:

```
# -t specifies number threads for `make`
python3 setup_mpspdz.py -b mascot shamir -t 2
```

- Online setup

This generates the required preprocessed data for the online phase. Note that `-DINSECURE` must be specified
in the mp-spdz `CONFIG.mine` otherwise this will fail. More details can be found in the MP-SPDZ instructions
(https://github.com/data61/MP-SPDZ.git).

```
# -n specifies number of players
python3 setup_mpspdz.py --setup-online -n 5
```

- Execute detection

Useful options (`--help`)
```
--raw-samples-dir: path to directory with raw samples
--detection-mode: opt_music, selest (default)
--single-output: denotes conditioned output instead of coarse (default: false)
--scaling-factor: scales inputs according to profiling (default: 25)
--nparties: parties for the MPC evaluation
--net-lat: network latency in ms (RTT=2*net-lat)
```

For example:

```
python3 ../secure_detection.py --raw-samples-dir ../data/real_doa90 --scaling-factor 30
```

Note: We found that `--scaling-factor 30` works best for the given input data.
