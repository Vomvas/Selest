""" High level MP-SPDZ setup API """
import argparse
import logging
from core import MPSPDZEngine
from core import Protocol


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
log = logging.getLogger('secure_detection')
log.setLevel(logging.DEBUG)


def process_setup(setup_online=False, nparties=3, lgp=128, build=None, threads=None):
    """ Setup mpspdz """

    mpe = MPSPDZEngine(n_parties=nparties, protocol="mascot")       # Protocol is not important at this point

    if setup_online:
        mpe.online_setup(nparties, lgp)

    if build:
        mpe.compile_protocol(build, threads)


def main():
    """ Parse args """
    parser = argparse.ArgumentParser(description="Provides a high level API for certain MP-SPDZ setup functionalities")
    parser.add_argument("--setup-online", action="store_true",
                        help="Generates preprocessed data required for offline only benchmarking.")
    parser.add_argument("--nparties", "-n", default=3,
                        help="Number of parties for online setup.")
    parser.add_argument("--lgp", default=128,
                        help="Prime field size for online setup.")
    parser.add_argument("--build", "-b", nargs="+", choices=Protocol.supported_protocols(),
                        help="Build MP-SPDZ protocols.")
    parser.add_argument("--build-multithreading", "-t", type=int,
                        help="Specifies make -j <t> for MP-SPDZ faster building.")
    args = parser.parse_args()

    process_setup(args.setup_online, args.nparties, args.lgp, args.build, args.build_multithreading)


if __name__ == "__main__":
    main()
