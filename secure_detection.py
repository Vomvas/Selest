""" Run secure detection by invoking MP-SPDZ """
import argparse
import logging
import sys
from core import SecureDetection as SecDec
from core import constants
from core import utils
from core import Protocol


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
log = logging.getLogger('secure_detection')
log.setLevel(logging.DEBUG)


def process_detection(raw_samples_dir, scaling_factor, n_parties, detection_mode, single_output, protocol,
                      net_lat, setup_online, offline, average_execution, skip_compilation, net_top, parallel):
    """ Run detection """

    sd = SecDec(n_parties, detection_mode=detection_mode, input_samples_dir=raw_samples_dir, protocol=protocol,
                preprocessing=not offline, single_output=single_output, scaling_factor=scaling_factor, net_lat=net_lat,
                setup_online=setup_online, net_top=net_top, parallel=parallel)

    if skip_compilation:
        sd.compile_prog = False

    if average_execution == 1:
        try:
            sd.run()
        except KeyboardInterrupt:
            log.info("User Interrupt.")
            sys.exit()
        log.info("Captured output: \n\n%s", utils.io.parse_subprocess_output(sd.output))
        if not single_output:
            sd.plot_pseudospectrum()
    else:
        log.info(f"Starting {average_execution} executions...")
        exec_times = [0] * average_execution

        for i in range(average_execution):
            if i % int(0.1 * average_execution) == 0 and i > 1:
                log.setLevel(logging.INFO)
                log.info(f"Completed {i} out of {average_execution} executions...")
                log.setLevel(logging.ERROR)
            try:
                sd.run()
            except KeyboardInterrupt:
                log.setLevel(logging.INFO)
                log.info(f"Results after {i} iterations:\n"
                         f"Min exec time: {min(exec_times[:i-1])}\n"
                         f"Max exec time: {max(exec_times[:i-1])}\n"
                         f"Avg exec time: {sum(exec_times[:i-1]) / float(len(i-1))}")
                log.info("User Interrupt.")
                sys.exit()
            sd.compile_prog = False         # Avoid recompiling; configuration did not change
            log.setLevel(logging.ERROR)     # Silence logging for readability
            stats = sd.parse_output()
            exec_times[i] = float(stats.get("Time", 0.0))
        log.setLevel(logging.INFO)

        log.info("Results:\n"
                 f"Min exec time: {min(exec_times)}\n"
                 f"Max exec time: {max(exec_times)}\n"
                 f"Avg exec time: {sum(exec_times) / float(len(exec_times))}")


def main():
    """Parse args"""
    parser = argparse.ArgumentParser(description="Detect the angle of arrival by operating on secret shared complex"
                                                 "samples.")
    parser.add_argument("--raw-samples-dir", type=str,
                        help="Path to directory with raw complex samples. All sample files will be processed.")
    parser.add_argument("--scaling-factor", type=float, default=25.0,
                        help="Scaling factor for raw inputs. Every complex input will be multiplied"
                             "by this factor before stored in the Player-Data inputs")
    parser.add_argument("--nparties", type=int, default=4,
                        help="Number of parties to perform the secure computation.")
    parser.add_argument("--detection-mode", type=str, default="selest",
                        choices=constants.DETECTION_MODES.keys(), help="Detection mode.")
    parser.add_argument("--single-output", action="store_true",
                        help="Output a single angle value instead of pseudospectrum power.")
    parser.add_argument("--protocol", type=str, choices=Protocol.supported_protocols(), default="mascot",
                        help="MP-SPDZ protocol to be invoked (mascot, shamir, mal-shamir, etc)")
    parser.add_argument("--net-lat", type=int, default=0,
                        help="Add artificial latency (in ms) to the loopback interface to simulate network connections"
                             " (sudo required). "
                             "Note that latency will result in twice the delay for round trip times.")
    parser.add_argument("--net-top", type=str, default="star", choices=["star", "direct"],
                        help="Network topology, star shaped or direct communication.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel executions to run concurrently.")
    parser.add_argument("--setup-online", action="store_true",
                        help="Perform online setup (chooses parameters based on number of players)")
    parser.add_argument("--offline", action='store_true',
                        help="Specify the offline setting to be used for the MPSPDZ framework")
    parser.add_argument("--average-execution", type=int, default=1,
                        help="Run multiple executions and output time statistics.")
    parser.add_argument("--skip-compilation", action='store_true',
                        help="Skip circuit compilation to save time. It's up to the user to ensure it's properly "
                             "compiled")
    args = parser.parse_args()

    process_detection(args.raw_samples_dir, args.scaling_factor, args.nparties, args.detection_mode,
                      args.single_output, args.protocol, args.net_lat, args.setup_online, args.offline,
                      args.average_execution, args.skip_compilation, args.net_top, args.parallel)


if __name__ == "__main__":
    main()
