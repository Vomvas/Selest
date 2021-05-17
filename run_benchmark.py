""" Creates a new benchmark """
import argparse
from datetime import datetime
import logging
import os
from configparser import ConfigParser
from core import constants
from core import Benchmark, SecureDetection as SecDet, MPSPDZEngine
from core import Protocol
from core import utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
log = logging.getLogger('secure_detection')
log.setLevel(logging.DEBUG)


def process_operations_benchmark(b: Benchmark):
    """ Handle operations benchmarks """
    for bench_case in (b.get_bench_cases()):
        log.info(f"Executing benchmark for {b.prog_name}: {bench_case}")

        mpe = MPSPDZEngine(
            prog_name=b.prog_name,
            protocol=bench_case.get("protocol"),
            n_parties=bench_case.get("n_parties"),
            preprocessing=bench_case.get("preprocessing"),
            comp_field=bench_case.get("comp_field")
        )

        config_file = os.path.join(os.path.dirname(mpe.root_dir), "benchmark_operations.conf")
        config = ConfigParser()

        config['benchmark'] = bench_case

        with open(config_file, 'w') as cf:
            config.write(cf)

        mpe.compile_circuit()
        mpe.execute_circuit()

        b.store_bench_outputs(bench_case, mpe.output)


def process_sec_det_bench(b: Benchmark, input_dir):
    """ Handle secure detection benchmarks """

    assert input_dir

    if b.resume:
        completed_cases = b.get_output_metadata().values()
    else:
        completed_cases = []

    for bench_case in (b.get_bench_cases()):

        if bench_case in completed_cases:
            continue

        curr_exec_avg = b.execution_averaging
        prot_name = bench_case.get("protocol")
        if Protocol.get_protocol_by_name(prot_name).slow_offline and \
                not utils.io.str_to_bool(bench_case.get("preprocessing")):
            log.info(f"Overriding averaging executions to (1) because of {prot_name} combined phase.")
            b.execution_averaging = 1
        log.info(f"Executing benchmark for {b.prog_name}:"
                 f"{bench_case} (averaged over {b.execution_averaging} executions)")
        sd = SecDet(input_samples_dir=input_dir, **bench_case)

        aggr_outputs = ""
        for i in range(b.execution_averaging):
            try:
                sd.run()
            except AssertionError as e:
                aggr_outputs += getattr(e, 'message', repr(e))
                break
            sd.compile_prog = False  # Don't recompile for averaged executions
            log.setLevel(logging.ERROR)  # Don't log any messages for consecutive identical executions
            aggr_outputs += utils.io.parse_subprocess_output(sd.output)
            aggr_outputs += constants.AVG_SEPARATOR

        log.setLevel(logging.DEBUG)
        b.store_bench_outputs(bench_case, aggr_outputs)
        b.execution_averaging = curr_exec_avg


def process_run_benchmarks(bench_dir, input_dir, resume=False, overwrite=False, exec_avg=100):
    """ Run the benchmark in the given directory """

    for b_dir in bench_dir:
        b = Benchmark.load_benchmark_from_dir(b_dir)
        b.execution_averaging = exec_avg
        b.store_bench_info({"exec_avg": str(b.execution_averaging)})

        if b.get_output_metadata():
            if not (overwrite or resume):
                log.error(f"Outputs already exist in {b.outputs_dir}. Exiting.")
                return
            elif resume:
                b.resume = True
            elif overwrite:
                b.delete_outputs()
                b = Benchmark.load_benchmark_from_dir(b_dir)

        b.store_bench_info({"last_ran": str(datetime.now())})

        if b.prog_name == "selest":
            process_sec_det_bench(b, input_dir)
        elif b.prog_name == "benchmark_operations":
            process_operations_benchmark(b)


def main():
    """ Parse args """
    parser = argparse.ArgumentParser(description="Run a benchmark.")
    parser.add_argument("benchmark_dir", nargs="*",
                        help="Directory of a created benchmark")
    parser.add_argument("--input-dir", required=True,
                        help="Directory to load input samples.")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from previous results")
    parser.add_argument("--overwrite", "-o", action="store_true",
                        help="Overwrite previous results")
    parser.add_argument("--exec-avg", "-e", type=int, default=100,
                        help="Results are averaged over that many executions")
    args = parser.parse_args()

    process_run_benchmarks(args.benchmark_dir, args.input_dir, args.resume, args.overwrite, args.exec_avg)


if __name__ == "__main__":
    main()
