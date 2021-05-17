""" Creates a new benchmark """
import argparse
import json
import logging
import os

import numpy as np

from core import Benchmark
from core import Plotter
from core import LatexTable


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
log = logging.getLogger('secure_detection')
log.setLevel(logging.DEBUG)


def process_print_table(bench_dir, options=None):
    """ Fill our table with results from the benchmark and print it out to be pasted in the LaTeX document """

    if len(bench_dir) != 1:
        log.error("Please provide exactly 1 benchmark for runtimes table printing at a time")
        return -1

    b = Benchmark.load_benchmark_from_dir(bench_dir[0])

    t = LatexTable(b.def_latex_format)

    b.fill_latex_table(t)


def process_print_acm_table(bench_dir, options=None):
    """ Print table rows according to ACM format """

    if len(bench_dir) != 1:
        log.error("Please provide exactly 1 benchmark for runtimes table printing at a time")
        return -1

    b = Benchmark.load_benchmark_from_dir(bench_dir[0])

    b.get_acm_rows(options=options)


def process_plot_benchmarks(bench_dir, options=None):
    """ Process plotting """

    for b_dir in bench_dir:

        b = Benchmark.load_benchmark_from_dir(b_dir)

        if not os.path.isfile(b.plot_config):
            raise NotImplementedError("Please maually provide a plot configuration for this benchmark")

        with open(b.plot_config) as pcr:
            plot_conf = json.load(pcr)

        sep_plots = plot_conf.get("separate")
        if sep_plots:
            for sep_value in b.get_benchmark_config().get(sep_plots):
                options = {
                    "figwidth": 20,
                    "figheight": 12,
                    "sep_type": sep_plots,
                    "separate": sep_value,
                }
                suff = options.get("separate")
                outfile = b.get_figure_filename(suffix=suff)
                Plotter.plot(b, outfile=outfile, figdir=b.outputs_dir, options=options)
                log.info(f"Figured saved in file {Plotter.gen_fig_filename(b, b.outputs_dir, outfile)}")
        else:
            options = {
                "figwidth": 12,
                "figheight": 10,
            }
            # lat_range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
            # lat_range2 = [0, "", 10, "", 20, "", 30, "", 40, "", 50, "", 60, "", 70, 75]
            lat_range = b.benchmark_config.get("net_lat")
            lat_range2 = np.linspace(0, int(max(b.benchmark_config.get("net_lat"))), len(b.benchmark_config.get("net_lat")))

            # lat_range2 = [0,  10,  20, 30, 40, 50, 60, 70, 75]
            # lat_range = [0 10, " ,20, 25, 30, 35, 40]
            # xticks = [0,20,40,60,80,100,120,140,160]
            # lat_range = range(6)
            plot_options = {
                "title": "Single execution time vs round-trip latency",
                "fontsize": 32,
                "titlesize": 30, #36,
                "x_ticks": lat_range,
                "x_ticklabels": [str(int(2*x)) for x in lat_range2],
                "xlabel": "Round-trip latency (ms)",
                "ylabel": "Execution time (s)",
                "legend": {"fontsize": 26}
            }

            options.update(plot_options)
            Plotter.plot(b, outfile=b.get_figure_filename(), figdir=b.outputs_dir, options=options)

            log.info(f"Figured saved in file {Plotter.gen_fig_filename(b, b.outputs_dir, b.get_figure_filename())}")


def main():
    """ Parse args """
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("benchmark_dir", nargs="*",
                        help="Directory of a created benchmark")
    parser.add_argument("--runtimes-table", action="store_true",
                        help="Parse the benchmark for execution runtimes and print the table according to our"
                             "LaTeX format.")
    parser.add_argument("--acm-runtimes", action="store_true",
                        help="Parse the benchmark for execution runtimes and print the table according to ACM format")
    parser.add_argument("--table", type=int,
                        help="Specify table in paper to generate")
    args = parser.parse_args()

    if args.runtimes_table:
        process_print_table(args.benchmark_dir)
    elif args.acm_runtimes:
        process_print_acm_table(args.benchmark_dir, options={'table': args.table})
    else:
        process_plot_benchmarks(args.benchmark_dir)


if __name__ == "__main__":
    main()
