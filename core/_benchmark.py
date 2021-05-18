""" Benchmarks class """
import json
import logging
import os
import time
from collections import defaultdict
from itertools import product
from . import constants
from ._chart import ChartData, CurveData, LineChart, BarData, BarGroup, BarChart
from ._protocols import *
from .utils import io, misc, networking
from tabulate import tabulate

log = logging.getLogger("secure_detection")


class Benchmark(object):
    """ Instance of benchmark class """

    @staticmethod
    def get_new_uid():
        """ Generate new UID from the hash of the current timestamp """
        return misc.hash_ts()[:6]

    @staticmethod
    def load_config_from_file(bench_config):
        """ Loads the config from a given file """
        assert os.path.isfile(bench_config)
        with open(bench_config) as bcr:
            return json.load(bcr)

    @classmethod
    def new_benchmark(cls, parent_dir, prog_name, alias='', prefix="bench_", notes=None):
        """ Create a new benchmark to be stored in the given parent dir """
        parent_dir = parent_dir or "./benchmarks"
        ws_name = networking.get_workstation_name()
        if ws_name:
            prefix += (ws_name + "_")
        if not io.dir_exists(parent_dir):
            raise NotADirectoryError(parent_dir)
        if not alias:
            bench_dir_name = os.path.join(os.path.abspath(parent_dir), prefix + cls.get_new_uid())
        else:
            bench_dir_name = os.path.join(os.path.abspath(parent_dir), prefix + alias + '_' + cls.get_new_uid())
        try:
            os.mkdir(bench_dir_name)
        except FileExistsError:
            return cls.new_benchmark(parent_dir, prefix)
        return cls(bench_dir_name, prog_name, alias)

    @classmethod
    def load_benchmark_from_dir(cls, bench_dir):
        """ Load existing benchmark """
        if not io.dir_exists(bench_dir):
            raise NotADirectoryError(bench_dir)
        with open(os.path.join(bench_dir, "bench_info.json")) as bir:
            bench_info = json.load(bir)
        assert (bench_info and bench_info.get("prog_name", None))
        return cls(bench_dir, bench_info.get("prog_name", None), bench_info.get("alias", None),
                   bench_info.get("notes", None), int(bench_info.get("exec_avg", 100)))

    def __init__(self, root_dir, prog_name, alias=None, notes=None, exec_avg=100):

        if io.dir_exists(root_dir):
            self.root_dir = os.path.abspath(root_dir)
        else:
            self.root_dir = os.path.abspath(root_dir)
            os.makedirs(self.root_dir)

        self.resume = False

        self.name = os.path.basename(self.root_dir)
        self.uid = self.parse_name_for_uid() or self.get_new_uid()
        self.prog_name = prog_name
        self.alias = alias
        self.notes = notes

        self.bench_info = os.path.join(self.root_dir, "bench_info.json")
        self.bench_file = os.path.join(self.root_dir, "bench_config.json")
        self.outputs_dir = os.path.join(self.root_dir, "bench_outputs")
        self.output_metadata = os.path.join(self.outputs_dir, "metadata.json")
        self.plot_config = os.path.join(self.root_dir, "plot_config.json")

        self.def_latex_format = os.path.join(os.path.dirname(self.root_dir), "default_latex_format.conf")
        def_conf_file = os.path.join(os.path.dirname(self.root_dir), "default_config_file.conf")
        self.default_config = io.load_default_config(def_conf_file)
        self.all_bench_params = constants.BENCHMARK_PARAMS.get(self.prog_name, None)
        self.execution_averaging = exec_avg  # Benchmarks are averaged over that many executions
        assert self.all_bench_params

        if not os.path.isfile(self.bench_info):
            self.store_bench_info()
        if not io.dir_exists(self.outputs_dir):
            os.mkdir(self.outputs_dir)

    @property
    def benchmark_config(self):
        """ Return dictionary of benchmark config """
        return self.get_benchmark_config()

    def parse_name_for_uid(self):
        """ Extract UID from name """
        if len(self.name.split()) == 1:
            return None
        else:
            return self.name.split()[-1]

    def delete_outputs(self):
        """ Delete files in output directory to enable overwrite """
        log.warning("\n\nDeleting benchmark outputs...\n\n")
        time.sleep(5)
        for outfile in self.get_output_metadata().keys():
            local_outfile = os.path.join(self.outputs_dir, outfile)
            os.remove(local_outfile)
        os.remove(self.output_metadata)
        return

    def store_bench_info(self, info=None):
        """ Stores benchmark information """

        if os.path.isfile(self.bench_info):
            with open(self.bench_info) as bir:
                bench_info = json.load(bir)
        else:
            bench_info = {'prog_name': self.prog_name,
                          'alias': self.alias,
                          'worksation': networking.get_workstation_name(),
                          'notes': self.notes or ""}

        if info:
            for key, val in info.items():
                bench_info[key] = val

        with open(self.bench_info, 'w') as biw:
            json.dump(bench_info, biw)

    def read_config_from_user(self):
        """ Request user input for benchmark config """
        prompt = f"Which parameters would you like to benchmark?\n[({' '.join(self.all_bench_params.keys())})] "
        user_input = input(prompt)
        bench_params = user_input.split(' ')

        assert all([x in self.all_bench_params for x in bench_params])

        bench_config = defaultdict(list)
        for par in bench_params:
            par_values = input(f"Provide values for parameter {par}: ")
            if par in ["preprocessing", "single_output"]:
                bench_config[par] = [io.str_to_bool(x) for x in par_values.split(' ')]
            else:
                bench_config[par] = [self.all_bench_params[par](x) for x in par_values.strip().split(' ')]
        return bench_config

    def setup_benchmark_config(self, bench_config=None):
        """ Initialize benchmark config from bench_config dict """

        if not self.default_config:
            log.warning(f"Default benchmark config not found: {self.default_config}")

        if not bench_config:
            log.warning(f"No config given, default benchmarks: {self.default_config}")

        for key, val in self.default_config.items():
            if key not in bench_config.keys():
                bench_config[key] = [val]

        # assert all([x in self.all_bench_params.keys() for x in bench_config.keys()]), \
        #     f"Missing: {[x for x in bench_config.keys() if x not in self.all_bench_params.keys()]} " \
        #     f"from {list(self.all_bench_params.keys())}."

        log.info(f"Missing values assumed default. Saving benchmark config: {json.dumps(bench_config)}.")
        with open(self.bench_file, 'w') as bfw:
            json.dump(bench_config, bfw)

    def get_benchmark_config(self):
        """ Return the dictionary holding the benchmark cases """
        with open(self.bench_file) as bfr:
            return json.load(bfr)

    def get_plot_config(self):
        """ Return the dictionary holding the plot configuration """
        with open(self.plot_config) as pcr:
            return json.load(pcr)

    def sort_bench_params(self):
        """ Returns a sorted list of the benchmark parameters based on compile requirement """
        return sorted(self.benchmark_config.keys(), reverse=True, key=lambda x: x in constants.REQUIRE_RECOMPILE)

    def get_bench_cases(self):
        """ Return a list of dicts, each dict representing a benchmark case """
        sorted_cases = [self.benchmark_config[x] for x in self.sort_bench_params()]

        all_combs = product(*sorted_cases)

        bench_cases = []
        for comb in all_combs:
            tmp_dict = {}
            compilation_required = False
            for param, value in zip(self.sort_bench_params(), comb):
                tmp_dict[param] = value

                if len(bench_cases) == 0:
                    compilation_required = True
                    continue

                prev_value = bench_cases[-1].get(param, None)
                if param == "protocol" and value != prev_value:
                    compilation_required = compilation_required or (Protocol.get_protocol_by_name(value).comp_domain !=
                                                                    Protocol.get_protocol_by_name(prev_value))
                else:
                    compilation_required = compilation_required or \
                                           (param in constants.REQUIRE_RECOMPILE and value != prev_value)
            tmp_dict["compile_prog"] = compilation_required
            bench_cases.append(tmp_dict)

        return bench_cases

    def store_bench_outputs(self, bench_case, output):
        """ Stores the benchmark outputs and metadata """

        if os.path.isfile(self.output_metadata):
            with open(self.output_metadata) as omr:
                bench_md = json.load(omr)
        else:
            bench_md = {}

        output_filename = os.path.join(self.outputs_dir, str(len(bench_md.keys())))
        bench_md[os.path.basename(output_filename)] = bench_case

        output = io.parse_subprocess_output(output)

        with open(output_filename, 'w') as ofw, open(self.output_metadata, 'w') as bfw:
            json.dump(bench_md, bfw)
            ofw.write(output)

    def get_output_metadata(self):
        """ Returns a dict containing the output metadata """
        if not os.path.isfile(self.output_metadata):
            return None
        with open(self.output_metadata) as bmr:
            md_dict = json.load(bmr)
        return md_dict

    def get_bench_outfile(self, param_values):
        """
        Given a set of parameter values, return the single execution output file that corresponds to these parameters
        alone when all the rest remains the same

        Example:

        get_bench_outfile([("net_lat", 7), ("net_top", "star")]) will look for the benchmark run with those exact
        parameter values and no other possible benchmark exists for the same values (the rest of the parameters are
        static).
        If found, returns the absolute filename, else returns None

        :param param_values: List of (param, value) tuples
        :returns absolute path to output file
        """
        bench_config = self.get_benchmark_config()
        # Make sure there are no missing parameters such that no unique outfile can be identified.
        for key, val in bench_config.items():
            if len(val) > 1 and key not in [x[0] for x in param_values]:
                return None
        # Return the unique file (key) that contains the required benchmarks (param values)
        for outfile, case in self.get_output_metadata().items():
            check_list = [case.get(x[0]) == x[1] for x in param_values]
            if all(check_list):
                return os.path.join(self.outputs_dir, os.path.basename(outfile))
        return None

    def get_output_from_outfile(self, outfile):
        """ Return output from given outfile """
        local_outfile = os.path.join(self.outputs_dir, os.path.basename(outfile))
        with open(local_outfile) as f:
            exec_output = f.read()
        return exec_output

    def check_outputs(self):
        """ Checks outputs for errors """
        for outfile, case in self.get_output_metadata().items():
            prot = case.get("protocol")
            total_par = case["parallel"]
            exec_output = self.get_output_from_outfile(outfile)
            failed = io.check_output_errors(exec_output, prot)
            if failed:
                print(f"Case: {outfile}, Total executions: {total_par}, Failed executions: {failed}, Exec info: {case}")

    def get_figure_filename(self, suffix=None):
        """ Generate a filename for the plots """
        suffix = ("_" + suffix.strip("_")) if suffix else ""
        return "plot_" + self.alias + suffix + ".pdf"

    def _plot_lines(self, subplot, options=None):
        plot_conf_dict = self.get_plot_config()
        bench_config = self.get_benchmark_config()
        style = plot_conf_dict.get("style")
        curves = plot_conf_dict.get(style)
        y_axis = plot_conf_dict.get("y")

        assert curves, f"Missing parameter values for {curves} from {repr(bench_config.keys())}"

        if plot_conf_dict.get("x") in bench_config.keys():
            x_axis = bench_config.get(plot_conf_dict.get("x"))
            assert x_axis, f"Missing parameter values for {plot_conf_dict.get('x')} from {repr(bench_config)}"
            assert y_axis in constants.OUTPUT_STATS_HELP.keys(), f"Result {y_axis} not in list of results" \
                                                                 f"{constants.OUTPUT_STATS_HELP.keys()}."
        else:
            raise NotImplementedError("Param not recognized or: "
                                      "Not implemented: parameter on y-axis and benchmark on x-axis...")

        # Get the y-point for every x-point for every line to be drawn and create the CurveData
        all_curves = []
        all_curve_params = [[x for x in bench_config.get(curve_type)] for curve_type in curves]
        permut = [x for x in product(*all_curve_params)]
        for p in permut:
            ext_list = [(curves[i], p[i]) for i in range(len(curves))]
            curve_y_axis = []
            for x_point in x_axis:
                bench_params = [(plot_conf_dict.get("x"), x_point)] + ext_list
                if options.get("separate"):
                    bench_params.append((options.get("sep_type"), options.get("separate")))
                outfile = self.get_bench_outfile(bench_params)
                prot = self.get_output_metadata().get(os.path.basename(outfile)).get("protocol")
                exec_output = self.get_output_from_outfile(outfile)
                y_point = io.extract_info_from_output(exec_output, [y_axis], prot).get(y_axis)
                curve_y_axis.append(y_point)

            # curve_label = ', '.join([f"{x[0]}: {x[1]}" for x in ext_list])
            curve_label = ', '.join([f"{x[1]}" for x in ext_list])
            curve = CurveData(x_axis, curve_y_axis, label=curve_label)

            if not all([not x for x in curve.y]):
                all_curves.append(curve)
            else:
                log.warning(f"Removing line {curve.label} because output could not be parsed")
        if options:
            y_lab =options.get("ylabel") or constants.OUTPUT_STATS_HELP.get(y_axis)
            x_lab =options.get("xlabel") or plot_conf_dict.get("x")
        else:
            y_lab = constants.OUTPUT_STATS_HELP.get(y_axis)
            x_lab = plot_conf_dict.get("x")
        chart = ChartData(curves=all_curves, xlabel=x_lab, ylabel=y_lab, options=options)

        LineChart.plot(subplot, chart, options=options)

    def _plot_bars(self, subplot, options=None):
        plot_conf_dict = self.get_plot_config()
        bench_config = self.get_benchmark_config()
        style = plot_conf_dict.get("style")
        bars = plot_conf_dict.get(style)
        y_axis = plot_conf_dict.get("y")

        if plot_conf_dict.get("x") in bench_config.keys():
            # x_axis = [bench_config.get(plot_conf_dict.get("x")) for _ in bench_config.get(bars)]
            # log.debug("X AXIS: ", x_axis)
            # assert x_axis, f"Missing parameter values for {plot_conf_dict.get('x')} from {repr(bench_config)}"
            assert y_axis in constants.OUTPUT_STATS_HELP.keys(), f"Result {y_axis} not in list of results" \
                                                                 f"{constants.OUTPUT_STATS_HELP.keys()}."
        else:
            raise NotImplementedError("Not implemented: parameter on y-axis and benchmark on x-axis...")

        # Get the y-point for every x-point for every line to be drawn and create the CurveData
        assert len(bars.keys()) == 1, "Max bar grouping level of (1) implemented"

        b_groups = []
        lvl1_group = list(bars.keys())[0]
        lvl2_group = bars.get(lvl1_group)
        bar_groups = [x for x in bench_config.get(lvl1_group)]
        for bgroup in bar_groups:
            bg = BarGroup([], label=bgroup)
            for b in bench_config.get(lvl2_group):
                bench_params = [(lvl1_group, bgroup), (plot_conf_dict.get("x"), b)]
                if options.get("separate"):
                    bench_params.append((options.get("sep_type"), options.get("separate")))
                outfile = self.get_bench_outfile(bench_params)
                prot = self.get_output_metadata().get(os.path.basename(outfile)).get("protocol")
                exec_output = self.get_output_from_outfile(outfile)
                y_point = io.extract_info_from_output(exec_output, [y_axis], prot).get(y_axis)
                bg.bars.append(BarData(None, y_point, label=b))

            b_groups.append(bg)
            # bar_label = ', '.join([f"{x[0]}: {x[1]}" for x in ext_list])
            # bar = CurveData(x_axis, bar_y_axis, label=bar_label)

        y_lab = constants.OUTPUT_STATS_HELP.get(y_axis)
        x_lab = plot_conf_dict.get("x")
        chart = ChartData(bargroups=b_groups, xlabel=x_lab, ylabel=y_lab, options=options)

        BarChart.plot(subplot, chart, options=options)

    def plot_benchmark(self, subplot, options=None):
        """
        Plot the results and saves the figure in the output directory. In order to do that, it creates several
        CurveData objects that hold the information about the lines to be plotted. Then, creates a ChartData object
        that holds the information about the overall figure and uses the helper class LineChart to draw the lines.
        """
        plot_conf_dict = self.get_plot_config()
        assert plot_conf_dict, f"Plot config could not be loaded: {self.plot_config}"
        if plot_conf_dict.get("style") == "lines":
            self._plot_lines(subplot, options)
        elif plot_conf_dict.get("style") == "bars":
            self._plot_bars(subplot, options)

    def get_acm_rows(self, latexstyle=False, options=None):
        """ Fills runtimes and costs for ACM paper format """

        if latexstyle:
            # Current configuration; subject to change
            n_parties = 3
            net_lat = 5
            parall = 60
            col_sep = " & "

            all_rows = []

            bench_config = self.get_benchmark_config()

            for prot in bench_config.get("protocol"):
                for prepr in bench_config.get("preprocessing"):
                    # prot_lines = defaultdict(dict)
                    prot_lines = {}
                    try:
                        # prot_lines[str(prepr)]["print_name"] = Protocol.get_protocol_by_name(prot).print_name
                        prot_lines["print_name"] = Protocol.get_protocol_by_name(prot).print_name
                    except AttributeError:
                        prot_lines["print_name"] = Protocol.get_protocol_by_name(prot).name
                    if not io.str_to_bool(prepr):
                        prot_lines["print_name"] += " (comb)"

                    for sing_out in bench_config.get("single_output"):
                        sing_out_str = sing_out
                        sing_out = io.str_to_bool(sing_out)
                        for par in [1, parall]:
                            case_conf = {
                                "protocol": prot,
                                "preprocessing": prepr,
                                "net_lat": net_lat,
                                "n_parties": n_parties,
                                "single_output": sing_out_str,
                                "parallel": par
                            }
                            outfile = self.get_bench_outfile(case_conf.items())
                            if not outfile:
                                log.warning(f"Case conflict or not found: {case_conf}")
                                continue
                            # assert outfile, f"Case conflict or not found: {case_conf} in {bench_config}"
                            case = self.get_output_metadata().get(os.path.basename(outfile))
                            # unique_case = self.isolate_case(case_conf)
                            exec_output = self.get_output_from_outfile(outfile)
                            info = io.extract_info_from_output(exec_output, ["Time", "Global_comm_size"], prot)

                            if sing_out and par == 1:
                                prot_lines["priv_times"] = self.calc_exec_per_sec(case,
                                                                                  info.get("Time"))
                                prot_lines["priv_traffic"] = self.calc_traffic_cost(case,
                                                                                    info.get("Global_comm_size"))
                            elif sing_out and par == parall:
                                prot_lines["priv_par_times"] = self.calc_exec_per_sec(case,
                                                                                      info.get("Time"))
                                prot_lines["priv_par_traffic"] = self.calc_traffic_cost(case,
                                                                                        info.get(
                                                                                            "Global_comm_size"))
                            elif not sing_out and par == 1:
                                prot_lines["coarse_times"] = self.calc_exec_per_sec(case,
                                                                                    info.get("Time"))
                                prot_lines["coarse_traffic"] = self.calc_traffic_cost(case,
                                                                                      info.get(
                                                                                          "Global_comm_size"))
                            elif not sing_out and par == 60:
                                prot_lines["coarse_par_times"] = self.calc_exec_per_sec(case,
                                                                                        info.get("Time"))
                                prot_lines["coarse_par_traffic"] = self.calc_traffic_cost(case,
                                                                                          info.get(
                                                                                              "Global_comm_size"))
                    col_data = ["coarse_times", "coarse_traffic", "coarse_par_times", "coarse_par_traffic",
                                "priv_times", "priv_traffic", "priv_par_times", "priv_par_traffic"]
                    line_list = [prot_lines.get("print_name")] + [str(round(float(prot_lines.get(x, -1)), 3)) for x in col_data]
                    app_line = col_sep.join(line_list) + " \\\\"
                    all_rows.append(app_line)

            all_rows_str = '\n'.join(all_rows)

            rows_output = os.path.join(self.outputs_dir, "acm_rows.txt")
            with open(rows_output, "w") as ow:
                ow.write(all_rows_str)
            log.info(f"Runtimes table written in: {rows_output}")
            return all_rows_str
        else:
            if options:
                papertable = options.get("table", 0)
            if papertable == 1:
                info = {
                    'Protocol': ['MASCOT', 'Lowgear', 'Cowgear', 'Semi', 'Hemi', 'Mal-Shamir', 'Sy-Shamir', 'Ps-Rep',
                                 'Shamir', 'Rep3'],
                    'Standard MUSIC': [],
                    'Opt-MUSIC': [],
                    'Selest': [],
                    'Speed-up': []
                    }
                column_names = {
                    'text_music': 'Standard MUSIC',
                    'opt_music': 'Opt-MUSIC',
                    'selest': 'Selest',
                }
                for prot in ['mascot', 'lowgear', 'cowgear', 'semi', 'hemi', 'mal-shamir', 'sy-shamir', 'ps-rep',
                                 'shamir', 'rep3']:
                    for det_mode in ["text_music", "opt_music", "selest"]:
                        det_mode_alias = "selest" if det_mode == "selest" else det_mode
                        params = [("protocol", prot), ("detection_mode", det_mode_alias)]
                        outfile = self.get_bench_outfile(params)
                        exec_output = self.get_output_from_outfile(outfile)
                        info[column_names[det_mode]].append(io.extract_info_from_output(exec_output, ["Time"], prot)["Time"])
                    info["Speed-up"].append(info["Standard MUSIC"][-1] / info["Selest"][-1])
                table = tabulate(info, headers='keys', showindex=False, tablefmt='fancy_grid')
                with open("table_1.txt", "w") as f:
                    f.write(table)
                log.info(f"Table saved in table_1.txt")

            elif papertable == 4:
                info = {
                    'Protocol': ['MASCOT', 'Lowgear', 'Cowgear', 'Semi', 'Hemi', 'Mal-Shamir', 'Sy-Shamir',
                                 'Ps-Rep', 'Shamir', 'Rep3'],
                    '1': [],
                    '2': [],
                    '3': [],
                    '4': [],
                    '5': [],
                    '6': [],
                    '7': [],
                    '8': [],
                    '9': [],
                    '10': [],
                    '11': [],
                    '12': [],
                    }
                legend = {
                    '1': "Coarse Single DoA/s",
                    '2': "Coarse Single Cost (c/h)",
                    '3': "Coarse Parallel DoA/s",
                    '4': "Coarse Parallel Cost (c/h)",
                    '5': "Conditioned Single DoA/s",
                    '6': "Conditioned Single Cost (c/h)",
                    '7': "Conditioned Parallel DoA/s",
                    '8': "Conditioned Parallel Cost (c/h)",
                    '9': "Coarse Offline Triples (1e6/h)",
                    '10': "Conditioned Offline Triples (1e6/h)",
                    '11': "Coarse Offline Cost ($/h)",
                    '12': "Conditioned Offline Cost ($/h)"
                }
                for prot in ['mascot', 'lowgear', 'cowgear', 'semi', 'hemi', 'mal-shamir', 'sy-shamir', 'ps-rep',
                             'shamir', 'rep3']:
                    for out_type in ['Coarse', 'Conditioned']:
                        for ex_par in ['Single', 'Parallel']:
                            params = [
                                ("protocol", prot),
                                ("single_output", True if out_type == 'Conditioned' else False)]
                            if ex_par == "Single":
                                params.append(("parallel", 1))
                            else:
                                params.append(("parallel", self.benchmark_config.get("parallel", 1)[-1]))
                            outfile = self.get_bench_outfile(params)
                            parse_output_info = io.extract_info_from_output(
                                self.get_output_from_outfile(outfile),
                                ["Time", "Global_comm_size"],
                                prot)
                            outfile_base = os.path.basename(outfile)
                            throughput = self.calc_exec_per_sec(self.get_output_metadata()[outfile_base],
                                                                parse_output_info.get("Time"))
                            comm_cost = self.calc_traffic_cost(self.get_output_metadata()[outfile_base],
                                                               parse_output_info.get("Global_comm_size"))
                            for k, v in legend.items():
                                if out_type in v and ex_par in v and "DoA/s" in v:
                                    info[k].append(throughput)
                                elif out_type in v and ex_par in v and "Cost (c/h)" in v:
                                    info[k].append(round(float(comm_cost*3600), 3))
                    coarse_triples = 114
                    conditioned_triples = 350
                    info['9'].append(coarse_triples * info['3'][-1] * 3600 / 1000000)
                    info['11'].append(round(info['9'][-1] * 3 / 190, 3))
                    info['10'].append(conditioned_triples * info['7'][-1] * 3600 / 1000000)
                    info['12'].append(round(info['11'][-1] * 3 / 190, 3))

                for i in range(1, 13):
                    info[legend[str(i)]] = info.pop(str(i))
                table = tabulate(info, headers='keys', showindex=False, tablefmt='fancy_grid')
                with open("table_4.txt", "w") as f:
                    f.write(table)
                log.info(f"Table saved in table_4.txt")

    def fill_latex_table(self, t):
        """ Fill runtimes in LaTeX table """

        overall_digits = 4
        float_precision = 3

        for (outfile, params) in self.get_output_metadata().items():
            approach = params.get("detection_mode", "")
            protocol = params.get("protocol", "").lower()
            if params.get("preprocessing", ""):
                protocol_full = protocol + " (online)"
            else:
                protocol_full = protocol + " (combined)"
            n_parties = params.get("n_parties", -1)
            exec_output = self.get_output_from_outfile(outfile)
            runtime = io.extract_info_from_output(exec_output, ["Time"], protocol).get("Time")

            # Try to round result
            try:
                runtime = float(runtime)
                int_digits = len(str(int(runtime)))
                if int_digits > 4:
                    pass
                elif int_digits > 1:
                    runtime = round(runtime, overall_digits - int_digits)
                else:
                    runtime = round(runtime, float_precision)
            except TypeError as e:
                pass

            t.fill_value(approach, protocol_full, n_parties, runtime)

        latex_output = os.path.join(self.outputs_dir, "latex_runtimes_table.txt")
        with open(latex_output, "w") as ow:
            ow.write(t.table)
        log.info(f"Runtimes table written in: {latex_output}")
        return t

    @staticmethod
    def calc_exec_per_sec(case, exec_time):
        """ Return executions / sec rate """
        conc_exec = case.get("parallel", 1)
        return round(conc_exec / float(exec_time), 1)

    @staticmethod
    def calc_traffic_cost(case, global_data):
        """ Return traffic cost rate for global data in MB """
        global_mb_exchanged = case.get("parallel", 1) * global_data
        traffic_cost = constants.NETWORK_COST * global_mb_exchanged / 1e3
        return traffic_cost

    def print_benchmark_info(self):
        """ Print bench info """
        with open(self.bench_info) as bir:
            bench_info = json.load(bir)
        log.info(f"Benchmark ({os.path.basename(self.root_dir)}) info: {bench_info}")

    def get_case_str(self, case, exclude=None):
        """ Return a string of the benchmark case non-default params and values """
        case_str = ""
        for param, value in case.items():
            if self.default_config.get(param) == value and param != "protocol":
                continue
            if param == "compile_prog" or (exclude and param in exclude):
                continue
            case_str += f"{param}: {value}, "
        return case_str.rstrip(", ")

    def filter_cases_from_str(self, case_str):
        """ Filter benchmark cases based on given string of param=value comma separated pairs """
        d = {x.split('=')[0]: x.split('=')[1].split(",") for x in case_str.split()}
        filtered_cases = []
        for case in self.get_output_metadata().values():
            for par, val in d.items():
                if str(case.get(par)) not in [str(x) for x in val]:
                    break
            else:
                filtered_cases.append(case)
        return filtered_cases

    def print_benchmark_results(self, cases=None):
        """ Print benchmark results """
        if cases:
            if isinstance(cases, str):
                case_list = self.filter_cases_from_str(cases)
            elif isinstance(cases, list):
                case_list = cases
            else:
                raise NotImplementedError
        else:
            case_list = None

        self.print_benchmark_info()
        s = "Cases and results:\n\n"

        non_def_params = []
        static_non_default = {}
        b_c = self.get_benchmark_config()
        for param, value_list in b_c.items():
            if len(value_list) == 1 and value_list[0] != self.default_config.get(param):
                static_non_default[param] = value_list[0]
        for outfile, case in self.get_output_metadata().items():
            if case_list and case not in case_list:
                continue
            non_def_params.append(self.get_case_str(case, exclude=static_non_default.keys()))

        max_conf_len = len(max(non_def_params, key=lambda x: len(x)))
        curr_def = {x: y for x, y in self.default_config.items() if x not in static_non_default.keys()}
        s += "Default parameters" + ' ' * 14 + f": {curr_def}\nNon-default (static) parameters : {static_non_default}"
        s += "\n\n"
        s += "Benchmarks" + ' ' * (
                    max_conf_len - 2) + "Time" + ' ' * 9 + "Exec/sec" + ' ' * 5 + "Traffic cost (cents/hour)"
        s += "\n" + '-' * 100 + "\n"

        for outfile, case in self.get_output_metadata().items():
            if case_list and case not in case_list:
                continue
            curr_conf = non_def_params.pop(0)
            s += curr_conf + ' ' * (max_conf_len - len(curr_conf) + 1) + ":     "
            prot = self.get_output_metadata().get(os.path.basename(outfile)).get("protocol")
            parse_output_info = io.extract_info_from_output(
                self.get_output_from_outfile(outfile),
                ["Time", "Global_comm_size"],
                prot)
            exec_time = parse_output_info.get("Time")
            if exec_time:
                exec_time = round(exec_time, 5)
                exec_per_sec = self.calc_exec_per_sec(case, exec_time)
                traffic_cost_per_sec = self.calc_traffic_cost(case, parse_output_info.get("Global_comm_size"))
                traffic_cost_per_hour = round(traffic_cost_per_sec * 3600, 3)
            else:
                exec_per_sec = " - "
                traffic_cost_per_hour = " - "
                if case.get("n_parties", -1) < Protocol.get_protocol_by_name(prot).min_parties:
                    exec_time = "  n/a"
                elif Protocol.get_protocol_by_name(prot).max_parties and \
                        case.get("n_parties", -1) > Protocol.get_protocol_by_name(prot).max_parties:
                    exec_time = "  n/a"
                else:
                    exec_time = f"Failed ({os.path.join(self.outputs_dir, outfile)})"
            case_str = str(exec_time) + ' ' * (16 - len(str(exec_time))) + str(exec_per_sec)
            case_str += ' ' * (18 - len(str(exec_per_sec))) + str(traffic_cost_per_hour)
            s += case_str
            s += '\n'

        log.info(s)
