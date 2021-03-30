""" Various I/O utility functions """
import glob
import json
import numpy as np
import os
import re
import struct
from configparser import ConfigParser
from subprocess import CompletedProcess
from core._protocols import *
from core import constants


def dir_exists(direct):
    """Check if directory exists"""
    return os.path.isdir(os.path.abspath(direct))


def read_json_str(s):
    """ Parse a json string """
    try:
        return json.loads(s)
    except json.decoder.JSONDecodeError:
        return None


def write_json_str(s):
    """ Serialize into json string """
    try:
        return json.dumps(s)
    except json.decoder.JSONDecodeError:
        return None


def str_to_bool(s):
    """ Parses a config string to determine the proper boolean value """
    if isinstance(s, bool):
        return s
    if not s or str(s).lower() == "false" or str(s) == '0':
        return False
    else:
        return True


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


def load_matrix_from_raw_samples(samples_dir, scaling_factor=1, discard_decimals=False, output_format="list_str"):
    """
    Loads all complex samples files in the given directory. The minimum size among all files is considered as input
    length and averaging factor, the rest of the inputs are truncated accordingly.

    Note: This functionality assumes relatively small inputs. Therefore there is no batching and results are stored
    in a list rather than an array. Not recommended for long streams of samples.
    """

    assert dir_exists(samples_dir)
    sample_files = sorted(glob.glob(os.path.join(os.path.abspath(samples_dir), "*.32fc")))

    if not sample_files:
        return None

    t = int if discard_decimals else float

    all_antenna_inputs = []
    for samp_file in sample_files:

        with open(samp_file, "rb") as rf:
            data = load_bytes_from_fd(rf)
        data = [complex(t(scaling_factor * data[i]), t(scaling_factor * data[i + 1]))
                for i in range(0, len(data), 2)]
        all_antenna_inputs.append(data)

    if output_format == "list_str":
        return [["{rp} {ip}".format(rp=t(x.real), ip=t(x.imag)) for x in y] for y in all_antenna_inputs]
    elif output_format == "numpy_cplx":
        return np.array(all_antenna_inputs)
    else:
        raise NotImplementedError


def parse_subprocess_output(output):
    """ Return stdout and stderr given a subprocess output """
    if isinstance(output, CompletedProcess):
        return output.stderr + output.stdout
    else:
        return output


def validate_parallel_executions(output, info_list, protocol_name):
    """ Parse output to detect errors in parallel executions """
    pass


def validate_averaged_executions(output, info_list, protocol_name):
    """ Parse output to detect errors in parallel executions """
    pass


def check_output_errors(output, protocol):
    """ Checks outputs for errors. Created for heavy parallelized executions """
    info_list = ["Time"]
    # Check if there is parallel executions and pick the largest time elapsed
    # parallel_exec = output.count(constants.PARALL_SEPARATOR) + 1
    # if parallel_exec > 1:
    failed_parallel = 0
    max_outputs = {}
    individ_outputs = [x for x in output.split(constants.PARALL_SEPARATOR) if x]
    for out in individ_outputs:
        exec_time = extract_info_from_single_output(out, info_list, protocol)["Time"]
        if not exec_time:
            failed_parallel += 1
    return failed_parallel


def extract_info_from_single_output(output, info_list, protocol_name):
    """ Parse a single execution output """
    results = {}

    # Check if there is parallel executions and pick the largest time elapsed
    parallel_exec = output.count(constants.PARALL_SEPARATOR) + 1
    if parallel_exec > 1:
        max_outputs = {}
        individ_outputs = [x for x in output.split(constants.PARALL_SEPARATOR) if x]
        for out in individ_outputs:
            for key, val in extract_info_from_single_output(out, info_list, protocol_name).items():
                if key not in max_outputs:
                    max_outputs[key] = val
                else:
                    if max_outputs.get(key) and val:
                        max_outputs[key] = max(max_outputs.get(key), val)
        return max_outputs

    for info in info_list:
        if "music spectrum" in info.lower():
            reg_exp = constants.REG_EXPS.get("Pseudospectrum")
        elif "doa" in info.lower():
            reg_exp = constants.REG_EXPS.get("DoA")
        else:
            reg_exp = Protocol.get_protocol_by_name(protocol_name).reg_exps.get(info, None)

        if not reg_exp:
            results[info] = None
            continue

        match = re.search(reg_exp, output, flags=re.MULTILINE)
        try:
            results[info] = float(match.group(1))
        except ValueError:
            results[info] = match.group(1)
        except (IndexError, AttributeError) as e:
            results[info] = None
    return results


def extract_info_from_output(output, info_list, protocol_name):
    """
    Extract execution information from output depending on protocol, handling parallel or averaged executions.

    Returns a dictionary with the results, info_list as keys and information as values.
    """

    assert isinstance(output, str), f"Unexpected output instance: {type(output)}"

    # Determine if there is execution averaging
    exec_avg = output.count(constants.AVG_SEPARATOR)

    averaged_results = {}
    if exec_avg > 1:
        indiv_res = []
        individ_outputs = [x for x in output.split(constants.AVG_SEPARATOR) if x]
        for out in individ_outputs:
            indiv_res.append(extract_info_from_single_output(out, info_list, protocol_name))
            for key, val in indiv_res[-1].items():
                if key not in averaged_results.keys():
                    averaged_results[key] = val
                else:
                    if key in constants.REG_EXPS:
                        continue
                    if val:
                        averaged_results[key] += val
        for key, val in averaged_results.items():
            if key in constants.REG_EXPS:
                continue
            elif averaged_results[key]:
                averaged_results[key] /= len(indiv_res)
        return averaged_results
    else:
        return extract_info_from_single_output(output, info_list, protocol_name)


def load_default_config(config_file):
    """ Load default benchmark configuration into a dictionary """
    config = ConfigParser()
    config.read(config_file)
    if not config:
        return {}
    all_dict = {k: v for sec in config.sections() for k, v in dict(config.items(sec)).items()}
    return all_dict
