""" Class representing the execution of a secure detection """
from configparser import ConfigParser
import logging

import matplotlib.pyplot as plt
import numpy as np

from . import utils
import os
import shutil
import sys

from . import constants
from ._mpspdz_engine import MPSPDZEngine
from ._protocols import Protocol

log = logging.getLogger('secure_detection')


class SecureDetection(object):
    """ Instance of secure detection execution """

    @staticmethod
    def find_root_dir():
        """ Try to find the root dir on the workstation """
        for root_dir in constants.SEC_COMP_ROOT:
            if utils.io.dir_exists(root_dir):
                return root_dir
        return None

    def __init__(self, n_parties=4, root_dir=None, mp_spdz_root=None, detection_mode="selest",
                 single_output=False, input_samples_dir=None, protocol='mascot', preprocessing=True,
                 scaling_factor=100, net_lat=0.0, setup_online=False, comp_field="arithmetic",
                 compile_prog=True, int_scalars=False, net_top="star", parallel=1,
                 batched_executions=1, angle_step=None, debug=False):

        if debug:
            log.setLevel(logging.DEBUG)

        if not root_dir or not utils.io.dir_exists(root_dir):
            root_dir = SecureDetection.find_root_dir()
            if not root_dir:
                raise NotADirectoryError("Root directory not found.")

        self.root_dir = os.path.abspath(root_dir)

        self.n_parties = int(n_parties)
        self.config_file = os.path.join(self.root_dir, "selest.conf")

        # By default, inputs are shared by Player 0. Specify split inputs to be shared by each player individually.
        self.split_inputs = False
        self.n_antennas = 4
        self.cov_mat_avg = None

        self.qr_iter = 1

        self.detection_mode = detection_mode
        self.single_output = single_output
        self.output = None

        self.net_lat = float(net_lat)
        self.net_top = net_top
        self.parallel = parallel
        self.batched_executions = batched_executions
        if self.detection_mode == "text_music":
            self.angle_step = 1
        else:
            self.angle_step = angle_step or 18

        self._setup_ssl = False
        self._setup_online = setup_online
        self._compile_prog = compile_prog

        self._discard_decimals = int_scalars
        self._scaling_factor = float(scaling_factor)
        self._raw_input_samples = None

        self.preprocessing = preprocessing
        self.protocol = protocol
        self.comp_field = comp_field
        self.mp_spdz_root = mp_spdz_root or os.path.join(self.root_dir, "MP_SPDZ_online")
        self.emp_toolkit_root = mp_spdz_root or os.path.join(self.root_dir, "emp_toolkit")
        self.prog_name = "selest"

        self.mpc_engine = self.get_mpc_engine()
        assert self.mpc_engine

        if input_samples_dir:
            if utils.io.dir_exists(input_samples_dir):
                self.input_samples_dir = input_samples_dir
                self.load_raw_samples()
                self.load_samples_to_inputs()
            else:
                raise NotADirectoryError("Input samples dir not found: " + input_samples_dir)

    @property
    def scaling_factor(self):
        """Returns the scaling factor used for input samples"""
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, val):
        assert isinstance(val, (float, int))
        self._scaling_factor = val

    @property
    def discard_decimals(self):
        """Returns the scaling factor used for input samples"""
        return self._discard_decimals

    @discard_decimals.setter
    def discard_decimals(self, val):
        assert isinstance(val, bool)
        self._discard_decimals = val

    @property
    def setup_ssl(self):
        """ Return selected setting """
        return self._setup_ssl

    @setup_ssl.setter
    def setup_ssl(self, val: bool):
        self._setup_ssl = val

    @property
    def setup_online(self):
        """ Return selected setting """
        return self._setup_online

    @setup_online.setter
    def setup_online(self, val: bool):
        self._setup_online = val

    @property
    def compile_prog(self):
        """ Return selected setting """
        return self._compile_prog

    @compile_prog.setter
    def compile_prog(self, val: bool):
        self._compile_prog = val

    def get_mpc_engine(self):
        """ Returns the mpc engine depending on MP-SPDZ or EMP-Toolkit protocol """
        if Protocol.get_protocol_by_name(self.protocol).mpc_engine == MPSPDZEngine.name:
            return MPSPDZEngine(root_dir=self.mp_spdz_root, prog_name="selest", n_parties=self.n_parties,
                                protocol=self.protocol, preprocessing=self.preprocessing, net_top=self.net_top)
        elif Protocol.get_protocol_by_name(self.protocol).mpc_engine == EMPEngine.name:
            return EMPEngine(root_dir=self.emp_toolkit_root, prog_name="selest", protocol=self.protocol)
        else:
            return None

    def load_raw_samples(self, dirname=None):
        """ Load recorded samples into a numpy array """
        dirname = dirname or self.input_samples_dir
        if not dirname:
            raise NotADirectoryError("Input samples dir missing.")

        self._raw_input_samples = utils.io.load_matrix_from_raw_samples(self.input_samples_dir,
                                                                        scaling_factor=self.scaling_factor,
                                                                        discard_decimals=self.discard_decimals)
        player_inputs_len = set(len(x) for x in self._raw_input_samples)
        assert len(player_inputs_len) == 1
        self.cov_mat_avg = list(player_inputs_len)[0]
        # print(self._raw_input_samples)
        # tmp_list = [x for y in self._raw_input_samples for x in y]
        # tmp_list = [x.split() for x in tmp_list]
        # tmp_list = [float(x) for y in tmp_list for x in y]
        # print(f"Length {len(tmp_list)}, min {min(tmp_list)}, max {max(tmp_list)}")
        # tmp_list = [int(x * 10**10) for x in tmp_list]
        # print(f"Length {len(tmp_list)}, min {min(tmp_list)}, max {max(tmp_list)}")
        # print(max(tmp_list) ** 2)
        # print(2**64)

    def load_samples_to_inputs(self, player_dirs=None):
        """ Store the samples as MP SPDZ player inputs """
        assert self.n_antennas == len(self._raw_input_samples)
        # print(self._raw_input_samples)
        # exit()

        if self.split_inputs:
            input_list = [" ".join([str(x) for x in row]) for row in self._raw_input_samples]
        else:
            input_list = ["\n".join([" ".join([str(x) for x in list(row)]) for row in list(self._raw_input_samples)])]

        input_list = [int(self.batched_executions) * (x + "\n") for x in input_list]

        self.mpc_engine.store_inputs(input_list, player_dirs)

    def update_config_file(self):
        """ Stores the required configuration for the circuit execution """
        config = ConfigParser()
        config.read(self.config_file)

        if self.mpc_engine.protocol.comp_domain == "binary":
            config['compile']['binary'] = "1"
        else:
            config['compile']['binary'] = "0"

        config['compile']['n_antennas'] = str(self.n_antennas)
        config['compile']['scalar_type'] = "sint" if self.discard_decimals else "sfix"
        config['music']['cov_mat_avg'] = str(self.cov_mat_avg)
        config['music']['single_output'] = str(int(utils.io.str_to_bool(self.single_output)))
        config['music']['detection_mode'] = str(self.detection_mode)
        config['music']['batched_executions'] = str(self.batched_executions)
        config['music']['angle_step'] = str(self.angle_step)
        config['qr_algo']['iterations'] = str(self.qr_iter)

        with open(self.config_file, 'w') as cf:
            config.write(cf)

    def parse_output(self, output=None, extract=None):
        """
        Parse circuit execution output to extract certain information. Give a list of information to extract, eg:
        extract = ["Time", "Time1", "Number of SPDZ gfp multiplications", "Data sent", "Global data sent"]
        """
        if extract is None:
            extract = ["Time"]
        output = output or self.output
        output = utils.io.parse_subprocess_output(output)
        assert output

        stats = utils.io.extract_info_from_output(output, extract, self.mpc_engine.protocol.name)
        return stats

    def _run_mp_spdz(self):
        """ Run in the mpspdz engine """
        mp_engine = self.mpc_engine

        if self.setup_online:
            mp_engine.online_setup()

        if self.setup_ssl:
            mp_engine.ssl_setup()

        self.update_config_file()

        if self.compile_prog:
            if mp_engine.protocol.comp_domain == "binary":
                mp_engine.comp_field = "binary"
            mp_engine.compile_circuit()

        mp_engine.execute_circuit(parallel=self.parallel)

        self.output = mp_engine.output

    def _run_emp(self):
        """ Run in the emp toolkit """

        # Currently reads inputs from hardcoded files, it's convenient to copy them to the cwd
        shutil.copy(os.path.join(self.root_dir, "steering_vector_avg_18.txt"),
                    os.path.join(os.getcwd(), "steering_vector_avg_18.txt"))
        shutil.copy(os.path.join(self.root_dir, "emp_inputs.txt"),
                    os.path.join(os.getcwd(), "emp_inputs.txt"))

        try:
            self.mpc_engine.execute_circuit(parallel=self.parallel)
            self.output = self.mpc_engine.output
        except Exception as e:
            raise e
        finally:
            os.remove(os.path.join(os.getcwd(), "emp_inputs.txt"))
            os.remove(os.path.join(os.getcwd(), "steering_vector_avg_18.txt"))

    def run(self):
        """ Start secure detection execution """

        assert self.input_samples_dir is not None

        if self.net_lat:
            utils.networking.clear_network_latency()
            log.info("Setting network latency (%s) to %d ms...", 'loopback', self.net_lat)
            exit_code = utils.networking.set_network_latency(self.net_lat)
            if exit_code:
                log.error("Elevated privileges required to change the interface latency.")
                sys.exit(-1)

        if isinstance(self.mpc_engine, MPSPDZEngine):
            self._run_mp_spdz()
        elif isinstance(self.mpc_engine, EMPEngine):
            self._run_emp()
        else:
            raise NotImplementedError

        if self.net_lat:
            log.info("Clearing network latency (%s)...", 'loopback')
            utils.networking.clear_network_latency()

    def get_pseudospectrum(self, raw=False):
        """ Return a list of power levels in dB across different angle points """
        pseudospectrum = self.parse_output(extract=["Music spectrum"]).get("Music spectrum")
        assert pseudospectrum
        pseudospectrum = utils.io.read_json_str(pseudospectrum)
        assert pseudospectrum
        if self.detection_mode in ["selest", "signal_subspace", "text_music"]:
            pseudospectrum = [float(x) for x in pseudospectrum]
        elif self.detection_mode in ["opt_music", "kernel_approximation"]:
            pseudospectrum = [1 / float(x) for x in pseudospectrum]
        if raw:
            return pseudospectrum
        pseudo_db = [utils.misc.pwr_to_db(x) for x in pseudospectrum]
        return pseudo_db

    def plot_pseudospectrum(self):
        """ Plots pseudospectrum curve for easier verification """
        pseudo_db = self.get_pseudospectrum()

        if self.detection_mode == "selest":
            figtitle = "SELEST Pseudospectrum"
        elif self.detection_mode == "opt_music":
            figtitle = "Opt-MUSIC Pseudospectrum"
        elif self.detection_mode == "text_music":
            figtitle = "Standard MUSIC Pseudospectrum"
        else:
            figtitle = "Pseudospectrum"

        fig, ax = plt.subplots()
        ax.grid("on")
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xticks(np.linspace(0, 5, 24))
        ax.set_xticklabels([int(18*x) for x in np.linspace(0, 5, 24)])
        ax.set_yticks(np.linspace(min(pseudo_db), max(pseudo_db), 15))
        ax.set_title(figtitle, size=22)
        ax.set_xlabel("Angle (degrees)", size=18)
        ax.set_ylabel("dB", size=18, rotation=90)
        ax.plot([0.5+x for x in range(len(pseudo_db))], pseudo_db, "-*")
        plt.show()
