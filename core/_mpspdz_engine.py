""" Class representing the MP-SPDZ functionalities """
from . import constants
import logging
import subprocess
from . import utils
import os
from ._protocols import *

log = logging.getLogger('secure_detection')


class MPSPDZEngine(object):
    """ Instance of mp spdz execution """

    name = "mp-spdz"

    @staticmethod
    def find_root_dir():
        """ Try to find the root dir on the workstation """
        for root_dir in constants.MP_SPDZ_ROOT:
            if utils.io.dir_exists(root_dir):
                return root_dir
        return None

    def __init__(self, root_dir=None, prog_name=None, n_parties=None, protocol=None, preprocessing=True,
                 comp_field=None, net_top="star"):

        if not root_dir or not utils.io.dir_exists(root_dir):
            root_dir = MPSPDZEngine.find_root_dir()
            if not root_dir:
                raise NotADirectoryError("Root directory not found: " + root_dir)

        self.root_dir = os.path.abspath(root_dir)
        self.player_data_dir = os.path.join(self.root_dir, "Player-Data")
        self.scripts_dir = os.path.join(self.root_dir, "Scripts")

        self.program_name = prog_name
        self.n_parties = int(n_parties)
        self._protocol = None
        self._preprocessing = utils.io.str_to_bool(preprocessing)

        self.protocol = protocol

        self._comp_field = comp_field or self.protocol.field_comp

        self._prime_field_size = 128
        self._circuit_int_size = 32

        self.net_top = net_top

        self.output = None

    @property
    def protocol(self):
        """ Returns specified protocol """
        return self._protocol

    @protocol.setter
    def protocol(self, protocol):
        if isinstance(protocol, Protocol):
            self._protocol = protocol
        elif isinstance(protocol, str):
            self._protocol = Protocol.get_protocol_by_name(protocol)()
        else:
            raise NotImplementedError

    @property
    def comp_field(self):
        """ Returns computation field """
        return self._comp_field

    @comp_field.setter
    def comp_field(self, val: str):
        assert val in constants.FIELD.values()
        self._comp_field = val

    @property
    def prime_field_size(self):
        """ Returns specified setting """
        return self._prime_field_size

    @prime_field_size.setter
    def prime_field_size(self, lgp: int):
        self._prime_field_size = int(lgp)

    @property
    def circuit_int_size(self):
        """ Returns specified setting """
        return self._circuit_int_size

    @circuit_int_size.setter
    def circuit_int_size(self, int_size: int):
        self._circuit_int_size = int(int_size)

    @property
    def preprocessing(self):
        """ Returns preprocessing on or off """
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, val: bool):
        self._preprocessing = val

    @property
    def prepr_mode(self):
        """ Returns online or offline computation depending on preprocessing """
        return "online" if self.preprocessing else "offline"

    def compile_protocol(self, prot_list, multithreading=None):
        """ Compile MP-SPDZ protocols, specifying multithreading number adds -j <multithreading> option """
        old_dir = os.getcwd()
        os.chdir(self.root_dir)
        cmd = ["make"]
        if multithreading:
            try:
                cmd.extend(["-j", str(int(multithreading))])
            except:
                pass
        prot_build_names = [Protocol.get_protocol_by_name(x).build_name for x in prot_list]
        cmd.extend(["online"] + prot_build_names)      # Always make target online just in case
        log.info(f"Compiling MP-SPDZ protocols ({' '.join(prot_build_names)}), this will take a while...")
        log.debug(" ".join(cmd))
        s = subprocess.run(cmd, capture_output=True, encoding="utf")
        if s.returncode:
            log.error(f"Error(s) in MP-SPDZ compilation (s.returncode):\n\n{s.stderr}")
        else:
            log.info("MP-SPDZ protocols compiled.")
        os.chdir(old_dir)

    def ssl_setup(self, n_parties=None):
        """ Setup ssl certificates """
        log.info("SSL Setup...")
        n_parties = str(n_parties or self.n_parties)
        cmd = os.path.join(self.scripts_dir, "setup-ssl.sh").split()
        cmd.append(str(n_parties))
        subprocess.run(cmd)
        log.info("SSL Setup Complete.")

    def online_setup(self, n_parties=None, prime_field_size=None):
        """ Setup online phase """
        log.info("Online phase setup...")
        old_dir = os.getcwd()
        os.chdir(self.root_dir)
        n_parties = str(n_parties or self.n_parties)
        prime_field_size = str(prime_field_size or self.prime_field_size)
        cmd = os.path.join(self.scripts_dir, "setup-online.sh").split()
        cmd.append(str(n_parties))
        cmd.append(str(prime_field_size))
        s = subprocess.run(cmd, capture_output=True, encoding="utf")
        if s.returncode:
            log.error(f"Error(s) in MP-SPDZ online setup ({s.returncode}):\n\n{s.stderr}")
        else:
            log.info("Online phase setup complete.")
        os.chdir(old_dir)

    def store_inputs(self, input_list, player_list=None):
        """
        Store a list of inputs in the specified player input directory. Default directories are assumed if not provided.
        Expects a list of space separated inputs to be directly copied in each player's directory.

        For example:

        input_list: ["2.3 1.1 -13.2 4", "0 1 2", 3, -0.04]
        player_list: ["Player0-0", "Player0-1", "Player1-0", "Player2-0"]
        stores:
        Player0-0: "2.3 1.1 -13.2 4"
        Player0-1: "0 1 2"
        Player1-0: "3"
        Player2-0: "-0.04"
        """

        if not player_list:
            player_list = ["Input-P%s-0" % x for x in range(len(input_list))]

        assert len(input_list) == len(player_list)
        for inp, pl in zip(input_list, player_list):
            with open(os.path.join(self.player_data_dir, pl), "w") as pl_inp_w:
                pl_inp_w.write(inp)

    def _validate_compilation(self):
        return not ((self._comp_field in ["prime", "power_two"] and self.protocol.comp_domain == "binary") or
                    (self._comp_field in ["binary"] and self.protocol.comp_domain == "arithmetic"))

    def _custom_compile(self, prog_name, options=None):
        """ Compile circuit with custom options """
        log.info(f"Compiling circuit with {options}...")
        cmd = os.path.join(self.root_dir, "compile.py").split()
        for key, val in options.items():
            cmd.extend([str(key), str(val)])
        cmd.append(prog_name)
        log.debug(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd)
        log.info("Circuit compilation complete.")

    def compile_circuit(self):
        """ Compile circuit """
        if not self._validate_compilation():
            log.warning(f"Requested protocol <{self.protocol.name}> does not support <{self._comp_field}> field.")
        options = {
            constants.FIELD_COMPILATION[self._comp_field]: str(self.circuit_int_size),
        }
        self._custom_compile(self.program_name, options)

    def _execute_in_parallel(self, cmd, parallel, cmd_env=None):
        """ Invoke parallel executions """
        if cmd_env:
            n_parties = int(cmd_env.get("PLAYERS", None)) or 5
        else:
            n_parties = 5
        base_port = 5000
        child_processes = []
        for i in range(parallel):
            port_number = base_port + 2 * n_parties * i     # Leave enough ports for at least n_parties
            child_command = [x for x in cmd] + ["-pn", str(port_number)]
            p = subprocess.Popen(child_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=cmd_env,
                                 encoding="utf")
            child_processes.append(p)

        outputs = []
        for cp in child_processes:
            try:
                outputs.append(cp.communicate(timeout=constants.PARALL_TIMEOUT[self.prepr_mode]))
            except subprocess.TimeoutExpired:
                cp.kill()
                flw = "Child timeout expired."
                outputs.append(flw)

        self.output = constants.PARALL_SEPARATOR.join([x[0] + x[1] for x in outputs])

    def _custom_execute(self, prog_name, protocol, n_parties=4, options=None, parallel=1):
        assertion_msg = f"Invalid number of parties {n_parties} for protocol {protocol.name}."
        assert protocol.min_parties <= n_parties, assertion_msg
        if protocol.max_parties:
            assert n_parties <= protocol.max_parties, assertion_msg
        prep_mode = "online" if self.preprocessing else "offline"
        log.info(f"Executing program {prog_name} using protocol {protocol.name} ({prep_mode}) with {n_parties} parties")
        my_env = os.environ.copy()
        my_env['PLAYERS'] = str(n_parties)
        cmd = os.path.join(self.scripts_dir, protocol.exec_script).split()
        cmd.append(prog_name)
        if options:
            for k, v in options.items():
                cmd.extend([str(k), str(v)])
        log.debug(f"Command ({parallel} in parallel): {' '.join(cmd)}")
        if parallel > 1:
            self._execute_in_parallel(cmd, parallel, my_env)
        else:
            self.output = subprocess.run(cmd, env=my_env, capture_output=True, encoding="utf")

    def execute_circuit(self, parallel=1):
        """ Execute circuit """

        options = {}
        if self.protocol.field_comp == "prime":
            options["-lgp"] = self.prime_field_size
        if self.preprocessing:
            options["-F"] = ""

        if self.net_top == "direct":
            options["-d"] = ""

        if self.protocol.name == "yao":
            self._custom_execute(self.program_name, self.protocol, 2, parallel=parallel)
        else:
            self._custom_execute(self.program_name, self.protocol, self.n_parties, options, parallel=parallel)
