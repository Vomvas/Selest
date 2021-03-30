""" Classes representing the protocols in MP-SPDZ """


class Protocol(object):
    """ MP_SPDZ Protocol base class """

    slow_offline = None

    @classmethod
    def supported_protocols(cls):
        """ Returns list of names of supported protocols """
        return [x.name for x in cls.get_subclasses()]

    @classmethod
    def get_subclasses(cls):
        """ Returns a list of subclasses """
        return cls.__subclasses__()

    @staticmethod
    def get_protocol_by_name(name):
        """ Return a protocol object by name """
        for prot in Protocol.get_subclasses():
            if prot.name == name:
                return prot
        else:
            raise NotImplementedError(f"Unrecognized protocol name: {name}")


class Shamir(Protocol):
    """ Protocol properties """
    name = "shamir"
    build_name = "shamir"
    min_parties = 3
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "semi-honest"
    majority = "honest"
    print_name = "Shamir"
    exec_script = "shamir.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of Shamir gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class MalShamir(Protocol):
    """ Protocol properties """
    name = "mal-shamir"
    build_name = "shamir"
    min_parties = 3
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "malicious"
    majority = "honest"
    print_name = "Mal-Shamir"
    exec_script = "mal-shamir.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of Shamir gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class SyShamir(Protocol):
    """ Protocol properties """
    name = "sy-shamir"
    build_name = "sy"
    min_parties = 3
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "malicious"
    majority = "honest"
    print_name = "Sy-Shamir"
    exec_script = "sy-shamir.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of Shamir gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Mascot(Protocol):
    """ Protocol properties """
    name = "mascot"
    build_name = "mascot"
    min_parties = 2
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "malicious"
    majority = "dishonest"
    print_name = "MASCOT"
    exec_script = "mascot.sh"
    slow_offline = True         # Avoid large execution averaging for this protocol
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of SPDZ gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Lowgear(Protocol):
    """ Protocol properties """
    name = "lowgear"
    build_name = "lowgear-party.x"
    min_parties = 2
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "malicious"
    majority = "dishonest"
    print_name = "Lowgear"
    exec_script = "lowgear.sh"
    slow_offline = True  # Avoid large execution averaging for this protocol
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of SPDZ gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Cowgear(Protocol):
    """ Protocol properties """
    name = "cowgear"
    build_name = "cowgear-party.x"
    min_parties = 2
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "covert"
    majority = "dishonest"
    print_name = "Cowgear"
    exec_script = "cowgear.sh"
    slow_offline = True  # Avoid large execution averaging for this protocol
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of SPDZ gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Semi(Protocol):
    """ Protocol properties """
    name = "semi"
    build_name = "semi-party.x"
    min_parties = 2
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "semi-honest"
    majority = "dishonest"
    print_name = "Semi"
    exec_script = "semi.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of SPDZ gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Hemi(Protocol):
    """ Protocol properties """
    name = "hemi"
    build_name = "hemi-party.x"
    min_parties = 2
    max_parties = 0
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "semi-honest"
    majority = "dishonest"
    print_name = "Hemi"
    exec_script = "hemi.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of SPDZ gfp multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Rep3(Protocol):
    """ Protocol properties """
    name = "rep3"
    build_name = "rep-field"
    min_parties = 3
    max_parties = 3
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "semi-honest"
    majority = "honest"
    print_name = "Rep3"
    exec_script = "rep-field.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of Z2^64^3 multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class PsRep(Protocol):
    """ Protocol properties """
    name = "ps-rep"
    build_name = "rep-field"
    min_parties = 3
    max_parties = 3
    comp_domain = "arithmetic"
    field_comp = "prime"
    advers_type = "malicious"
    majority = "honest"
    print_name = "Ps-Rep"
    exec_script = "ps-rep-field.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of Z2^64^3 multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class Rep4(Protocol):
    """ Protocol properties """
    name = "rep4"
    build_name = "rep4-ring-party.x"
    min_parties = 4
    max_parties = 4
    comp_domain = "arithmetic"
    field_comp = "power_two"
    advers_type = "malicious"
    majority = "honest"
    print_name = "Rep4"
    exec_script = "rep4-ring.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of Z2^64^3 multiplications: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Global data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"


class YaoGC(Protocol):
    """ Protocol properties """
    name = "yao"
    build_name = "yao"
    min_parties = 2
    max_parties = 2
    comp_domain = "binary"
    field_comp = "binary"
    advers_type = "semi-honest"
    majority = "n/a"
    print_name = "Yao's GC (2pc)"
    exec_script = "yao.sh"
    reg_exps = {
        "Time": "Time = (.[0-9.]*) seconds",
        "Time#": "Time# = (.[0-9.]*) seconds",
        "N_mults": "Number of AND gates: (.[0-9]*)",
        "Comm_size": "Data sent = (.[0-9.]*) MB",
        "Global_comm_size": "Data sent = (.[0-9.]*) MB",
    }
    mpc_engine = "mp-spdz"
