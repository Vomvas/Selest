""" Various networking utility functions """
import subprocess
from os import devnull
from socket import gethostname


def set_network_latency(latency, bw_limit=None, interface='lo'):
    """Sets the network latency (loopback interface by default) to the requested value in ms"""
    # log.info("Setting network latency (%s) to %d...", interface, latency)
    cmd = f"tc qdisc add dev {interface} root netem delay {latency}ms"
    sp = subprocess.run(cmd.split(), capture_output=True)
    if "Operation not permitted" in str(sp.stderr):
        return -1
    return 0


def clear_network_latency(interface='lo'):
    """Clears the network latency (default interface: Loopback)"""
    # log.info("Clearing network latency (%s)...", interface)
    cmd = f"tc qdisc del dev {interface} root netem"
    with open(devnull, 'w') as devnul:
        subprocess.run(cmd.split(), stderr=devnul)
    return 0
