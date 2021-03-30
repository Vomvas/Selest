""" Misc utilities """
import time
from hashlib import sha1
from numpy import log10


# Power utils
def pwr_to_db(x):
    """ Return power in db """
    return 10 * log10(x)
