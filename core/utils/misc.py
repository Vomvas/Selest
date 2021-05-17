""" Misc utilities """
import time
from hashlib import sha1
from numpy import log10


# Time utils
def hash_ts():
    """ Returns a SHA-1 hash of the current timestamp to be used as a name UID """
    ts_hash = sha1()
    ts_hash.update(str(time.time()).encode("utf-8"))
    ts_hash.hexdigest()
    return ts_hash.hexdigest()


# Power utils
def pwr_to_db(x):
    """ Return power in db """
    return 10 * log10(x)


def dict_depth(dic, level=1):
    """ Find nested dict depth """
    if not isinstance(dic, dict) or not dic:
        return level
    return max(dict_depth(dic[key], level + 1)
               for key in dic)
