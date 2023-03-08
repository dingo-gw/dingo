import numpy as np


def get_version():
    try:
        from dingo import __version__

        return __version__
    except ImportError:
        return None


def recursive_check_dicts_are_equal(dict_a, dict_b):
    if dict_a.keys() != dict_b.keys():
        return False
    else:
        for k, v_a in dict_a.items():
            v_b = dict_b[k]
            if type(v_a) != type(v_b):
                return False
            if type(v_a) == dict:
                if not recursive_check_dicts_are_equal(v_a, v_b):
                    return False
            elif not np.all(v_a == v_b):
                return False
    return True
