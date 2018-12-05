from __future__ import absolute_import

from .Models import *

__factory = {
    'base': Base,
    'base_spatial': Base_spatial,
    'base_temporal': Base_temporal,
    'ours': Ours,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)