import numpy as np
from astropy.table import Table


def read_catalogue(filename, format='', columns=None):
    """ """
    args = {}
    if format.startswith("ascii"):
        args['names'] = columns
    else:
        pass
    table = Table.read(filename, format=format, **args)

    return table