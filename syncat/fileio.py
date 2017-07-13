import numpy as np
from astropy.table import Table


def read_catalogue(filename, format='', columns=None, quick=0):
    """ """
    args = {}
    if format.startswith("ascii"):
        args['names'] = columns
    else:
        pass
    table = Table.read(filename, format=format, **args)

    if quick > 0:
        table = table[:quick]

    return table