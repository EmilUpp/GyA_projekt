"""A collections of useful decorators"""

from functools import wraps
from time import time


def timing(f):
    """A decorator for timing function calls"""
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

