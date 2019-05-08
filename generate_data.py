"""
generate_data.py

Core script for generating training/test addition data. First, generates random pairs of numbers,
then steps through an execution trace, computing the exact order of subroutines that need to be
called.
"""
import pickle

import numpy as np

from trace import Trace

from config import COUNTRY_REGION

def create_trace(prefix, num_examples):
    """
    Generates addition data with the given string prefix (i.e. 'train', 'test') and the specified
    number of examples.

    :param prefix: String prefix for saving the file ('train', 'test')
    :param num_examples: Number of examples to generate.
    """
    data = []
    for j in range(num_examples):
        rand_int = np.random.randint(len(COUNTRY_REGION))
        c_find = COUNTRY_REGION[rand_int][0]
        trace = Trace(c_find).trace
        data.append(( c_find, trace ))

    with open('{}.pik'.format(prefix), 'wb') as f:
        pickle.dump(data, f)

    print('Data Generated.')
