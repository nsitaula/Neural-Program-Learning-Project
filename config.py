"""
config.py

Configuration Variables for the Addition NPI Task => Stores Scratch-Pad Dimensions, Vector/Program
Embedding Information, etc.
"""
import numpy as np
import sys
import time

# COUNTRY_REGION = [('Nepal', 'Asia'),
#                           ('India', 'Asia'),
#                           ('Sudan', 'Africa'),
#                           ('Angola', 'Africa')]

NUM_CODE = {'Nepal': 1, 'Asia': 2, 'India': 3, 'Sudan': 4, 'Africa': 5, 'Angola': 6, 'Sweden': 7, 'Europe': 8, 'Italy': 9, 'France': 0, 'China': 11, 'Brazil': 12, 'Peru': 13, 'USA': 14, 'Mexico': 15, 'UK': 16, 'North America': 17, 'Bangladesh': 18, 'Japan': 19, 'Egypt': 20, 'Nigeria': 21}

COUNTRY_REGION = [(NUM_CODE['Nepal'], NUM_CODE['Asia']),
                  (NUM_CODE['India'], NUM_CODE['Asia']),
                  (NUM_CODE['China'], NUM_CODE['Asia']),
                  (NUM_CODE['Peru'], NUM_CODE['North America']),
                  (NUM_CODE['Brazil'], NUM_CODE['North America']),
                  (NUM_CODE['UK'], NUM_CODE['Europe']),
                  (NUM_CODE['Sudan'], NUM_CODE['Africa']),
                  (NUM_CODE['Italy'], NUM_CODE['Europe']),
                  (NUM_CODE['Sweden'], NUM_CODE['Europe']),
                  (NUM_CODE['Mexico'], NUM_CODE['North America']),
                  (NUM_CODE['USA'], NUM_CODE['North America']),
                  (NUM_CODE['Angola'], NUM_CODE['Africa']),
                  (NUM_CODE['Bangladesh'], NUM_CODE['Asia']),
                  (NUM_CODE['Japan'], NUM_CODE['Asia']),
                  (NUM_CODE['Egypt'], NUM_CODE['Africa']),
                  (NUM_CODE['Nigeria'], NUM_CODE['Africa']),
                  ]


COUNTRY_REGION_CODE = {1: 'Nepal',
                       2: 'Asia',
                       3: 'India',
                       4: 'Sudan',
                       5: 'Africa',
                       6: 'Angola',
                       7: 'Sweden',
                       8: 'Europe',
                       9: 'Italy',
                       0: 'France',
                       11: 'China',
                       12: 'Brazil',
                       13: 'Peru',
                       14: 'USA',
                       15: 'Mexico',
                       16: 'UK',
                       17: 'North America',
                       18: 'Bangladesh',
                       19: 'Japan',
                       20: 'Egypt',
                       21: 'Nigeria'
                       }
CONFIG = {
    "ENVIRONMENT_ROW": len(COUNTRY_REGION)+2,         # Input 1 to store array
    "ENVIRONMENT_COL": 2,        # 10-Digit Maximum for Addition Task
    "ENVIRONMENT_DEPTH": 10,      # 2 country 2 region => One-Hot, Options: 0-9

    "PROGRAM_NUM": 3,             # Maximum Number of Subroutines
    "PROGRAM_KEY_SIZE": 3,        # Size of the Program Keys
    "PROGRAM_EMBEDDING_SIZE": 3  # Size of the Program Embeddings
}

PROGRAM_SET = [("FIND",),  ("INCREMENT", ), ("END",)]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}

class ScratchPad():           # Addition Environment
    def __init__(self, c_find, rows=CONFIG["ENVIRONMENT_ROW"], cols=CONFIG["ENVIRONMENT_COL"]):
        # Setup Internal ScratchPad
        self.rows, self.cols = rows, cols
        # self.scratchpad = np.chararray((self.rows, self.cols), itemsize=10)
        self.scratchpad = np.zeros((self.rows, self.cols), dtype=np.int8)

        # Initialize ScratchPad In1
        self.c_find = c_find
        self.init_scratchpad(c_find)

        # Initially pointer is at the top
        self.ptr = 0

    def init_scratchpad(self, c_find):
        """
        Initialize the scratchpad with the given input numbers (to be added together).
        """
        self.scratchpad[0, 0] = c_find
        self.scratchpad[0, 1] = -1

        for i in range(len(COUNTRY_REGION)):
            for k in range(2):
                self.scratchpad[i+1, k] = COUNTRY_REGION[i][k]

        self.scratchpad[i+2, 0] = -1
        self.scratchpad[i+2, 1] = -1

    def done(self):

        # cur_val = self.scratchpad[self.ptr][0].decode('utf-8')
        cur_val = self.scratchpad[self.ptr][0]

        if self.c_find == cur_val:
            self.scratchpad[self.rows-1][0] = self.scratchpad[self.ptr][1]
            return True
        else:
            return False

    def increment_ptr(self):
        self.ptr = self.ptr + 1


    def pretty_print(self):
        for i in self:
            country = COUNTRY_REGION_CODE.get(i[0], 'None')
            region = COUNTRY_REGION_CODE.get(i[1], 'None')

            print ('> ', country, ' ', region)
        print('')

    def get_env(self):
        return np.append(self.scratchpad.flatten() , np.array([self.ptr]))

    def __getitem__(self, item):
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        self.scratchpad[key] = value
