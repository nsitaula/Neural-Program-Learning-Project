"""
trace.py

Core class definition for a trace object => given a pair of integers to add, builds the execution
trace, calling the specified subprograms.
"""
# from tasks.addition.env.config import ScratchPad, PROGRAM_ID as P
from config import ScratchPad, PROGRAM_ID as P, COUNTRY_REGION

FIND, INCREMENT, END, ACT = "FIND", "INCREMENT", "END", "ACT"
PTR = "PTR"


class Trace():
    def __init__(self, c_find, debug=False):
        """
        Instantiates a trace object, and builds the exact execution pipeline for adding the given
        parameters.
        """
        self.c_find, self.debug = c_find, debug
        self.trace, self.scratch = [], ScratchPad(c_find)

        #######################
        # Build Execution Trace
        self.build()

        # Check answer
        true_ans = None
        for i in COUNTRY_REGION:
            if i[0] == c_find:
                true_ans = i[1]
                break
        # trace_ans = self.scratch[self.scratch.rows-1][0].decode("utf-8")
        trace_ans = self.scratch[self.scratch.rows-1][0]

        assert(true_ans == trace_ans)

    def build(self):
        """
        Builds execution trace, adding individual steps to the instance variable trace. Each
        step is represented by a triple (program_id : Integer, args : List, terminate: Boolean). If
        a subroutine doesn't take arguments, the empty list is returned.
        """
        # Seed with the starting subroutine call
        self.trace.append(((FIND, P[FIND]), [], False))

        # Execute Trace
        self.scratch.increment_ptr()
        self.trace.append(((INCREMENT, P[INCREMENT]), [], False))
        while not self.scratch.done():
            self.scratch.increment_ptr()
            # self.trace.append(((END, P[END]), [], False))
            self.trace.append(((INCREMENT, P[INCREMENT]), [], False))

        self.trace.append(((END, P[END]), [], True))
