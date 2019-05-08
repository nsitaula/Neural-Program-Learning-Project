"""
eval.py

Loads in an Addition NPI, and starts a REPL.
"""
from npi import NPI
from f_encoder import F_Encoder
from config import CONFIG, PROGRAM_SET, ScratchPad, COUNTRY_REGION, NUM_CODE, COUNTRY_REGION_CODE
import numpy as np
import pickle
import tensorflow as tf

CKPT_PATH = "model/model.ckpt"
TEST_PATH = "test.pik"

def evaluate():

    with tf.Session() as sess:
        # Load Data
        with open(TEST_PATH, 'rb') as f:
            data = pickle.load(f)

        # Initialize Core
        core = F_Encoder()

        # Initialize NPI Model
        npi = NPI(core, CONFIG)

        # Restore from Checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)

        # Run REPL
        repl(sess, npi, data)


def repl(session, npi, data):
    while True:
        inpt = input('Hit Enter for Random country: ')

        if inpt == "":
            x = data[np.random.randint(len(data))][0]

        else:
          break

        print('To find: ', COUNTRY_REGION_CODE[x])
        # Reset NPI States
        print("")
        npi.reset_state()

        # Setup Environment
        scratch = ScratchPad(x)
        prog_name, prog_id, arg, term = 'FIND', 0, [], False
        print('--------  Environment ----------' )
        scratch.pretty_print()
        print('-----------------------------------')
        cont = 'c'
        while cont == 'c' or cont == 'C':
            a_str = "[]"

            print('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term)))

            if prog_id == 1:
                scratch.increment_ptr()


            env_in, prog_in = [scratch.get_env()],  [[prog_id]]

            t, n_p = session.run([npi.terminate, npi.program_distribution],
                                         feed_dict={npi.env_in: env_in,
                                                    npi.prg_in: prog_in})
            if np.argmax(t) == 1:
                print('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(True)))

                scratch.pretty_print()

                true_ans = None
                for i in COUNTRY_REGION:
                  if i[0] == x:
                      true_ans = COUNTRY_REGION_CODE[i[1]]
                      break
                if(scratch.ptr<scratch.rows-1):
                    output = COUNTRY_REGION_CODE[scratch[scratch.ptr][1]]
                    scratch[scratch.rows-1][0] = NUM_CODE[output]
                else:
                    output = 'Not found'

                print('Pointer at ', scratch.ptr)
                print('-------- KB Environment ----------' )
                scratch.pretty_print()
                print('-'*100)
                print("Model Output: ", output)
                print("Correct Out : " , true_ans)
                print("Correct!" if output == true_ans else "Incorrect!")

            else:
                prog_id = np.argmax(n_p)
                prog_name = PROGRAM_SET[prog_id][0]
                term = False

            cont = input('Continue? enter "c"')

if __name__ == "__main__":
    evaluate()
