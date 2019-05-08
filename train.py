from npi import NPI
from f_encoder import F_Encoder
from config import CONFIG, ScratchPad
import pickle
import tensorflow as tf

INCREMENT = 1
DATA_PATH = "train.pik"

def train_model(epochs):
    """
    Instantiates an Addition Core, NPI, then loads and fits model to data.

    :param epochs: Number of epochs to train for.
    """
    # Load Data
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Initialize Addition Core
    print('Initializing Core!')
    core = F_Encoder()

    # Initialize NPI Model
    print('Initializing NPI Model!')
    npi = NPI(core, CONFIG)

    # Initialize TF Saver
    saver = tf.train.Saver()

    # Initialize TF Session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Start Training
    training_error = {}

    for ep in range(1, epochs + 1):
        print(ep)
        training_error[ep] = []
        for i in range(len(data)):
            # Reset NPI States
            npi.reset_state()

            # Setup Environment
            c_find, steps = data[i]
            scratch = ScratchPad(c_find)
            x, y = steps[:-1], steps[1:]
           # Run through steps, and fit!
            step_def_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0

            for j in range(len(x)):
                (prog_name, prog_in_id), arg, term = x[j]
                (_, prog_out_id), arg_out, term_out = y[j]

                if prog_in_id == 1:
                    scratch.increment_ptr()
                # Get Environment, Argument Vectors
                env_in = [scratch.get_env()]
                # arg_in, arg_out = [get_args(arg, arg_in=True)], get_args(arg_out, arg_in=False)
                prog_in, prog_out = [[prog_in_id]], [prog_out_id]
                term_out = [1] if term_out else [0]

                # Fit!
                loss, t_acc, p_acc, _ = sess.run(
                        [npi.default_loss, npi.t_metric, npi.p_metric, npi.default_train_op],
                        feed_dict={npi.env_in: env_in,  npi.prg_in: prog_in,
                                   npi.y_prog: prog_out, npi.y_term: term_out})
                step_def_loss += loss
                term_acc += t_acc
                prog_acc += p_acc

            training_error[ep].append((i, step_def_loss / len(x), term_acc / len(x), prog_acc / len(x)))
    # Save Model
    print("Epoch {} Step {} Default Step Loss {},Term: {}, Prog: {}".format(ep, i, step_def_loss / len(x), term_acc / len(x), prog_acc / len(x)))

    saver.save(sess, 'model/model.ckpt')
    return training_error

if __name__ == "__main__":
    training_error = train_model(epochs=100)
