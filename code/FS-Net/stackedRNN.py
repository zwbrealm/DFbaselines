import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    dtype):
    states_fw = []
    states_bw = []
    prev_layer = inputs
    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
        outputs, (state_fw, state_bw) = bidirectional_dynamic_rnn(cell_fw,cell_bw,prev_layer,dtype=dtype)
        # Concat the outputs to create the new input.
        prev_layer = tf.concat(outputs, 2)
        states_fw.append(state_fw)
        states_bw.append(state_bw)
    return prev_layer, tuple(states_fw), tuple(states_bw)
