"""
copied from https://github.com/HawkAaron/warp-transducer/blob/master/tensorflow_binding/warprnnt_tensorflow/__init__.py
Changes: RNN-T -> RNA.
"""
import imp
import tensorflow as tf
from tensorflow.python.framework import ops

lib_file = imp.find_module('kernels', __path__)[1]
_warprna = tf.load_op_library(lib_file)


def rna_loss(log_probs, labels, input_lengths, label_lengths, blank_label=0):
    '''Computes the RNA loss between a sequence of activations and a
    ground truth labeling.
    Args:
        log_probs: A 4-D Tensor of floats.  The dimensions
                     should be (B, T, U, V), where B is the minibatch index,
                     T is the time index, U is the prediction network sequence
                     length, and V indexes over activations for each
                     symbol in the alphabet.
        labels: A 2-D Tensor of ints, a padded label sequences to make sure
                     labels for the minibatch are same length.
        input_lengths: A 1-D Tensor of ints, the number of time steps
                       for each sequence in the minibatch.
        label_lengths: A 1-D Tensor of ints, the length of each label
                       for each example in the minibatch.
        blank_label: int, the label value/index that the RNA
                     calculation should use as the blank label
    Returns:
        1-D float Tensor, the cost of each example in the minibatch
        (as negative log probabilities).
    '''
    loss, _ = _warprna.warp_rna(log_probs, labels, input_lengths,
                                label_lengths, blank_label)
    return loss


@ops.RegisterGradient("WarpRNA")
def _RNALossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    # NOTE since here we are batch first, cannot use _BroadcastMul
    grad_loss = tf.reshape(grad_loss, (-1, 1, 1, 1))
    return [grad_loss * grad, None, None, None]


@ops.RegisterShape("WarpRNA")
def _RNALossShape(op):
    inputs_shape = op.inputs[0].get_shape().with_rank(4)
    batch_size = inputs_shape[0]
    return [batch_size, inputs_shape]
