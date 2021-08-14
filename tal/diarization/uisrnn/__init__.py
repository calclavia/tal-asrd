"""
The module for Unbounded Interleaved-State Recurrent Neural Network.

An introduction is available at [README.md].

[README.md]: https://github.com/google/uis-rnn/blob/master/README.md

Source: https://github.com/google/uis-rnn
"""
from wildspeech.diarization.uisrnn import arguments, evals, utils, uisrnn

#pylint: disable=C0103
parse_arguments = arguments.parse_arguments
compute_sequence_match_accuracy = evals.compute_sequence_match_accuracy
output_result = utils.output_result
UISRNN = uisrnn.UISRNN
