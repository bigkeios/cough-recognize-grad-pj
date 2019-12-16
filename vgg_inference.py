from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

def extract_wav_features(f_dir):
  examples_batch = vggish_input.wavfile_to_examples(f_dir)

  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess,  'vggish_model.ckpt')
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    print(embedding_batch)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    print(postprocessed_batch)
    return postprocessed_batch
