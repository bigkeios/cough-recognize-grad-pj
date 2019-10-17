from __future__ import print_function
import tensorflow as tf

def parse_TFRecord_file(dir, file_name):
  """ Convert data from TFRecord file to an array of SequenceExample

  Params:
    dir: String of the directory contains the file to be converted
    file_name: String of the file name

  Return:
    Array of SequenceExamples saved in the file
  """
  # ==DEPRECATED==
  # # Read TFRecord file with python.io record iterator
  # record_iterator = tf.python_io.tf_record_iterator(path=file_dir)

  # # Parse string read into SequenceExample
  # for string_record in record_iterator:
  #   example = tf.train.SequenceExample()
  #   example.ParseFromString(string_record)
  # # print one record
  # print(example.context.feature['labels'].int64_list)

  # ==ALSO DEPRECATED==
  # # Read TFRecord file with TFRECORDREADER
  # sess = tf.compat.v1.InteractiveSession()
  # reader = tf.TFRecordReader()
  # file_dir_queue = tf.train.string_input_producer([file_dir])
  # _, example = reader.read(file_dir_queue)

  # # Context features
  # context_features = {
  #   'video_id': tf.io.FixedLenFeature([1], dtype=tf.string),
  #   'start_time_seconds': tf.io.FixedLenFeature([1], dtype=tf.float32),
  #   'end_time_seconds': tf.io.FixedLenFeature([1], dtype=tf.float32),
  #   'labels': tf.io.VarLenFeature(dtype=tf.int64)
  # }

  # # Sequence features
  # sequence_features = {
  #   'audio_embedding': tf.io.VarLenFeature(dtype=tf.string)
  # }

  # # Read context and sequence data of one record (first one)
  # context_data, sequence_data = tf.io.parse_single_sequence_example(
  #   serialized=example,
  #   context_features=context_features,
  #   sequence_features=sequence_features)

  # tf.train.start_queue_runners(sess)

  # # Print the context data
  # print('\nContext: ')
  # for name, tensor in context_data.items():
  #   print('{}: {}'.format(name, tensor.eval()))

  # # Print the sequence data
  # print('\nData: \n')
  # for name, tensor in sequence_data.items():
  #   print('{}: {}'.format(name, tensor.eval()))

  # ==RECOMMENDED IN 1.4 VERSION==
  # Read TFRecord file with tf.data.TFRECORDDATASET
  raw_data = tf.data.TFRecordDataset(dir+file_name)

  # Context features
  context_features = {
    # 'video_id': tf.io.FixedLenFeature([1], dtype=tf.string),
    # 'start_time_seconds': tf.io.FixedLenFeature([1], dtype=tf.float32),
    # 'end_time_seconds': tf.io.FixedLenFeature([1], dtype=tf.float32),
    'labels': tf.io.VarLenFeature(dtype=tf.int64)
  }

  # Sequence features
  sequence_features = {
    'audio_embedding': tf.io.VarLenFeature(dtype=tf.string)
  }

  def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    parsed_example = tf.io.parse_single_sequence_example(
      example_proto, 
      context_features, 
      sequence_features)
    return parsed_example

  parsed_data = raw_data.map(_parse_function)
  return parsed_data
  
def extract_data_by_label(parsed_data, label, output_dir, file_name):
  """Extract SequenceExamples by their label (can extract Examples NOT having a label)

    Params:
      parsed_data: Array of SequenceExamples to extract from
      label: Int value of the label
      output_dir: String of the directory to write the output file having one extracted 
      Example to
      file_name: String of the file's name

    Return: None
  """
  count = 0
  for parsed_record in parsed_data:
    # PARSED_RECORD FORMART: 
    # ({'labels': <tensorflow.python.framework.sparse_tensor (tf.sparse.Sparse.Tensor)>,
    # 'start_time_seconds': <tf.Tensor>, 'video_id': <tf.Tensor>,
    # 'end_time_seconds': <tf.Tensor>}, 
    # {'audio_embedding': <tensorflow.python.framework.sparse_tensor>})
    label_value_np = parsed_record[0]['labels'].values.numpy()
    # video_id = parsed_record[0]['video_id']
    embeddings_np = parsed_record[1]['audio_embedding'].values.numpy()
    # tf.print("Labels: {} /n Embedding: {}".format(label_value_np, embeddings_np))
    if label in label_value_np:
      context = {
        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=label_value_np))
      }
      featureList = tf.train.FeatureList(
        feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=embeddings_np))]
        )

      sequence_example = tf.train.SequenceExample(
        context = tf.train.Features(feature=context), 
        feature_lists = tf.train.FeatureLists(
          feature_list={
          'audio_embedding': featureList
        }))
      sequence_example_str = sequence_example.SerializeToString()  
      # write the sequence example
      file_name_split = file_name.split(".")
      name_only = file_name_split[0]
      output_file = "bal_"+name_only + "_" + str(count) + ".tfrecord"
      with tf.io.TFRecordWriter(output_dir + output_file) as writer:
        writer.write(sequence_example_str)
    count = count + 1 

def build_train_data_from_TFRecord(dir, file_name, cough_positive):
  """ Build the array of training data for SVM

  Params: 
    dir: String of the directory contains the file 
    file_name: String of the file's name
    cough_positive: Boolean indicating the data is cough or not

  Return:
    Array of training data with each example on one line
    Array of label corresponding to the data
  """
  data = []
  label = []
  parsed_data = parse_TFRecord_file(dir, file_name)
  for parsed_record in parsed_data:
    data.append(parsed_record[1]['audio_embedding'].values.numpy())
    label_value_np = parsed_record[0]['labels'].values.numpy()
    if 47 in label_value_np:
      label.append(0)
    else:
      label.append(1)
  return data, label

def main():
  tf.compat.v1.enable_eager_execution()
  # Directory of data from AudioSet
  tfrecord_dir = "../audioset/audioset_v1_embeddings/bal_train/"
  # Data file to extract from
  tfrecord_file = "1P.tfrecord"
  parsed_data = parse_TFRecord_file(tfrecord_dir, tfrecord_file)
  # Directory to put extracted data
  output_dir = "../audioset/cough/"
  label = 47
  extract_data_by_label(parsed_data, label, output_dir, tfrecord_file)

if __name__ == "__main__":
    main()
