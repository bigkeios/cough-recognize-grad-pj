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

  # ==RECOMMENDED IN tf 1.14 VERSION==
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
      # might need to add checking length of audio extracted (end_time - start time)
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

def array_from_TFRecord(dir, file_name):
  """ Extract data from TFRecord to an array

  Params: 
    dir: String of the directory contains the file 
    file_name: String of the file's name

  Return:
    Array of training data with each example on one line
    Array of label corresponding to the data
  """
  data = []
  label = []
  parsed_data = parse_TFRecord_file(dir, file_name)
  for parsed_record in parsed_data:
    embeddings_np = parsed_record[1]['audio_embedding'].values.numpy()
    # convert embedding of one record in bytestring to int (unsigned) and append it
    int_embeddings =[]
    # each element in embeddings array corresponds to a second of the audio
    for embedding in embeddings_np:
      hexembed = embedding.hex()
      #128-element embeddings for one second is added
      int_embeddings.extend([int(hexembed[i:i+2],16) for i in range(0,len(hexembed),2)])
    data.append(int_embeddings)
    # append class's label
    label_value_np = parsed_record[0]['labels'].values.numpy()
    if 47 in label_value_np:
      label.append(1)
    else:
      label.append(-1)
  return data, label

def build_train_data(dir):
  """ Build training data array from data saved in a folder

    Params:
      dir: String of directory storing traing data
    Return:
      Array of training data with each example on one row
      Array of label corresponding with training data
  """
  folder = os.fsencode(dir)
  data = []
  label = []
  for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith('.tfrecord'):
      data_from_file, label_from_file = array_from_TFRecord(dir, filename)
      data.extend(data_from_file)
      label.extend(label_from_file)
  return data, label

def main():
  tf.compat.v1.enable_eager_execution()
  # Directory of data from AudioSet. Name is to be changed
  tfrecord_dir = "../audioset/audioset_v1_embeddings/bal_train/"
  # Data file to extract from. Name is to be changed
  tfrecord_file = "0q.tfrecord"
  parsed_data = parse_TFRecord_file(tfrecord_dir, tfrecord_file)
  # Directory to put extracted data. Name is to be changed
  output_dir = "../audioset/not cough/"
  label = 47
  extract_data_by_label(parsed_data, label, output_dir, tfrecord_file)

if __name__ == "__main__":
  main()

