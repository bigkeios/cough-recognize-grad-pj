from read_tfrecord import build_train_data
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import pickle

def main():
  tf.compat.v1.enable_eager_execution()
  print("Building test data...")
  # Directory of test file(s). Is to be changed
  dir_test = "../Test not in training/iso/"
  test_data, test_label = build_train_data(dir_test)
  test_data_scaled = preprocessing.normalize(test_data)
  print("Predicting... ")
  svm = pickle.load(open('svm_mix.pkl', 'rb'))
  test_predict = svm.predict(test_data_scaled)
  # print test_predict to show raw results
  print("Raw result:")
  print(test_predict)
  print("Confusion matrix:")
  print(confusion_matrix(test_label, test_predict))
  print("Classification report: ")
  print(classification_report(test_label, test_predict))

if __name__ == "__main__":
  main()
