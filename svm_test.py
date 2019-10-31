from read_tfrecord import build_train_data
from sklearn import preprocessing
from sklearn.metrics import classification_report
import tensorflow as tf
import pickle

def main():
  tf.compat.v1.enable_eager_execution()
  print("Building test data...")
  dir_test = "../audioset/test/"
  test_data, test_label = build_train_data(dir_test)
  test_data_scaled = preprocessing.scale(test_data)
  print("Predicting... ")
  svm, svm_coef = pickle.load(open('svm_model.pkl', 'rb'))
  test_predict = svm.predict(test_data_scaled)
  print("Classification report: ")
  print(classification_report(test_label, test_predict))

if __name__ == "__main__":
  main()
