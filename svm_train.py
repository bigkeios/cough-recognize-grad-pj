from read_tfrecord import array_from_TFRecord, build_train_data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf

def main():
  tf.compat.v1.enable_eager_execution()
  # Load data
  dir = "../audioset/train/"
  data, label = build_train_data(dir)
  # print(label)
  train_data, eval_data, train_label, eval_label = train_test_split(data, label,
  test_size=0.2, random_state=1)
  # create and train the model
  svm_model = SVC(kernel = 'linear', C=1e5)
  svm_model.fit(train_data, train_label)
  # calculate accuracy
  predicted_label = svm_model.predict(eval_data)
  print(predicted_label)
  print("Accuracy score: {0:.2f}%".format(100 * svm_model.score(eval_data, eval_label)))
  # save model trained
  model_filename = "svm_model.pkl"
  with open(model_filename, "wb") as file:
    pickle.dump((svm_model, svm_model.coef_), file)

if __name__ == "__main__":
  main()
