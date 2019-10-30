from read_tfrecord import array_from_TFRecord, build_train_data
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
import tensorflow as tf

def simple_train(data, label):
  
  train_data, eval_data, train_label, eval_label = train_test_split(data_scaled, label,
  test_size=0.2, random_state=1)

  # SIMPLE INIT TRAIN AND EVAL 
  # create and fit svm model
  svm_model = SVC(kernel = 'linear', C=1e5)
  svm_model.fit(train_data, train_label)

  # evaluate on eval_data
  predicted_label = svm_model.predict(eval_data)
  print(predicted_label)
  print("Accuracy score: {0:.2f}%".format(100 * svm_model.score(eval_data, eval_label)))

def cv_train_simple(data, label):
  # TRAIN AND EVAL EACH MODEL TRAINED ON DIFFRENT SPLITS IN CROSS VALIADTION
  # create and fit svm model
  svm_model = SVC(kernel = 'linear', C=1e5)
  scores = cross_val_score(svm_model, data, label, cv=10)
  # evaluate
  print(scores)
  print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def grid_search_cv_train(data, label):
  # TRAIN AND FIND BEST HYPERPARAMS WITH CROSS VALIDATION AND GRID SEARCH
  # train and estimate best hyperparameters for SVM model
  print("Training model...")
  hyperparams = [{'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000, 1e5]},
                  {'kernel': ['linear'], 'C': [1, 10, 100, 1000, 1e5]}]
  grid_search = GridSearchCV(SVC(), hyperparams, cv=5, scoring='accuracy')
  grid_search.fit(data, label)
  # evaluate
  print("Best parameters found: ")
  print(grid_search.best_params_)
  print("Detailed scores on each set: ")
  means = grid_search.cv_results_['mean_test_score']
  stds = grid_search.cv_results_['std_test_score']
  for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

def main():
  tf.compat.v1.enable_eager_execution()
  # Load data
  dir = "../audioset/train/"
  data, label = build_train_data(dir)
  data_scaled = preprocessing.scale(data)

  # simple_train(data_scaled, label)
  # cv_train_simple(data_scaled, label)
  grid_search_cv_train(data_scaled, label)
  #save model trained
  # print("Saving model...")
  # model_filename = "svm_model.pkl"
  # with open(model_filename, "wb") as file:
  #   pickle.dump((svm_model, svm_model.coef_), file)

if __name__ == "__main__":
  main()
