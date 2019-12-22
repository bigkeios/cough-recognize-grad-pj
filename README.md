# This is the repo to store what I used in my graduation project 

**Tested with:**  
python 3.6  
tensorflow 1.14  
numpy 1.17  
scipy 1.3  
scikit-learn 0.21
on Ubuntu 18.04

**To run VGGish (and data prep), checkpoint and PCA params are required:**  
[VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt)  
[Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz)  
Put them in the same directory with README

**NOTED: svm_train.py is out-dated, please use this notebook: [SVM_train.ipynb](https://colab.research.google.com/drive/1ei4g1uIxxnFNEw3ChynyXXaZe_JZnHaD)**

**To run inference (cough recognization) with single file and new record:** inference_gui.py (required tkinter)  
**To run bulk inference from tfrecord file:** svm_test.py (remember to change dir_test)
