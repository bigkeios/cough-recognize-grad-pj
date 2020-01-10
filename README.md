# Posts and Telecommunications Institute of Technology - IT Graduation Project (D15 2015-2020)

**Tested with:**  
python 3.6  
tensorflow 1.14  
numpy 1.17  
scipy 1.3  
scikit-learn 0.21
on Ubuntu 18.04

**This project uses VGGish so model checkpoints and PCA params are required (for post-processing) when running inference or training SVM:**  
[VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt)  
[Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz)  
Put them in the same directory with README  
More details about the model can be found [here](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)

**NOTED: svm_train.py is out-dated, please use this notebook: [SVM_train.ipynb](https://colab.research.google.com/drive/1ei4g1uIxxnFNEw3ChynyXXaZe_JZnHaD)**

**To run inference (cough recognization) with single files and new records:** inference_gui.py (required tkinter)  
**To run bulk inference from tfrecord file:** svm_test.py (remember to change dir_test)
