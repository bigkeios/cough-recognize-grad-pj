from tkinter import filedialog
from tkinter import *
import sounddevice as sd
from scipy.io.wavfile import write
import os.path
from sklearn import preprocessing
import pickle
from tensorflow.compat.v1 import enable_eager_execution
from read_tfrecord import array_from_TFRecord
from vgg_inference import extract_wav_features

def svm_predict_file(f_dir):
  data, label = array_from_TFRecord(f_dir)
  data = preprocessing.normalize(data)
  svm = pickle.load(open('svm_mix.pkl', 'rb'))
  return svm.predict(data)

def svm_predict_embedding(embedding_arr):
  embedding_arr = preprocessing.normalize(embedding_arr.reshape(1, -1))
  svm = pickle.load(open('svm_mix.pkl', 'rb'))
  return svm.predict(embedding_arr)

def process_predict_wav(f_dir):
  # Use VGGish to extract embedding
  emb = extract_wav_features(f_dir)
  # Predict w/ svm
  # Flatten bc emb is 2-D (10x128)
  return svm_predict_embedding(emb.flatten())

class MainApp(Frame):
  def __init__(self, master):
    super().__init__(master)
    self.master = master
    master.title("Cough recognize demo")
    master.minsize(300, 200)
    self.pack()
    self.load_widgets()

  def load_widgets(self):
    self.label = Label(self, text="Choose the audio source: ")
    self.label.pack()

    self.file_button = Button(self, text="From existed audio file(s)", 
                                command=self.open_file_frm)
    self.file_button.pack()

    self.record_button = Button(self, text="From new recording", command=self.record_frm)
    self.record_button.pack()

  def open_file_frm(self):
    def select_file():
      open_file_frm.f_dir = filedialog.askopenfilename(
      initialdir = "/home/phuong/",
      title = "Select file",
      filetypes = (("wav files","*.wav"),("tfrecord files","*.tfrecord"), ("all files","*.*"))
      )
      if type(open_file_frm.f_dir) != tuple:
        # Process directory -> file name and directory leads to it (in relation w
        # folder running this)
        # f_dir = os.path.dirname(open_file_frm.f_dir)
        f_name = os.path.basename(open_file_frm.f_dir)
        rel_file_dir = os.path.relpath(open_file_frm.f_dir)
        # Start predicting
        if f_name.endswith('.tfrecord'):
          predict = svm_predict_file(rel_file_dir)
          self.print_predict(predict)
        elif f_name.endswith('.wav'):
          predict = process_predict_wav(rel_file_dir)
          self.print_predict(predict)
        else:
          f_error_frm = Toplevel(open_file_frm)
          f_error_frm.title("File error")
          f_error_frm.label = Label(f_error_frm, 
                                    text="Please select wav or tfrecord file", 
                                    fg="red")
          f_error_frm.label.pack()

      open_file_frm.destroy()

    open_file_frm = Toplevel(self.master, width=200, height=100)
    open_file_frm.title("Select file")
    open_file_frm.select_btn = Button(open_file_frm, text="Browse file", command=select_file)
    open_file_frm.select_btn.pack()

  def record_frm(self):
    def record():
      sr = 44100
      seconds = 10

      myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=2, dtype='int16')
      # Wait until recording is finished
      sd.wait()
      # Save as WAV file
      write('output.wav', sr, myrecording)
      # Close the recording prompt and start inference
      record_frm.fin_label = Label(record_frm, text="Finished recording")
      record_frm.fin_label.pack()
      record_frm.destroy()
      predict = process_predict_wav(f_dir="output.wav")
      self.print_predict(predict)

    record_frm = Toplevel(self.master, width=200, height=100)
    record_frm.title("Record audio")
    record_frm.button = Button(record_frm, text="Press to start", command=record)
    record_frm.button.pack()
    

  def print_predict(self, predict):
    self.predict_label = Label(self)
    if 1 in predict:
      self.predict_label['text'] = "The audio contains COUGH"
    elif -1 in predict:
      self.predict_label['text'] = "The audio does NOT contain COUGH"
    else:
      self.predict_label['text'] = "There was something wrong"
    self.predict_label.pack()

enable_eager_execution()
root = Tk()
app = MainApp(master=root)
root.mainloop()
