import sounddevice as sd
from scipy.io.wavfile import write

sr = 44100
seconds = 10

myrecording = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
# Wait until recording is finished
sd.wait()
# Save as WAV file
write('output.wav', sr, myrecording)
