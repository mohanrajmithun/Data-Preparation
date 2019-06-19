import pyaudio
import wave
from array import array
import numpy as np
FORMAT=pyaudio.paInt16
CHANNELS=2
RATE=44100
CHUNK=1024
RECORD_SECONDS=5
FILE_NAME="nRECORDING.wav"

audio=pyaudio.PyAudio() #instantiate the pyaudio

#recording prerequisites
stream=audio.open(format=FORMAT,channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK)
import librosa
import numpy as np
#starting recording
frames=[]
x1,sr1 = librosa.load('rec.wav')

mfccs1 = np.mean(librosa.feature.mfcc(y=x1, sr=sr1, n_mfcc=40).T,axis=0)
mfccs1=np.reshape(mfccs1,(1,40))
from sklearn.metrics.pairwise import cosine_similarity


for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
    # data=stream.read(CHUNK)
    # print(data)
    data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    frames.append(data)

    wavfile = wave.open(FILE_NAME, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(audio.get_sample_size(FORMAT))
    wavfile.setframerate(RATE)
    wavfile.writeframes(b''.join(frames))  # append frames recorded to file
    wavfile.close()
    x, sr = librosa.load(FILE_NAME)
    # print(type(x), np.shape(x))
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
    mfccs = np.reshape(mfccs, (1, 40))
    sim = cosine_similarity(mfccs, mfccs1)
    print(sim)




    # data_chunk=array('h',data)
    # print(data_chunk)
    # vol=max(data_chunk)
    # if(vol>=0):
    #     print("something said")
    #     frames.append(data)
    # else:
    #     print("nothing")
    # print("\n")

# frames=np.array(frames)
# print(frames)
# print(np.shape(frames))

stream.stop_stream()
stream.close()
audio.terminate()
#writing to file
wavfile=wave.open(FILE_NAME,'wb')
wavfile.setnchannels(CHANNELS)
wavfile.setsampwidth(audio.get_sample_size(FORMAT))
wavfile.setframerate(RATE)
wavfile.writeframes(b''.join(frames))#append frames recorded to file
wavfile.close()


