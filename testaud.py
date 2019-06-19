# import pyaudio
# import wave
#
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# RECORD_SECONDS = 5
# WAVE_OUTPUT_FILENAME = "output.wav"
#
# p = pyaudio.PyAudio()
#
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)
#
# print("* recording")
#
# frames = []
#
# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)
#
# print("* done recording")
#
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# print(type(frames))

import librosa
import numpy as np

x,sr = librosa.load('samp_audio.wav')
print(type(x),np.shape(x))

mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T,axis=0)

# print(type(x))
# print(x[0:10])
# print(mfccs[0])
# print(np.shape(mfccs))

x1,sr1 = librosa.load('rec.wav')

mfccs1 = np.mean(librosa.feature.mfcc(y=x1, sr=sr1, n_mfcc=40).T,axis=0)
mfccs=np.reshape(mfccs,(1,40))
mfccs1=np.reshape(mfccs1,(1,40))
# print(np.shape(mfccs1))
# print(mfccs)
# print(mfccs1)
from sklearn.metrics.pairwise import cosine_similarity

sim=cosine_similarity(mfccs,mfccs1)
print(sim)

stft = np.abs(librosa.stft(x))
stft1 = np.abs(librosa.stft(x1))

chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
chroma1 = np.mean(librosa.feature.chroma_stft(S=stft1, sr=sr1).T,axis=0)
print(np.shape(chroma),np.shape(chroma1))

chroma=np.reshape(chroma,(1,12))

chroma1=np.reshape(chroma1,(1,12))

sim=cosine_similarity(chroma,chroma1)
print(sim)
# play=pyaudio.PyAudio()
# stream_play=play.open(format=FORMAT,
#                       channels=CHANNELS,
#                       rate=RATE,
#                       output=True)
# for data in frames:
#     stream_play.write(data)
# stream_play.stop_stream()
# stream_play.close()
# play.terminate()