import os 
import librosa # for reading wav files and transform it to numpy array and apply rate ? 
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout 
from sklearn.metrics import confusion_matrix, classification_report
#from PCM.PCM import plot_confusion_matrix
from tensorflow.keras.utils import to_categorical
sample = "background_sound/1.wav" 
data, sample_rate = librosa.load(sample)
plt.title("Wave form")
librosa.display.waveplot(data, sr= sample_rate)
plt.show()
mfccs= librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)
plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis ='time')
plt.show()
all_data =[]
data_path_dict = {
    0: ["background_sound/" + file_path for file_path in os.listdir("background_sound/")],
    1: ["audio_data/" + file_path for file_path in os.listdir("audio_data/")],}

for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        data, sample_rate = librosa.load(single_file)
        mfccs = librosa.feature.mfcc(y=data, sr =sample_rate, n_mfcc = 40)
        mfcc_processed = np.mean(mfccs.T, axis = 0)
        all_data.append([mfcc_processed, class_label])
    print(f"INFo : successfully preprocessed class label {class_label}")
    
df = pd.DataFrame(all_data, columns =["features", "class_label"])
df.to_pickle("final_audio_data/audio_data.csv")


df = pd.read_pickle("final_audio_data/audio_data.csv")
X = df["features"].values
X=np.concatenate(X,axis=0).reshape(len(X),40)
X = df["features"].values
X = np.concatenate(X, axis=0).reshape(len(X), 40)

y = np.array(df["class_label"].tolist())
y = to_categorical(y)

####### train test split ############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation('relu'),
    Dropout(0.5),
    Dense(256),
    Activation('relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

print("Model Score: \n")
history = model.fit(X_train, y_train, epochs=1000)
model.save("saved_model/WWD.h5")
score = model.evaluate(X_test, y_test)
print(score)

#### Evaluating our model ###########
print("Model Classification Report: \n")
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
#plot_confusion_matrix(cm, classes=["Does not have Wake Word", "Has Wake Word"])

import threading
import time
import sounddevice as sd
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

#### SETTING UP TEXT TO SPEECH ###
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    engine.endLoop()

##### CONSTANTS ################
fs = 22050
seconds = 2

model = load_model("D:\Deeplearning specialization\DeepLearning_Triggerworddetection\saved_model\WWD.h5")

##### LISTENING THREAD #########
def listener():
    while True:
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        mfcc = librosa.feature.mfcc(y=myrecording.ravel(), sr=fs, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        prediction_thread(mfcc_processed)
        time.sleep(0.001)


def voice_thread():
    listen_thread = threading.Thread(target=listener, name="ListeningFunction")
    listen_thread.start()

##### PREDICTION THREAD #############
def prediction(y):
    prediction = model.predict(np.expand_dims(y, axis=0))
    if prediction[:, 1] > 0.93:
        if engine._inLoop:
            engine.endLoop()

        speak("Hello, What can I do for you?")
        

    time.sleep(0.1)

def prediction_thread(y):
    pred_thread = threading.Thread(target=prediction, name="PredictFunction", args=(y,))
    pred_thread.start()

voice_thread()
