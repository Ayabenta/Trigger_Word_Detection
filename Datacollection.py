import sounddevice as sd #used for recording our sounds and making an numpy array of itf
from scipy.io.wavfile import write # it wil take that numpy array and save it a wav file 
def record_audio_and_save(save_path, n_times): # for recording our voice that contains the wake world 
    input("To start audio recording press Enter")
    
    for i in range(100):# on listen for 100 times 
        fs = 44100
        seconds = 2 # the audio will be 2 seconds long 
        #recording using soundevice 
        myrecording  = sd.rec(int(seconds*fs), samplerate = fs, channels=2)
       #wait for the reccording to be finished 
        sd.wait()
        write(save_path+ str(i)+".wav", fs, myrecording) # save the recorded audio
        input(f"Press to record next or press ctrl +C ({i+1}/n_times)")
def record_background_sound(save_path, n_times): # for recording background sounds, so that our model can classify the background sound from the sound that conatins the wake word 
    input("To start recording your background sounds press Enter: ")
    for i in range(100):
        fs = 44100
        seconds = 2 
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
    
        write(save_path + str(i) + ".wav", fs, myrecording)
        print(f"Currently on {i+1}/{n_times}")

# Step 1: Record yourself saying the Wake Word
print("Recording the Wake Word:\n")
record_audio_and_save("audio_data/",100) 
print("Recording the Background sounds:\n")
record_background_sound("background_sound/", n_times=100)