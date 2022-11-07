import pydub
import os
import moviepy.editor as mp
import speech_recognition as sr

# Accuracy is fking shit dont use

def m4a_to_wav(filepath):
    filename = os.path.splitext(filepath)[0]
    pydub.AudioSegment.from_file(filename + '.m4a').export('./tmp/' + filename + '.wav', format='wav')

def video_to_wav(filepath):
    filename = os.path.splitext(filepath)[0]
    video_clip = mp.VideoFileClip(r"" + filepath)
    video_clip.audio.write_audiofile(r"" + './tmp/' + filename + '.wav')
    if os.path.exists('./tmp/' + filename + '.wav'):
        print("Converting Success")
    else:
        print("Converting Failure")
        exit()

def speech_small_file(filepath):
    if os.path.exists(filepath):
        pass # do nothing
    else:
        print("File does not exist")
        exit()
    r = sr.Recognizer()
    with sr.AudioFile(filepath) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        print(text)


if __name__ == "__main__":
    speech_small_file('./tmp/test3.wav')