from preparation import chunk_audio, get_audio
from translation.translation import translate
from models_enum import Models

import os

filename_video = 'phone-message-jacko-its-pete.mp4'
filename_audio = filename_video.split('.')[0] + '-sound.wav'

wav_path = 'sound_files'
str_path = 'str_files'

if not os.path.exists(wav_path):
    os.mkdir(wav_path)

if not os.path.exists(str_path):
    os.mkdir(str_path)


get_audio(filename_video, filename_audio)
chunk_audio(filename_audio, wav_path, str_path)

# define languages here
# the program expects to find vocabulary files in /translations/vocabulary/vocab_{lang}
# the first argument is the main language (translate_from)
translate(Models.SIMPLE, 'english', 'french')