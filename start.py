from speech_recognition.audio_preparation import AudioAnalisys, construct_subtitres

from translation.translation import translate
from models_enum import Models

filename_video = 'speech.mp4'
speech_recognition_directory = 'speech_recognition/'

wav_path = 'sound_files/'
str_path = 'str_files/'
wav_path_full = speech_recognition_directory + wav_path

aa = AudioAnalisys(filename_video, wav_path_full, str_path)
aa.chunk_audio()

# str_path = 'dummy_str_files/'
# construct_subtitres(filename_video, str_path)

# define languages here
# the program expects to find vocabulary files in /translations/vocabulary/vocab_{lang}
# the first argument is the main language (translate_from)
translate(Models.EMBEDDING_BIDIRECTIONAL, 'english', 'french')