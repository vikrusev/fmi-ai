from speech_recognition.audio_preparation import AudioAnalisys, construct_subtitres

filename_video = 'phone-message-jacko-its-pete.mp4'
speech_recognition_directory = 'speech_recognition/'

wav_path = 'sound_files/'
str_path = 'str_files/'
wav_path_full = speech_recognition_directory + wav_path

aa = AudioAnalisys(filename_video, wav_path_full, str_path)
aa.chunk_audio()

str_path = 'dummy_str_files/'
construct_subtitres(filename_video, str_path)
