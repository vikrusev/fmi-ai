from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import moviepy.editor as mp


def miliseconds_to_time_str(miliseconds):
    hours, minutes, seconds = 0, 0, 0
    clock_parts = 60
    seconds = miliseconds / 1000
    if seconds > clock_parts:
        minutes = seconds // clock_parts
        seconds %= clock_parts
    if minutes > clock_parts:
        hours = minutes // clock_parts
        seconds %= clock_parts

    seconds_str = f'{0 if seconds < 10 else ""}{seconds:.3f}'
    return f'{hours:02}:{minutes:02}:{seconds_str}'


def generate_str_file(chunk_number, time, path):
    chunk_name = f'chunk{chunk_number}'
    str_file = f'{path}/{chunk_name}.str'
    f = open(str_file, 'w')
    f.write(f'{chunk_number + 1}\n')
    begin = miliseconds_to_time_str(time[0])
    end = miliseconds_to_time_str(time[1])
    f.write(f'{begin} --> {end}\n')
    f.close()


def generate_wav_file(chunk_number, time, path, sound_file):
    chunk_name = f'chunk{chunk_number}'
    out_file = f'{path}/{chunk_name}.wav'
    print("exporting", out_file)
    chunk = sound_file[time[0]:time[1]]
    chunk.export(out_file, format='wav')


def chunk_audio(filename, wav_path, str_path):

    sound_file = AudioSegment.from_wav(filename)
    sound_file = sound_file.set_frame_rate(8000)
    silences = detect_silence(sound_file, min_silence_len=500, silence_thresh=-40)

    audio_chunks_times = [(0 if i == 0 else silences[i - 1][1], silences[i][0])
                          for i in range(len(silences))]

    for i, time in enumerate(audio_chunks_times):
        generate_wav_file(i, time, wav_path, sound_file)
        generate_str_file(i, time, str_path)


def get_audio(filename_video, filename_audio):
    my_clip = mp.VideoFileClip(filename_video)
    my_clip.audio.write_audiofile(filename_audio)
