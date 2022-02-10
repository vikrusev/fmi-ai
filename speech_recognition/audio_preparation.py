from pydub import AudioSegment
from pydub.silence import detect_silence
import moviepy.editor as mp
import os


def miliseconds_to_time_str(miliseconds):
    # Transforms minisecond to a time string
    # in format hh:mm:ss

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


def get_name(filename):
    # Gets the name of a file without the extension

    return filename.split('.')[0]


def get_audio_file(filename, sample_rate):
    # Gets an pydub audio segment from a file

    sound_file = AudioSegment.from_wav(filename)
    sound_file = sound_file.set_frame_rate(sample_rate)
    return sound_file


def get_audio(filename_audio, filename_video):
    # Extracts the audio from a video file

    my_video = mp.VideoFileClip(filename_video)
    my_video.audio.write_audiofile(filename_audio)


def create_emty_directory(dir_name):
    # Creates an empty directory
    # If a directory with the same name exists, deletes its content

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        for x in os.listdir(dir_name):
            os.remove(dir_name + x)


class AudioAnalisys:
    # Prepares an audio segments from a video to be estimated

    def __init__(self, video_file, wav_path, str_path, sample_rate=16000):
        name = get_name(video_file)
        self.sound_format = 'wav'
        self.str_format = 'str'
        self.filename = name + '-sound.' + self.sound_format
        get_audio(self.filename, video_file)

        self.wav_path = wav_path
        self.str_path = str_path
        create_emty_directory(self.wav_path)
        create_emty_directory(self.str_path)

        self.audio = get_audio_file(self.filename, sample_rate)

    def __del__(self):
        os.remove(self.filename)

    def generate_str_file(self, chunk_number, time):
        # Generates a str file template for an audio file

        chunk_name = f'chunk{chunk_number:03}'
        str_file = f'{self.str_path}{chunk_name}.{self.str_format}'
        f = open(str_file, 'w')
        f.write(f'{chunk_number + 1}\n')
        begin = miliseconds_to_time_str(time[0])
        end = miliseconds_to_time_str(time[1])
        f.write(f'{begin} --> {end}\n')
        f.close()

    def generate_wav_file(self, chunk_number, time):
        # Extract a part form an audio file (from:to time)

        chunk_name = f'chunk{chunk_number:03}'
        out_file = f'{self.wav_path}{chunk_name}.{self.sound_format}'
        chunk = self.audio[time[0]:time[1]]
        chunk.export(out_file, format=self.sound_format)

    def chunk_audio(self, min_silence_len=500, silence_thresh=-40):
        # Detects the silence parts of an audio file
        # and extracts the different speach/noide parts between the pauses

        silences = detect_silence(self.audio,
                                  min_silence_len=min_silence_len,
                                  silence_thresh=silence_thresh)

        audio_chunks_times = [(0 if i == 0 else silences[i - 1][1], silences[i][0])
                              for i in range(len(silences))]

        for i, time in enumerate(audio_chunks_times):
            self.generate_wav_file(i, time)
            self.generate_str_file(i, time)


def construct_subtitres(video_file, str_path):
    filename = get_name(video_file) + '.str'
    file_write = open(filename, 'w')
    for file in os.listdir(str_path):
        with open(str_path + file) as file_read:
            for line in file_read.readlines():
                file_write.write(line)
            file_write.write('\n')
    file_write.close()
