import numpy as np

# read and parse subtitles file
def get_data_from_file(path):
    sentences = []
    timings = []

    # 0 - index
    # 1 - time
    # 2 - sentence
    read = 0

    with open(path) as file:
        for line in file:
            line = line.strip()

            if read == 2:
                sentences.append(line)
                read = 0
            if read == 1:
                timings.append(line)
                read = 2
            if line.isdigit():
                read = 1

    return sentences, timings


# write indexes, timings and translations to file
# overwrite file if exists
def build_translated_file(srt_file_path, predictions, timings, y_tk):
    y_id_to_word = { value: key for key, value in y_tk.word_index.items() }
    y_id_to_word[0] = '<PAD>'

    with open(srt_file_path, 'w') as srt_file:
        for i in range(0, len(predictions)):
            srt_file.write(str(i + 1) + '\n')
            srt_file.write(timings[i] + '\n')
            srt_file.write(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[i]]) + '\n')
            srt_file.write('\n')