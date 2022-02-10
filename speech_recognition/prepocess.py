import torch
from torch import nn
import torchaudio
from string import ascii_lowercase

_blank_code = 28


def get_order(char):
    return ord(char) - ord('a')


class DataManipulation:
    @staticmethod
    def transpose_position(item, position, contigues=False):
        item = item.transpose(position, position + 1)
        return item.contigues() if contigues else item

    @staticmethod
    def squeeze(item, squeeze=True):
        return item.squeeze(0) if squeeze else item.unsqueeze(1)

    def specgram(waveform, sample_rate=1600, n_mels=128,
                 freq_mask_param=15, time_mask_param=35):
        # Returns a spectogram of a wave

        spectogram = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                 n_mels=n_mels),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        )(waveform)
        spectogram = DataManipulation.squeeze(spectogram)
        spectogram = DataManipulation.transpose_position(spectogram, 0)

        return spectogram


class TextTransform:
    # Maps characters to integers and vice versa

    def __init__(self):
        self.char_map = {
            '\'': 0,
            ' ': 1
        }
        self.char_map.update({x: get_order(x) + 1 for x in ascii_lowercase})
        self.index_map = {value: key for key, value in self.char_map.items()}

    def text_to_int(self, text):
        # Uses a character map and converts text to an integer sequence

        return [self.char_map[c] for c in text]

    def int_to_text(self, labels):
        # Uses a character map and converts integer labels to an text sequence

        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')


def training_data_processing(data):
    text_transform = TextTransform()
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (sound, _, text, _, _, _) in data:
        spectogram = DataManipulation.specgram(sound)
        spectrograms.append(spectogram)
        label = torch.Tensor(text_transform.text_to_int(text.lower()))
        labels.append(label)
        input_lengths.append(spectogram.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    spectrograms = DataManipulation.squeeze(spectrograms, False)
    spectrograms = DataManipulation.transpose_position(spectrograms, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths
