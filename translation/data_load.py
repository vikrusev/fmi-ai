from os.path import join as path_join

from collections import Counter

# load dataset from file
def read_dataset(path):
    input_file = path_join(path)

    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


# read vocabularies
def load_data(vocabulary_path, sentences):
    for key in sentences.keys():
        sentences[key] = read_dataset(path_join(vocabulary_path, 'vocab_{}'.format(key)))


# print first 2 tuples of sentences
def sample_sentences(sentences):
    for sample_i in range(2):
        for key in sentences.keys():
            print('vocab_{} Line {}: {}'.format(key, sample_i + 1, sentences[key][sample_i]))

    print()


# count and print total, unique and 10 most common words
def complexity_of_data(sentences):
    for key in sentences.keys():
        word_counter = Counter([word for sentence in sentences[key] for word in sentence.split()])

        print('{} {} words.'.format(key, len([word for sentence in sentences[key] for word in sentence.split()])))
        print('{} unique {} words.'.format(len(word_counter), key))
        print('10 Most common words in the {} dataset:'.format(key))
        print('"' + '" "'.join(list(zip(*word_counter.most_common(10)))[0]) + '"')

        print()