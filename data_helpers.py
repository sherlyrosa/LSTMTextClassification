import numpy as np
import pandas as pd
import nltk
import re
import time
from tensorflow.contrib import learn

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# def load_data_and_labels(incident_data_file, nonincident_data_file):
def load_data(file_path, sw_path=None, min_frequency=0, max_length=0, vocab_processor=None, shuffle=True):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :param max_length: the max document length
    :param vocab_processor: the predefined vocabulary processor
    :param shuffle: whether to shuffle the data
    :return data, labels, lengths, vocabulary processor
    """
    # with open(file_path, 'r', encoding='utf-8') as f:
    print('Building dataset ...')
    start = time.time()
    # incsv = csv.reader(f)
    # header = next(incsv)  # Header
    # label_idx = header.index('label')
    # content_idx = header.index('content')

    labels = []
    sentences = []

    if sw_path is not None:
        sw = _stop_words(sw_path)
    else:
        sw = None

    labelsMapping = {'Non-KejadianPenting': 0,
                 'Jalan-Rusak': 1,
                 'Internet': 2, 'PDAM': 3, 'Mati-Listrik': 4}
    lines = [line.strip() for line in open(file_path)]

    for idx in range(0, len(lines), 3):
        sent = lines[idx]
        sent = sent.lower()
        sent = clean_str(sent)
        sentences.append(sent)
        labels.append(labelsMapping[lines[idx + 1]])


        # for line in incsv:
        #     sent = line[content_idx].strip()

        #     if language == 'ch':
        #         sent = _tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
        #     elif language == 'en':
        #         sent = sent.lower()
        #     else:
        #         raise ValueError('language should be one of [ch, en].')

        #     sent = _clean_data(sent, sw, language=language)  # Remove stop words and special characters

        #     if len(sent) < 1:
        #         continue

        #     if language == 'ch':
        #         sent = _word_segmentation(sent)
        #     sentences.append(sent)

        #     if int(line[label_idx]) < 0:
        #         labels.append(2)
        #     else:
        #         labels.append(int(line[label_idx]))

    labels = np.array(labels)
    # Real lengths
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in sentences])))

    if max_length == 0:
        max_length = max(lengths)

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(sentences)))
    else:
        data = np.array(list(vocab_processor.transform(sentences)))

    data_size = len(data)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]
        lengths = lengths[shuffle_indices]

    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
    print('Max document length: {}\n'.format(vocab_processor.max_document_length))
    
    return data, labels, lengths, vocab_processor
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # incident_examples = list(open(incident_data_file, "r", encoding="UTF-8").readlines())
    # incident_examples = [s.strip() for s in incident_examples]
    # nonincident_examples = list(open(nonincident_data_file, "r", encoding="UTF-8").readlines())
    # nonincident_examples = [s.strip() for s in nonincident_examples]
    # # Split by words
    # x_text = incident_examples + nonincident_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # # Generate labels
    # incident_labels = [[0, 1] for _ in incident_examples]
    # nonincident_labels = [[1, 0] for _ in nonincident_examples]
    # y = np.concatenate([incident_labels, nonincident_labels], 0)
    # return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)
if __name__ == "__main__":
    inc_dir = "data/rt-polaritydata/KejadianPenting"
    non_dir = "data/rt-polaritydata/NonKejadianPenting"

    load_data_and_labels(inc_dir, non_dir)
