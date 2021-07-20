import numpy as np
import pandas as pd
import nltk
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z!\?\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r"\.", " \. ", string)
    # string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    print(string)
    return string.strip().lower()


def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path)]
    for idx in range(0, len(lines), 3):
        # id = lines[idx].split("\t")[0]
        classIn = lines[idx + 1]
        sentence = lines[idx]

        sentence = clean_str(sentence)

        data.append([sentence, classIn])

    df = pd.DataFrame(data=data, columns=["sentence", "classIn"])
    labelsMapping = {'Non-KejadianPenting': 0,
                     'Lalu-Lintas': 1,
                     'Kebakaran': 2, 'Bencana-Alam': 3}
    df['label'] = [labelsMapping[r] for r in df['classIn']]

    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']
    # print('var y = ', y)
    labels_flat = y.values.ravel()
    # print('after ravel = ', labels_flat)
    labels_count = np.unique(labels_flat).shape[0]

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    print(labels)
    return x_text, labels


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


if __name__ == "__main__":
    trainFile = 'TRAIN_FILE.TXT'
    testFile = 'TEST_FILE_FULL.TXT'

    load_data_and_labels(trainFile)
