import numpy as np
import regex
import codecs


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = regex.sub(r"<[^>]+>", "", string)
    string = regex.sub(r"[^\s\p{Latin}']", "", string)
    # string = regex.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = regex.sub(r"\\'s", " \'s", string)
    # string = regex.sub(r"\\'ve", " \'ve", string)
    # string = regex.sub(r"n\\'t", " n\'t", string)
    # string = regex.sub(r"\\'re", " \'re", string)
    # string = regex.sub(r"\\'d", " \'d", string)
    # string = regex.sub(r"\\'ll", " \'ll", string)
    # string = regex.sub(r",", " , ", string)
    # string = regex.sub(r"!", " ! ", string)
    # string = regex.sub(r"\(", " \( ", string)
    # string = regex.sub(r"\)", " \) ", string)
    # string = regex.sub(r"\?", " \? ", string)
    # string = regex.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_train_data(source_path, target_path, MAX_LENGTH=10):
    source_sents = []
    target_sents = []

    for s, t in zip(codecs.open(source_path, 'r', 'utf-8').read().split("\n"),
                    codecs.open(target_path, 'r', 'utf-8').read().split("\n")):
        if (s and s[0] != "<") and (t and t[0] != "<"):
            slen = len(s.split())
            tlen = len(t.split())
            if slen < MAX_LENGTH and tlen < MAX_LENGTH:
                source_sents.append(clean_str(s) + " __EOS__" + (" __PAD__" * (MAX_LENGTH - slen - 1)))
                target_sents.append(clean_str(t) + " __EOS__" + (" __PAD__" * (MAX_LENGTH - tlen - 1)))

    return source_sents, target_sents


def load_test_data(source_path, target_path, MAX_LENGTH=10):
    source_sents = []
    target_sents = []

    for s, t in zip(codecs.open(source_path, 'r', 'utf-8').read().split("\n"),
                    codecs.open(target_path, 'r', 'utf-8').read().split("\n")):
        if (s and s[:4] == "<seg") and (t and t[:4] == "<seg"):
            slen = len(s.split())
            tlen = len(t.split())
            if slen < MAX_LENGTH and tlen < MAX_LENGTH:
                source_sents.append(clean_str(s) + " </S>" + (" <PAD>" * (MAX_LENGTH - slen - 1)))
                target_sents.append(clean_str(t) + " </S>" + (" <PAD>" * (MAX_LENGTH - tlen - 1)))

    return source_sents, target_sents


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
    load_train_data("corpora/train.tags.de-en.de", "corpora/train.tags.de-en.en")