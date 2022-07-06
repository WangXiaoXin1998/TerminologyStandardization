from model.config import hyper
from common.Loader import getText, getCode
import json

def getWord2ID(path):
    word2id = {}
    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 1
    id = 2

    data, _ = getText(path)

    for line in data:
        for word in line:
            if word2id.get(word, -1) == -1:
                word2id[word] = id
                id += 1

    data = getCode()
    for line in data:
        for word in line:
            if word2id.get(word, -1) == -1:
                word2id[word] = id
                id += 1

    hyper['num_word'] = len(word2id)
    return word2id

# test
if __name__ == '__main__':
    a = getWord2ID('../' + hyper['train_path'])
    print(a)
    print(len(a))
