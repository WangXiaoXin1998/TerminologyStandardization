import numpy as np
from model.config import hyper
from common.Word2ID import getWord2ID

word2id = getWord2ID(hyper['train_path'])

# 将句子转为 id
def sentence2id_txt(sentence):
    tt = np.zeros(hyper['max_len'])
    sen = sentence
    for i in range(len(sen)):
        if i < hyper['max_len']:
            if word2id.get(sen[i], -1) != -1:
                tt[i] = word2id[sen[i]]
            else:
                tt[i] = word2id['<UNK>']
    return tt

