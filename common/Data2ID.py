from common.Sentence2ID import sentence2id_txt
from common.Loader import getText, getCode
from model.config import hyper
import numpy as np

def getData2ID(path):
    X_data, y_data = getText(path)
    X_data0 = []
    for i in range(len(X_data)):
        X_data0.append(sentence2id_txt(X_data[i]))
    return X_data0, y_data

def getData2ID_Bert(path):
    from pytorch_transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(hyper['bert_path'])

    X_data, y_data = getText(path)
    X_data0, y_data0 = [], []

    for i in range(len(X_data)):
        tt1 = np.zeros(hyper['max_len'], dtype=int)
        xx1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(X_data[i]))
        if len(xx1) < hyper['max_len']:
            tt1[:len(xx1)] = xx1
        else:
            tt1 = xx1[:hyper['max_len']]
        X_data0.append(tt1)
        y_data0.append(y_data[i])
    return X_data0, y_data0

def getCodeID():
    code = getCode()
    hyper['code_len'] = len(code)
    X_code = []
    for i in range(len(code)):
        X_code.append(sentence2id_txt(code[i]))

    return X_code

def getCodeID_Bert():
    from pytorch_transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(hyper['bert_path'])
    code = getCode()
    hyper['code_len'] = len(code)
    X_code = []

    for i in range(len(code)):
        tt1 = np.zeros(hyper['max_len'], dtype=int)
        xx1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(code[i]))
        if len(xx1) < hyper['max_len']:
            tt1[:len(xx1)] = xx1
        else:
            tt1 = xx1[:hyper['max_len']]
        X_code.append(tt1)

    return X_code





