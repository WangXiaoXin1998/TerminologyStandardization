import json
from model.config import hyper
import numpy as np
import pandas as pd

def getText(path):
    data = pd.read_csv(path)
    ans1 = np.array(data['org'])
    ans2 = np.array(data['stad'])
    return ans1, ans2

def getCode():
    data = pd.read_csv(hyper['code_path'])['code']
    return np.array(data)

# test
if __name__ == '__main__':
    a, b = getText('../' + hyper['train_path'])
    print(a[0:10])
    print(b[0:10])
    max_len = 0
    for i in range(len(a)):
        c = a[i]
        max_len = max(max_len, len(c))
    print(max_len)
