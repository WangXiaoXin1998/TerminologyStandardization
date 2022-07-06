from common.Data2ID import getCodeID, getCode
from common.Sentence2ID import sentence2id_txt
from model.config import hyper
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model.Lstm import Lstm
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

code = getCodeID()
code = (Variable(torch.tensor(code)).long()).to(device)

net = Lstm()
net.load_state_dict(torch.load('save_model/params8.pkl'))
net = net.to(device)
net.eval()

while True:
    test_text = input('请输入：')
    test_id = (Variable(torch.tensor(sentence2id_txt(test_text))).long()).to(device)
    with torch.no_grad():
        output = net(test_id, code)
    output = torch.argmax(output, dim=1)
    code_txt = getCode()
    print(code_txt[output])

