import torch
import torch.nn as nn
from model.config import hyper
from pytorch_transformers import BertModel
import numpy as np
import torch.nn.functional as F
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda")


class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()
        self.max_len = hyper['max_len']
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.bert = BertModel.from_pretrained(hyper['bert_path'])
        self.f_tag1 = nn.Linear(2, 64)
        self.f_tag2 = nn.Linear(64, 1)

    def forward(self, input_x1, input_x2):
        input_x1 = torch.unsqueeze(input_x1, dim=0)
        mask1_sum = (input_x1 != 0).type(torch.bool)
        mask2_sum = (input_x2 != 0).type(torch.bool)
        mask1_sum = mask1_sum.sum(dim=1)
        mask2_sum = mask2_sum.sum(dim=1)
        mask1_sum = torch.unsqueeze(mask1_sum, dim=1)
        mask2_sum = torch.unsqueeze(mask2_sum, dim=1)
        mask1 = (input_x1 == 0).type(torch.bool)
        mask2 = (input_x2 == 0).type(torch.bool)
        mask1 = torch.unsqueeze(mask1, dim=2)
        mask2 = torch.unsqueeze(mask2, dim=2)

        mask = (input_x1 != 0).type(torch.long)
        input_x1 = self.bert(input_x1, mask)[0]
        mask = (input_x2 != 0).type(torch.long)
        input_x2 = self.bert(input_x2, mask)[0]

        input_x1 = input_x1.masked_fill(mask1, 0)
        input_x1 = self.dropout1(input_x1)
        input_x1 = input_x1.sum(dim=1) / mask1_sum

        input_x2 = input_x2.masked_fill(mask2, 0)
        input_x2 = self.dropout1(input_x2)
        input_x2 = input_x2.sum(dim=1) / mask2_sum

        dis1 = torch.pairwise_distance(input_x1, input_x2, p=1)
        dis1 = torch.unsqueeze(dis1, dim=1)
        ang = torch.cosine_similarity(input_x1, input_x2, dim=1)
        ang = torch.unsqueeze(ang, dim=1)
        input_x = torch.cat((dis1, dis1, ang), dim=1)
        input_x = self.dropout2(torch.relu(self.f_tag1(input_x)))
        input_x = self.f_tag2(input_x)
        input_x = torch.squeeze(input_x)
        input_x = torch.unsqueeze(input_x, dim=0)
        return input_x





