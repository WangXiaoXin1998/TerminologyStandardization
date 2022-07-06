import torch
import torch.nn as nn
from model.config import hyper
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cpu")


class Lstm(nn.Module):

    def __init__(self):
        super(Lstm, self).__init__()
        self.hidden_dim = hyper['lstm_hidden']
        self.max_len = hyper['max_len']
        self.dropout = nn.Dropout(0.2)

        self.embedding = nn.Embedding(hyper['num_word'], hyper['word_dim'])
        self.lstm = nn.LSTM(hyper['word_dim'], self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.f_tag1 = nn.Linear(2, 128)
        self.f_tag2 = nn.Linear(128, 1)

    def init_hidden_lstm(self, batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
                torch.randn(2, batch_size, self.hidden_dim // 2).to(device))

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

        lengths1 = (torch.sum((input_x1 != 0), dim=1).long()).cpu()
        input_x1 = self.embedding(input_x1)
        self.hidden1 = self.init_hidden_lstm(input_x1.shape[0])
        input_x1 = pack_padded_sequence(input_x1, lengths1, batch_first=True, enforce_sorted=False)
        input_x1, self.hidden1 = self.lstm(input_x1, self.hidden1)
        input_x1, _ = pad_packed_sequence(input_x1, batch_first=True, total_length=self.max_len)
        # input_x1 = self.dropout(self.emission1(input_x1))
        input_x1 = input_x1.masked_fill(mask1, 0)
        input_x1 = input_x1.sum(dim=1) / mask1_sum

        lengths2 = (torch.sum((input_x2 != 0), dim=1).long()).cpu()
        input_x2 = self.embedding(input_x2)
        self.hidden2 = self.init_hidden_lstm(input_x2.shape[0])
        input_x2 = pack_padded_sequence(input_x2, lengths2, batch_first=True, enforce_sorted=False)
        input_x2, self.hidden2 = self.lstm(input_x2, self.hidden2)
        input_x2, _ = pad_packed_sequence(input_x2, batch_first=True, total_length=self.max_len)
        # input_x2 = self.dropout(self.emission1(input_x2))
        input_x2 = input_x2.masked_fill(mask2, 0)
        input_x2 = input_x2.sum(dim=1) / mask2_sum

        dis1 = torch.pairwise_distance(input_x1, input_x2, p=1)
        dis1 = torch.unsqueeze(dis1, dim=1)
        ang = torch.cosine_similarity(input_x1, input_x2, dim=1)
        ang = torch.unsqueeze(ang, dim=1)
        input_x = torch.cat((dis1, ang), dim=1)
        input_x = self.dropout(torch.relu(self.f_tag1(input_x)))
        input_x = self.f_tag2(input_x)
        input_x = torch.squeeze(input_x)
        input_x = torch.unsqueeze(input_x, dim=0)
        return input_x