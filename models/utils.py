from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

import config

cfg = SimpleNamespace(**vars(config))


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(MLPRegressor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, d_model, token_dim, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.w_q = nn.Parameter(torch.Tensor(d_model + token_dim, d_model))
        self.w_k = nn.Parameter(torch.Tensor(d_model + token_dim, d_model))
        self.w_v = nn.Parameter(torch.Tensor(d_model, d_model))

        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)

    def forward(self, query, key, value, mask=None):
        Q = torch.matmul(query, self.w_q)
        K = torch.matmul(key, self.w_k)
        V = torch.matmul(value, self.w_v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            Q.size(-1)
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae_loss = nn.L1Loss()

    def forward(self, output, target):
        return self.mae_loss(output, target)
