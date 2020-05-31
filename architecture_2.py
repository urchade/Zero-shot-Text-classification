import torch
from torch import nn


class Arch2(nn.Module):
    def __init__(self, num_emb, dim_emb, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(num_emb, dim_emb, pad_idx)
        self.rnn = nn.LSTM(dim_emb, dim_emb, batch_first=True)
        self.linear = nn.Linear(2 * dim_emb, 1)

    def forward(self, text_ids, tag_id):
        emb_text = self.embedding(text_ids)  # (batch_size, seq_len, emb_dim)
        emb_tag = self.embedding(tag_id)  # (batch_size, emb_dim)

        _, (rep_text, _) = self.rnn(emb_text)  # (1, batch_size, emb_dim)
        rep_text = rep_text.squeeze()  # (batch_size, emb_dim)

        concatenates = torch.cat([rep_text, emb_tag], dim=1)  # (batch_size, 2*emb_dim)
        out = self.linear(concatenates)  # (batch_size, 1)

        return torch.sigmoid(out)
