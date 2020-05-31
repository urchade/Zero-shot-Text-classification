import torch
from torch import nn


class Arch2(nn.Module):
    def __init__(self, num_emb, dim_emb, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(num_emb, dim_emb, pad_idx)
        self.rnn = nn.LSTM(dim_emb * 2, dim_emb, batch_first=True)
        self.linear = nn.Linear(dim_emb, 1)

    def forward(self, text_ids, tag_id):
        emb_text = self.embedding(text_ids)  # (batch_size, seq_len, emb_dim)
        emb_tag = self.embedding(tag_id)  # (batch_size, emb_dim)

        seqs = []
        for i in range(emb_text.size()[1]):
            concat = torch.cat([emb_text[:, i, :], emb_tag], dim=1)  # (batch_size, emb_dim*2)
            seqs.append(concat)

        seqs = torch.stack(seqs)  # (seq_len, batch_size, emb_dim*2)
        seqs = seqs.transpose(0, 1)  # (batch_size, seq_len, emb_dim*2)

        _, (h, _) = self.rnn(seqs)
        h = h.squeeze()  # (batch_size, emb_dim)

        out = self.linear(h)  # (batch_size, 1)
        return torch.sigmoid(out)
