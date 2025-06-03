from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=768, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Thêm dropout để tăng khả năng khái quát
        self.dropout = nn.Dropout(dropout)

        # Thay GRU bằng LSTM để có cell state
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Thêm lớp pre-output để tăng khả năng biểu diễn
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, y, h_0=None):
        y = self.embedding(y)  # (B, U, Embed_dim)
        y = self.dropout(y)

        if h_0 is not None:
            if isinstance(h_0, tuple):  # LSTM
                y, (h_n, c_n) = self.rnn(y, h_0)
            else:  # Tương thích ngược với code cũ (GRU)
                h_0_lstm = (h_0, torch.zeros_like(h_0))
                y, (h_n, c_n) = self.rnn(y, h_0_lstm)
        else:
            # random state sampling
            h_0 = torch.randn(1, y.size(0), self.hidden_dim, device=y.device)
            c_0 = torch.randn(1, y.size(0), self.hidden_dim, device=y.device)
            y, (h_n, c_n) = self.rnn(y, (h_0, c_0))

        y = self.pre_output(y)
        return y, (h_n, c_n)
