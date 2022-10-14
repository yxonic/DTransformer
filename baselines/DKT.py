import torch
import torch.nn as nn
import torch.nn.functional as F


class DKT(nn.Module):
    def __init__(self, n_questions, d_model=100):
        super().__init__()
        self.n_questions = n_questions
        self.d_model = d_model
        self.rnn = nn.RNN(n_questions * 2 + 1, d_model, batch_first=True)
        self.fc = nn.Linear(d_model, n_questions + 1)

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.d_model).to(self.device())
        pad_start = torch.zeros(x.size(0), 1, x.size(2)).to(self.device())
        out, _ = self.rnn(
            torch.cat([pad_start, x], dim=1),
            h0,
        )
        res = torch.sigmoid(self.fc(out))[:, :-1, :]
        return res

    def predict(self, q, s, pid=None, n=1):
        assert pid is None, "DKT does not support pid input"
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        x = (
            F.one_hot(q + s * self.n_questions, self.n_questions * 2 + 1)
            .float()
            .to(self.device())
        )
        h = self(x)
        y = (
            h[:, : h.size(1) - n + 1, :]
            .gather(-1, q[:, n - 1 :].unsqueeze(-1))
            .squeeze(-1)
        )
        return y, h

    def get_loss(self, q, s, pid=None):
        assert pid is None, "DKT does not support pid input"
        logits, _ = self.predict(q, s)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return F.binary_cross_entropy_with_logits(masked_logits, masked_labels)
