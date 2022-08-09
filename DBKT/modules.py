import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import device


class Architecture(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.05):
        super().__init__()
        self.n_heads = n_heads
        self.block1 = TransformerLayer(d_model, n_heads, dropout)
        self.block2 = TransformerLayer(d_model, n_heads, dropout)
        self.block3 = TransformerLayer(d_model, n_heads, dropout, kq_same=False)
        self.linear_k = nn.Linear(d_model // n_heads, d_model)
        self.linear_v = nn.Linear(d_model // n_heads, d_model)
        self.know_params = nn.Parameter(torch.empty(1, 1, d_model))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)

    def forward(self, q_emb, qa_emb):
        hq = self.block1(q_emb, q_emb, q_emb, peek_cur=True)
        ha = self.block2(qa_emb, qa_emb, qa_emb, peek_cur=True)
        query = self.know_params.expand_as(hq)
        h = self.block3(query, hq, ha, peek_cur=False)

        bs, seqlen, d_model = hq.size()

        key = torch.sigmoid(
            self.linear_k(
                self.know_params.expand(bs, seqlen, -1).view(
                    bs, seqlen, self.n_heads, d_model // self.n_heads
                )
            )
        ).view(bs * seqlen, self.n_heads, -1)
        value = torch.sigmoid(
            self.linear_v(h.view(bs, seqlen, self.n_heads, d_model // self.n_heads))
        ).view(bs * seqlen, self.n_heads, -1)

        beta = torch.bmm(key, q_emb.view(bs * seqlen, -1, 1)).view(
            bs * seqlen, 1, self.n_heads
        )
        alpha = torch.softmax(beta, -1)
        h = torch.bmm(alpha, value).view(bs, seqlen, -1)

        return h


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, dropout, kq_same)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, values, peek_cur=False):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=peek_cur)
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to()
        if peek_cur:
            query_ = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False
            )
        else:
            query_ = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True
            )

        query = query + self.dropout(query_)
        return self.layer_norm(query)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, zero_pad, self.dropout, self.gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, zero_pad, dropout, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = (
            torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        )
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )
    scores = scores * total_effect

    scores.masked_fill(mask == 0, -1e23)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
