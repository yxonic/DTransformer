import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DTransformer(nn.Module):
    def __init__(self, n_questions, d_model=256, d_fc=512, n_heads=8, dropout=0.05):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.qa_embed = nn.Embedding(2 * n_questions + 1, d_model)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout, kq_same=False)
        self.linear_k = nn.Linear(d_model // n_heads, d_model)
        self.linear_v = nn.Linear(d_model // n_heads, d_model)
        self.know_params = nn.Parameter(torch.empty(1, 1, d_model))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

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

        beta = torch.matmul(
            key,
            q_emb.view(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.n_heads)
        alpha = torch.softmax(beta, -1)
        h = torch.matmul(alpha, value).view(bs, seqlen, -1)

        return h

    def predict(self, q, s):
        # for qa embedding
        qa = s * self.n_questions + q

        # set prediction mask
        q[s < 0] = 0
        qa[s < 0] = 0

        q_emb = self.q_embed(q)
        qa_emb = self.qa_embed(qa)
        h = self(q_emb, qa_emb)
        return h, self.out(torch.cat([q_emb, h], dim=-1)).squeeze(-1)

    def get_loss(self, q, s):
        _, logits = self.predict(q, s)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="sum"
        )


class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, values, peek_cur=False):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :]

        # apply transformer layer
        query_ = self.masked_attn_head(query, key, values, mask)
        query = query + self.dropout(query_)
        return self.layer_norm(query)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
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

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(
            q,
            k,
            v,
            mask,
            self.gammas,
        )

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, mask, gamma=None):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # include temporal effect
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage

    # calculate output
    output = torch.matmul(scores, v)
    return output
