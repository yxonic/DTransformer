import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_SEQ_LEN = 5


class DTransformer(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=256,
        d_fc=512,
        n_heads=8,
        dropout=0.05,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

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

        self.dropout_rate = dropout

    def forward(self, q_emb, s_emb, lens):
        hq = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True)
        hs = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True)

        # AKT
        # return self.block3(hq, hq, hs, peek_cur=False)

        query = self.know_params.expand_as(hq)
        h = self.block3(query, hq, hs, lens, peek_cur=False)

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

    def predict(self, q, s, pid=None):
        lens = (s >= 0).sum(dim=1)

        # set prediction mask
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)

            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff

        h = self(q_emb, s_emb, lens)
        y = self.out(torch.cat([q_emb, h], dim=-1)).squeeze(-1)

        if pid is not None:
            return y, h, (p_diff**2).sum() * 1e-5
        else:
            return y, h, 0.0

    def get_loss(self, q, s, pid=None):
        logits, _, reg_loss = self.predict(q, s, pid)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )

    def get_cl_loss(self, q, s, pid=None):
        bs = s.size(0)
        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            # skip CL for batches that are too short
            return self.get_loss(q, s, pid)

        masked_labels = s[s >= 0].float()

        # augmentation
        q_ = q.clone()
        s_ = s.clone()
        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        for b in range(bs):
            # manipulate score
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_[b, i] = 1 - s_[b, i]
            # manipulate order
            idx = random.sample(
                range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        # model
        logits_1, h_1, reg_loss_1 = self.predict(q, s, pid)
        masked_logits_1 = logits_1[s >= 0]

        logits_2, h_2, reg_loss_2 = self.predict(q, s, pid)
        masked_logits_2 = logits_2[s >= 0]

        reg_loss = (reg_loss_1 + reg_loss_2) / 2

        # CL loss
        input = F.cosine_similarity(
            h_1[:, None, :minlen, :], h_2[None, :, :minlen, :], dim=-1
        )
        target = torch.arange(s.size(0))[:, None].expand(-1, minlen)
        cl_loss = F.cross_entropy(input, target)

        # prediction loss
        pred_loss_1 = F.binary_cross_entropy_with_logits(
            masked_logits_1, masked_labels, reduction="mean"
        )
        pred_loss_2 = F.binary_cross_entropy_with_logits(
            masked_logits_2, masked_labels, reduction="mean"
        )
        pred_loss = (pred_loss_1 + pred_loss_2) / 2

        # TODO: weights
        return cl_loss * 0.1 + pred_loss + reg_loss


class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, values, lens, peek_cur=False):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :]

        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1)

            for b in range(query.size(0)):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0

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
        v_ = attention(
            q,
            k,
            v,
            mask,
            self.gammas,
        )

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

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
