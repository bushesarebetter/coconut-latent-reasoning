import argparse
import copy
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

VARS = list("abcdefghijklmnopqrstuvwxyz")
VOCAB = ["<pad>", "<eos>", "<bot>", "<eot>", "<lat>"] + VARS + [str(i) for i in range(10)] + list("=+-*;?#")
STOI = {s: i for i, s in enumerate(VOCAB)}
OPS = {"+": lambda x, c: (x + c) % 10, "-": lambda x, c: (x - c) % 10, "*": lambda x, c: (x * c) % 10}


def make_example(rng, hops=4):
    """'a = 3 ; b = a + 2 ; c = b * 4 ; ... ? e' with `hops` operations, all mod 10.
    The CoT is one token per intermediate value, in order ('5 0 9'); the last value is the answer."""
    names = rng.sample(VARS, hops + 1)
    val = rng.randrange(10)
    question = [names[0], "=", str(val), ";"]
    steps = []
    for prev, cur in zip(names, names[1:]):
        op, c = rng.choice("+-*"), rng.randrange(1, 10)
        val = OPS[op](val, c)
        question += [cur, "=", prev, op, str(c), ";"]
        steps.append([str(val)])
    return question + ["?", names[-1]], steps[:-1], str(val)


def build(question, steps, answer, mode, k=0):
    """Token sequence and the index of the first supervised token.
    nocot:   Q # ans <eos>
    cot:     Q <bot> step_1 .. step_n <eot> # ans <eos>
    coconut: Q <bot> <lat>*k step_k+1 .. step_n <eot> # ans <eos>   (the <lat> slots get hidden states)
    so cot is stage 0 of the curriculum and at the last stage only <eot> # ans <eos> is left to write."""
    if mode == "nocot":
        return [STOI[t] for t in question + ["#", answer, "<eos>"]], len(question)
    prompt = question + ["<bot>"] + ["<lat>"] * k
    target = [t for s in steps[k:] for t in s] + ["<eot>", "#", answer, "<eos>"]
    return [STOI[t] for t in prompt + target], len(prompt)


def batch(rng, n, mode, k=0, hops=4):
    # every example of a given (mode, k) has the same length, so no padding is needed
    examples = [make_example(rng, hops) for _ in range(n)]
    seqs = [build(*ex, mode, k) for ex in examples]
    return torch.tensor([s for s, _ in seqs]), seqs[0][1], [ex[2] for ex in examples]


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, past=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        S = k.shape[2]
        mask = torch.ones(T, S, dtype=torch.bool).tril(S - T)  # causal, offset by the cached prefix
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x)), (k, v)


class GPT(nn.Module):
    def __init__(self, d=64, layers=3, heads=4, max_len=96):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(VOCAB), bias=False)
        # keep nn.Embedding's N(0, 1) init: with GPT-2's tiny 0.02 init the position signal is swamped by the
        # fed-back hidden states (norm ~sqrt(d)), and thoughts past the second lose track of which step they are

    def hidden(self, emb, cache=None):
        """Last-layer hidden states (after the final LayerNorm, i.e. what the LM head sees) for input
        embeddings `emb` that continue the sequence held in the key/value `cache`."""
        start = 0 if cache is None else cache[0][0].shape[2]
        x = emb + self.pos.weight[start : start + emb.shape[1]]
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if cache is None else cache[i])
            new_cache.append(kv)
        return self.ln_f(x), new_cache










