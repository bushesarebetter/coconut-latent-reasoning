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


def run(model, ids, k=0):
    """Forward pass over a batch. With k > 0 the k <lat> slots after <bot> are continuous thoughts: the
    input at each slot is the last hidden state of the previous position, so they are computed one at a
    time (k extra sequential steps, reusing the key/value cache). Returns the input embeddings that were
    actually used, the hidden states, and the cache."""
    emb = model.tok(ids)
    if k == 0:
        return emb, *model.hidden(emb)
    p0 = (ids[0] == STOI["<bot>"]).nonzero().item() + 1
    h, cache = model.hidden(emb[:, :p0])
    inputs, hs = [emb[:, :p0]], [h]
    for _ in range(k):
        thought = hs[-1][:, -1:]
        h, cache = model.hidden(thought, cache)
        inputs.append(thought)
        hs.append(h)
    if ids.shape[1] > p0 + k:
        h, cache = model.hidden(emb[:, p0 + k :], cache)
        inputs.append(emb[:, p0 + k :])
        hs.append(h)
    return torch.cat(inputs, 1), torch.cat(hs, 1), cache


def train(model, mode, steps, k=0, bs=64, lr=3e-3, seed=0, hops=4):
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # fresh optimizer per stage
    # short warmup, then linear decay to 10% of lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * (1 - 0.9 * s / steps))
    model.train()
    for _ in range(steps):
        ids, start, _ = batch(rng, bs, mode, k, hops)
        _, h, _ = run(model, ids, k)
        # loss only on the language tokens after the prompt (question, <bot>, thoughts, <eot>)
        logits = model.head(h[:, start - 1 : -1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, start:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, mode, n=1000, k=0, seed=12345, hops=4, max_new=16):
    """Greedy decoding. Returns (answer accuracy, mean decoded tokens incl. <eos>)."""
    model.eval()
    ids, start, answers = batch(random.Random(seed), n, mode, k, hops)
    _, h, cache = run(model, ids[:, :start], k)
    out = []
    for _ in range(max_new):
        nxt = model.head(h[:, -1]).argmax(-1)
        out.append(nxt)
        h, cache = model.hidden(model.tok(nxt)[:, None], cache)
    out = torch.stack(out, 1).tolist()
    correct, lengths = 0, []
    for row, ans in zip(out, answers):
        toks = [VOCAB[i] for i in row]
        n_gen = toks.index("<eos>") + 1 if "<eos>" in toks else len(toks)
        lengths.append(n_gen)
        toks = toks[:n_gen]
        if "#" in toks and toks.index("#") + 1 < len(toks):
            correct += toks[toks.index("#") + 1] == ans
    return correct / n, sum(lengths) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    ap.add_argument("--hops", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(min(4, os.cpu_count()))
    steps, stage_steps, n_eval = (10, 5, 20) if args.quick else (1000, 1000, 1000)
    hops, results, t0 = args.hops, [], time.time()

    def report(name, model, mode, k=0):
        acc, n_tok = evaluate(model, mode, n_eval, k, hops=hops)
        results.append((name, acc, n_tok, k))
        print(f"  {name}: acc={acc:.3f} out_tokens={n_tok:.1f} latent={k}  [{time.time() - t0:.0f}s]", flush=True)

    # every variant gets the same number of optimizer steps: Coconut is `steps` of CoT plus the stages
    total = steps + (hops - 1) * stage_steps
    print("no-CoT")
    nocot = GPT()
    train(nocot, "nocot", total, hops=hops)
    report("no-CoT", nocot, "nocot")

    print("language CoT")
    cot = GPT()
    train(cot, "cot", steps, hops=hops)
    coconut = copy.deepcopy(cot)  # stage 0 of the curriculum is plain CoT
    train(cot, "cot", total - steps, seed=100, hops=hops)
    report("CoT", cot, "cot")

    print("Coconut curriculum")
    for k in range(1, hops):  # hops-1 intermediate values, one thought each
        train(coconut, "coconut", stage_steps, k=k, seed=k, hops=hops)
        report(f"Coconut stage {k}", coconut, "coconut", k)

    print("\n| method | accuracy | output tokens | continuous thoughts |\n|---|---|---|---|")
    for name, acc, n_tok, k in [results[0], results[1], results[-1]]:
        print(f"| {name} | {acc:.1%} | {n_tok:.1f} | {k} |")


if __name__ == "__main__":
    main()
