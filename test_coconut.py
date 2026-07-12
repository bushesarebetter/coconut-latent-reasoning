import random
import subprocess
import sys

import torch

from coconut import GPT, STOI, VOCAB, batch, make_example, run


def solve(question):
    # independent evaluator: run the program with Python's own arithmetic
    env = {}
    for stmt in " ".join(question[:-2]).split(" ;")[:-1]:
        var, expr = stmt.strip().split(" = ")
        for name, value in env.items():
            expr = expr.replace(name, str(value))
        env[var] = eval(expr) % 10
    return env[question[-1]], env


def test_data_answers_and_steps():
    rng = random.Random(0)
    for hops in (1, 4, 6):
        for _ in range(300):
            q, steps, ans = make_example(rng, hops)
            value, env = solve(q)
            assert ans == str(value)
            assert len(steps) == hops - 1 and len(env) == hops + 1
            # the CoT lists the true intermediate values in order; the last value is the answer
            assert [int(v) for (v,) in steps] == list(env.values())[1:-1]


def test_latent_slots_hold_previous_hidden_state():
    torch.manual_seed(0)
    model = GPT().eval()
    k = 3
    ids, _, _ = batch(random.Random(0), 4, "coconut", k)
    p0 = (ids[0] == STOI["<bot>"]).nonzero().item() + 1
    assert (ids[:, p0 : p0 + k] == STOI["<lat>"]).all() and (ids[:, p0 + k] == STOI["<eot>"]).all()
    with torch.no_grad():
        inputs, h, _ = run(model, ids, k)
        tok = model.tok(ids)
        assert torch.equal(inputs[:, :p0], tok[:, :p0])
        assert torch.equal(inputs[:, p0 + k :], tok[:, p0 + k :])
        for p in range(p0, p0 + k):
            # recompute the prefix from scratch without the cache: slot p must hold the hidden state at p-1
            prev = model.hidden(inputs[:, :p])[0][:, -1]
            assert torch.allclose(inputs[:, p], prev, atol=1e-5)
            # and it is not any row of the token embedding table
            assert torch.cdist(inputs[:, p], model.tok.weight).min() > 1.0
        # the cached sequential pass matches one plain pass over the same inputs
        assert torch.allclose(h, model.hidden(inputs)[0], atol=1e-5)
        # thoughts depend on the question, unlike a fixed <lat> embedding
        assert not torch.allclose(inputs[0, p0], inputs[1, p0])


def test_lat_token_embedding_is_never_used():
    torch.manual_seed(0)
    model = GPT()
    ids, _, _ = batch(random.Random(0), 2, "coconut", 2)
    model.head(run(model, ids, 2)[1]).sum().backward()
    # slots hold hidden states, so the <lat> embedding gets no gradient...
    assert model.tok.weight.grad[STOI["<lat>"]].abs().sum() == 0
    # ...while gradient still reaches the question tokens through the thoughts and attention
    assert model.tok.weight.grad[STOI["?"]].abs().sum() > 0


def test_quick_run():
    out = subprocess.run([sys.executable, "coconut.py", "--quick"], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert "| Coconut" in out.stdout and "| CoT" in out.stdout and "| no-CoT" in out.stdout
    assert len(VOCAB) == len(STOI)
