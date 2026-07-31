# coconut-latent-reasoning

A small reimplementation of Coconut from [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) (Hao et al., 2024). Instead of a pretrained LLM, it uses a 3-layer, 64-dim GPT trained from scratch on a synthetic multi-step arithmetic task. In Coconut, the model's last hidden state goes straight back in as the next input embedding, so it can take reasoning steps without writing them out as tokens.

Each example is a chain of 4 operations mod 10:

```
question:  k = 7 ; q = k * 3 ; b = q + 8 ; x = b - 4 ; m = x * 6 ; ? m
CoT:       <bot> 1 9 5 <eot> # 0 <eos>
Coconut:   <bot> [h] [h] [h] <eot> # 0 <eos>
```

I train three variants:

- no-CoT: question → `# answer`
- language CoT: question → the three intermediate values → answer
- Coconut: start from the CoT model, then run a curriculum where stage k swaps the first k written values for k continuous thoughts. A thought is the last-layer hidden state at the previous position, so each thought needs its own sequential forward step (a KV cache keeps that cheap). Loss is only on the language tokens that remain.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python coconut.py          # full run, about 13 min on my CPU while it was busy with other jobs
python coconut.py --quick  # a few seconds, smoke test only
```

## Results

All three variants get the same 4000 optimizer steps (batch 64, fresh samples every step). Accuracy is greedy decoding on 1000 held-out problems, and output tokens counts every token the model decodes after the prompt, including `<eos>`:

| method | accuracy | output tokens | continuous thoughts |
|---|---|---|---|
| no-CoT | 16.2% | 3.0 | 0 |
| language CoT | 100.0% | 7.0 | 0 |
| Coconut (stage 3) | 99.7% | 4.0 | 3 |

Without intermediate steps the model is barely above chance (10%). Written CoT solves the task completely. Coconut does just as well (99.3%, 100%, 99.7% after stages 1–3) while writing only `<eot> # answer <eos>`, so all of the intermediate values are carried in hidden states.

I had to change two things from the paper before this worked:

- In intermediate stages, the remaining written steps sit before `<eot>` (`<bot> [h] 9 5 <eot> # 0`), so CoT is literally stage 0 and every thought replaces a token at the same position. The final stage is the paper's format.
- Positional embeddings use PyTorch's default N(0, 1) init. With GPT-2's 0.02 init, the fed-back hidden states (norm around √d) drown out the position signal, and in my runs the third thought never learned.

## Test

```
python -m pytest -q
```

The tests check that the synthetic problems and CoT values match an independent evaluator, and that each latent slot holds exactly the hidden state from the position before it. They recompute that hidden state without the cache, check that it isn't a token embedding, and check that the `<lat>` embedding gets no gradient. They also check that `--quick` runs end to end.
