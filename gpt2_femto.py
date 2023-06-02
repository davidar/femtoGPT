#!/usr/bin/env python3

import json
import numpy as np
from scipy.special import erf
import safetensors.numpy

from encoder import get_encoder

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

encoder = get_encoder('', '')
hparams = json.loads(open("models/gpt2/config.json").read())

n_vocab = hparams["vocab_size"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
D = int(n_embd / n_head)

params = safetensors.numpy.load_file("models/gpt2/model.safetensors")
for k, v in params.items():
    print(k, v.shape)

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]


def normalise(x):
    return x / np.linalg.norm(x)


class TransformerBlock:
    def __init__(self, b):
        self.qkv = np.zeros((0, 3 * n_embd))
        self.b_attn1 = params[f"h.{b}.attn.c_attn.bias"]
        self.b_attn1 += params[f"h.{b}.ln_1.bias"] @ params[f"h.{b}.attn.c_attn.weight"]
        self.w_attn1 = params[f"h.{b}.attn.c_attn.weight"]
        self.w_attn1 *= params[f"h.{b}.ln_1.weight"][:, None] * np.sqrt(n_embd)
        self.b_attn2 = params[f"h.{b}.attn.c_proj.bias"]
        self.w_attn2 = params[f"h.{b}.attn.c_proj.weight"]
        self.b_mlp1 = params[f"h.{b}.mlp.c_fc.bias"]
        self.b_mlp1 += params[f"h.{b}.ln_2.bias"] @ params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 = params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 *= params[f"h.{b}.ln_2.weight"][:, None] * np.sqrt(n_embd)
        self.b_mlp2 = params[f"h.{b}.mlp.c_proj.bias"]
        self.w_mlp2 = params[f"h.{b}.mlp.c_proj.weight"]

    def __call__(self, x):
        self.qkv = np.vstack(
            [self.qkv, normalise(x - np.mean(x)) @ self.w_attn1 + self.b_attn1]
        )
        attn = np.zeros(n_embd, dtype=x.dtype)
        for i in range(n_head):
            q = self.qkv[-1, D * i : D * (i + 1)]
            k = self.qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
            v = self.qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
            A = np.exp(q @ k.T / np.sqrt(D))
            A /= np.sum(A)
            attn[D * i : D * (i + 1)] = A @ v
        x += attn @ self.w_attn2 + self.b_attn2
        h = normalise(x - np.mean(x)) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        x += h @ self.w_mlp2 + self.b_mlp2
        return x


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    n_tokens_to_generate: int = 40,
):
    tokens = encoder.encode(prompt)
    print(encoder.decode([tokens[0]]), end="", flush=True)
    total = len(tokens) + n_tokens_to_generate
    assert total < n_ctx
    blocks = [TransformerBlock(b) for b in range(n_layer)]
    for posn in range(total):
        token = tokens[posn]
        x = wte[token] + wpe[posn]
        for block in blocks:
            x = block(x)
        final = normalise(x - np.mean(x)) * w_ln + b_ln
        logits = final @ wte.T
        if posn + 1 >= len(tokens):
            tokens.append(int(np.argmax(logits)))
        print(encoder.decode([tokens[posn + 1]]), end="", flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
