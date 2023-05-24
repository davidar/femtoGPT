#!/usr/bin/env python3

import json
import numpy as np
import scipy.stats as stats
import safetensors.numpy

from encoder import get_encoder

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

model_size: str = "124M"
models_dir: str = "models"
encoder = get_encoder(model_size, models_dir)
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
w_ln = params["ln_f.weight"]
b_ln = params["ln_f.bias"]


def gelu(x):
    return x * stats.norm.cdf(x)


def normalise_rows(x):
    return x / np.sum(x, axis=-1, keepdims=True)


def standardise_rows(x):
    return (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)


def gpt2(inputs):
    n_seq = len(inputs)
    x = wte[inputs] + wpe[:n_seq]
    mask = np.tri(n_seq, dtype=x.dtype)
    for b in range(n_layer):
        w_attn1 = (
            params[f"h.{b}.ln_1.weight"][:, None] * params[f"h.{b}.attn.c_attn.weight"]
        )
        b_attn1 = (
            params[f"h.{b}.ln_1.bias"] @ params[f"h.{b}.attn.c_attn.weight"]
            + params[f"h.{b}.attn.c_attn.bias"]
        )
        w_attn2 = params[f"h.{b}.attn.c_proj.weight"]
        b_attn2 = params[f"h.{b}.attn.c_proj.bias"]
        w_mlp1 = (
            params[f"h.{b}.ln_2.weight"][:, None] * params[f"h.{b}.mlp.c_fc.weight"]
        )
        b_mlp1 = (
            params[f"h.{b}.ln_2.bias"] @ params[f"h.{b}.mlp.c_fc.weight"]
            + params[f"h.{b}.mlp.c_fc.bias"]
        )
        w_mlp2 = params[f"h.{b}.mlp.c_proj.weight"]
        b_mlp2 = params[f"h.{b}.mlp.c_proj.bias"]

        qkv = standardise_rows(x) @ w_attn1 + b_attn1
        attn = np.zeros((n_seq, n_embd), dtype=x.dtype)
        for i in range(n_head):
            q = qkv[:, D * i : D * (i + 1)]
            k = qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
            v = qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
            attn[:, D * i : D * (i + 1)] = (
                normalise_rows(np.exp(q @ k.T / np.sqrt(D)) * mask) @ v
            )
        x += attn @ w_attn2 + b_attn2
        x += gelu(standardise_rows(x) @ w_mlp1 + b_mlp1) @ w_mlp2 + b_mlp2
    return (standardise_rows(x) * w_ln + b_ln) @ wte.T


def main(
    prompt: str = "Alan Turing theorized that computers would one day become",
    n_tokens_to_generate: int = 40,
):
    print(prompt, end="", flush=True)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < n_ctx
    for _ in range(n_tokens_to_generate):
        logits = gpt2(input_ids)
        next_id = np.argmax(logits[-1])
        input_ids.append(int(next_id))
        print(encoder.decode([next_id]), end="", flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
