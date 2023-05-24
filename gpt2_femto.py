#!/usr/bin/env python3

import numpy as np
import scipy.stats as stats

from utils import load_encoder_hparams_and_params

model_size: str = "124M"
models_dir: str = "models"
encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
n_vocab = hparams["n_vocab"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]


def gelu(x):
    return x * stats.norm.cdf(x)


def normalise_rows(x):
    return x / np.sum(x, axis=-1, keepdims=True)


def standardise_rows(x):
    return (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)


def gpt2(inputs):
    wte = params["wte"]
    wpe = params["wpe"]
    g_f = params["ln_f"]["g"]
    b_f = params["ln_f"]["b"]
    dk = int(n_embd / n_head)
    n_seq = len(inputs)
    x = wte[inputs] + wpe[:n_seq]
    mask = np.tri(n_seq, dtype=x.dtype)
    for block in params["blocks"]:
        w_attn2 = block["ln_1"]["g"][:, None] * block["attn"]["c_attn"]["w"]
        b_attn2 = (
            block["ln_1"]["b"] @ block["attn"]["c_attn"]["w"]
            + block["attn"]["c_attn"]["b"]
        )
        w_attn_proj = block["attn"]["c_proj"]["w"]
        b_attn_proj = block["attn"]["c_proj"]["b"]
        w_fc2 = block["ln_2"]["g"][:, None] * block["mlp"]["c_fc"]["w"]
        b_fc2 = (
            block["ln_2"]["b"] @ block["mlp"]["c_fc"]["w"] + block["mlp"]["c_fc"]["b"]
        )
        w_proj = block["mlp"]["c_proj"]["w"]
        b_proj = block["mlp"]["c_proj"]["b"]

        qkv = standardise_rows(x) @ w_attn2 + b_attn2
        out_heads = np.zeros((n_seq, n_embd), dtype=x.dtype)
        for i in range(n_head):
            q = qkv[:, dk * i : dk * (i + 1)]
            k = qkv[:, dk * (n_head + i) : dk * (n_head + i + 1)]
            v = qkv[:, dk * (2 * n_head + i) : dk * (2 * n_head + i + 1)]
            out_heads[:, dk * i : dk * (i + 1)] = (
                normalise_rows(np.exp(q @ k.T / np.sqrt(dk)) * mask) @ v
            )
        x += out_heads @ w_attn_proj + b_attn_proj
        x += gelu(standardise_rows(x) @ w_fc2 + b_fc2) @ w_proj + b_proj
    return (standardise_rows(x) * g_f + b_f) @ wte.T


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
