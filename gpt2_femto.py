#!/usr/bin/env python3

import json
import jax
import jax.numpy as np
from jax.scipy.special import erf
import safetensors.numpy

from print_color import print

from encoder import get_encoder

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

encoder = get_encoder("", "")
hparams = json.loads(open("models/gpt2/config.json").read())

n_vocab = hparams["vocab_size"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
D = int(n_embd / n_head)

params = safetensors.numpy.load_file("models/gpt2/model.safetensors")
# for k, v in params.items(): print(k, v.shape)

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]

# centering
w_unembed = wte.T.copy()
w_unembed -= w_unembed.mean(axis=1, keepdims=True)
wte -= wte.mean(axis=1, keepdims=True)
wpe -= wpe.mean(axis=1, keepdims=True)


def normalise(x):
    return x / np.linalg.norm(x)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class TransformerBlock:
    def __init__(self, b):
        self.qkv = np.zeros((0, 3 * n_embd))
        self.b_qkv = params[f"h.{b}.attn.c_attn.bias"]
        self.b_qkv += params[f"h.{b}.ln_1.bias"] @ params[f"h.{b}.attn.c_attn.weight"]
        self.w_qkv = params[f"h.{b}.attn.c_attn.weight"]
        self.w_qkv *= params[f"h.{b}.ln_1.weight"][:, None] * np.sqrt(n_embd)
        self.b_out = params[f"h.{b}.attn.c_proj.bias"]
        self.b_out -= self.b_out.mean(keepdims=True)
        self.w_out = params[f"h.{b}.attn.c_proj.weight"]
        self.w_out -= self.w_out.mean(axis=1, keepdims=True)
        self.b_mlp1 = params[f"h.{b}.mlp.c_fc.bias"]
        self.b_mlp1 += params[f"h.{b}.ln_2.bias"] @ params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 = params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 *= params[f"h.{b}.ln_2.weight"][:, None] * np.sqrt(n_embd)
        self.b_mlp2 = params[f"h.{b}.mlp.c_proj.bias"]
        self.b_mlp2 -= self.b_mlp2.mean(keepdims=True)
        self.w_mlp2 = params[f"h.{b}.mlp.c_proj.weight"]
        self.w_mlp2 -= self.w_mlp2.mean(axis=1, keepdims=True)

    def __call__(self, x):
        self.qkv = np.vstack([self.qkv, normalise(x) @ self.w_qkv + self.b_qkv])
        attn = []
        for i in range(n_head):
            q = self.qkv[-1, D * i : D * (i + 1)]
            k = self.qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
            v = self.qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
            A = np.exp(q @ k.T / np.sqrt(D))
            A /= np.sum(A)
            # A[A < 0.04] = 0
            # A /= np.sum(A)
            attn.append(A @ v)
        x += np.hstack(attn) @ self.w_out + self.b_out
        h = normalise(x) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        x += h @ self.w_mlp2 + self.b_mlp2
        # assert x.mean() < 1e-5
        return x


blocks = [TransformerBlock(b) for b in range(n_layer)]


def gpt2(x):
    for block in blocks:
        x = block(x)
    final = normalise(x) * w_ln + b_ln
    logits = final @ w_unembed
    return logits


def main():
    # prompt_tokens = [50256] + encoder.encode("When Mary and John went to the store, John gave a drink to")
    prompt_tokens = encoder.encode(
        "Alan Turing theorized that computers would one day become"
    )
    tokens = prompt_tokens[:]
    print(encoder.decode([tokens[0]]), end="", flush=True)
    total = len(tokens) + 40
    assert total < n_ctx
    for posn in range(total):
        token = tokens[posn]
        x = wte[token] + wpe[posn]
        logits = gpt2(x)
        token = int(np.argmax(logits))

        temp = 0.7
        exp_logits = np.exp((logits - np.max(logits)) / temp)
        probs = exp_logits / np.sum(exp_logits)

        # top k sampling
        k = 5
        top_k = list(np.argsort(probs)[-k:])
        top_k.reverse()
        top_k_probs = probs[np.array(top_k)]
        top_k_probs /= np.sum(top_k_probs)
        # token = np.random.choice(top_k, p=top_k_probs)

        if posn + 1 >= len(tokens):
            tokens.append(token)
            # for token, prob in zip(top_k, top_k_probs):
            #     print(encoder.decode([token]), prob, end="; ", flush=True)
        print(encoder.decode([tokens[posn + 1]]), end="", flush=True)


if __name__ == "__main__":
    main()
    # for block in blocks: block.qkv = np.zeros((0, 3 * n_embd))
