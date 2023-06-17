#!/usr/bin/env python3

from collections import namedtuple
import functools
import json
import jax
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as np
from jax.scipy.special import erf
import safetensors.flax
import einops
from tqdm import trange

from print_color import print

from encoder import get_encoder

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

# jax.config.update("jax_log_compiles", True)

jax.experimental.compilation_cache.compilation_cache.initialize_cache("jax_cache")

encoder = get_encoder("", "")
hparams = json.loads(open("models/gpt2/config.json").read())

n_vocab = hparams["vocab_size"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
D = int(n_embd / n_head)

params = safetensors.flax.load_file("models/gpt2/model.safetensors")

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]

# centering
w_unembed = wte.T.copy()
w_unembed -= w_unembed.mean(axis=-1, keepdims=True)
wte -= wte.mean(axis=-1, keepdims=True)
wpe -= wpe.mean(axis=-1, keepdims=True)


def normalise_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def print_top_k(probs, k):
    top_k = np.argsort(probs)[-1 : -k - 1 : -1]
    top_k_probs = probs[top_k]
    top_k_probs /= np.sum(top_k_probs)

    for token, prob in zip(top_k, top_k_probs):
        print(
            f"'{encoder.decode([int(token)])}' {100 * prob:.1f}%",
            end="; ",
            flush=True,
        )
    print()


n_batch = 1


TransformerBlock = namedtuple(
    "TransformerBlock",
    [
        "layer",
        "b_qkv",
        "w_qkv",
        "b_out",
        "w_out",
        "b_mlp1",
        "w_mlp1",
        "b_mlp2",
        "w_mlp2",
    ],
)


def new_block(layer):
    b_qkv = params[f"h.{layer}.attn.c_attn.bias"]
    b_qkv += params[f"h.{layer}.ln_1.bias"] @ params[f"h.{layer}.attn.c_attn.weight"]
    w_qkv = params[f"h.{layer}.attn.c_attn.weight"]
    w_qkv *= params[f"h.{layer}.ln_1.weight"][:, None] * np.sqrt(n_embd)
    b_out = params[f"h.{layer}.attn.c_proj.bias"]
    b_out -= b_out.mean(keepdims=True)
    w_out = params[f"h.{layer}.attn.c_proj.weight"]
    w_out -= w_out.mean(axis=-1, keepdims=True)
    b_mlp1 = params[f"h.{layer}.mlp.c_fc.bias"]
    b_mlp1 += params[f"h.{layer}.ln_2.bias"] @ params[f"h.{layer}.mlp.c_fc.weight"]
    w_mlp1 = params[f"h.{layer}.mlp.c_fc.weight"]
    w_mlp1 *= params[f"h.{layer}.ln_2.weight"][:, None] * np.sqrt(n_embd)
    b_mlp2 = params[f"h.{layer}.mlp.c_proj.bias"]
    b_mlp2 -= b_mlp2.mean(keepdims=True)
    w_mlp2 = params[f"h.{layer}.mlp.c_proj.weight"]
    w_mlp2 -= w_mlp2.mean(axis=-1, keepdims=True)
    return TransformerBlock(
        layer, b_qkv, w_qkv, b_out, w_out, b_mlp1, w_mlp1, b_mlp2, w_mlp2
    )


def nested_vmap(f, X, *Xs):
    if len(Xs) == 0:
        return jax.vmap(f)(X)
    else:
        return jax.vmap(lambda x: nested_vmap(functools.partial(f, x), *Xs))(X)


@jax.jit
def call_block(b, x):
    n_seq = x.shape[1]
    causal_mask = np.tri(n_seq, dtype=np.float32)
    qkv = normalise_rows(x) @ b.w_qkv + b.b_qkv
    # qkv = normalise_rows(x) * head_activations[0, :, 0, None] @ b.w_qkv + b.b_qkv
    # head_activations = np.ones((n_batch, n_seq, n_head))
    Q, K, V = np.split(qkv, 3, axis=-1)
    sig = "batch posn (head D) -> head batch posn D"
    dims = {"batch": n_batch, "posn": n_seq, "head": n_head, "D": D}
    qs = einops.rearrange(Q, sig, **dims)
    ks = einops.rearrange(K, sig, **dims)
    vs = einops.rearrange(V, sig, **dims)

    def attention(batch, posn, head, raw=False):
        q = qs[head, batch, posn]
        k = ks[head, batch]
        v = vs[head, batch]
        A = np.exp(q @ k.T / np.sqrt(D)) * causal_mask[posn]
        A /= np.sum(A, axis=-1, keepdims=True)
        return A if raw else A @ v

    attn = nested_vmap(
        attention, np.arange(n_batch), np.arange(n_seq), np.arange(n_head)
    )
    attn = einops.rearrange(attn, "batch posn head D -> batch posn (head D)", **dims)
    x += attn @ b.w_out + b.b_out

    cache_attn = nested_vmap(
        functools.partial(attention, raw=True),
        np.arange(n_batch),
        np.arange(n_seq),
        np.arange(n_head),
    )

    h = normalise_rows(x) @ b.w_mlp1 + b.b_mlp1
    # h *= scipy.stats.norm.cdf(h)  # gelu
    h *= (1 + erf(h / np.sqrt(2))) / 2
    x += h @ b.w_mlp2 + b.b_mlp2

    return x, cache_attn


blocks = [new_block(layer) for layer in range(n_layer)]


def gpt2(prompt_tokens, output_batch=0):
    n_seq = prompt_tokens.shape[0]
    x = wte[prompt_tokens] + wpe[:n_seq]
    x = np.stack([x] * n_batch)
    cache_attn = []
    for block in blocks:
        x, cache_attn_layer = call_block(block, x)
        cache_attn.append(cache_attn_layer)
        # assert x.mean() < 1e-5
    final = normalise_rows(x[output_batch]) * w_ln + b_ln
    logits = final @ w_unembed
    return logits, cache_attn


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def main():
    prompt_tokens = encoder.encode(
        "Alan Turing theorized that computers would one day become"
    )
    tokens = prompt_tokens[:]
    print(encoder.decode(tokens), end="", flush=True)
    total = len(tokens) + 40
    assert total < n_ctx
    for posn in range(len(prompt_tokens) - 1, total):
        logits, cache_attn = gpt2(np.array(tokens + [0] * (total - len(tokens))))
        logits = logits[posn]
        token = int(np.argmax(logits))

        if posn + 1 >= len(tokens):
            tokens.append(token)
            # for token, prob in zip(top_k, top_k_probs):
            #     print(encoder.decode([token]), prob, end="; ", flush=True)
        print(encoder.decode([tokens[posn + 1]]), end="", flush=True)


if __name__ == "__main__":
    main()
