#!/usr/bin/env python3

from dotenv import load_dotenv

load_dotenv()

import functools
import json
import jax
import jax.experimental.compilation_cache.compilation_cache
import jax.numpy as np
from jax.scipy.special import erf
import einops
import pandas as pd
from print_color import print
import streamlit as st
import streamlit.components.v1 as components

from encoder import get_encoder
from gpt2_weights import load_gpt2

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)

jax.experimental.compilation_cache.compilation_cache.initialize_cache("jax_cache")

encoder = get_encoder("", "")

config = json.loads(open("models/gpt2/config.json").read())
n_vocab = config["vocab_size"]
n_ctx = config["n_ctx"]
n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
D = int(n_embd / n_head)
n_batch = 1


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


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


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
        k = ks[head, batch] * causal_mask[posn, :, None]
        v = vs[head, batch] * causal_mask[posn, :, None]
        A = causal_mask[posn]
        A *= np.exp(q @ k.T / np.sqrt(D))
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


def gpt2(params, prompt_tokens, output_batch=0):
    n_seq = prompt_tokens.shape[0]
    x = params.wte[prompt_tokens] + params.wpe[:n_seq]
    x = np.stack([x] * n_batch)
    cache_attn = []
    cache_x = []
    for block in params.blocks:
        x, cache_attn_layer = call_block(block, x)
        cache_attn.append(cache_attn_layer)
        cache_x.append(x)
        # assert x.mean() < 1e-5
    final = normalise_rows(x[output_batch]) * params.w_ln + params.b_ln
    logits = final @ params.w_unembed
    return logits, cache_attn, cache_x


def main():
    st.set_page_config(layout="wide")
    params = load_gpt2()
    tokens = encoder.encode("Mr Dursley said something to Mrs Dursley")
    logits, cache_attn, cache_x = gpt2(params, np.array(tokens))
    divs = ""
    renders = ""

    st.markdown("## Logit lens")
    table = []
    for posn, token in enumerate(tokens):
        row = []
        row.append(encoder.decode([int(token)]))
        for layer in range(n_layer):
            x = cache_x[layer][0, posn, :]
            final = normalise_rows(x) * params.w_ln + params.b_ln
            logits = final @ params.w_unembed
            top_token = np.argmax(logits)
            row.append(encoder.decode([int(top_token)]))
        table.append(row)
    st.table(
        pd.DataFrame(table, columns=["Input"] + [f"Layer {i}" for i in range(n_layer)])
    )

    for layer in range(n_layer):
        params = {
            "tokens": [encoder.decode([int(t)]) for t in tokens],
            "attention": cache_attn[layer][0, :, :, :].swapaxes(0, 1).tolist(),
        }
        divs += f'<h2>Layer {layer}</h2><div id="layer-{layer}"></div>'
        renders += f'render("layer-{layer}", AttentionPatterns, {json.dumps(params)});'
    components.html(
        f"""
        {divs}
        <script crossorigin type="module">
        import {{ render, AttentionPatterns }} from "https://unpkg.com/circuitsvis@1.40.0/dist/cdn/esm.js";
        {renders}
        </script>
        """,
        height=900,
        scrolling=True,
    )


if __name__ == "__main__":
    main()
