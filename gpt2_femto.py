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

jax.experimental.compilation_cache.compilation_cache.initialize_cache("jax_cache")

encoder = get_encoder("", "")
hparams = json.loads(open("models/gpt2/config.json").read())

n_vocab = hparams["vocab_size"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
D = int(n_embd / n_head)

params = safetensors.numpy.load_file("models/gpt2/model.safetensors")
for k, v in params.items():
    params[k] = np.array(v)

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]

# centering
w_unembed = wte.T.copy()
w_unembed -= w_unembed.mean(axis=1, keepdims=True)
wte -= wte.mean(axis=1, keepdims=True)
wpe -= wpe.mean(axis=1, keepdims=True)

force_enable_layers = 4


def normalise_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class TransformerBlock:
    def __init__(self, b):
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

    def __call__(self, x, layer, head_activations):
        mask = np.tri(n_seq, dtype=x.dtype)
        qkv = normalise_rows(x) @ self.w_qkv + self.b_qkv
        attn = []
        for i in range(n_head):
            q = qkv[:, D * i : D * (i + 1)]
            k = qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
            v = qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
            A = np.exp(q @ k.T / np.sqrt(D)) * mask
            A /= np.sum(A, axis=1, keepdims=True)
            # A[A < 0.04] = 0
            # A /= np.sum(A)
            if layer < force_enable_layers:
                attn.append(A @ v)
            else:
                attn.append((A @ v) * head_activations[:, layer, i].reshape(n_seq, 1))
        x += np.hstack(attn) @ self.w_out + self.b_out
        h = normalise_rows(x) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        x += h @ self.w_mlp2 + self.b_mlp2
        # assert x.mean() < 1e-5
        return x


blocks = [TransformerBlock(b) for b in range(n_layer)]


@jax.jit
def gpt2(inputs, head_activations):
    x = wte[inputs] + wpe[:n_seq]
    for layer, block in enumerate(blocks):
        x = block(x, layer, head_activations)
    final = normalise_rows(x) * w_ln + b_ln
    logits = final @ w_unembed
    return logits


prompt_tokens = [50256] + encoder.encode(
    "When Mary and John went to the store, John gave a drink to"
    # "Alan Turing theorized that computers would one day become"
)
prompt_tokens = np.array(prompt_tokens, dtype=np.int32)

n_seq = len(prompt_tokens)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


@jax.grad
def grad_objective(head_activations, probs_ref):
    logits = gpt2(prompt_tokens, head_activations)[-1]
    probs = softmax(logits)
    return kl_divergence(probs_ref, probs) + 0.1 * np.sum(head_activations)


if __name__ == "__main__":
    probs_ref = softmax(gpt2(prompt_tokens, np.ones((n_seq, n_layer, n_head)))[-1])

    head_activations = np.ones((n_seq, n_layer, n_head))
    while True:
        head_grad = grad_objective(head_activations, probs_ref)
        head_activations -= 0.1 * head_grad
        head_activations = np.clip(head_activations, 0, 1)
        # print(head_grad)
        # head_enable = (head_activations > 0.01).astype(np.float32)
        # print(repr(head_enable))
        num_enabled = head_activations[force_enable_layers:].sum()
        total = len(prompt_tokens) * (n_layer - force_enable_layers) * n_head
        print(f"{100 * num_enabled / total:.1f}% of heads enabled")

        logits = gpt2(prompt_tokens, head_activations)[-1]

        temp = 1
        exp_logits = np.exp((logits - np.max(logits)) / temp)
        probs = exp_logits / np.sum(exp_logits)

        # top k sampling
        k = 5
        top_k = np.argsort(probs)[-1 : -k - 1 : -1]
        top_k_probs = probs[top_k]
        top_k_probs /= np.sum(top_k_probs)
        # token = np.random.choice(top_k, p=top_k_probs)

        for token, prob in zip(top_k, top_k_probs):
            print(encoder.decode([int(token)]), prob, end="; ", flush=True)
        print()
