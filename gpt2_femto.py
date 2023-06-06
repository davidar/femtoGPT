#!/usr/bin/env python3

import functools
import json
import jax
import jax.numpy as np
from jax.scipy.special import erf
import safetensors.numpy
from tqdm import tqdm

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

params = safetensors.numpy.load_file("models/gpt2/model.safetensors")
for k, v in params.items():
    params[k] = np.array(v)

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]

# centering
w_unembed = wte.T.copy()
w_unembed -= w_unembed.mean(axis=-1, keepdims=True)
wte -= wte.mean(axis=-1, keepdims=True)
wpe -= wpe.mean(axis=-1, keepdims=True)

force_enable_layers = 0

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

prompt_tokens = [np.array([50256] + encoder.encode(s), dtype=np.int32) for s in prompts]

n_seq = 15
causal_mask = np.tri(n_seq, dtype=np.float32)

for toks in prompt_tokens:
    assert toks.size == n_seq


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


n_streams = 2


class TransformerBlock:
    def __init__(self, b):
        self.layer = b
        self.b_qkv = params[f"h.{b}.attn.c_attn.bias"]
        self.b_qkv += params[f"h.{b}.ln_1.bias"] @ params[f"h.{b}.attn.c_attn.weight"]
        self.w_qkv = params[f"h.{b}.attn.c_attn.weight"]
        self.w_qkv *= params[f"h.{b}.ln_1.weight"][:, None] * np.sqrt(n_embd)
        self.b_out = params[f"h.{b}.attn.c_proj.bias"]
        self.b_out -= self.b_out.mean(keepdims=True)
        self.w_out = params[f"h.{b}.attn.c_proj.weight"]
        self.w_out -= self.w_out.mean(axis=-1, keepdims=True)
        self.b_mlp1 = params[f"h.{b}.mlp.c_fc.bias"]
        self.b_mlp1 += params[f"h.{b}.ln_2.bias"] @ params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 = params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 *= params[f"h.{b}.ln_2.weight"][:, None] * np.sqrt(n_embd)
        self.b_mlp2 = params[f"h.{b}.mlp.c_proj.bias"]
        self.b_mlp2 -= self.b_mlp2.mean(keepdims=True)
        self.w_mlp2 = params[f"h.{b}.mlp.c_proj.weight"]
        self.w_mlp2 -= self.w_mlp2.mean(axis=-1, keepdims=True)

    def attention(self, threshold, q, k, v, h):
        A = np.exp(q @ k.T / np.sqrt(D)) * causal_mask
        A /= np.sum(A, axis=-1, keepdims=True)
        A *= A > threshold
        A /= np.sum(A, axis=-1, keepdims=True)
        return (A @ v) * h

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(self, x, head_activations, threshold):
        qkv = normalise_rows(x) @ self.w_qkv + self.b_qkv
        Q, K, V = np.split(qkv, 3, axis=-1)
        qs = np.split(Q, n_head, axis=-1)
        ks = np.split(K, n_head, axis=-1)
        vs = np.split(V, n_head, axis=-1)
        hs = np.split(head_activations[:, :, self.layer, :], n_head, axis=-1)
        attn = np.stack(
            [
                np.hstack(
                    [
                        self.attention(
                            threshold,
                            np.vstack(
                                [
                                    qs[i][0, :-5],
                                    qs[i][1, -5],
                                    qs[i][0, -4:],
                                ]
                            ),
                            ks[i][0],
                            vs[i][0],
                            hs[i][0],
                        )
                        if (self.layer == 5 and i == 5)
                        or (self.layer == 5 and i == 8)
                        or (self.layer == 5 and i == 9)
                        or (self.layer == 6 and i == 9)
                        else self.attention(
                            threshold, qs[i][0], ks[i][0], vs[i][0], hs[i][0]
                        )
                        for i in range(n_head)
                    ]
                ),
                np.hstack(
                    [
                        self.attention(
                            threshold, qs[i][0], ks[i][0], vs[i][0], hs[i][1]
                        )
                        for i in range(n_head)
                    ]
                ),
            ]
        )
        x += attn @ self.w_out + self.b_out

        h = normalise_rows(x) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        x += h @ self.w_mlp2 + self.b_mlp2

        return x


blocks = [TransformerBlock(b) for b in range(n_layer)]

warmup = True


def gpt2(which_prompt, head_activations, threshold):
    x = wte[prompt_tokens[which_prompt]] + wpe[:n_seq]
    x = np.stack([x] * n_streams)
    for block in tqdm(blocks) if warmup else blocks:
        x = block(x, head_activations, threshold)
        # assert x.mean() < 1e-5
    final = normalise_rows(x[0]) * w_ln + b_ln
    logits = final @ w_unembed
    return logits


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def kl_divergence(p, q):
    # p = p[important_tokens]
    # q = q[important_tokens]
    return np.sum(p * np.log(p / q))


@jax.grad
def grad_objective(head_activations, probs_ref):
    logits = gpt2(head_activations, 0)[-1]
    probs = softmax(logits)
    return (
        kl_divergence(probs_ref, probs)
        + 10 * np.sum(head_activations) / head_activations.size
    )


@jax.grad
def grad_logit_diff(head_activations):
    sum = 0
    for p in range(len(prompts)):
        logits = gpt2(p, head_activations, 0)[-1]
        correct, incorrect = answers[p]
        sum += (
            logits[int(encoder.encode(correct)[0])]
            - logits[int(encoder.encode(incorrect)[0])]
        )
    return sum / len(prompts)


if __name__ == "__main__":
    head_activations = np.ones((n_streams, n_seq, n_layer, n_head), dtype=np.float32)
    """
    probs_ref = softmax(gpt2(head_activations, 0)[-1])
    for i in range(250):
        print(f"Step {i}", end="; ")
        head_grad = grad_objective(head_activations, probs_ref)
        head_activations -= 5 * (1 - i / 250) * head_grad
        head_activations = np.clip(head_activations, 0, 1)
        # print(head_grad)
        # head_enable = (head_activations > 0.01).astype(np.float32)
        # print(repr(head_enable))
        num_enabled = head_activations[force_enable_layers:].sum()
        total = head_activations.size - force_enable_layers * n_head
        print(f"{100 * num_enabled / total:.1f}% total activation", end="; ")

        head_enable = (head_activations > 0.1).astype(np.float32)
        num_enabled = head_enable[force_enable_layers:].sum()
        print(f"{100 * num_enabled / total:.1f}% of heads enabled", end="; ")

        probs = softmax(gpt2(head_enable, 0.04)[-1])
        print(f"KL divergence: {kl_divergence(probs_ref, probs)}")
        print_top_k(probs, 5)

        warmup = False
    """

    # jax.config.update("jax_disable_jit", True)
    # analyse = True

    sensitivity = grad_logit_diff(head_activations)
    # logits = gpt2(head_activations, 0)[-1]
    # probs = softmax(logits)
    # print_top_k(probs, 5)
    # print(sensitivity[-1])
    warmup = False

    for i in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[0][i])]), end=" ")
        analyse_posn = i
        analyse_heads = []
        absmax = max(np.abs(sensitivity[1, i, :, :]).max(), 0.5)
        for j in range(n_layer):
            for k in range(n_head):
                s = np.abs(sensitivity[1, i, j, k]) / absmax
                if s > 0.1:
                    print(
                        f"{j}.{k} -- {sensitivity[1, i, j, k]:.2f}",
                        # end=" ",
                        colour="green" if s > 0.5 else "yellow" if s > 0.25 else "red",
                    )
                    analyse_heads.append((j, k))
        print()
        # if len(analyse_heads) > 0:
        #     gpt2(head_activations, 0.04)
