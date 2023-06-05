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
w_unembed -= w_unembed.mean(axis=1, keepdims=True)
wte -= wte.mean(axis=1, keepdims=True)
wpe -= wpe.mean(axis=1, keepdims=True)

force_enable_layers = 0

prompt_tokens = [50256] + encoder.encode(
    "When Mary and John went to the store, John gave a drink to"
    # "Alan Turing theorized that computers would one day become"
)
prompt_tokens = np.array(prompt_tokens, dtype=np.int32)

n_seq = len(prompt_tokens)
causal_mask = np.tri(n_seq, dtype=np.float32)

# important_tokens = np.array(encoder.encode(" Mary them John the her"))

analyse = False
analyse_posn = 0
analyse_heads = []


def normalise_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


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
        self.w_out -= self.w_out.mean(axis=1, keepdims=True)
        self.b_mlp1 = params[f"h.{b}.mlp.c_fc.bias"]
        self.b_mlp1 += params[f"h.{b}.ln_2.bias"] @ params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 = params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 *= params[f"h.{b}.ln_2.weight"][:, None] * np.sqrt(n_embd)
        self.b_mlp2 = params[f"h.{b}.mlp.c_proj.bias"]
        self.b_mlp2 -= self.b_mlp2.mean(keepdims=True)
        self.w_mlp2 = params[f"h.{b}.mlp.c_proj.weight"]
        self.w_mlp2 -= self.w_mlp2.mean(axis=1, keepdims=True)

    def attention(self, threshold, pure, i, q, k, v, h):
        A = np.exp(q @ k.T / np.sqrt(D)) * causal_mask
        A /= np.sum(A, axis=1, keepdims=True)
        A *= A > threshold
        A /= np.sum(A, axis=1, keepdims=True)

        if analyse and ((self.layer, i) in analyse_heads or analyse_posn == n_seq - 1):
            print(
                f"{self.layer}.{i}",
                end=": ",
                flush=True,
                format="bold" if (self.layer, i) in analyse_heads else None,
            )
            for token, amt in zip(
                prompt_tokens[1 : analyse_posn + 1],
                A[analyse_posn, 1 : analyse_posn + 1],
            ):
                token = int(token)
                if amt > 0:
                    print(
                        encoder.decode([token]),
                        end="",
                        flush=True,
                        colour="green"
                        if amt > 0.5
                        else "yellow"
                        if amt > 0.1
                        else "red",
                    )
                else:
                    print(encoder.decode([token]), end="", flush=True)

            # logit lens just for the output of this head
            x_head = self.x_copy
            x_head += (A @ v) @ self.w_out[D * i : D * (i + 1), :] + self.b_out
            logits = (normalise_rows(x_head) * w_ln + b_ln) @ w_unembed
            probs = softmax(logits[analyse_posn])
            print("| ", end="", flush=True)
            print_top_k(probs, 5)

        if pure or self.layer < force_enable_layers:
            return A @ v
        else:
            return (A @ v) * h

    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(self, pure_x, impure_x, head_activations, threshold):
        if analyse:
            self.x_copy = pure_x.copy()

        qkv = normalise_rows(pure_x) @ self.w_qkv + self.b_qkv
        Q, K, V = np.split(qkv, 3, axis=1)
        qs = np.split(Q, n_head, axis=1)
        ks = np.split(K, n_head, axis=1)
        vs = np.split(V, n_head, axis=1)
        hs = np.split(head_activations[:, self.layer, :], n_head, axis=1)
        qkv_impure = normalise_rows(impure_x) @ self.w_qkv + self.b_qkv
        Q_impure, K_impure, V_impure = np.split(qkv_impure, 3, axis=1)
        qs_impure = np.split(Q_impure, n_head, axis=1)
        ks_impure = np.split(K_impure, n_head, axis=1)
        vs_impure = np.split(V_impure, n_head, axis=1)
        attn = np.hstack(
            [
                np.vstack(
                    [
                        self.attention(threshold, True, i, qs[i], ks[i], vs[i], hs[i])[
                            :-1
                        ],
                        self.attention(
                            threshold, True, i, qs_impure[i], ks[i], vs[i], hs[i]
                        )[-1],
                    ]
                )
                if self.layer == 9 and (i == 6 or i == 9)
                else self.attention(threshold, True, i, qs[i], ks[i], vs[i], hs[i])
                for i in range(n_head)
            ]
        )
        pure_x += attn @ self.w_out + self.b_out

        attn = np.hstack(
            [
                self.attention(threshold, self.layer >= 9, i, *args)
                for i, args in enumerate(zip(qs, ks, vs, hs))
            ]
        )
        impure_x += attn @ self.w_out + self.b_out

        if analyse and analyse_posn == n_seq - 1:
            # logit lens
            logits = (normalise_rows(x) * w_ln + b_ln) @ w_unembed
            probs = softmax(logits[analyse_posn])
            print(f"Layer {self.layer} prediction midway: ", end="", flush=True)
            print_top_k(probs, 5)

        h = normalise_rows(pure_x) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        pure_x += h @ self.w_mlp2 + self.b_mlp2

        h = normalise_rows(impure_x) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        impure_x += h @ self.w_mlp2 + self.b_mlp2

        if analyse and analyse_posn == n_seq - 1:
            # logit lens
            logits = (normalise_rows(x) * w_ln + b_ln) @ w_unembed
            probs = softmax(logits[analyse_posn])
            print(f"Layer {self.layer} prediction updated to: ", end="", flush=True)
            print_top_k(probs, 5)
        return pure_x, impure_x


blocks = [TransformerBlock(b) for b in range(n_layer)]

warmup = True


def gpt2(head_activations, threshold):
    x = wte[prompt_tokens] + wpe[:n_seq]
    pure_x = x.copy()
    impure_x = x.copy()
    for block in tqdm(blocks) if warmup else blocks:
        pure_x, impure_x = block(pure_x, impure_x, head_activations, threshold)
        # assert x.mean() < 1e-5
    final = normalise_rows(pure_x) * w_ln + b_ln
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
    logits = gpt2(head_activations, 0)[-1]
    return (
        logits[int(encoder.encode(" Mary")[0])]
        - logits[int(encoder.encode(" John")[0])]
    )


if __name__ == "__main__":
    head_activations = np.ones((n_seq, n_layer, n_head), dtype=np.float32)
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

    jax.config.update("jax_disable_jit", True)
    analyse = True

    sensitivity = grad_logit_diff(head_activations)
    # logits = gpt2(head_activations, 0)[-1]
    # probs = softmax(logits)
    # print_top_k(probs, 5)
    # print(sensitivity[-1])
    warmup = False

    for i in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[i])]), end=" ")
        analyse_posn = i
        analyse_heads = []
        for j in range(n_layer):
            for k in range(n_head):
                if abs(sensitivity[i, j, k]) > 0.05:
                    print(
                        f"{j}.{k} -- {sensitivity[i, j, k]:.2f}",
                        # end=" ",
                        colour="green"
                        if abs(sensitivity[i, j, k]) > 1.0
                        else "yellow"
                        if abs(sensitivity[i, j, k]) > 0.5
                        else "red",
                    )
                    analyse_heads.append((j, k))
        print()
        # if len(analyse_heads) > 0:
        #     gpt2(head_activations, 0.04)
